"""Complete local OCR -> durable, compact inventory -> bound-account HTTP sync."""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.library import Library
from custom.action.inventory_reporting import AUTO_UPLOAD_MODE, TOKEN_REQUIRED_HINT, get_bound_account, read_upload_settings, resolve_token
from custom.action.star_full_local_ocr_shadow import FullShadowSession, SYNC_DIAGNOSTICS_RETENTION_VERSION
from custom.action.star_diagnostics import start_star_diagnostics
from custom.action.toolbox_diagnostics import call
from utils import logger
from utils.star_local_ocr_shadow import REPO, write

__all__ = ['StarInventorySync']
SYNC_PATH = '/open-api/star/inventory-sync'
QUALITY = {'橙': 'orange', '紫': 'purple', '蓝': 'blue', '绿': 'green', '白': 'white',
           **{q: q for q in ('orange', 'purple', 'blue', 'green', 'white')}}
MAX_LEVEL = {'orange': 60, 'purple': 60, 'blue': 50, 'green': 40, 'white': 30}
MAX_COUNT = 2_147_483_647


def compact_inventory(local: dict, scan_id: str) -> dict:
    """Preserve every positional instance; never deduplicate by attributes."""
    if str(uuid.UUID(scan_id)) != scan_id:
        raise ValueError('invalid scan_id')
    if local.get('schemaVersion') != 2 or local.get('complete') is not True:
        raise ValueError('本地扫描未完整完成，禁止同步')
    entries = []
    counts = {}
    for kind in ('main', 'support'):
        section = local.get('sections', {}).get(kind, {})
        items = section.get('items')
        if section.get('complete') is not True or not isinstance(items, list) or type(section.get('count')) is not int or section['count'] != len(items):
            raise ValueError('主辅星分区未完成或数量不一致')
        counts[kind] = len(items)
        for item in items:
            quality = QUALITY.get(item.get('quality'))
            level = item.get('level')
            name = item.get('name')
            if item.get('section') != kind or not isinstance(name, str) or not name.strip() or quality is None or type(level) is not int or not 1 <= level <= MAX_LEVEL[quality]:
                raise ValueError('星石字段不完整或无效，禁止同步')
            entries.append(dict(kind=kind, name=name, quality=quality, level=level))
    exp = local.get('experience', {})
    experience = {kind: exp.get(kind + 'Count') for kind in ('orange', 'purple', 'white')}
    if any(value is not None and (type(value) is not int or not 0 <= value <= MAX_COUNT)
           for value in experience.values()) or len(entries) > 1000:
        raise ValueError('经验数量非法或星石数量超限，禁止同步')
    payload = dict(format='maayuan-star-inventory', version=1, scan_id=scan_id, complete=True,
                   counts=counts, inventory=entries, experience=experience)
    # Historical saved batches omit bag unchanged, preserving their retry identity.
    if 'bag' in local:
        bag = local['bag']
        capacity = None
        if bag.get('complete') is True:
            current, capacity = bag.get('currentCount'), bag.get('capacity')
            if type(current) is not int or current != len(entries) or type(capacity) is not int or not current <= capacity <= MAX_COUNT or capacity <= 0:
                raise ValueError('背包数量或容量与完整扫描不一致，禁止同步')
        payload['bag'] = dict(current_count=len(entries), capacity=capacity)
    return payload


class NoRedirect(urllib_request.HTTPRedirectHandler):
    # Do not forward a bound-account bearer credential to a redirected host.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_inventory(payload: dict, settings, timeout_seconds=30.0) -> dict:
    """One POST. An uncertain result keeps the same on-disk batch for the next run."""
    request = urllib_request.Request(
        settings.base_url + SYNC_PATH, json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        {'Authorization': 'Bearer ' + settings.token, 'Accept': 'application/json',
         'Content-Type': 'application/json; charset=utf-8', 'User-Agent': 'MaaYuan-StarInventory/1'}, method='POST')
    try:
        with urllib_request.build_opener(NoRedirect()).open(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode('utf-8'))
            receipt = body.get('data', {})
            if receipt.get('status') != 'synced' or receipt.get('scan_id') != payload['scan_id'] or type(receipt.get('generation')) is not int:
                return dict(success=False, status='cloud_sync_failed', message='云端响应无法确认；已保留原批次，下次任务安全重试')
            return dict(success=True, status='synced', receipt=receipt)
    except urllib_error.HTTPError as exc:
        # Do not persist arbitrary server text, URLs, headers or exception strings.
        code = ''
        try:
            parsed = json.loads(exc.read().decode('utf-8'))
            code = parsed.get('error', {}).get('code', '') if isinstance(parsed.get('error'), dict) else parsed.get('code', '')
        except (ValueError, AttributeError):
            pass
        message = {
            401: '连接码无效；' + TOKEN_REQUIRED_HINT,
            403: '连接码权限不足，请在 YuanHub 连接管理补齐 MaaYuan 权限（星石自动同步）',
            409: '云端背包已变化或旧计划待继承，请先处理云端状态；本地扫描和原批次已保留',
            422: '扫描数据未通过云端校验，本地完整数据已保留',
        }.get(exc.code, '云端同步失败，已保留原批次，下次任务可安全重试')
        return dict(success=False, status='cloud_sync_failed', http_status=exc.code,
                    error_code=code if code in {'star_plan_inheritance_pending', 'star_state_revision_conflict', 'star_sync_scan_conflict'} else '', message=message)
    except (urllib_error.URLError, TimeoutError, OSError, ValueError, AttributeError):
        return dict(success=False, status='cloud_sync_failed', message='网络或响应异常，无法确认云端结果；本地数据和原批次已保留，下次任务安全重试')


def sync_saved_inventory(context, directory: Path) -> dict:
    """The upload document is written before HTTP and is immutable across retries."""
    payload_path = directory / 'inventory-sync.json'
    if payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding='utf-8'))
        # Detect any changed local source before a retry instead of minting a new ID.
        if payload != compact_inventory(json.loads((directory / 'inventory.json').read_text(encoding='utf-8')), payload['scan_id']):
            raise ValueError('已保存批次与本地扫描不一致，禁止改写或生成新批次')
    else:
        payload = compact_inventory(json.loads((directory / 'inventory.json').read_text(encoding='utf-8')), str(uuid.uuid4()))
        with payload_path.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
    settings = replace(read_upload_settings(context), mode=AUTO_UPLOAD_MODE)
    settings = replace(settings, token=resolve_token(settings.token))
    if not settings.token:
        raise ValueError(TOKEN_REQUIRED_HINT + '，并补齐星石自动同步权限')
    # Reuse the existing token/account lookup. Persist identity only locally, never
    # in the compact transfer document, so changing token cannot redirect a retry.
    account = get_bound_account(settings, max_attempts=1)
    binding = dict(account_id=account.id, base_url=settings.base_url)
    binding_path = directory / 'sync-binding.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding='utf-8')) != binding:
            raise ValueError('连接账号或服务地址已变化，禁止将原扫描重试到另一账号')
    else:
        with binding_path.open('x', encoding='utf-8') as stream:
            json.dump(binding, stream, ensure_ascii=False)
    result = post_inventory(payload, settings)
    if result.get('success') and result['receipt'].get('account_id') != account.id:
        result = dict(success=False, status='cloud_sync_failed', message='响应账号不一致，未确认同步成功；原批次已保留')
    write(directory / 'sync-result.json', result)
    return result


def prune_synced_diagnostics(directory: Path, local: dict) -> None:
    """Retain the durable batch and compact summary after a healthy verified sync."""
    report_path=directory/'result.json'
    experience=local.get('experience',{})
    if (local.get('complete') is not True or experience.get('captured') is not True
            or experience.get('complete') is not True or (directory/'error.json').exists()
            or report_path.is_symlink() or not report_path.is_file()):
        return
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('diagnostics_retention_version')!=SYNC_DIAGNOSTICS_RETENTION_VERSION
            or report.get('run_id')!=directory.name or report.get('outcome')!='completed'):
        return

    # Use the generated capture and page indexes, never a recursive sweep.
    temporary={directory/name for name in (
        'progress.json','navigation.json','bag-results.json','experience-results.json',
        'tail-merge.json','main-results.json','support-results.json','timing.json',
        'completed-fields.jsonl','context-ocr.jsonl','position-recovery.json','pages/checkpoints.jsonl',
    )}
    temporary.update(directory/capture['file'] for capture in report['captures']
                     if re.fullmatch(r'[0-9]+-[a-z0-9-]+\.png',capture['file']))
    temporary.update(directory/'pages'/page['file'] for page in report['pages']
                     if re.fullmatch(r'(?:main|support)-[a-z0-9-]+\.json',page['file']))
    for event in report['events']:
        section,swipe=event.get('section'),event.get('swipe')
        if section in ('main','support') and type(swipe) is int and 1<=swipe<=30:
            temporary.add(directory/f'{section}-swipe-{swipe:03d}-claim.json')

    section_keys=('complete','final_count','stop_reason','swipes','recovery_attempts','bottom_attempts')
    sections={name:{key:section[key] for key in section_keys if key in section}
              for name,section in report['sections'].items()}
    for name,section in report['sections'].items():
        if 'terminal_selection' in section:
            sections[name]['terminal_row_offset']=section['terminal_selection']['selected_row_offset']
    compact={'run_id':report['run_id'],'outcome':'synced','complete':True,
             'sections':sections,'action_wall_ms':report['action_wall_ms'],'diagnostics_pruned':True}
    # Keep the full index until removal succeeds so a failed cleanup leaves a
    # report and the already-persisted inventory, request and receipt intact.
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=directory,
                                     prefix='result-',suffix='.tmp',delete=False) as stream:
        pending=Path(stream.name)
        json.dump(compact,stream,ensure_ascii=False,indent=2)
    try:
        for path in temporary:
            # Treat symlinks/junction targets and unindexed files as unowned.
            if path.resolve()==path and not path.is_symlink() and path.is_file():
                path.unlink()
        os.replace(pending,report_path)
    finally:
        pending.unlink(missing_ok=True)


def sync_success_message(local: dict) -> str:
    main = len(local['sections']['main']['items'])
    support = len(local['sections']['support']['items'])
    count = main + support
    capacity = local.get('bag', {}).get('capacity')
    bag = f'背包{count}/{capacity}。' if type(capacity) is int and capacity >= count and capacity > 0 else f'背包{count}颗，容量未知。'
    return f'星石背包同步完成。\n主星{main}颗，辅星{support}颗，{bag}'


@AgentServer.custom_action('StarInventorySync')
class StarInventorySync(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg):
        if not isinstance(context, Context) or not Library.is_agent_server():
            raise RuntimeError('actual Agent Context required')
        params = json.loads(argv.custom_action_param)
        root = (REPO / params.get('output', 'debug/star-local-ocr')).resolve()
        if not root.is_relative_to(REPO):
            raise ValueError('output must remain inside MaaYuan')
        directory = None
        diagnostics = start_star_diagnostics(context, params)
        result = None
        failure = None
        local = None
        try:
            latest = json.loads((root / 'latest.json').read_text(encoding='utf-8')) if (root / 'latest.json').exists() else {}
            if latest.get('outcome') in ('cloud_sync_pending', 'cloud_sync_failed'):
                pending_directory = (root / latest['directory']).resolve()
                if pending_directory.parent != root:
                    raise ValueError('invalid pending directory')
                directory = pending_directory
                logger.info('正在重试上次完整扫描的同步，保留原批次编号')
            else:
                session = FullShadowSession(context, params)
                directory = session.out
                if not session.execute():
                    logger.error('星石背包识别未完成，本次未上传。诊断已保留。')
                    return False
            # A crash or lost response after this write resumes the original scan.
            write(root / 'latest.json', dict(directory=directory.name, complete=True, outcome='cloud_sync_pending'))
            result = sync_saved_inventory(context, directory)
            write(root / 'latest.json', dict(directory=directory.name, complete=True, outcome='synced' if result['success'] else 'cloud_sync_failed'))
            if result['success']:
                local = json.loads((directory / 'inventory.json').read_text(encoding='utf-8'))
                logger.info(sync_success_message(local))
            else:
                logger.error('星石背包识别已完成，但同步未完成。完整结果与原批次已保留，下次开始会重试。' + result['message'])
            return result['success']
        except Exception as exc:
            failure = exc
            # Underlying HTTP/config exceptions may contain credentials. Store no
            # traceback or raw exception text in upload records or task logs.
            result = dict(success=False, status='cloud_sync_failed', message='同步准备失败：请检查账号和本地完整结果；' + TOKEN_REQUIRED_HINT + '；数据已保留，下次任务重试原批次')
            if directory is not None:
                write(directory / 'sync-result.json', result)
                write(root / 'latest.json', dict(directory=directory.name, complete=True, outcome='cloud_sync_failed'))
            logger.error(result['message'])
            return False
        finally:
            stopped = getattr(getattr(context, 'tasker', None), 'stopping', False) or getattr(context, 'stop', False) is True
            call(diagnostics, 'finish_star', directory, stopped=stopped, sync_result=result, exception=failure)
            if not stopped and failure is None and local is not None and result.get('status') == 'synced':
                try:
                    prune_synced_diagnostics(directory, local)
                except (OSError, ValueError, KeyError, TypeError):
                    # The upload is already confirmed. Cleanup is best effort.
                    logger.warning('星石同步已完成，本地诊断文件清理未完成')
