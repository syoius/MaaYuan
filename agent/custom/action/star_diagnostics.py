"""Allowlisted star scan summaries and list-only evidence on the shared transport."""
from __future__ import annotations

import hashlib
import json
import math
import re
from urllib.error import HTTPError, URLError

from custom.action.inventory_reporting import read_upload_settings, resolve_token
from custom.action.toolbox_diagnostics import (
    ROOT, DiagnosticSession, _STAR_WORKER, boundary, encode_manifest,
)
from custom.action.toolbox_diagnostics_transport import MAX_IMAGE, trusted_base
from utils.star_geometry import MAAYUAN_MUMU_LAYOUT

SCAN_REASONS = frozenset({
    'navigation_failed', 'forward_settle_timeout', 'position_uncertain',
    'recovery_insufficient_rows', 'name_anchor_unconfirmed',
    'experience_not_captured', 'ocr_field_invalid', 'stopped_unsafe', 'unknown',
})
ISSUES = {
    'star_scan_incomplete': ({'star_navigation', 'star_main', 'star_sub', 'star_experience'}, SCAN_REASONS),
    'star_capacity_mismatch': ({'star_capacity'}, {'count_mismatch', 'capacity_unread'}),
    'star_sync_failed': ({'star_sync'}, {'http_401', 'http_403', 'http_409', 'http_422', 'network'}),
}
WARNINGS = frozenset({
    'unknown', 'experience_not_started', 'experience_scan_failed', 'experience_counts_incomplete',
    'experience_tab_selection_unverified', 'experience_page_coverage_unconfirmed',
    'experience_icon_incomplete', 'experience_invalid_count_roi', 'experience_count_outside_content',
    'experience_icon_outside_content', 'experience_page_unverified', 'experience_icon_unclassified',
    'experience_tab_bounds_unverified', 'experience_portrait_profile_unverified',
    'experience_too_many_icons', 'experience_row_not_left_compacted', 'experience_content_unaccounted',
    'experience_icons_not_found', 'experience_order_conflict', 'experience_viewport_cropped',
    'experience_kind_duplicate', 'experience_count_unparsed', 'bag_not_started',
    'bag_frame_size_invalid', 'bag_numbers_unconfirmed', 'bag_numbers_conflict', 'bag_count_inventory_mismatch',
})
STAGE = re.compile(r'(?:start|unified_navigation|bag_ocr|scan_main|scan_support|switch_support|'
                   r'switch_experience|experience_ocr|position_recovery|completed|failed_stop|'
                   r'(?:main|support)_page_[0-9]{1,2})\Z')


def _integer(value, maximum, nullable=False):
    return (nullable and value is None) or (type(value) is int and 0 <= value <= maximum)


def validate_star_manifest(manifest):
    """Validate both fresh events and persisted retries without admitting free text."""
    issues, images = manifest['issues'], manifest['images']
    if (manifest['outcome'] not in {'failed', 'upload_failed'} or manifest.get('trace', []) != []
            or not isinstance(issues, list) or not 1 <= len(issues) <= 3
            or not isinstance(images, list) or len(images) > 4):
        raise ValueError('diagnostics_star_schema')
    codes = set()
    empty = {'operator_ids', 'candidate_operator_ids', 'error_type', 'node', 'stack',
             'blocking_errors', 'section_status', 'ocr_samples'}
    for item in issues:
        if (not isinstance(item, dict) or set(item) - empty != {'code', 'stage', 'reason', 'count'}
                or any(item.get(key) for key in empty) or item['code'] not in ISSUES
                or item['code'] in codes or type(item['count']) is not int or item['count'] != 1):
            raise ValueError('diagnostics_star_issue')
        stages, reasons = ISSUES[item['code']]
        if item['stage'] not in stages or item['reason'] not in reasons:
            raise ValueError('diagnostics_star_issue')
        codes.add(item['code'])
    if (manifest['outcome'] == 'upload_failed') != (codes == {'star_sync_failed'}):
        raise ValueError('diagnostics_star_outcome')
    for index, meta in enumerate(images):
        if (not isinstance(meta, dict) or set(meta) != {'index', 'role', 'roi'}
                or type(meta['index']) is not int or meta['index'] != index
                or meta['role'] != 'star_page' or meta['roi'] != _list_roi()
                or any(type(value) is not int for value in meta['roi'])
                or manifest['client']['resolution'] != '720x1280'):
            raise ValueError('diagnostics_star_image')
    summary = manifest.get('star_summary')
    if 'star_summary' not in manifest:
        return
    if not isinstance(summary, dict) or set(summary) != {
        'main', 'sub', 'experience', 'capacity', 'warnings', 'error', 'timing', 'settle',
    }:
        raise ValueError('diagnostics_star_summary')
    for section in ('main', 'sub'):
        values = summary[section]
        if (not isinstance(values, dict) or set(values) != {'count', 'complete', 'pages'}
                or type(values['complete']) is not bool
                or not all(_integer(values[key], 999) for key in ('count', 'pages'))):
            raise ValueError('diagnostics_star_summary')
    for key, fields, maximum in (
        ('experience', {'orange', 'purple', 'white', 'captured'}, 99999999),
        ('capacity', {'current', 'capacity'}, 999),
    ):
        values = summary[key]
        if (not isinstance(values, dict) or set(values) != fields
                or any(not _integer(value, maximum, True) for field, value in values.items() if field != 'captured')
                or key == 'experience' and type(values['captured']) is not bool):
            raise ValueError('diagnostics_star_summary')
    warnings = summary['warnings']
    if (not isinstance(warnings, list) or len(warnings) > 16
            or any(not isinstance(value, str) or value not in WARNINGS for value in warnings)):
        raise ValueError('diagnostics_star_summary')
    error = summary['error']
    if error is not None and (not isinstance(error, dict) or set(error) != {'stage', 'code'}
                              or not isinstance(error['stage'], str)
                              or not (STAGE.fullmatch(error['stage']) or error['stage'] in set().union(*(stages for stages, _ in ISSUES.values())))
                              or error['code'] not in set().union(*(reasons for _, reasons in ISSUES.values()))):
        raise ValueError('diagnostics_star_summary')
    for key, fields in (('timing', {'navigation_ms', 'total_ms'}),
                        ('settle', {'max_ms', 'last_settle_ms', 'comparisons'})):
        values = summary[key]
        if values is not None and (not isinstance(values, dict) or set(values) != fields
                                   or any(not _integer(value, 600000) for value in values.values())):
            raise ValueError('diagnostics_star_summary')


def _list_roi():
    layout = MAAYUAN_MUMU_LAYOUT
    top = round(layout['imageHeight'] * layout['contentTopRatio'])
    bottom = round(layout['imageHeight'] * layout['contentBottomRatio'])
    return [0, top, layout['imageWidth'], bottom - top]


def _scan_reason(error, stage):
    # Only fixed symbols leave this function; exception/OCR text stays local.
    if 'forward page did not settle' in error or 'page did not settle' in error:
        return 'forward_settle_timeout'
    if 'recovery needs four complete' in error or 'recovery needs calibrated row spacing' in error:
        return 'recovery_insufficient_rows'
    if 'name compensation' in error or 'incomplete name on confirmation' in error:
        return 'name_anchor_unconfirmed'
    if any(part in error for part in ('position', 'positional', 'checkpoint', 'recovery')):
        return 'position_uncertain'
    if error in {'unconfirmed or invalid required OCR field', 'Context ROI OCR failed',
                 'page has invalid or missing field geometry'}:
        return 'ocr_field_invalid'
    if (any(part in error for part in ('time limit', 'section limit', 'maximum 30 swipes'))
            or error in {'native swipe failed', 'one successful native Swipe required',
                         'backpack identity or selected section changed'}):
        return 'stopped_unsafe'
    if stage in {'start', 'unified_navigation', 'switch_support', 'switch_experience'}:
        return 'navigation_failed'
    return 'unknown'


def _sync_reason(result, exception):
    code = (result or {}).get('http_status')
    if code is not None:
        return f'http_{code}' if code in (401, 403, 409, 422) else ''
    if exception is None:
        return 'network' if result and result.get('success') is False else ''
    original = exception
    seen = set()
    while exception is not None and id(exception) not in seen:
        seen.add(id(exception))
        if isinstance(exception, HTTPError):
            return f'http_{exception.code}' if exception.code in (401, 403, 409, 422) else ''
        if (isinstance(exception, (URLError, TimeoutError, ConnectionError))
                or exception is not original and isinstance(exception, OSError)
                and not isinstance(exception, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError))):
            return 'network'
        exception = exception.__cause__ or exception.__context__
    return ''


def _read(directory, name):
    path = directory / name
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def _number(value, maximum, default=None):
    return value if _integer(value, maximum) else default


def _milliseconds(value):
    return max(0, min(600000, round(value))) if type(value) in (int, float) and math.isfinite(value) else 0


class StarDiagnosticSession(DiagnosticSession):
    def __init__(self, token, base_url, resource):
        super().__init__(token, base_url, 'images', resource, 'star_backpack_scan')
        self.finished = False

    @boundary
    def observe_star(self, directory, result, stage):
        import cv2

        # Collapse stability probes to one frame per page; keep the problem page first.
        pages = {}
        for capture in result.get('captures', []):
            name = capture.get('file', '')
            label = re.sub(r'-stable-[0-9]+(?=\.png$)', '', name)
            match = re.fullmatch(r'[0-9]+-((?:main|support)-(?:page-[0-9]+|bottom-confirmation|'
                                 r'(?:main|support)-[0-9]+(?:-(?:recovery|name-compensation|confirmation))*))\.png', label)
            if match:
                key = match[1]
            elif re.fullmatch(r'[0-9]+-(?:initial-main|bag-stable-retry|selected-support|selected-experience)(?:-[a-z0-9-]+)?\.png', name):
                key = 'experience' if 'experience' in name else 'support-initial' if 'support' in name else 'main-initial'
            elif re.fullmatch(r'[0-9]+-nav[0-9]+-entry-main-backpack\.png', name):
                key = 'main-initial'
            else:
                continue
            pages.pop(key, None)
            pages[key] = name
        if stage == 'star_navigation':
            return  # Never upload a crop from an unconfirmed home/observatory screen.
        names = list(pages.values())[-2:][::-1]
        for name in names:
            if len(self.images) >= 4:
                break
            path = (directory / name).resolve()
            if path.parent != directory.resolve() or path.is_symlink():
                continue
            frame = cv2.imread(str(path))
            if frame is None or frame.shape != (1280, 720, 3):
                continue
            box = _list_roi()
            x, y, w, h = box
            crop = frame[y:y+h, x:x+w].copy()
            for quality in (*range(90, 0, -10), 1):
                ok, encoded = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok and encoded.nbytes <= MAX_IMAGE:
                    self.client['resolution'] = '720x1280'
                    self.images.append(({'index': len(self.images), 'role': 'star_page', 'roi': box}, encoded.tobytes()))
                    break

    @boundary
    def finish_star(self, directory, *, stopped=False, sync_result=None, exception=None):
        if self.finished:
            return
        self.finished = True
        if stopped or directory is None:
            return
        local, result = _read(directory, 'inventory.json'), _read(directory, 'result.json')
        error, timing = _read(directory, 'error.json'), _read(directory, 'timing.json')
        if error.get('error') in {'Shadow stopped', 'navigation stopped'}:
            return
        exp, bag = local.get('experience', {}), local.get('bag', {})
        sections = local.get('sections', {})
        issues = []
        def issue(code, stage, reason):
            issues.append(dict(code=code, stage=stage, reason=reason, count=1))
        stage = error.get('stage', '')
        raw_error = error.get('error', '')
        incomplete = next((key for key in ('main', 'support') if sections.get(key, {}).get('complete') is not True), None)
        if incomplete or error and raw_error not in {
            'game backpack count/capacity unconfirmed; complete scan cannot upload',
            'game backpack count differs from complete scanned instance count',
        }:
            reason = _scan_reason(raw_error, stage)
            issue('star_scan_incomplete', 'star_navigation' if stage in {'start', 'unified_navigation'}
                  else 'star_experience' if 'experience' in stage
                  else 'star_sub' if incomplete == 'support' else 'star_main', reason)
        elif exp.get('captured') is not True or exp.get('complete') is not True:
            issue('star_scan_incomplete', 'star_experience',
                  'experience_not_captured' if exp.get('captured') is not True else 'ocr_field_invalid')
        if incomplete is None:
            if bag.get('complete') is not True:
                issue('star_capacity_mismatch', 'star_capacity', 'capacity_unread')
            elif bag.get('currentCount') != sum(sections.get(key, {}).get('count', 0) for key in ('main', 'support')):
                issue('star_capacity_mismatch', 'star_capacity', 'count_mismatch')
        sync_reason = _sync_reason(sync_result, exception)
        if sync_reason:
            issue('star_sync_failed', 'star_sync', sync_reason)
        if not issues:
            return
        summary = {}
        for key, source in (('main', 'main'), ('sub', 'support')):
            section = sections.get(source, {})
            summary[key] = dict(count=_number(section.get('count'), 999, 0), complete=section.get('complete') is True,
                                pages=min(999, sum(page.get('image', '').startswith(source + '-') for page in result.get('pages', []))))
        warnings = [warning.split(':', 1)[0] for warning in (*exp.get('warnings', []), *bag.get('warnings', []))]
        warnings = list(dict.fromkeys(warning if warning in WARNINGS else 'unknown' for warning in warnings))[:16]
        primary = issues[0]
        summary.update(
            experience={**{kind: _number(exp.get(kind + 'Count'), 99999999) for kind in ('orange', 'purple', 'white')},
                        'captured': exp.get('captured') is True},
            capacity=dict(current=_number(bag.get('currentCount'), 999), capacity=_number(bag.get('capacity'), 999)),
            warnings=warnings,
            error=dict(stage=stage if primary['code'] == 'star_scan_incomplete' and isinstance(stage, str)
                       and STAGE.fullmatch(stage) else primary['stage'], code=primary['reason']),
            timing=dict(navigation_ms=_milliseconds(timing.get('navigation_ms')),
                        total_ms=_milliseconds(timing.get('action_wall_ms'))), settle=None,
        )
        settles = [event['forward_stability'] for event in result.get('events', []) if event.get('forward_stability')]
        if settles:
            last = settles[-1]
            summary['settle'] = dict(max_ms=_milliseconds(last.get('window_ms')),
                                     last_settle_ms=_milliseconds(last.get('settle_ms')),
                                     comparisons=_number(last.get('comparisons'), 600000, 0))
        catalog_path = ROOT / 'agent/runtime/star_ocr/star_catalog.json'
        raw = catalog_path.read_bytes()
        version = json.loads(raw).get('version')
        self.catalog = dict(source='builtin', version=self.safe_version(str(version)) if version is not None
                            else hashlib.sha256(raw).hexdigest()[:16])
        self.observe_star(directory, result, primary['stage'])
        manifest = dict(schema_version=1, event_id=self.event_id, run_id=self.run_id, feature='star_backpack_scan',
                        client=self.client, catalog=self.catalog, occurred_at=self.occurred_at,
                        outcome='upload_failed' if all(item['code'] == 'star_sync_failed' for item in issues) else 'failed',
                        issues=issues, images=[meta for meta, _ in self.images], star_summary=summary)
        encode_manifest(manifest, self.secrets)
        _STAR_WORKER.submit(self, (manifest, [data for _, data in self.images], None))


@boundary
def start_star_diagnostics(context, params):
    node = context.get_node_data('星石本地OCR-完整背包')
    attach = node.get('attach', {}) if isinstance(node, dict) else {}
    if attach.get('star_scan_diagnostics', 'images') != 'images':
        _STAR_WORKER.configure('off')
        return None
    if getattr(context, 'stop', False) is True or context.tasker.stopping:
        return None
    settings = read_upload_settings(context)
    token = resolve_token(settings.token)
    if not token or not trusted_base(settings.base_url):
        return None
    return StarDiagnosticSession(token, settings.base_url, params.get('resource', attach.get('resource')))
