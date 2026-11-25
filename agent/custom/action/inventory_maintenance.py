"""Compose inventory reports, the backpack scanner and the existing training tasks."""

from __future__ import annotations

import re
import tempfile
import uuid
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from custom.action.fixed_reward_reporting import _item_names
from custom.action.inventory_reporting import (
    AUTH_NODE_NAME,
    AUTO_UPLOAD_MODE,
    LOCAL_ONLY_MODE,
    UploadSettings,
    append_inventory_report,
    bound_account_report_filename,
    build_exchange_document,
    get_bound_account,
    inventory_recovery_hint,
    read_upload_settings,
    resolve_inventory_report_destination,
    update_upload_status,
    upload_inventory_document,
)
from custom.action.maintenance_stock import (
    ReportTarget,
    Stock,
    SweepJournal,
    cloud_stock,
    local_stock,
    maintenance_lock,
    now,
    report_records,
    timestamp,
)
from custom.action.maintenance_training import (
    BOOK_XP,
    CONFIG,
    Goal,
    Stage,
    book_experience,
    load_goals,
    merge_fields,
)
from custom.action.paged_item_recognition import _parse_params
from custom.reco.agent_item import REPO_ROOT, _ocr_results
from utils import logger


ENTRY = "库存保持启动"
SCAN = "背包识别-已在养成页-道具"
START = "点击开始历练扫荡"
SETTLED = "历练扫荡-固定奖励上报"
ENERGY = "体力不足check"
QUOTA = "历练次数不足check"
LOCKED = "库存保持-关卡不可扫"
CLOSED = "库存保持-历练未开放"
MAXIMUM = "历练扫荡-自适应次数检测"
RED_COST = "历练扫荡-检测行动力是否充足"
COUNTER = "历练扫荡-实际次数4"
SWEEP_COUNT_PATTERN = r"^[1-9][0-9]*$"
HOME = "返回主界面左半屏"
NO_ACTION = {"type": "DoNothing", "param": {}}
GROUP_LIMIT = 6


def terminal(context, entry, overrides=None):
    if context.tasker.stopping:
        raise RuntimeError("任务已停止")
    detail = context.run_task(entry, overrides or {})
    if (
        not detail
        or not detail.status.succeeded
        or not detail.nodes
        or not detail.nodes[-1].completed
    ):
        raise RuntimeError(f"{entry} 未完成")
    return detail.nodes[-1].name


def progress(context, message):
    logger.info(f"【库存保持】{message}")
    terminal(context, "库存保持-进度", {"库存保持-进度": {"focus": message}})


def go_home(context):
    terminal(context, HOME)


def document(counts, names, channel, kind, account, instant, scope=None):
    results = [
        {"entity_type": "item", "item_id": key, "item_name": names[key], "count": count}
        for key, count in counts.items()
    ]
    return build_exchange_document(
        results, uuid.uuid4().hex, instant, instant, channel, kind, scope, account
    )


def scan_backpack(context, target, names):
    child = context.clone()
    params = _parse_params(
        child.get_node_data(SCAN)["action"]["param"]["custom_action_param"]
    )
    with tempfile.TemporaryDirectory() as directory:
        capture = Path(directory) / "StockReport.txt"
        overrides = {
            AUTH_NODE_NAME: {
                "attach": {"mode": LOCAL_ONLY_MODE, "inventory_report_filename": ""}
            },
            "背包识别-已在物品页": {"action": NO_ACTION},
            SCAN: {
                "next": [],
                "action": {
                    "param": {
                        "custom_action_param": {
                            **params,
                            "inventory_report_path": str(capture),
                        }
                    }
                },
            },
        }
        if terminal(child, "背包识别-start", overrides) != SCAN:
            raise ValueError("背包未完成养成页扫描")
        rows = list(report_records(capture, names))
        if (
            len(rows) != 1
            or rows[0].kind != "库存快照"
            or rows[0].channel != "背包-道具"
        ):
            raise ValueError("本次背包扫描没有生成养成页库存")
        row = rows[0]
    if not row.counts:
        raise ValueError("本次养成页没有可用的数量")
    snapshot = document(
        row.counts,
        names,
        row.channel,
        "stock_snapshot",
        target.account,
        row.time.isoformat(),
        "listed",
    )
    append_inventory_report(target.rewards, snapshot, LOCAL_ONLY_MODE)
    return Stock(row.counts, dict.fromkeys(row.counts, row.time))


def open_stage(child, goal: Goal, stage: Stage):
    ready = f"想扫荡几次{goal.kind}历练"
    options = deepcopy(stage.override)
    # Stop the original adaptive chain at the dialog, before its automatic submit.
    merge_fields(
        options,
        {
            f"进入{goal.kind}历练扫荡界面": {"next": [MAXIMUM, ready]},
            MAXIMUM: {"next": [ready], "on_error": []},
            ready: {"next": [], "on_error": []},
            f"选择你的{goal.kind}历练": {"timeout": 5000, "on_error": [CLOSED]},
            f"进入你的{goal.label}": {"timeout": 5000, "on_error": [LOCKED]},
            f"找到{goal.kind}历练具体层数": {"timeout": 5000, "on_error": [LOCKED]},
            "下滑-密探列表-两行": {"max_hit": 6},
            ENERGY: {"next": []},
            QUOTA: {"next": []},
        },
    )
    end = terminal(child, f"刷{goal.kind}历练启动", options)
    if end not in (ready, ENERGY, QUOTA, LOCKED, CLOSED):
        raise RuntimeError(f"{goal.label}没有进入扫荡设置")
    return end


def affordable_count(child, wanted):
    read_count = {
        COUNTER: {"recognition": {"param": {"expected": [SWEEP_COUNT_PATTERN]}}}
    }
    expected_after_minus = None
    while not child.tasker.stopping:
        image = child.tasker.controller.post_screencap().wait().get()
        if image is None:
            raise RuntimeError("读取扫荡界面截图失败")
        detail = child.run_recognition(COUNTER, image, read_count)
        digits = (
            {r.text.strip() for r in _ocr_results(detail)}
            if detail and detail.hit
            else set()
        )
        number = digits.pop() if len(digits) == 1 else ""
        if not re.fullmatch(SWEEP_COUNT_PATTERN, number):
            raw = [r.text for r in getattr(detail, "all_results", [])]
            raise ValueError(
                f"无法确认扫荡次数，计划 {wanted} 次，次数区域 OCR 候选 {raw!r}"
            )
        count = int(number)
        if expected_after_minus is not None and count != expected_after_minus:
            raise ValueError(
                f"减少扫荡次数后应为 {expected_after_minus}，界面读到 {count}"
            )
        start = child.run_recognition(START, image)
        red = child.run_recognition(RED_COST, image)
        if not start or not start.hit or red is None:
            raise ValueError("无法确认扫荡按钮或体力状态")
        if not red.hit and count <= wanted:
            logger.info(f"【库存保持】界面已确认 {count} 次，体力足够，准备扫荡")
            return count
        if count == 1:
            logger.info("【库存保持】界面已减至 1 次，体力仍不足，本批跳过")
            return 0
        if expected_after_minus is None:
            reasons = []
            if count > wanted:
                reasons.append(f"超过本批计划 {wanted} 次")
            if red.hit:
                reasons.append("当前次数所需体力不足")
            logger.info(
                f"【库存保持】界面选中 {count} 次，{'，'.join(reasons)}，自动减次"
            )
        logger.debug(
            f"【库存保持】次数 {count} → {count - 1}，体力{'不足' if red.hit else '足够'}"
        )
        expected_after_minus = count - 1
        terminal(
            child,
            RED_COST,
            {
                RED_COST: {
                    "recognition": {"type": "DirectHit"},
                    "next": [],
                    "on_error": [],
                }
            },
        )
    raise RuntimeError("任务已停止")


def sync_sweep_report(path: Path, report: dict, settings: UploadSettings) -> bool:
    result = upload_inventory_document(report, settings)
    status = "自动上报成功" if result.success else "自动上报失败"
    try:
        update_upload_status(
            path, report["records"][0]["record_id"], f"{status}（{result.message}）"
        )
    except (OSError, ValueError) as exc:
        logger.warning(f"【库存保持】无法更新本地上报状态。{exc}")
    if result.success:
        logger.info(f"【库存保持】扫荡奖励已同步到 YuanHub（{result.message}）")
    else:
        permission_hint = (
            "请检查连接码的「导入并更新库存」权限和绑定账号。"
            if result.status_code == 403
            else ""
        )
        logger.warning(
            f"【库存保持】扫荡奖励上报失败（{result.message}），已保留本地报告。"
            f"{permission_hint}{inventory_recovery_hint(result)}"
        )
    return result.success


def submit(child, stage, count, stock, source, journal, target, names, upload_settings):
    gains = {key: n * count for key, n in stage.rewards.items()}
    instant = now().isoformat(timespec="milliseconds")
    report = document(
        gains,
        names,
        stage.override[CONFIG]["attach"]["acquisition_channel"],
        "reward_delta",
        target.account,
        instant,
    )
    receipt = report["records"][0]["record_id"]
    previous = journal.entries.copy()
    for key, gain in gains.items():
        baseline = source.baselines.get(key)
        journal.entries[key] = {
            "time": instant,
            "id": receipt,
            "confirmed": False,
            "source_count": source.counts.get(key),
            "source_time": baseline.isoformat() if baseline else None,
            "after": stock.counts[key] + gain,
        }
    journal.save()
    # Only this call may submit. A timeout leaves the write-ahead entry pending.
    try:
        end = terminal(
            child,
            START,
            {
                START: {
                    "next": [ENERGY, QUOTA, SETTLED],
                    "on_error": [],
                    "timeout": 12000,
                },
                ENERGY: {"next": []},
                QUOTA: {"next": []},
                SETTLED: {"action": NO_ACTION, "next": [], "on_error": []},
            },
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"本批设置 {count} 次，扫荡结算未确认，实际次数需用新库存核对"
        ) from exc
    if end in (ENERGY, QUOTA):
        journal.entries = previous
        journal.save()
        return end
    if end != SETTLED:
        raise RuntimeError(
            f"本批设置 {count} 次，扫荡结算未确认，实际次数需用新库存核对"
        )
    status = "等待自动上报" if upload_settings else LOCAL_ONLY_MODE
    append_inventory_report(target.rewards, report, status)
    for key, gain in gains.items():
        stock.counts[key] += gain
        journal.entries[key]["confirmed"] = True
    journal.save()
    if upload_settings and sync_sweep_report(target.rewards, report, upload_settings):
        for key, gain in gains.items():
            if key in source.counts:
                source.counts[key] += gain
            # Only trust the cloud outright if it accounts for the whole local
            # estimate, including any batches that previously failed to upload.
            if (
                source.counts.get(key) == stock.counts[key]
                and source.baselines.get(key) == stock.baselines[key]
            ):
                journal.entries[key]["cloud_synced"] = True
        journal.save()
    return end


def log_inventory(goal, counts, names):
    if goal.experience:
        books = "、".join(f"{names[k]} {counts.get(k, '未知')}" for k in BOOK_XP)
        total = str(book_experience(counts))
        if not BOOK_XP.keys() <= counts.keys():
            total = f"至少 {total}" if BOOK_XP.keys() & counts.keys() else "未知"
        logger.info(
            f"【库存保持】{goal.label}库存：{books}。折合经验 {total}，目标 {goal.experience}"
        )
    else:
        items = "、".join(
            f"{names[k]} {counts.get(k, '未知')}/{target}"
            for k, target in goal.targets.items()
        )
        logger.info(f"【库存保持】{goal.label}库存（当前/目标）：{items}")


def train(context, goals, stock, source, journal, target, names, upload_settings):
    for goal in goals:
        unavailable = set()
        left = GROUP_LIMIT
        while True:
            gaps = goal.gaps(stock.counts)
            if gaps is None:
                progress(
                    context,
                    f"{goal.label}库存未知，无法计算所需次数，本轮已刷 {GROUP_LIMIT - left} 次",
                )
                break
            if not gaps:
                progress(
                    context,
                    f"{goal.label}库存已达标，还需 0 次，本轮已刷 {GROUP_LIMIT - left} 次",
                )
                break
            missing = "、".join(f"{names.get(k, k)} {v}" for k, v in gaps.items())
            plan = goal.plan_batches(gaps, unavailable)
            remaining_runs = sum(n for _, n in plan)
            if not left or not plan:
                reason = (
                    f"本次已扫满 {GROUP_LIMIT} 次"
                    if not left
                    else f"没有可扫关卡，本轮已刷 {GROUP_LIMIT - left} 次"
                )
                estimate = f"，按可扫关卡还需 {remaining_runs} 次" if plan else ""
                progress(context, f"{goal.label}{reason}，还缺{missing}{estimate}")
                break
            stage, needed = plan[0]
            wanted = min(needed, left)
            stage_label = f"第{stage.name}层" if goal.kind == "修为" else stage.name
            batch_label = f"{goal.label} · {stage_label}"
            logger.info(
                f"【库存保持】{goal.label}还缺{missing}。按可扫关卡预计还需 {remaining_runs} 次，本批计划{stage_label} {wanted} 次"
            )
            child = context.clone()
            end = open_stage(child, goal, stage)
            if end == CLOSED:
                progress(
                    context,
                    f"{goal.label}未开放，本轮已刷 {GROUP_LIMIT - left} 次，还缺{missing}",
                )
                go_home(context)
                break
            if end == LOCKED:
                logger.info(f"【库存保持】{batch_label}不可扫，重新选关")
                unavailable.add(stage.name)
            elif end not in (ENERGY, QUOTA):
                count = affordable_count(child, wanted)
                if count:
                    end = submit(
                        child,
                        stage,
                        count,
                        stock,
                        source,
                        journal,
                        target,
                        names,
                        upload_settings,
                    )
                    if end == SETTLED:
                        left -= count
                        updated = "、".join(
                            f"{names[k]} {stock.counts[k]}" for k in stage.rewards
                        )
                        progress(
                            context,
                            f"{batch_label}计划 {wanted} 次，实际结算 {count} 次，本轮已刷 {GROUP_LIMIT - left} 次。记账后库存：{updated}",
                        )
                else:
                    end = ENERGY
            go_home(context)
            if end == ENERGY:
                progress(
                    context,
                    f"体力不足，结束库存保持。{goal.label}本轮已刷 {GROUP_LIMIT - left} 次，还缺{missing}，按可扫关卡还需 {remaining_runs} 次",
                )
                return
            if end == QUOTA:
                progress(
                    context,
                    f"{goal.label}次数已用完，本轮已刷 {GROUP_LIMIT - left} 次，还缺{missing}，按可扫关卡还需 {remaining_runs} 次",
                )
                break


def execute(context, config, root=REPO_ROOT):
    goals = load_goals(root, config)
    if not goals:
        logger.info("【库存保持】未勾选目标，本轮扫荡 0 次")
        return
    mode = config["stock_source"]
    if (
        mode not in ("local_report", "yuanhub")
        or type(config["scan_if_missing"]) is not bool
    ):
        raise ValueError("库存来源或背包识别配置无效")
    names = _item_names(root / "agent/items.json")
    wanted = {k: names[k] for goal in goals for k in (goal.targets or BOOK_XP)}
    directory = root / "data/inventory-maintenance"
    with ExitStack() as locks:
        target = None
        profile = mode
        stock = Stock()
        try:
            settings = read_upload_settings(context)
        except (ValueError, RuntimeError) as exc:
            settings = None
            progress(context, f"读取库存配置失败：{exc}")
        if settings:
            profile += ":" + (
                sha256(settings.token.encode()).hexdigest()
                if mode == "yuanhub"
                else (settings.report_filename or "")
            )
        locks.enter_context(maintenance_lock(directory, "unresolved:" + profile))
        if settings:
            try:
                if mode == "yuanhub":
                    account = get_bound_account(settings)
                    path = resolve_inventory_report_destination(
                        None, bound_account_report_filename(account), root
                    )
                    target = ReportTarget(path, account.id)
                else:
                    target, _ = local_stock(root, settings, wanted)
            except Exception as exc:
                progress(context, f"读取库存账号失败：{exc}")
        if target is not None:
            locks.enter_context(maintenance_lock(directory, target.key))
            try:
                if mode == "yuanhub":
                    stock = cloud_stock(target.account, settings, wanted)
                else:
                    # A second profile or token can resolve to the same account.
                    # Take the actual stock reading under the account lock.
                    checked_target, stock = local_stock(root, settings, wanted)
                    if checked_target != target:
                        target = None
                        raise ValueError("盘点期间本地报告账号发生变化")
            except HTTPError as exc:
                reason = (
                    "连接码缺少查看当前库存权限"
                    if exc.code == 403
                    else f"HTTP {exc.code}"
                )
                progress(context, f"读取 YuanHub 库存失败：{reason}")
            except Exception as exc:
                stock = Stock()
                if mode == "local_report":
                    target = None
                progress(context, f"读取库存失败：{exc}")
        source = deepcopy(stock)
        # A scan can still settle this run when the source cannot identify an
        # account. Keep its receipts separate; on recovery require newer readings
        # for affected items instead of assigning rewards to an unverified owner.
        unresolved = SweepJournal(directory, "unresolved:" + profile)
        if target is None:
            target = ReportTarget(
                directory / "unassigned" / (uuid.uuid4().hex + ".txt"), None
            )
            journal = unresolved
        else:
            for key, entry in unresolved.entries.items():
                baseline = stock.baselines.get(key)
                if not baseline or baseline <= timestamp(entry["time"]):
                    stock.counts.pop(key, None)
            journal = SweepJournal(directory, target.key)
        journal.reconcile(stock, cloud=mode == "yuanhub")
        stock_source = "本地库存报告" if mode == "local_report" else "YuanHub 云库存"
        if config["scan_if_missing"] and any(
            goal.gaps(stock.counts) is None for goal in goals
        ):
            logger.info(f"【库存保持】{stock_source}缺少可用读数，开始识别养成背包")
            try:
                stock = scan_backpack(context, target, names)
                stock_source = "本次养成背包盘点"
                journal.reconcile(stock)
            except Exception as exc:
                progress(context, f"背包盘点失败：{exc}")
            go_home(context)
        logger.info(f"【库存保持】库存来源：{stock_source}")
        for goal in goals:
            log_inventory(goal, stock.counts, names)
        upload_settings = (
            replace(settings, mode=AUTO_UPLOAD_MODE)
            if mode == "yuanhub" and settings and settings.token and target.account
            else None
        )
        train(context, goals, stock, source, journal, target, names, upload_settings)


@AgentServer.custom_action("InventoryMaintenance")
class InventoryMaintenance(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        try:
            execute(context, context.get_node_data(ENTRY)["attach"])
            return CustomAction.RunResult(success=True)
        except Exception as exc:
            logger.exception(f"库存保持已停止：{exc}")
            return CustomAction.RunResult(success=False)
