"""Region-based processing for the 鸢报 task screens."""

from __future__ import annotations

import json
import threading
import time
from typing import Iterable

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition

from utils import logger


# The four card ROI nodes are defined next to the 待办公务 pipeline.  Keep
# their order aligned with the visual order in the screenshot.
_CARD_REGION_NODES = ("区域1", "区域2", "区域3", "区域4")

# Existing recognition nodes are deliberately reused.  The second item is the
# first task to run after the card action has opened its detail screen.
_CARD_TASKS = (
    ("识别包含紫云英的请求出战", "已在请求出战战斗页面"),
    ("识别包含瑛琼瑶的请求出战", "已在请求出战战斗页面"),
    ("识别包含金错刀的请求出战", "已在请求出战战斗页面"),
    ("识别请求出战", "已在请求出战战斗页面"),
    ("识别包含建材的物资支援", "物资支援提交"),
    ("识别决策事件", "决策事件选择"),
    ("识别物资支援", "物资支援提交"),
)

_REQUEST_RECOGNITIONS = frozenset(name for name, _ in _CARD_TASKS[:4])
_ENERGY_RESTORE = "气力值回复"
_CATEGORY_ENERGY_RESTORES = {
    "突发情况自动回复气力值": ("突发情况", "突发情况不吃鸟食"),
    "小道消息自动回复气力值": ("小道消息", "小道消息不吃鸟食"),
    "他的传闻自动回复气力值": ("他的传闻", "他的传闻不吃鸟食"),
    "待办公务自动回复气力值": ("待办公务", "待办公务不吃鸟食"),
}
_ENERGY_RESTORE_HANDLERS = (_ENERGY_RESTORE, *_CATEGORY_ENERGY_RESTORES)
_ENERGY_STOP = "new不吃鸟食"
_CATEGORY_ENERGY_STOPS = (
    "突发情况不吃鸟食",
    "小道消息不吃鸟食",
    "他的传闻不吃鸟食",
    "待办公务不吃鸟食",
)
_ENERGY_STOP_HANDLERS = (_ENERGY_STOP, *_CATEGORY_ENERGY_STOPS)
_OFFICE_ENERGY_STOP = "待办公务不吃鸟食"
_OFFICE_EXHAUSTED_MARKER = "待办公务鸟食耗尽标记"
_OFFICE_QUERY_NODES = ("查询鸡炙剩余数量", "查询鸡炙剩余数量2")
_CARD_BUTTON_OFFSET = (75, 425, 115, 25)
_PAGE_READY_TIMEOUT = 30.0
_PAGE_READY_POLL_INTERVAL = 0.5

_INCIDENT_REGION_NODES = ("突发情况区域1", "突发情况区域2")
_INCIDENT_RECOGNITION = "点击前往调查"
_INCIDENT_FOLLOWUP = "突发情况调查后续"
_INCIDENT_READY_NODES = ("开始调查", "开始快速调查")
_INCIDENT_BUTTON_OFFSET = (201, 379, 210, 9)

# Nodes whose `enabled` flag gates whether a category is still checked by the
# all-in-one rotation chains (开始检查大礼包ABCD/BCDA/CDAB/DABC).
_CATEGORY_DISABLED_NODES = {
    "突发情况": ("查询蛇肉剩余数量", "查询蛇肉剩余数量2"),
    "小道消息": ("小道消息收菜", "查询麻籽剩余数量", "查询麻籽剩余数量2"),
    "他的传闻": ("他的传闻做吗", "他的传闻做吗2"),
    "待办公务": _OFFICE_QUERY_NODES,
}
_ALL_CATEGORIES = tuple(_CATEGORY_DISABLED_NODES)

# --- run-scoped exhaustion registry -----------------------------------------
#
# MaaFramework pipeline overrides are bound to the Context that applied them
# and only propagate downwards (Context::run_task clones its overrides into
# child contexts; they never propagate back up to the outer task).  The
# exhaustion handler therefore cannot disable a category for the outer
# all-in-one rotation: it runs inside a nested run_task whose overrides die
# with it (verified against logs: outer task_id stayed 200000128 while the
# handler task ids incremented per cycle, and the outer marker check never
# fired).
#
# This registry provides the missing cross-task state for ONE all-in-one run:
# - written by BirdFoodCategoryExhausted (nested context is fine);
# - read by the gate recognition and the two scanners, which re-apply the
#   `enabled: false` overrides from their own outer-task context;
# - reset by BirdFoodRunReset, attached to the 鸟食大礼包启动 entry node, so a
#   fresh all-in-one run starts with a clean slate.
#
# Mid-run loop-back caveat: the pipeline can re-enter 鸟食大礼包启动 during
# the SAME all-in-one task (interface.json task-level override
# `end_是否继续他的传闻 -> 鸟食大礼包启动`; upstream interface_v1.json also
# routes `鸟食耗尽check -> 鸟食大礼包启动`). Resetting there would wipe the
# exhausted registry and breaker counters and reopen the loop. BirdFoodRunReset
# therefore only resets when the incoming task_id differs from the one recorded
# for the active run; task ids are monotonically increasing per tasker session,
# so a new interface task always yields a fresh id.
#
# Lifecycle / isolation notes:
# - thread safety: all mutations go through _STATE_LOCK (agent actions can be
#   dispatched from different threads);
# - reset: the entry node always runs before any gate/scanner of the run;
# - isolation: each MaaYuan instance spawns its own agent process, so module
#   state is not shared between separate game accounts/instances.
_STATE_LOCK = threading.Lock()
_RUN_STATE = {
    "active": False,
    "run_task_id": None,
    "exhausted": set(),
    "gate_miss": {},
    "no_progress": {},
}
# Breaker thresholds: how many consecutive no-progress events are tolerated
# before the protection disables the category / all categories.
_GATE_MISS_LIMIT = 3
_NO_PROGRESS_LIMIT = 3


def _reset_run_state(task_id: int | None = None) -> bool:
    """Reset run-scoped state for a fresh all-in-one run.

    With a `task_id` (the normal path), a reset only happens when the state
    does not already belong to the same task: pipeline loop-backs re-enter the
    entry node within one task and must NOT wipe exhausted categories or
    breaker counters. Returns True when a reset actually happened.
    """
    with _STATE_LOCK:
        if (
            task_id is not None
            and _RUN_STATE["active"]
            and _RUN_STATE["run_task_id"] == task_id
        ):
            return False
        _RUN_STATE["active"] = True
        _RUN_STATE["run_task_id"] = task_id
        _RUN_STATE["exhausted"] = set()
        _RUN_STATE["gate_miss"] = {}
        _RUN_STATE["no_progress"] = {}
        return True


def _ensure_run_state() -> None:
    """Defensive: treat a gate/scanner hit without a reset as a fresh run."""
    with _STATE_LOCK:
        if not _RUN_STATE["active"]:
            _RUN_STATE["active"] = True
            _RUN_STATE["run_task_id"] = None
            _RUN_STATE["exhausted"] = set()
            _RUN_STATE["gate_miss"] = {}
            _RUN_STATE["no_progress"] = {}


def _mark_exhausted(category: str) -> None:
    with _STATE_LOCK:
        _RUN_STATE["active"] = True
        _RUN_STATE["exhausted"].add(category)
        _RUN_STATE["gate_miss"].pop(category, None)
        _RUN_STATE["no_progress"].pop(category, None)


def _is_exhausted(category: str) -> bool:
    with _STATE_LOCK:
        return category in _RUN_STATE["exhausted"]


def _exhausted_categories() -> tuple[str, ...]:
    with _STATE_LOCK:
        return tuple(_RUN_STATE["exhausted"])


def _bump_gate_miss(category: str) -> int:
    with _STATE_LOCK:
        count = int(_RUN_STATE["gate_miss"].get(category, 0)) + 1
        _RUN_STATE["gate_miss"][category] = count
        return count


def _bump_no_progress(category: str) -> int:
    with _STATE_LOCK:
        count = int(_RUN_STATE["no_progress"].get(category, 0)) + 1
        _RUN_STATE["no_progress"][category] = count
        return count


def _exhausted_disable_override() -> dict:
    """Pipeline override disabling every exhausted category's check nodes."""
    override: dict = {}
    for category in _exhausted_categories():
        for node_name in _CATEGORY_DISABLED_NODES.get(category, ()):
            override[node_name] = {"enabled": False}
    return override


def _is_hit(detail) -> bool:
    return bool(detail and getattr(detail, "hit", False))


def _is_enabled(context: Context, node_name: str) -> bool:
    """Read the effective node definition so interface options still apply."""
    try:
        data = context.get_node_data(node_name)
    except Exception:
        logger.exception(f"读取待办公务节点状态失败: {node_name}")
        return True

    if not isinstance(data, dict):
        return True
    return data.get("enabled", True) is not False


def _recognition_override(node_name: str, roi: Iterable[int]) -> dict:
    return {
        node_name: {
            "recognition": {
                "param": {"roi": list(roi)},
            }
        }
    }


def _next_names(data: dict) -> list[str]:
    names = []
    for item in data.get("next", []):
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


def _energy_handler(
    context: Context, recognition_name: str, followup_name: str
) -> str | None:
    # All request-card variants share the interface option configured on the
    # generic request node.  Material/decision options are configured on their
    # own follow-up nodes.
    option_node = (
        "识别请求出战"
        if recognition_name in _REQUEST_RECOGNITIONS
        else followup_name
    )
    data = context.get_node_data(option_node)
    if not isinstance(data, dict):
        return None

    next_names = _next_names(data)
    for stop_handler in _ENERGY_STOP_HANDLERS:
        if stop_handler in next_names:
            return stop_handler
    for restore_handler in _ENERGY_RESTORE_HANDLERS:
        if restore_handler in next_names:
            return restore_handler
    return None


def _is_allinone_run(context: Context) -> bool:
    try:
        data = context.get_node_data("end_当前没有可办公务")
    except Exception:
        logger.exception("读取待办公务四合一状态失败")
        return False

    if not isinstance(data, dict):
        return False
    return any(
        name.startswith("开始检查大礼包") for name in _next_names(data)
    )


def _card_rois(
    context: Context,
) -> tuple[tuple[int, tuple[int, int, int, int]], ...]:
    rois = []
    for card_index, node_name in enumerate(_CARD_REGION_NODES, start=1):
        if not _is_enabled(context, node_name):
            continue

        data = context.get_node_data(node_name)
        roi = data["recognition"]["param"]["roi"]
        if len(roi) != 4:
            raise ValueError(f"{node_name} 的 ROI 必须包含四个值")
        rois.append((card_index, tuple(int(value) for value in roi)))
    return tuple(rois)


def _exclude_empty_card_region(context: Context, card_index: int) -> bool:
    region_name = _CARD_REGION_NODES[card_index - 1]
    if not context.override_pipeline({region_name: {"enabled": False}}):
        logger.warning(f"禁用待办公务第 {card_index} 个区域失败")
        return False

    logger.info(f"本次四合一不再检查待办公务第 {card_index} 个区域")
    if any(_is_enabled(context, name) for name in _CARD_REGION_NODES):
        return True

    override = {name: {"enabled": False} for name in _OFFICE_QUERY_NODES}
    if not context.override_pipeline(override):
        logger.warning("全部区域排除后禁用待办公务类别失败")
        return False

    logger.info("待办公务四个区域均已排除，本次四合一不再检测待办公务")
    return True


def _run_action(
    context: Context,
    node_name: str,
    box,
    card_roi,
    button_offset: tuple[int, int, int, int] = _CARD_BUTTON_OFFSET,
) -> bool:
    # Recognition results can appear at different positions within a card.
    # Click the card's fixed 前往 button instead of using the matched content.
    x, y, _, _ = card_roi
    offset_x, offset_y, width, height = button_offset
    target = [x + offset_x, y + offset_y, width, height]
    override = {
        node_name: {
            "action": {
                "param": {
                    "target": target,
                    "target_offset": [0, 0, 0, 0],
                }
            }
        }
    }
    action = context.run_action(node_name, box, "", override)
    status = getattr(action, "status", None) if action else None
    return bool(
        action
        and not (status is not None and getattr(status, "failed", False))
    )


def _post_action_delay(context: Context, node_name: str) -> None:
    post_delay = 500
    node_data = context.get_node_data(node_name)
    if isinstance(node_data, dict):
        try:
            post_delay = max(post_delay, int(node_data.get("post_delay", post_delay)))
        except (TypeError, ValueError):
            pass
    time.sleep(max(post_delay, 0) / 1000)


def _run_followup(context: Context, task_name: str, allinone_run: bool) -> str:
    # Do not let a failed follow-up jump back into the card scanner.  The card
    # has already been consumed and must not be recognized a second time.
    override = {
        task_name: {"on_error": []},
        _OFFICE_ENERGY_STOP: {"next": []},
    }
    detail = context.run_task(task_name, override)
    status = getattr(detail, "status", None) if detail else None
    succeeded = bool(status and getattr(status, "succeeded", False))
    if not succeeded:
        logger.warning(f"待办公务后续节点执行失败: {task_name}")
        return "failed"

    # Registry check is allinone-only: the registry is reset per all-in-one
    # run and must not leak into the standalone 待办公务 task.
    if allinone_run and _is_exhausted("待办公务"):
        logger.info("待办公务鸟食已耗尽，本次四合一不再检测待办公务")
        return "stop"
    # Legacy pipeline-marker check kept as a fallback: the marker is enabled
    # via override_pipeline, which is context-scoped, so it is only visible
    # here when the exhaustion happened in a context this check can see.
    if _is_enabled(context, _OFFICE_EXHAUSTED_MARKER):
        logger.info("待办公务鸟食已耗尽，本次四合一不再检测待办公务")
        return "stop"
    return "continue"


def _run_energy_handler(
    context: Context,
    handler: str,
    screen_name: str,
) -> str:
    """Execute a previously recognized energy popup handler."""

    logger.info(f"{screen_name}检测到气力弹窗，执行: {handler}")
    override = {handler: {"on_error": []}}
    category_restore = _CATEGORY_ENERGY_RESTORES.get(handler)
    if handler in _ENERGY_RESTORE_HANDLERS:
        # Exit back to the card list without using the original global card
        # recognition.  The saved hit box is clicked again by the caller.
        override["退出气力回复页面"] = {"next": [], "on_error": []}
        if category_restore is not None:
            _, stop_handler = category_restore
            # The category stop action navigates back to 鸢报.  End this nested
            # handler there so the outer region scanner can return cleanly.
            override[stop_handler] = {"next": []}
    elif handler in _CATEGORY_ENERGY_STOPS:
        # birdfood1/4 invoke the handler as a nested task.  Let the outer task's
        # existing end node resume the all-in-one loop after navigation.
        override[handler]["next"] = []

    detail = context.run_task(handler, override)
    status = getattr(detail, "status", None) if detail else None
    if not bool(status and getattr(status, "succeeded", False)):
        logger.warning(f"{screen_name}气力弹窗处理失败: {handler}")
        return "failed"

    if category_restore is not None:
        category, _ = category_restore
        if not any(
            _is_enabled(context, name)
            for name in _CATEGORY_DISABLED_NODES[category]
        ):
            return "stop"

    if handler in _ENERGY_STOP_HANDLERS or context.tasker.stopping:
        return "stop"
    return "retry"


def _wait_for_page(
    context: Context,
    recognition_name: str,
    followup_name: str,
    ready_nodes: Iterable[str],
    screen_name: str,
) -> str:
    """Wait until an energy popup or one of the target pages is fully loaded."""
    handler = _energy_handler(context, recognition_name, followup_name)
    ready_names = tuple(ready_nodes)
    ready_label = " / ".join(ready_names)
    deadline = time.monotonic() + _PAGE_READY_TIMEOUT

    logger.info(f"{screen_name}等待页面加载: 气力弹窗或 {ready_label}")
    while not context.tasker.stopping:
        try:
            image = context.tasker.controller.post_screencap().wait().get()
        except Exception:
            logger.exception(f"{screen_name}等待页面时截图失败")
            return "failed"

        if image is None:
            logger.warning(f"{screen_name}等待页面时没有截图")
            return "failed"

        if handler is not None:
            popup = context.run_recognition(handler, image)
            if _is_hit(popup):
                return _run_energy_handler(context, handler, screen_name)

        for ready_name in ready_names:
            ready = context.run_recognition(ready_name, image)
            if _is_hit(ready):
                logger.info(f"{screen_name}页面已加载: {ready_name}")
                return "continue"

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(f"{screen_name}等待页面加载超时: {ready_label}")
            return "failed"
        time.sleep(min(_PAGE_READY_POLL_INTERVAL, remaining))

    return "stop"


def _begin_category_scan(context: Context, category: str) -> str:
    """Outer-task-side guard executed when a category scan is about to run.

    Re-applies the exhausted-category disables from the scanner's own context
    (which belongs to the running outer all-in-one task, so the override
    persists for the rest of the rotation) and skips the scan entirely when
    the category was already confirmed exhausted during this run.
    """
    _ensure_run_state()
    override = _exhausted_disable_override()
    if override:
        if context.override_pipeline(override):
            logger.info(
                "已同步耗尽类别禁用覆盖: " + "、".join(_exhausted_categories())
            )
        else:
            logger.error("同步耗尽类别禁用覆盖失败")

    if _is_exhausted(category):
        logger.warning(
            f"{category}本次四合一已确认鸟食耗尽，跳过本次调度（防死循环保护）"
        )
        return "skip"
    return "run"


def _end_category_scan(
    context: Context, category: str, allinone_run: bool, made_progress: bool
) -> None:
    """No-progress breaker: stop the rotation when a scan keeps achieving nothing."""
    if not allinone_run or context.tasker.stopping:
        return
    if made_progress or _is_exhausted(category):
        with _STATE_LOCK:
            _RUN_STATE["no_progress"].pop(category, None)
        return

    count = _bump_no_progress(category)
    if count < _NO_PROGRESS_LIMIT:
        logger.info(f"{category}本次扫描无进展（{count}/{_NO_PROGRESS_LIMIT}）")
        return

    logger.error(
        f"{category}连续 {_NO_PROGRESS_LIMIT} 次扫描无进展，触发无进展熔断："
        "禁用该类别检查节点以中止循环"
    )
    override = {
        name: {"enabled": False}
        for name in _CATEGORY_DISABLED_NODES.get(category, ())
    }
    if override and not context.override_pipeline(override):
        logger.error("无进展熔断禁用覆盖失败")


@AgentServer.custom_action("BirdFood4TaskScan")
class BirdFood4TaskScan(CustomAction):
    """Scan each 待办公务 card once and execute its matching existing task."""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        allinone_run = _is_allinone_run(context)
        if allinone_run and _begin_category_scan(context, "待办公务") == "skip":
            return CustomAction.RunResult(success=True)
        result = self._scan(context, argv)
        _end_category_scan(
            context,
            "待办公务",
            allinone_run,
            bool(result and getattr(result, "success", False)),
        )
        return result

    def _scan(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        handled_any = False
        allinone_run = _is_allinone_run(context)

        try:
            card_rois = _card_rois(context)
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(f"读取待办公务 ROI 失败: {exc}")
            return CustomAction.RunResult(success=False)

        for card_index, card_roi in card_rois:
            try:
                image = context.tasker.controller.post_screencap().wait().get()
            except Exception:
                logger.exception(f"待办公务第 {card_index} 个区域截图失败")
                return CustomAction.RunResult(success=False)

            if image is None:
                logger.warning(f"待办公务第 {card_index} 个区域没有截图")
                return CustomAction.RunResult(success=False)

            card_handled = False
            for recognition_name, followup_name in _CARD_TASKS:
                if not _is_enabled(context, recognition_name):
                    continue

                result = context.run_recognition(
                    recognition_name,
                    image,
                    _recognition_override(recognition_name, card_roi),
                )
                if not _is_hit(result):
                    continue

                logger.info(f"待办公务第 {card_index} 个区域命中: {recognition_name}")
                if not _run_action(
                    context, recognition_name, result.box, card_roi
                ):
                    logger.warning(f"待办公务卡片 action 执行失败: {recognition_name}")
                    return CustomAction.RunResult(success=False)

                _post_action_delay(context, recognition_name)

                while True:
                    page_result = _wait_for_page(
                        context,
                        recognition_name,
                        followup_name,
                        (followup_name,),
                        "待办公务",
                    )
                    if page_result == "failed":
                        return CustomAction.RunResult(success=False)
                    if page_result == "stop":
                        return CustomAction.RunResult(success=True)
                    if page_result != "retry":
                        break

                    logger.info(
                        f"待办公务复用第 {card_index} 个区域首次命中位置重新点击"
                    )
                    if not _run_action(
                        context, recognition_name, result.box, card_roi
                    ):
                        logger.warning(
                            f"待办公务补充气力后重新点击失败: {recognition_name}"
                        )
                        return CustomAction.RunResult(success=False)
                    _post_action_delay(context, recognition_name)

                # The action changes the screen.  Complete that card before
                # taking the next card screenshot, so each card is examined once.
                followup_result = _run_followup(
                    context, followup_name, allinone_run
                )
                if followup_result == "failed":
                    return CustomAction.RunResult(success=False)
                if followup_result == "stop":
                    return CustomAction.RunResult(success=True)

                handled_any = True
                card_handled = True
                break

            if not card_handled:
                logger.info(f"待办公务第 {card_index} 个区域未命中可执行任务")
                if allinone_run and not _exclude_empty_card_region(
                    context, card_index
                ):
                    return CustomAction.RunResult(success=False)
        return CustomAction.RunResult(success=handled_any)


def _incident_rois(context: Context) -> tuple[tuple[int, int, int, int], ...]:
    rois = []
    for node_name in _INCIDENT_REGION_NODES:
        data = context.get_node_data(node_name)
        roi = data["recognition"]["param"]["roi"]
        if len(roi) != 4:
            raise ValueError(f"{node_name} 的 ROI 必须包含四个值")
        rois.append(tuple(int(value) for value in roi))
    return tuple(rois)


def _run_incident_followup(context: Context) -> bool:
    # Each card is owned by the scanner.  Stop at its reward instead of letting
    # the old full-screen loop search for another 调查 button.
    override = {
        _INCIDENT_FOLLOWUP: {"on_error": []},
        "获取情报收获": {"next": [], "on_error": []},
    }
    detail = context.run_task(_INCIDENT_FOLLOWUP, override)
    status = getattr(detail, "status", None) if detail else None
    succeeded = bool(status and getattr(status, "succeeded", False))
    if not succeeded:
        logger.warning("突发情况后续节点执行失败: 突发情况调查后续")
    return succeeded


@AgentServer.custom_action("BirdFood1TaskScan")
class BirdFood1TaskScan(CustomAction):
    """Scan both 突发情况 cards once and investigate each matching card."""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        allinone_run = _is_allinone_run(context)
        if allinone_run and _begin_category_scan(context, "突发情况") == "skip":
            return CustomAction.RunResult(success=True)
        result = self._scan(context, argv)
        _end_category_scan(
            context,
            "突发情况",
            allinone_run,
            bool(result and getattr(result, "success", False)),
        )
        return result

    def _scan(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        handled_any = False

        try:
            card_rois = _incident_rois(context)
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(f"读取突发情况 ROI 失败: {exc}")
            return CustomAction.RunResult(success=False)

        for card_index, card_roi in enumerate(card_rois, start=1):
            try:
                image = context.tasker.controller.post_screencap().wait().get()
            except Exception:
                logger.exception(f"突发情况第 {card_index} 个区域截图失败")
                return CustomAction.RunResult(success=False)

            if image is None:
                logger.warning(f"突发情况第 {card_index} 个区域没有截图")
                return CustomAction.RunResult(success=False)

            result = context.run_recognition(
                _INCIDENT_RECOGNITION,
                image,
                _recognition_override(_INCIDENT_RECOGNITION, card_roi),
            )
            if not _is_hit(result):
                logger.info(f"突发情况第 {card_index} 个区域未命中可执行任务")
                continue

            logger.info(
                f"突发情况第 {card_index} 个区域命中: {_INCIDENT_RECOGNITION}"
            )
            if not _run_action(
                context,
                _INCIDENT_RECOGNITION,
                result.box,
                card_roi,
                _INCIDENT_BUTTON_OFFSET,
            ):
                logger.warning(
                    f"突发情况卡片 action 执行失败: {_INCIDENT_RECOGNITION}"
                )
                return CustomAction.RunResult(success=False)

            _post_action_delay(context, _INCIDENT_RECOGNITION)

            while True:
                page_result = _wait_for_page(
                    context,
                    _INCIDENT_RECOGNITION,
                    _INCIDENT_RECOGNITION,
                    _INCIDENT_READY_NODES,
                    "突发情况",
                )
                if page_result == "failed":
                    return CustomAction.RunResult(success=False)
                if page_result == "stop":
                    return CustomAction.RunResult(success=True)
                if page_result != "retry":
                    break

                logger.info(f"突发情况复用第 {card_index} 个区域首次命中位置重新点击")
                if not _run_action(
                    context,
                    _INCIDENT_RECOGNITION,
                    result.box,
                    card_roi,
                    _INCIDENT_BUTTON_OFFSET,
                ):
                    logger.warning("突发情况补充气力后重新点击失败")
                    return CustomAction.RunResult(success=False)
                _post_action_delay(context, _INCIDENT_RECOGNITION)

            if not _run_incident_followup(context):
                return CustomAction.RunResult(success=False)
            handled_any = True

        return CustomAction.RunResult(success=handled_any)


@AgentServer.custom_action("BirdFoodCategoryExhausted")
class BirdFoodCategoryExhausted(CustomAction):
    """Disable one exhausted category for the current all-in-one run."""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        try:
            raw_param = argv.custom_action_param
            param = json.loads(raw_param) if isinstance(raw_param, str) else raw_param
            category = param["category"]
            disabled_nodes = _CATEGORY_DISABLED_NODES[category]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.exception("读取鸢报鸟食耗尽类别失败")
            return CustomAction.RunResult(success=False)

        override = {name: {"enabled": False} for name in disabled_nodes}
        if category == "待办公务":
            override[_OFFICE_EXHAUSTED_MARKER] = {"enabled": True}

        # Record the exhaustion in the run-scoped registry FIRST: the
        # override above is bound to the current (often nested) context and
        # does not reach the outer all-in-one rotation, while the registry is
        # read back by the outer-chain gates and scanners.
        _mark_exhausted(category)

        if not context.override_pipeline(override):
            logger.warning(f"禁用鸢报类别检查节点失败: {category}")
            return CustomAction.RunResult(success=False)

        logger.info(f"{category}鸟食已耗尽，本次四合一不再检测该类别")
        detail = context.run_task("进入界面-鸢报")
        status = getattr(detail, "status", None) if detail else None
        if not bool(status and getattr(status, "succeeded", False)):
            logger.warning(f"{category}鸟食耗尽后返回鸢报失败")
            return CustomAction.RunResult(success=False)

        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("BirdFoodRunReset")
class BirdFoodRunReset(CustomAction):
    """Reset the run-scoped exhaustion registry at the all-in-one entry.

    Attached to the 鸟食大礼包启动 pipeline node. The entry node can be
    re-entered mid-run by pipeline loop-backs (e.g. task-level override
    `end_是否继续他的传闻 -> 鸟食大礼包启动`; upstream also routes
    `鸟食耗尽check -> 鸟食大礼包启动`), so the reset is scoped by task_id:
    only a genuinely new task (monotonically increasing ids per tasker
    session) starts with a clean slate.
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        task_id = None
        task_detail = getattr(argv, "task_detail", None)
        if task_detail is not None:
            task_id = getattr(task_detail, "task_id", None)

        if task_id is None:
            # Task id unavailable (should not happen with the bundled
            # binding). Fail safe: never wipe an active run state, otherwise
            # a mid-run loop-back would re-enable exhausted categories.
            with _STATE_LOCK:
                already_active = bool(_RUN_STATE["active"])
            if already_active:
                logger.warning(
                    "鸟食大礼包启动：无法获取任务 ID 且运行状态已激活，"
                    "跳过重置以防同轮回环误清空耗尽状态"
                )
            else:
                logger.warning(
                    "鸟食大礼包启动：无法获取任务 ID，按全新运行重置鸟食耗尽状态跟踪"
                )
                _reset_run_state(None)
            return CustomAction.RunResult(success=True)

        if _reset_run_state(task_id):
            logger.info(
                f"鸢报四合一运行开始（task_id={task_id}），已重置鸟食耗尽状态跟踪"
            )
        else:
            logger.info(
                f"鸢报四合一同轮回环至入口节点（task_id={task_id}），"
                "保留耗尽状态与熔断计数"
            )
        return CustomAction.RunResult(success=True)


@AgentServer.custom_recognition("BirdFoodCategoryGate")
class BirdFoodCategoryGate(CustomRecognition):
    """Outer-chain gate in front of each category's start node.

    Hit  -> the category was not exhausted during this run, scheduling may
            proceed normally.
    Miss -> the category is already exhausted for this run; the gate node
            fails and its on_error routes the rotation to the next chain.
            Repeated misses while exhausted indicate the rotation is stuck on
            this category; after _GATE_MISS_LIMIT misses a breaker disables
            every category's check nodes so the rotation terminates through
            the existing 鸢报四合一-无任务 -> stop path.
    """

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
        try:
            raw_param = argv.custom_recognition_param
            param = json.loads(raw_param) if isinstance(raw_param, str) else raw_param
            category = param["category"]
            if category not in _CATEGORY_DISABLED_NODES:
                raise KeyError(category)
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.exception("读取鸢报类别门控参数失败")
            # Fail closed: a malformed gate must not schedule a category.
            return None

        _ensure_run_state()

        if not _is_exhausted(category):
            with _STATE_LOCK:
                _RUN_STATE["gate_miss"].pop(category, None)
            roi = argv.roi
            box = [roi.x, roi.y, roi.w, roi.h] if roi else [0, 0, 0, 0]
            return CustomRecognition.AnalyzeResult(
                box=box, detail={"gate": "pass", "category": category}
            )

        count = _bump_gate_miss(category)
        logger.warning(
            f"{category}本次四合一已确认鸟食耗尽但仍被调度（第 {count} 次），门控拦截"
        )
        if count >= _GATE_MISS_LIMIT:
            logger.error(
                f"{category}耗尽后被重复调度达 {_GATE_MISS_LIMIT} 次，触发循环熔断："
                "禁用全部类别检查节点，鸢报四合一将通过无任务路径终止"
            )
            override = {}
            for exhausted_category in _ALL_CATEGORIES:
                for node_name in _CATEGORY_DISABLED_NODES[exhausted_category]:
                    override[node_name] = {"enabled": False}
            if override and not context.override_pipeline(override):
                logger.error("循环熔断禁用覆盖失败")
        return None


__all__ = [
    "BirdFood1TaskScan",
    "BirdFood4TaskScan",
    "BirdFoodCategoryExhausted",
    "BirdFoodCategoryGate",
    "BirdFoodRunReset",
]
