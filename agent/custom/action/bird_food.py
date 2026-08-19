"""Region-based processing for the 鸢报 task screens."""

from __future__ import annotations

import time
from typing import Iterable

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

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
_ENERGY_STOP = "new不吃鸟食"
_CARD_BUTTON_OFFSET = (72, 421, 120, 32)

_INCIDENT_REGION_NODES = ("突发情况区域1", "突发情况区域2")
_INCIDENT_RECOGNITION = "点击前往调查"
_INCIDENT_FOLLOWUP = "突发情况调查后续"
_INCIDENT_BUTTON_OFFSET = (201, 354, 210, 25)


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
    if _ENERGY_STOP in next_names:
        return _ENERGY_STOP
    if _ENERGY_RESTORE in next_names:
        return _ENERGY_RESTORE
    return None


def _card_rois(context: Context) -> tuple[tuple[int, int, int, int], ...]:
    rois = []
    for node_name in _CARD_REGION_NODES:
        data = context.get_node_data(node_name)
        roi = data["recognition"]["param"]["roi"]
        if len(roi) != 4:
            raise ValueError(f"{node_name} 的 ROI 必须包含四个值")
        rois.append(tuple(int(value) for value in roi))
    return tuple(rois)


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


def _run_followup(context: Context, task_name: str) -> bool:
    # Do not let a failed follow-up jump back into the card scanner.  The card
    # has already been consumed and must not be recognized a second time.
    detail = context.run_task(task_name, {task_name: {"on_error": []}})
    status = getattr(detail, "status", None) if detail else None
    succeeded = bool(status and getattr(status, "succeeded", False))
    if not succeeded:
        logger.warning(f"待办公务后续节点执行失败: {task_name}")
    return succeeded


def _handle_energy_popup(
    context: Context,
    recognition_name: str,
    followup_name: str,
    image,
    screen_name: str = "待办公务",
) -> str:
    """Return ``continue``, ``retry``, ``stop``, or ``failed``."""
    handler = _energy_handler(context, recognition_name, followup_name)
    if handler is None:
        return "continue"

    popup = context.run_recognition(handler, image)
    if not _is_hit(popup):
        return "continue"

    logger.info(f"{screen_name}检测到气力弹窗，执行: {handler}")
    override = {handler: {"on_error": []}}
    if handler == _ENERGY_RESTORE:
        # Exit back to the card list without using the original global card
        # recognition.  The saved hit box is clicked again by the caller.
        override["退出气力回复页面"] = {"next": [], "on_error": []}

    detail = context.run_task(handler, override)
    status = getattr(detail, "status", None) if detail else None
    if not bool(status and getattr(status, "succeeded", False)):
        logger.warning(f"{screen_name}气力弹窗处理失败: {handler}")
        return "failed"

    if handler == _ENERGY_STOP or context.tasker.stopping:
        return "stop"
    return "retry"


@AgentServer.custom_action("BirdFood4TaskScan")
class BirdFood4TaskScan(CustomAction):
    """Scan each 待办公务 card once and execute its matching existing task."""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        handled_any = False

        try:
            card_rois = _card_rois(context)
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(f"读取待办公务 ROI 失败: {exc}")
            return CustomAction.RunResult(success=False)

        for card_index, card_roi in enumerate(card_rois, start=1):
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

                popup_image = context.tasker.controller.post_screencap().wait().get()
                if popup_image is None:
                    logger.warning("待办公务点击卡片后没有截图")
                    return CustomAction.RunResult(success=False)

                popup_result = _handle_energy_popup(
                    context,
                    recognition_name,
                    followup_name,
                    popup_image,
                )
                if popup_result == "failed":
                    return CustomAction.RunResult(success=False)
                if popup_result == "stop":
                    return CustomAction.RunResult(success=True)
                if popup_result == "retry":
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
                if not _run_followup(context, followup_name):
                    return CustomAction.RunResult(success=False)

                handled_any = True
                card_handled = True
                break

            if not card_handled:
                logger.info(f"待办公务第 {card_index} 个区域未命中可执行任务")
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

            popup_image = context.tasker.controller.post_screencap().wait().get()
            if popup_image is None:
                logger.warning("突发情况点击卡片后没有截图")
                return CustomAction.RunResult(success=False)

            popup_result = _handle_energy_popup(
                context,
                _INCIDENT_RECOGNITION,
                _INCIDENT_RECOGNITION,
                popup_image,
                "突发情况",
            )
            if popup_result == "failed":
                return CustomAction.RunResult(success=False)
            if popup_result == "stop":
                return CustomAction.RunResult(success=True)
            if popup_result == "retry":
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


__all__ = ["BirdFood1TaskScan", "BirdFood4TaskScan"]
