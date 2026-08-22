from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from custom.action.inventory_reporting import (
    AUTO_UPLOAD_MODE,
    LOCAL_ONLY_MODE,
    append_inventory_report,
    build_exchange_document,
    read_upload_settings,
    update_upload_status,
    upload_inventory_document,
)
from custom.action.paged_item_recognition import (
    SnapshotList,
    _parse_params,
    _resolve_inventory_report_context,
)
from custom.reco.agent_item import recognize_count
from utils import logger


ACQUISITION_CHANNEL = "密探升级"
SETTLE_DELAY_MS = 1800
COUNT_RECOGNITION_PARAMS = {
    "count_box": [-35, 15, 95, 50],
    "count_binary_threshold": 165,
    "count_max_digits": 6,
}


@dataclass(frozen=True)
class ExperienceBookSlot:
    item_id: str
    item_name: str
    center: tuple[float, float]


EXPERIENCE_BOOK_SLOTS = (
    ExperienceBookSlot("bingshucanjuan", "兵书残卷", (94.0, 1145.0)),
    ExperienceBookSlot("bingshuquanjuan", "兵书全卷", (212.0, 1144.0)),
    ExperienceBookSlot("liutaobingshu", "六韬兵书", (327.0, 1145.0)),
)
EXPERIENCE_BOOK_SNAPSHOT = SnapshotList(
    "experience-books",
    "item",
    tuple(slot.item_id for slot in EXPERIENCE_BOOK_SLOTS),
)


def recognize_experience_book_counts(
    image: np.ndarray,
    params: dict | None = None,
) -> dict[str, int]:
    recognition_params = dict(COUNT_RECOGNITION_PARAMS)
    if params:
        recognition_params.update(params)

    counts: dict[str, int] = {}
    for slot in EXPERIENCE_BOOK_SLOTS:
        count, score, raw, count_box = recognize_count(
            None,
            image,
            slot.center,
            recognition_params,
        )
        if count is None or count < 0:
            raise ValueError(
                f"无法识别{slot.item_name}数量: raw={raw!r}, "
                f"score={score:.4f}, box={count_box}"
            )
        counts[slot.item_id] = count
    return counts


def has_confirmed_consumption(
    before: dict[str, int],
    after: dict[str, int],
) -> bool:
    expected = {slot.item_id for slot in EXPERIENCE_BOOK_SLOTS}
    if set(before) != expected or set(after) != expected:
        raise ValueError("经验书数量必须完整包含三种固定道具")
    return all(after[item_id] <= before[item_id] for item_id in expected) and any(
        after[item_id] < before[item_id] for item_id in expected
    )


def build_experience_book_snapshot_results(counts: dict[str, int]) -> list[dict]:
    expected = {slot.item_id for slot in EXPERIENCE_BOOK_SLOTS}
    if set(counts) != expected:
        raise ValueError("经验书库存快照必须完整包含三种固定道具")
    return [
        {
            "entity_type": "item",
            "item_id": slot.item_id,
            "item_name": slot.item_name,
            "count": counts[slot.item_id],
        }
        for slot in EXPERIENCE_BOOK_SLOTS
    ]


def _save_and_upload_stock_snapshot(
    context: Context,
    params: dict,
    counts: dict[str, int],
) -> Path:
    upload_settings = read_upload_settings(context)
    bound_account, report_path = _resolve_inventory_report_context(
        params,
        EXPERIENCE_BOOK_SNAPSHOT,
        upload_settings,
    )
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    document = build_exchange_document(
        build_experience_book_snapshot_results(counts),
        uuid.uuid4().hex,
        timestamp,
        timestamp,
        ACQUISITION_CHANNEL,
        "stock_snapshot",
        "listed",
        bound_account.id if bound_account is not None else None,
        None,
    )
    initial_status = (
        "等待自动上报"
        if upload_settings.mode == AUTO_UPLOAD_MODE
        else "仅保存到本地"
    )
    append_inventory_report(report_path, document, initial_status)
    if upload_settings.mode == LOCAL_ONLY_MODE:
        return report_path

    upload_result = upload_inventory_document(document, upload_settings)
    record_ids = [record["record_id"] for record in document["records"]]
    final_status = (
        f"自动上报成功（{upload_result.message}）"
        if upload_result.success
        else f"自动上报失败（{upload_result.message}）"
    )
    try:
        update_upload_status(report_path, record_ids, final_status)
    except Exception as exc:
        logger.warning(
            "【广陵库房】无法更新经验书库存 TXT 上报状态，"
            f"原始记录仍然有效: {exc}"
        )
    if not upload_result.success:
        logger.warning(
            "【广陵库房】经验书库存自动上报失败，"
            f"response={upload_result.message}；记录已保存至 {report_path}"
        )
    return report_path


def _click_point(argv: CustomAction.RunArg, params: dict) -> tuple[int, int]:
    box = getattr(argv, "box", None)
    if box is not None and int(box.w) > 0 and int(box.h) > 0:
        return int(box.x + box.w // 2), int(box.y + box.h // 2)

    target = params.get("click_target", [473, 1137, 171, 77])
    if not isinstance(target, (list, tuple)) or len(target) != 4:
        raise ValueError("click_target 必须为 [x, y, width, height]")
    x, y, width, height = (int(value) for value in target)
    if width <= 0 or height <= 0:
        raise ValueError("click_target 的宽高必须大于 0")
    return x + width // 2, y + height // 2


@AgentServer.custom_action("ExperienceBookStockReporting")
class ExperienceBookStockReporting(CustomAction):
    """Click level-up once and report the three experience-book stocks if consumed."""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        try:
            params = _parse_params(argv.custom_action_param)
            click_x, click_y = _click_point(argv, params)
        except Exception as exc:
            logger.exception(f"ExperienceBookStockReporting 参数无效: {exc}")
            return CustomAction.RunResult(success=False)

        controller = context.tasker.controller
        before: dict[str, int] | None = None
        try:
            image = controller.post_screencap().wait().get()
            before = recognize_experience_book_counts(image, params)
        except Exception as exc:
            logger.warning(f"密探升级前无法读取经验书库存，将仅执行升级: {exc}")

        try:
            click_job = controller.post_click(click_x, click_y)
        except Exception as exc:
            logger.exception(f"密探升级点击提交失败: {exc}")
            return CustomAction.RunResult(success=False)

        # The click may already have reached the game even if waiting for its job fails.
        # Do not return failure after submission, otherwise the pipeline may click twice.
        try:
            click_job.wait()
        except Exception as exc:
            logger.warning(f"密探升级点击已提交，但等待动作完成失败: {exc}")
            return CustomAction.RunResult(success=True)

        if before is None:
            return CustomAction.RunResult(success=True)

        try:
            delay_ms = int(params.get("settle_delay_ms", SETTLE_DELAY_MS))
            if delay_ms < 0 or delay_ms > 5000:
                raise ValueError("settle_delay_ms 必须位于 0 至 5000")
            if delay_ms:
                time.sleep(delay_ms / 1000)
            image = controller.post_screencap().wait().get()
            after = recognize_experience_book_counts(image, params)
            if not has_confirmed_consumption(before, after):
                logger.info(
                    "密探升级后未确认经验书库存下降，跳过库存同步："
                    f"before={before}, after={after}"
                )
                return CustomAction.RunResult(success=True)

            report_path = _save_and_upload_stock_snapshot(context, params, after)
            logger.info(
                "【广陵库房】密探升级后的经验书库存已记录："
                f"before={before}, after={after}, report={report_path}"
            )
        except Exception as exc:
            logger.exception(
                "密探升级点击已完成，但经验书库存同步失败；为避免重复升级，"
                f"继续后续任务: {exc}"
            )
        return CustomAction.RunResult(success=True)
