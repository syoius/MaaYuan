from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from custom.reco.agent_item import (
    DEFAULT_INDEX_PATH,
    LayoutCell,
    _clip_rect,
    _load_index,
    _parse_auto_grid_hint,
    _parse_rect,
    _record_results,
    _resolve_path,
    _resolve_record_path,
    auto_recognition_params,
    detect_auto_layout,
    recognize_item_grid,
)
from utils import logger


@dataclass
class PageScan:
    results: list[dict]
    rejected: list[dict]
    row_features: list[dict[int, np.ndarray]]
    column_count: int
    layout: dict


def _parse_params(raw: Any) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        params = raw
    elif isinstance(raw, str):
        params = json.loads(raw)
    else:
        raise ValueError("custom_action_param 必须是 JSON 对象")
    if not isinstance(params, dict):
        raise ValueError("custom_action_param 必须是 JSON 对象")
    return params


def _should_stop(context: Context) -> bool:
    try:
        if bool(getattr(context, "stop", False)):
            return True
        tasker = getattr(context, "tasker", None)
        if tasker is not None:
            if bool(getattr(tasker, "stopping", False)):
                return True
            if not tasker.running:
                return True
    except Exception:
        return False
    return False


def _circle_feature(image: np.ndarray, cell: LayoutCell) -> np.ndarray:
    x, y, width, height = cell.box
    patch = image[y : y + height, x : x + width]
    resized = cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA)
    vector = resized.astype(np.float32).reshape(-1)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        raise ValueError("物品圆形区域没有有效图像特征")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def scan_page(
    image: np.ndarray,
    index,
    roi: tuple[int, int, int, int],
    grid_hint: tuple[int | None, int | None],
    params: dict,
    context: Context | None,
) -> PageScan:
    cells, layout = detect_auto_layout(image, index, roi, grid_hint, params)
    if not cells:
        raise RuntimeError(f"分页物品识别未检测到完整圆形物品: {layout}")

    effective_params = auto_recognition_params(params)
    results, rejected = recognize_item_grid(
        image,
        index,
        roi,
        (1, 1),
        effective_params,
        context,
        layout_cells=cells,
    )
    row_count = max(cell.row for cell in cells) + 1
    row_features: list[dict[int, np.ndarray]] = [dict() for _ in range(row_count)]
    for cell in cells:
        row_features[cell.row][cell.column] = _circle_feature(image, cell)
    column_count = max(cell.column for cell in cells) + 1
    return PageScan(results, rejected, row_features, column_count, layout)


def _row_similarity(
    previous: dict[int, np.ndarray], current: dict[int, np.ndarray]
) -> float | None:
    common_columns = sorted(set(previous).intersection(current))
    required = min(2, len(previous), len(current))
    if len(common_columns) < required:
        return None
    scores = [
        float(previous[column] @ current[column]) for column in common_columns
    ]
    return float(np.mean(scores))


def find_row_overlap(
    previous: list[dict[int, np.ndarray]],
    current: list[dict[int, np.ndarray]],
    threshold: float,
) -> tuple[int, list[float]]:
    for count in range(min(len(previous), len(current)), 0, -1):
        scores: list[float] = []
        for previous_row, current_row in zip(previous[-count:], current[:count]):
            score = _row_similarity(previous_row, current_row)
            if score is None or score < threshold:
                break
            scores.append(score)
        if len(scores) == count:
            return count, scores
    return 0, []


def _parse_swipe(
    params: dict,
) -> tuple[tuple[int, int, int, int, int] | None, int]:
    default_duration = int(params.get("swipe_duration", 500))
    if default_duration <= 0:
        raise ValueError("swipe_duration 必须大于 0")
    configured = params.get("swipe")
    if configured is not None:
        if not isinstance(configured, (list, tuple)) or len(configured) not in {4, 5}:
            raise ValueError("swipe 必须为 [x1, y1, x2, y2] 或再追加 duration")
        values = [int(value) for value in configured]
        duration = values[4] if len(values) == 5 else default_duration
        if duration <= 0:
            raise ValueError("swipe duration 必须大于 0")
        return (values[0], values[1], values[2], values[3], duration), duration
    return None, default_duration


def automatic_swipe(
    layout: dict,
    roi: tuple[int, int, int, int],
    duration: int,
) -> tuple[int, int, int, int, int]:
    roi_x, roi_y, roi_width, roi_height = roi
    columns = sorted(float(value) for value in layout.get("column_centers", []))
    rows = sorted(float(value) for value in layout.get("row_centers", []))
    if not columns or not rows:
        raise ValueError("自动滑动缺少行列中心")

    roi_center_x = roi_x + roi_width / 2
    if len(columns) >= 2:
        gaps = [(left + right) / 2 for left, right in zip(columns, columns[1:])]
        swipe_x = min(gaps, key=lambda value: abs(value - roi_center_x))
    else:
        radius_range = layout.get("circle_radius", [0, 0])
        radius = sum(float(value) for value in radius_range) / 2
        left_edge = columns[0] - radius
        right_edge = columns[0] + radius
        left_width = max(0.0, left_edge - roi_x)
        right_width = max(0.0, roi_x + roi_width - right_edge)
        if left_width >= right_width:
            swipe_x = roi_x + left_width / 2
        else:
            swipe_x = right_edge + right_width / 2

    if len(rows) >= 2:
        start_y, end_y = rows[-1], rows[0]
    else:
        start_y = roi_y + roi_height * 0.82
        end_y = roi_y + roi_height * 0.37
    return (
        int(round(swipe_x)),
        int(round(start_y)),
        int(round(swipe_x)),
        int(round(end_y)),
        duration,
    )


def _prefer_result(candidate: dict, current: dict) -> bool:
    candidate_score = (
        float(candidate.get("match_score", 0.0)),
        float(candidate.get("count_score", 0.0)),
    )
    current_score = (
        float(current.get("match_score", 0.0)),
        float(current.get("count_score", 0.0)),
    )
    return candidate_score > current_score


@AgentServer.custom_action("PagedItemRecognition")
class PagedItemRecognition(CustomAction):
    """Scan a scrollable circular-item list and write one de-duplicated report."""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        try:
            params = _parse_params(argv.custom_action_param)
            if str(params.get("layout_mode", "auto")).lower() != "auto":
                raise ValueError("PagedItemRecognition 只支持 layout_mode: auto")
            acquisition_channel_value = params.get("acquisition_channel", "")
            if not isinstance(acquisition_channel_value, str):
                raise ValueError("acquisition_channel 必须是字符串")
            acquisition_channel = acquisition_channel_value.strip()
            if len(acquisition_channel) > 64:
                raise ValueError("acquisition_channel 不能超过 64 个字符")

            roi = _parse_rect(params.get("roi"), "roi")
            grid_hint = _parse_auto_grid_hint(params.get("grid"))
            index_path = _resolve_path(
                params.get(
                    "index_path", params.get("npz_path", params.get("npx_path"))
                ),
                DEFAULT_INDEX_PATH,
            )
            record_path = _resolve_record_path(
                params.get("record_path", params.get("output_path"))
            )
            index = _load_index(index_path)
            max_pages = int(params.get("max_pages", 10))
            if max_pages < 2:
                raise ValueError("max_pages 必须至少为 2")
            overlap_threshold = float(params.get("overlap_threshold", 0.90))
            if not math.isfinite(overlap_threshold) or not 0 < overlap_threshold <= 1:
                raise ValueError("overlap_threshold 必须位于 (0, 1]")
            settle_seconds = max(0.0, float(params.get("swipe_wait_ms", 700)) / 1000)
            configured_swipe, swipe_duration = _parse_swipe(params)

            previous: PageScan | None = None
            previous_row_ids: list[int] = []
            next_row_id = 0
            max_columns = 1
            collected: dict[tuple[int, int], dict] = {}
            reached_bottom = False

            for page_number in range(max_pages):
                if _should_stop(context):
                    logger.info("PagedItemRecognition: 任务已停止")
                    return CustomAction.RunResult(success=False)
                image = context.tasker.controller.post_screencap().wait().get()
                if image is None:
                    raise RuntimeError("分页物品识别截图失败")
                if _clip_rect(roi, image) != roi:
                    raise ValueError(f"roi 超出截图范围: {roi}, image={image.shape}")

                page = scan_page(image, index, roi, grid_hint, params, context)
                max_columns = max(max_columns, page.column_count)
                if previous is None:
                    overlap = 0
                    overlap_scores: list[float] = []
                    row_ids = list(range(len(page.row_features)))
                    next_row_id = len(row_ids)
                else:
                    overlap, overlap_scores = find_row_overlap(
                        previous.row_features,
                        page.row_features,
                        overlap_threshold,
                    )
                    row_ids = previous_row_ids[-overlap:] if overlap else []
                    new_row_count = len(page.row_features) - overlap
                    row_ids.extend(range(next_row_id, next_row_id + new_row_count))
                    next_row_id += new_row_count

                for result in page.results:
                    page_row = int(result["row"])
                    virtual_row = row_ids[page_row]
                    column = int(result["column"])
                    candidate = dict(result)
                    candidate["row"] = virtual_row
                    key = (virtual_row, column)
                    current = collected.get(key)
                    if current is None or _prefer_result(candidate, current):
                        collected[key] = candidate

                logger.info(
                    "PagedItemRecognition: "
                    f"page={page_number + 1}, circles={page.layout['detected_count']}, "
                    f"recognized={len(page.results)}, overlap={overlap}, "
                    f"overlap_scores={[round(score, 3) for score in overlap_scores]}"
                )

                if previous is not None and overlap == len(page.row_features):
                    reached_bottom = True
                    break
                if page_number + 1 >= max_pages:
                    break

                previous = page
                previous_row_ids = row_ids
                swipe = configured_swipe or automatic_swipe(
                    page.layout, roi, swipe_duration
                )
                logger.info(f"PagedItemRecognition: swipe={swipe}")
                context.tasker.controller.post_swipe(*swipe).wait()
                if settle_seconds:
                    time.sleep(settle_seconds)

            if not reached_bottom:
                logger.error(
                    f"PagedItemRecognition: 达到 max_pages={max_pages} 仍未确认列表底部，"
                    "本次不写入报告"
                )
                return CustomAction.RunResult(success=False)

            results: list[dict] = []
            for (row, column), result in sorted(collected.items()):
                output = dict(result)
                output["slot"] = row * max_columns + column
                output["acquisition_channel"] = acquisition_channel
                results.append(output)

            timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            invocation_id = uuid.uuid4().hex
            _record_results(
                record_path,
                timestamp,
                invocation_id,
                "paged",
                index_path,
                results,
            )
            logger.info(
                f"PagedItemRecognition: 完成，共 {next_row_id} 个物理行、"
                f"{len(results)} 个目标，已记录至 {record_path}"
            )
            return CustomAction.RunResult(success=True)
        except Exception as exc:
            logger.exception(f"PagedItemRecognition 失败: {exc}")
            return CustomAction.RunResult(success=False)
