from __future__ import annotations

import csv
import json
import math
import re
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from utils import logger

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_PATH = REPO_ROOT / "agent" / "dispatch-reward-index.npz"
DEFAULT_DIGIT_INDEX_PATH = REPO_ROOT / "agent" / "agent-item-digit-index.npz"
DEFAULT_RECORD_PATH = REPO_ROOT / "AgentItemReport.csv"
COUNT_OCR_NODE = "AgentItemCountOCR_Internal"
CSV_FIELDS = [
    "timestamp",
    "invocation_id",
    "mode",
    "acquisition_channel",
    "slot",
    "row",
    "column",
    "entity_type",
    "item_id",
    "item_name",
    "agent_id",
    "operator_id",
    "operator_name",
    "count",
    "count_score",
    "count_raw",
    "coarse_score",
    "match_score",
    "match_scale",
    "refined",
    "refine_group",
    "refine_score",
    "refine_margin",
    "cell_box",
    "match_box",
    "item_center",
    "count_box",
    "index_path",
]

_CACHE_LOCK = threading.Lock()
_RECORD_LOCK = threading.Lock()
_INDEX_CACHE: dict[Path, tuple[int, int, "AgentIndex"]] = {}
_DIGIT_INDEX_CACHE: dict[Path, tuple[int, int, "DigitIndex"]] = {}


@dataclass(frozen=True)
class RefineGroup:
    group_id: str
    member_indices: tuple[int, ...]
    box: tuple[int, int, int, int]
    threshold: float
    min_margin: float
    expand_candidates: bool


@dataclass(frozen=True)
class AgentIndex:
    path: Path
    entity_type: str
    entity_types: np.ndarray
    agent_ids: np.ndarray
    operator_ids: np.ndarray
    operator_names: np.ndarray
    features: np.ndarray
    variants: tuple[tuple[float, np.ndarray], ...]
    feature_size: tuple[int, int]
    refine_groups: tuple[RefineGroup, ...]


@dataclass(frozen=True)
class DigitIndex:
    path: Path
    labels: np.ndarray
    features: np.ndarray
    feature_size: tuple[int, int]
    content_size: tuple[int, int]
    count_box: tuple[int, int, int, int]
    binary_threshold: int
    badge_threshold: int
    badge_close_kernel: tuple[int, int]


@dataclass
class AgentMatch:
    index: int
    entity_type: str
    agent_id: str
    operator_id: str
    operator_name: str
    coarse_score: float
    match_score: float
    match_scale: float
    match_box: tuple[int, int, int, int]
    item_center: tuple[float, float]
    refined: bool = False
    refine_group: str = ""
    refine_score: float = 0.0
    refine_margin: float = 0.0


@dataclass(frozen=True)
class LayoutCell:
    slot: int
    row: int
    column: int
    box: tuple[int, int, int, int]
    center: tuple[float, float]


def _resolve_path(value: Any, default: Path) -> Path:
    if value in (None, ""):
        return default.resolve()
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _resolve_record_path(value: Any) -> Path:
    path = _resolve_path(value, DEFAULT_RECORD_PATH)
    if not path.suffix:
        path = path.with_suffix(".csv")
    return path


def _template_array_key(scale: float) -> str:
    scale_percent = int(round(float(scale) * 100))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"NPZ 包含无效缩放比例: {scale}")
    if not math.isclose(float(scale) * 100, scale_percent, abs_tol=1e-4):
        raise ValueError(f"NPZ 缩放比例最多支持两位小数: {scale}")
    return f"templates_{scale_percent:03d}"


def _load_index(path: Path) -> AgentIndex:
    stat = path.stat()
    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(path)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]

    with np.load(path, allow_pickle=False) as data:
        required = {
            "agent_ids",
            "operator_ids",
            "operator_names",
            "features_gray",
            "feature_size",
            "scales",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"NPZ 缺少字段: {sorted(missing)}")

        scales = [float(value) for value in data["scales"]]
        if not scales:
            raise ValueError("NPZ 至少需要一个模板缩放比例")
        template_keys = tuple(_template_array_key(scale) for scale in scales)
        if len(template_keys) != len(set(template_keys)):
            raise ValueError("NPZ scales 生成了重复模板键")
        missing_templates = set(template_keys) - set(data.files)
        if missing_templates:
            raise ValueError(f"NPZ 缺少模板数组: {sorted(missing_templates)}")

        entity_type = (
            str(data["entity_type"].item()) if "entity_type" in data.files else "agent"
        )
        agent_ids = (
            data["item_ids"].astype(str, copy=True)
            if "item_ids" in data.files
            else data["agent_ids"].astype(str, copy=True)
        )
        operator_names = (
            data["item_names"].astype(str, copy=True)
            if "item_names" in data.files
            else data["operator_names"].astype(str, copy=True)
        )
        operator_ids = (
            data["operator_ids"].astype(str, copy=True)
            if "operator_ids" in data.files
            else agent_ids.copy()
        )
        entity_types = (
            data["entity_types"].astype(str, copy=True)
            if "entity_types" in data.files
            else np.full(len(agent_ids), entity_type)
        )
        id_to_index = {
            str(item_id): position for position, item_id in enumerate(agent_ids)
        }
        refine_groups: list[RefineGroup] = []
        if "refine_groups_json" in data.files:
            raw_groups = json.loads(str(data["refine_groups_json"].item()))
            if not isinstance(raw_groups, list):
                raise ValueError("NPZ refine_groups_json 必须是数组")
            occupied: set[int] = set()
            for raw_group in raw_groups:
                group_id = str(raw_group.get("id", ""))
                member_ids = [str(value) for value in raw_group.get("item_ids", [])]
                unknown = set(member_ids) - set(id_to_index)
                if unknown:
                    raise ValueError(
                        f"NPZ 复核组 {group_id} 包含未知道具: {sorted(unknown)}"
                    )
                members = tuple(id_to_index[item_id] for item_id in member_ids)
                if len(members) < 2 or len(members) != len(set(members)):
                    raise ValueError(f"NPZ 复核组成员异常: {group_id}")
                overlap = occupied.intersection(members)
                if overlap:
                    raise ValueError(f"NPZ 复核组成员重复归组: {group_id}")
                occupied.update(members)
                box = tuple(int(value) for value in raw_group.get("box", []))
                if len(box) != 4:
                    raise ValueError(f"NPZ 复核组 box 异常: {group_id}")
                threshold = float(raw_group.get("threshold", 0.9))
                min_margin = float(raw_group.get("min_margin", 0.08))
                if not (
                    0 <= threshold <= 1
                    and 0 <= min_margin <= 1
                    and str(raw_group.get("mode", "color_ncc")) == "color_ncc"
                ):
                    raise ValueError(f"NPZ 复核组参数异常: {group_id}")
                refine_groups.append(
                    RefineGroup(
                        group_id=group_id,
                        member_indices=members,
                        box=box,
                        threshold=threshold,
                        min_margin=min_margin,
                        expand_candidates=bool(
                            raw_group.get("expand_candidates", True)
                        ),
                    )
                )

        index = AgentIndex(
            path=path,
            entity_type=entity_type,
            entity_types=entity_types,
            agent_ids=agent_ids,
            operator_ids=operator_ids,
            operator_names=operator_names,
            features=np.ascontiguousarray(data["features_gray"], dtype=np.float32),
            variants=tuple(
                (
                    scale,
                    np.ascontiguousarray(data[key], dtype=np.uint8),
                )
                for scale, key in zip(scales, template_keys)
            ),
            feature_size=tuple(int(value) for value in data["feature_size"]),
            refine_groups=tuple(refine_groups),
        )

    count = len(index.agent_ids)
    if not (
        len(index.entity_types)
        == len(index.operator_ids)
        == len(index.operator_names)
        == index.features.shape[0]
        == count
    ):
        raise ValueError("NPZ 中对象类型、元数据与特征数量不一致")
    invalid_entity_types = sorted(set(index.entity_types) - {"agent", "item"})
    if invalid_entity_types:
        raise ValueError(f"NPZ 包含无效对象类型: {invalid_entity_types}")
    if index.features.shape[1] != index.feature_size[0] * index.feature_size[1]:
        raise ValueError("NPZ 中 feature_size 与特征维数不一致")
    for _, templates in index.variants:
        if len(templates) != count or templates.ndim != 4 or templates.shape[-1] != 3:
            raise ValueError("NPZ 中完整模板数组尺寸异常")
        for group in index.refine_groups:
            x, y, width, height = group.box
            scale_width = int(round((x + width) * templates.shape[2] / 70))
            scale_height = int(round((y + height) * templates.shape[1] / 58))
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(f"NPZ 复核框异常: {group.group_id}")
            if scale_width > templates.shape[2] or scale_height > templates.shape[1]:
                raise ValueError(f"NPZ 复核框超出模板: {group.group_id}")

    with _CACHE_LOCK:
        _INDEX_CACHE[path] = (stat.st_mtime_ns, stat.st_size, index)
    logger.info(
        f"角色物品识别：已加载索引 {path}，类型 {index.entity_type}，条目数 {count}"
    )
    return index


def _load_digit_index(path: Path) -> DigitIndex:
    stat = path.stat()
    with _CACHE_LOCK:
        cached = _DIGIT_INDEX_CACHE.get(path)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]

    with np.load(path, allow_pickle=False) as data:
        required = {
            "labels",
            "features",
            "feature_size",
            "glyph_content_size",
            "count_box",
            "binary_threshold",
            "badge_threshold",
            "badge_close_kernel",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"数字 NPZ 缺少字段: {sorted(missing)}")
        index = DigitIndex(
            path=path,
            labels=data["labels"].astype(str, copy=True),
            features=np.ascontiguousarray(data["features"], dtype=np.float32),
            feature_size=tuple(int(value) for value in data["feature_size"]),
            content_size=tuple(int(value) for value in data["glyph_content_size"]),
            count_box=tuple(int(value) for value in data["count_box"]),
            binary_threshold=int(data["binary_threshold"].item()),
            badge_threshold=int(data["badge_threshold"].item()),
            badge_close_kernel=tuple(
                int(value) for value in data["badge_close_kernel"]
            ),
        )

    if index.features.ndim != 2 or len(index.labels) != len(index.features):
        raise ValueError("数字 NPZ 的标签与特征数量不一致")
    if index.features.shape[1] != index.feature_size[0] * index.feature_size[1]:
        raise ValueError("数字 NPZ 的 feature_size 与特征维数不一致")
    if set(index.labels) != set("0123456789"):
        raise ValueError("数字 NPZ 必须包含 0 至 9 的全部字形")
    if len(index.count_box) != 4:
        raise ValueError("数字 NPZ 的 count_box 尺寸异常")

    with _CACHE_LOCK:
        _DIGIT_INDEX_CACHE[path] = (stat.st_mtime_ns, stat.st_size, index)
    logger.info(f"角色物品识别：已加载数字索引 {path}，字形数 {len(index.labels)}")
    return index


def _parse_grid(value: Any) -> tuple[int, int]:
    if value in (None, 1, "1"):
        return 1, 1
    if isinstance(value, (list, tuple)) and len(value) == 2:
        columns, rows = int(value[0]), int(value[1])
        if columns > 0 and rows > 0:
            return columns, rows
    raise ValueError("grid 必须为 1 或 [列数, 行数]，例如 [4, 2]")


def _parse_auto_grid_hint(value: Any) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        columns = int(value)
        if columns > 0:
            return columns, None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        columns, rows = int(value[0]), int(value[1])
        if columns > 0 and rows > 0:
            return columns, rows
    raise ValueError("auto 模式的 grid 必须省略、为列数或为 [列数, 行数]")


def _parse_rect(value: Any, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} 必须为 [x, y, w, h]")
    rect = tuple(int(item) for item in value)
    if rect[2] <= 0 or rect[3] <= 0:
        raise ValueError(f"{name} 的宽高必须大于 0")
    return rect


def _parse_feature_y_offsets(value: Any) -> tuple[int, ...]:
    if value is None:
        return (0,)
    if isinstance(value, (int, float)):
        values = [int(value)]
    elif isinstance(value, (list, tuple)):
        values = [int(item) for item in value]
    else:
        raise ValueError("feature_y_offsets 必须是整数或整数数组")
    offsets = tuple(dict.fromkeys(values))
    if not offsets or len(offsets) > 32:
        raise ValueError("feature_y_offsets 必须包含 1 至 32 项")
    if any(abs(offset) > 64 for offset in offsets):
        raise ValueError("feature_y_offsets 的绝对值不能超过 64")
    return offsets


def _clip_rect(
    rect: tuple[int, int, int, int], image: np.ndarray
) -> tuple[int, int, int, int] | None:
    x, y, width, height = rect
    image_height, image_width = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _relative_rect(
    center: tuple[float, float], value: Any, name: str
) -> tuple[int, int, int, int]:
    offset_x, offset_y, width, height = _parse_rect(value, name)
    return (
        int(round(center[0] + offset_x)),
        int(round(center[1] + offset_y)),
        width,
        height,
    )


def _crop(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = rect
    return image[y : y + height, x : x + width]


def _normalized_gray_feature(
    image: np.ndarray, expected_size: tuple[int, int]
) -> np.ndarray:
    expected_width, expected_height = expected_size
    if image.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            f"特征裁剪应为 {expected_width}x{expected_height}，实际为 "
            f"{image.shape[1]}x{image.shape[0]}"
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vector = gray.astype(np.float32).reshape(-1)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        raise ValueError("特征裁剪没有有效对比度")
    return vector / norm


def _normalized_color_feature(image: np.ndarray) -> np.ndarray:
    vector = image.astype(np.float32).reshape(-1)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        raise ValueError("彩色复核区域没有有效对比度")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(max(1, count), len(scores))
    if count == len(scores):
        return np.argsort(scores)[::-1]
    partition = np.argpartition(scores, -count)[-count:]
    return partition[np.argsort(scores[partition])[::-1]]


def recognize_agent_in_cell(
    image: np.ndarray,
    center: tuple[float, float],
    index: AgentIndex,
    params: dict,
) -> tuple[AgentMatch | None, dict]:
    feature_rect = _parse_rect(
        params.get("feature_box", [-24, -27, 48, 44]),
        "feature_box",
    )
    if feature_rect[2:] != index.feature_size:
        return None, {"reason": "feature-size-mismatch"}

    coarse_scores: np.ndarray | None = None
    coarse_offsets = np.zeros(len(index.agent_ids), dtype=np.int32)
    for offset_y in _parse_feature_y_offsets(params.get("feature_y_offsets")):
        feature_box = _relative_rect(
            center,
            [feature_rect[0], feature_rect[1] + offset_y, *feature_rect[2:]],
            "feature_box",
        )
        feature_box = _clip_rect(feature_box, image)
        if feature_box is None or feature_box[2:] != index.feature_size:
            continue
        feature = _normalized_gray_feature(
            _crop(image, feature_box), index.feature_size
        )
        scores = feature @ index.features.T
        if coarse_scores is None:
            coarse_scores = scores
            coarse_offsets.fill(offset_y)
            continue
        improved = scores > coarse_scores
        coarse_scores[improved] = scores[improved]
        coarse_offsets[improved] = offset_y

    if coarse_scores is None:
        return None, {"reason": "feature-box-out-of-range"}
    top_k = int(params.get("top_k", 5))
    candidates = [int(value) for value in _top_indices(coarse_scores, top_k)]
    candidate_set = set(candidates)
    for group in index.refine_groups:
        if group.expand_candidates and candidate_set.intersection(group.member_indices):
            for member in group.member_indices:
                if member not in candidate_set:
                    candidates.append(member)
                    candidate_set.add(member)

    match_box = _relative_rect(
        center,
        params.get("match_search_box", [-41, -38, 82, 71]),
        "match_search_box",
    )
    match_box = _clip_rect(match_box, image)
    if match_box is None:
        return None, {"reason": "match-box-out-of-range"}
    search = _crop(image, match_box)

    best: tuple[float, int, float, tuple[int, int], tuple[int, int]] | None = None
    best_by_candidate: dict[
        int, tuple[float, int, float, tuple[int, int], tuple[int, int]]
    ] = {}
    for candidate in candidates:
        candidate_index = int(candidate)
        for scale, templates in index.variants:
            template = templates[candidate_index]
            if (
                search.shape[0] < template.shape[0]
                or search.shape[1] < template.shape[1]
            ):
                continue
            matched = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(matched)
            current = (
                float(score),
                candidate_index,
                float(scale),
                (int(location[0]), int(location[1])),
                (template.shape[1], template.shape[0]),
            )
            previous = best_by_candidate.get(candidate_index)
            if previous is None or current[0] > previous[0]:
                best_by_candidate[candidate_index] = current
            if best is None or current[0] > best[0]:
                best = current

    if best is None:
        return None, {"reason": "no-template-fit"}

    match_score, best_index, best_scale, location, template_size = best
    refined = False
    refine_group_id = ""
    refine_score = 0.0
    refine_margin = 0.0
    refine_group = next(
        (group for group in index.refine_groups if best_index in group.member_indices),
        None,
    )
    if refine_group is not None and bool(params.get("enable_refine", True)):
        provisional_box = (
            match_box[0] + location[0],
            match_box[1] + location[1],
            template_size[0],
            template_size[1],
        )
        query_template = _crop(image, provisional_box)
        variant_templates = next(
            templates
            for scale, templates in index.variants
            if math.isclose(scale, best_scale, abs_tol=1e-6)
        )
        base_x, base_y, base_width, base_height = refine_group.box
        template_width, template_height = template_size
        refine_x1 = int(round(base_x * template_width / 70))
        refine_y1 = int(round(base_y * template_height / 58))
        refine_x2 = int(round((base_x + base_width) * template_width / 70))
        refine_y2 = int(round((base_y + base_height) * template_height / 58))
        refine_x2 = max(refine_x1 + 1, min(template_width, refine_x2))
        refine_y2 = max(refine_y1 + 1, min(template_height, refine_y2))
        query_feature = _normalized_color_feature(
            query_template[refine_y1:refine_y2, refine_x1:refine_x2]
        )
        refine_scores = sorted(
            (
                float(
                    query_feature
                    @ _normalized_color_feature(
                        variant_templates[member][
                            refine_y1:refine_y2,
                            refine_x1:refine_x2,
                        ]
                    )
                ),
                member,
            )
            for member in refine_group.member_indices
        )
        refine_scores.sort(reverse=True)
        refine_score, refined_index = refine_scores[0]
        refine_margin = refine_score - refine_scores[1][0]
        refine_group_id = refine_group.group_id
        diagnostics = {
            "best_agent_id": str(index.agent_ids[best_index]),
            "coarse_score": float(coarse_scores[best_index]),
            "match_score": match_score,
            "match_scale": best_scale,
            "refined": True,
            "refine_group": refine_group_id,
            "refine_score": refine_score,
            "refine_margin": refine_margin,
            "refine_best_item_id": str(index.agent_ids[refined_index]),
            "refine_runner_up_item_id": str(index.agent_ids[refine_scores[1][1]]),
            "refine_runner_up_score": refine_scores[1][0],
        }
        refine_threshold = float(params.get("refine_threshold", refine_group.threshold))
        min_refine_margin = float(
            params.get("refine_min_margin", refine_group.min_margin)
        )
        if refine_score < refine_threshold:
            diagnostics["reason"] = "refine-score-below-threshold"
            return None, diagnostics
        if refine_margin < min_refine_margin:
            diagnostics["reason"] = "refine-margin-below-threshold"
            return None, diagnostics
        selected = best_by_candidate.get(refined_index)
        if selected is None:
            raise RuntimeError(
                f"复核候选未进入完整匹配: {index.agent_ids[refined_index]}"
            )
        best = selected
        match_score, best_index, best_scale, location, template_size = best
        refined = True

    coarse_score = float(coarse_scores[best_index])
    diagnostics = {
        "best_agent_id": str(index.agent_ids[best_index]),
        "coarse_score": coarse_score,
        "match_score": match_score,
        "match_scale": best_scale,
        "refined": refined,
        "refine_group": refine_group_id,
        "refine_score": refine_score,
        "refine_margin": refine_margin,
    }
    if coarse_score < float(params.get("coarse_threshold", 0.0)):
        diagnostics["reason"] = "coarse-score-below-threshold"
        return None, diagnostics
    if match_score < float(params.get("match_threshold", 0.90)):
        diagnostics["reason"] = "match-score-below-threshold"
        return None, diagnostics

    absolute_match_box = (
        match_box[0] + location[0],
        match_box[1] + location[1],
        template_size[0],
        template_size[1],
    )
    template_feature_x = (template_size[0] - index.feature_size[0]) // 2
    template_feature_y = (template_size[1] - index.feature_size[1]) // 2
    item_center = (
        float(absolute_match_box[0] + template_feature_x - feature_rect[0]),
        float(absolute_match_box[1] + template_feature_y - feature_rect[1]),
    )
    diagnostics["coarse_feature_y_offset"] = int(coarse_offsets[best_index])
    diagnostics["item_center"] = list(item_center)
    return (
        AgentMatch(
            index=best_index,
            entity_type=str(index.entity_types[best_index]),
            agent_id=str(index.agent_ids[best_index]),
            operator_id=str(index.operator_ids[best_index]),
            operator_name=str(index.operator_names[best_index]),
            coarse_score=coarse_score,
            match_score=match_score,
            match_scale=best_scale,
            match_box=absolute_match_box,
            item_center=item_center,
            refined=refined,
            refine_group=refine_group_id,
            refine_score=refine_score,
            refine_margin=refine_margin,
        ),
        diagnostics,
    )


def _ocr_results(detail: Any) -> list[Any]:
    if not detail:
        return []
    results = getattr(detail, "filtered_results", None)
    if results is None:
        results = getattr(detail, "filterd_results", None)
    if not results:
        best = getattr(detail, "best_result", None)
        return [best] if best else []
    return list(results)


def _normalize_digits(text: str) -> str:
    output: list[str] = []
    for character in str(text):
        try:
            output.append(str(unicodedata.digit(character)))
        except (TypeError, ValueError):
            output.append(character)
    match = re.search(r"\d+", "".join(output))
    return match.group(0) if match else ""


def _normalized_digit_feature(
    glyph: np.ndarray,
    feature_size: tuple[int, int],
    content_size: tuple[int, int],
) -> np.ndarray:
    height, width = glyph.shape
    content_width, content_height = content_size
    scale = min(content_width / width, content_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        glyph,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    feature_width, feature_height = feature_size
    canvas = np.zeros((feature_height, feature_width), dtype=np.uint8)
    x = (feature_width - resized_width) // 2
    y = (feature_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    vector = canvas.astype(np.float32).reshape(-1) / 255.0
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        raise ValueError("数量字形没有有效前景")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def recognize_count_digits(
    image: np.ndarray,
    center: tuple[float, float],
    params: dict,
) -> tuple[Optional[int], float, str, tuple[int, int, int, int]]:
    digit_index_path = _resolve_path(
        params.get("digit_index_path"), DEFAULT_DIGIT_INDEX_PATH
    )
    index = _load_digit_index(digit_index_path)
    count_box = _relative_rect(
        center,
        params.get("count_box", list(index.count_box)),
        "count_box",
    )
    clipped_box = _clip_rect(count_box, image)
    if clipped_box is None or clipped_box[2:] != count_box[2:]:
        return None, 0.0, "", count_box

    crop = _crop(image, clipped_box)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    binary_threshold = int(params.get("count_binary_threshold", index.binary_threshold))
    badge_threshold = int(params.get("count_badge_threshold", index.badge_threshold))
    kernel_width, kernel_height = index.badge_close_kernel
    dark_badge = np.where(gray < badge_threshold, 255, 0).astype(np.uint8)
    badge = cv2.morphologyEx(
        dark_badge,
        cv2.MORPH_CLOSE,
        np.ones((kernel_height, kernel_width), dtype=np.uint8),
    )
    binary = np.where((gray >= binary_threshold) & (badge > 0), 255, 0).astype(np.uint8)
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        binary
    )

    groups: list[dict] = []
    width, height = clipped_box[2], clipped_box[3]
    for component in range(1, component_count):
        glyph_x, glyph_y, glyph_width, glyph_height, area = (
            int(value) for value in stats[component]
        )
        touches_edge = (
            glyph_x == 0
            or glyph_y == 0
            or glyph_x + glyph_width == width
            or glyph_y + glyph_height == height
        )
        if touches_edge:
            continue
        if not (
            glyph_y >= 11
            and 1 <= glyph_width <= 18
            and 5 <= glyph_height <= 30
            and 8 <= area <= 250
        ):
            continue
        target = next(
            (
                group
                for group in groups
                if min(glyph_x + glyph_width, group["x"] + group["width"])
                - max(glyph_x, group["x"])
                >= 1
            ),
            None,
        )
        if target is None:
            groups.append(
                {
                    "x": glyph_x,
                    "y": glyph_y,
                    "width": glyph_width,
                    "height": glyph_height,
                    "area": area,
                    "components": [component],
                }
            )
            continue
        x1 = min(target["x"], glyph_x)
        y1 = min(target["y"], glyph_y)
        x2 = max(target["x"] + target["width"], glyph_x + glyph_width)
        y2 = max(target["y"] + target["height"], glyph_y + glyph_height)
        target.update(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            area=target["area"] + area,
        )
        target["components"].append(component)

    groups = [
        group
        for group in groups
        if 1 <= group["width"] <= 18
        and 12 <= group["height"] <= 20
        and 8 <= group["area"] <= 250
    ]
    groups.sort(key=lambda group: group["x"])
    if groups:
        run = [groups[-1]]
        for previous in reversed(groups[:-1]):
            following = run[-1]
            gap = following["x"] - (previous["x"] + previous["width"])
            previous_bottom = previous["y"] + previous["height"]
            following_bottom = following["y"] + following["height"]
            if gap > 5 or abs(previous_bottom - following_bottom) > 5:
                break
            run.append(previous)
        groups = list(reversed(run))

    glyphs: list[np.ndarray] = []
    for group in groups:
        glyph_x, glyph_y = group["x"], group["y"]
        glyph_width, glyph_height = group["width"], group["height"]
        region = component_labels[
            glyph_y : glyph_y + glyph_height,
            glyph_x : glyph_x + glyph_width,
        ]
        glyphs.append(
            np.where(np.isin(region, group["components"]), 255, 0).astype(np.uint8)
        )

    max_digits = max(1, int(params.get("count_max_digits", 6)))
    if not glyphs or len(glyphs) > max_digits:
        return None, 0.0, "", clipped_box

    digit_threshold = float(params.get("count_digit_threshold", 0.45))
    digits: list[str] = []
    scores: list[float] = []
    for glyph in glyphs:
        feature = _normalized_digit_feature(
            glyph, index.feature_size, index.content_size
        )
        similarities = index.features @ feature
        per_digit = {
            digit: float(similarities[index.labels == digit].max())
            for digit in "0123456789"
        }
        digit, score = max(per_digit.items(), key=lambda item: item[1])
        if score < digit_threshold:
            return None, score, "".join(digits), clipped_box
        digits.append(digit)
        scores.append(score)

    raw = "".join(digits)
    try:
        return int(raw), min(scores), raw, clipped_box
    except ValueError:
        return None, min(scores), raw, clipped_box


def _run_count_ocr(
    context: Context,
    image: np.ndarray,
    model: str,
    threshold: float,
) -> tuple[str, float, str] | None:
    override = {
        COUNT_OCR_NODE: {
            "recognition": {
                "type": "OCR",
                "param": {
                    "model": model,
                    "only_rec": True,
                    "threshold": 0.1,
                },
            }
        }
    }
    detail = context.run_recognition(COUNT_OCR_NODE, image, override)
    candidates: list[tuple[str, float, str]] = []
    for result in _ocr_results(detail):
        raw = str(getattr(result, "text", ""))
        digits = _normalize_digits(raw)
        score = float(getattr(result, "score", 0.0))
        if digits and score >= threshold:
            candidates.append((digits, score, raw))
    return (
        max(candidates, key=lambda item: (item[1], len(item[0])))
        if candidates
        else None
    )


def recognize_count_ocr(
    context: Context,
    image: np.ndarray,
    center: tuple[float, float],
    params: dict,
) -> tuple[Optional[int], float, str, tuple[int, int, int, int]]:
    wide_box = _relative_rect(
        center,
        params.get("count_box", [5, 23, 47, 28]),
        "count_box",
    )
    clipped_wide = _clip_rect(wide_box, image)
    if clipped_wide is None:
        return None, 0.0, "", wide_box

    scale = max(1, int(params.get("count_ocr_scale", 4)))
    threshold = float(params.get("count_ocr_threshold", 0.45))
    model = str(params.get("count_ocr_model", ""))

    wide = _crop(image, clipped_wide)
    wide_gray = cv2.cvtColor(wide, cv2.COLOR_BGR2GRAY)
    wide_prepared = cv2.resize(
        cv2.cvtColor(wide_gray, cv2.COLOR_GRAY2BGR),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    candidates: list[tuple[str, float, str]] = []
    wide_result = _run_count_ocr(context, wide_prepared, model, threshold)
    if wide_result:
        candidates.append(wide_result)

    tight_box = _relative_rect(
        center,
        params.get("count_tight_box", [25, 26, 27, 24]),
        "count_tight_box",
    )
    clipped_tight = _clip_rect(tight_box, image)
    if clipped_tight is not None:
        tight = _crop(image, clipped_tight)
        tight_gray = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
        tight_gray = cv2.copyMakeBorder(
            tight_gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0
        )
        tight_prepared = cv2.resize(
            cv2.cvtColor(tight_gray, cv2.COLOR_GRAY2BGR),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        tight_result = _run_count_ocr(context, tight_prepared, model, threshold)
        if tight_result:
            candidates.append(tight_result)

    if not candidates:
        return None, 0.0, "", clipped_wide

    # Length bonus preserves 12/300 from the wide crop when a tight crop sees
    # only the last digit with nearly identical confidence.
    digits, score, raw = max(
        candidates, key=lambda item: (item[1] + 0.015 * len(item[0]), len(item[0]))
    )
    try:
        return int(digits), score, raw, clipped_wide
    except ValueError:
        return None, score, raw, clipped_wide


def recognize_count(
    context: Context | None,
    image: np.ndarray,
    center: tuple[float, float],
    params: dict,
) -> tuple[Optional[int], float, str, tuple[int, int, int, int]]:
    mode = str(params.get("count_mode", "digit_template")).strip().lower()
    if mode in {"digit", "digits", "digit_template", "template"}:
        return recognize_count_digits(image, center, params)
    if mode == "ocr":
        if context is None:
            count_box = _relative_rect(
                center,
                params.get("count_box", [5, 23, 47, 28]),
                "count_box",
            )
            return None, 0.0, "", count_box
        return recognize_count_ocr(context, image, center, params)
    raise ValueError(f"不支持的 count_mode: {mode}")


def _cell_boxes(
    roi: tuple[int, int, int, int], columns: int, rows: int
) -> list[tuple[int, int, int, int, int, int]]:
    roi_x, roi_y, roi_width, roi_height = roi
    boxes: list[tuple[int, int, int, int, int, int]] = []
    for row in range(rows):
        y1 = roi_y + round(row * roi_height / rows)
        y2 = roi_y + round((row + 1) * roi_height / rows)
        for column in range(columns):
            x1 = roi_x + round(column * roi_width / columns)
            x2 = roi_x + round((column + 1) * roi_width / columns)
            boxes.append((x1, y1, x2 - x1, y2 - y1, row, column))
    return boxes


def _fixed_layout_cells(
    roi: tuple[int, int, int, int], grid: tuple[int, int]
) -> list[LayoutCell]:
    columns, rows = grid
    return [
        LayoutCell(
            slot=slot,
            row=row,
            column=column,
            box=(x, y, width, height),
            center=(x + width / 2.0, y + height / 2.0),
        )
        for slot, (x, y, width, height, row, column) in enumerate(
            _cell_boxes(roi, columns, rows)
        )
    ]


def _auto_circle_radius_range(index: AgentIndex, params: dict) -> tuple[int, int]:
    configured = params.get("auto_circle_radius")
    if configured is not None:
        if not isinstance(configured, (list, tuple)) or len(configured) != 2:
            raise ValueError("auto_circle_radius 必须为 [最小半径, 最大半径]")
        min_radius, max_radius = (int(configured[0]), int(configured[1]))
        if min_radius <= 0 or max_radius < min_radius:
            raise ValueError("auto_circle_radius 的半径范围无效")
        return min_radius, max_radius

    template_widths = [int(templates.shape[2]) for _, templates in index.variants]
    return (
        max(8, int(round(min(template_widths) * 0.60))),
        max(10, int(round(max(template_widths) * 0.98))),
    )


def _cluster_circles(
    circles: list[tuple[float, float, float]],
    axis: int,
    tolerance: float,
) -> list[list[tuple[float, float, float]]]:
    groups: list[list[tuple[float, float, float]]] = []
    for circle in sorted(circles, key=lambda item: item[axis]):
        if not groups:
            groups.append([circle])
            continue
        center = sum(item[axis] for item in groups[-1]) / len(groups[-1])
        if abs(circle[axis] - center) <= tolerance:
            groups[-1].append(circle)
        else:
            groups.append([circle])
    return groups


def detect_auto_layout(
    image: np.ndarray,
    index: AgentIndex,
    roi: tuple[int, int, int, int],
    grid_hint: tuple[int | None, int | None],
    params: dict,
) -> tuple[list[LayoutCell], dict]:
    roi_x, roi_y, roi_width, roi_height = roi
    crop = _crop(image, roi)
    gray = cv2.medianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 5)
    min_radius, max_radius = _auto_circle_radius_range(index, params)
    min_distance = float(params.get("auto_circle_min_distance", min_radius * 1.7))
    accumulator_threshold = float(params.get("auto_circle_threshold", 35))
    detected = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_distance,
        param1=80,
        param2=accumulator_threshold,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if detected is None:
        return [], {
            "reason": "auto-layout-no-circles",
            "circle_radius": [min_radius, max_radius],
        }

    circles = [
        (float(x + roi_x), float(y + roi_y), float(radius))
        for x, y, radius in detected[0]
        if x - radius >= 0
        and y - radius >= 0
        and x + radius < roi_width
        and y + radius < roi_height
    ]
    if not circles:
        return [], {
            "reason": "auto-layout-no-complete-circles",
            "circle_radius": [min_radius, max_radius],
        }

    median_radius = float(np.median([circle[2] for circle in circles]))
    row_tolerance = float(
        params.get("auto_row_tolerance", max(12.0, median_radius * 0.55))
    )
    column_tolerance = float(
        params.get("auto_column_tolerance", max(12.0, median_radius * 0.75))
    )
    rows = _cluster_circles(circles, axis=1, tolerance=row_tolerance)
    columns = _cluster_circles(circles, axis=0, tolerance=column_tolerance)
    row_centers = [sum(item[1] for item in group) / len(group) for group in rows]
    column_centers = [sum(item[0] for item in group) / len(group) for group in columns]

    columns_hint, rows_hint = grid_hint
    if columns_hint is not None and len(columns) > columns_hint:
        raise ValueError(
            f"auto 模式检测到 {len(columns)} 列，超过 grid 提示的 {columns_hint} 列"
        )
    if rows_hint is not None and len(rows) > rows_hint:
        raise ValueError(
            f"auto 模式检测到 {len(rows)} 行，超过 grid 提示的 {rows_hint} 行"
        )

    slot_columns = columns_hint or len(columns)
    cells: list[LayoutCell] = []
    for row, group in enumerate(rows):
        for center_x, center_y, radius in sorted(group, key=lambda item: item[0]):
            column = int(
                np.argmin([abs(center_x - anchor) for anchor in column_centers])
            )
            diameter = max(1, int(round(radius * 2)))
            cells.append(
                LayoutCell(
                    slot=row * slot_columns + column,
                    row=row,
                    column=column,
                    box=(
                        int(round(center_x - radius)),
                        int(round(center_y - radius)),
                        diameter,
                        diameter,
                    ),
                    center=(center_x, center_y),
                )
            )
    cells.sort(key=lambda cell: (cell.row, cell.column))
    detail = {
        "circle_radius": [min_radius, max_radius],
        "detected_count": len(cells),
        "inferred_grid": [len(columns), len(rows)],
        "row_centers": [round(value, 2) for value in row_centers],
        "column_centers": [round(value, 2) for value in column_centers],
        "centers": [
            [round(cell.center[0], 2), round(cell.center[1], 2)] for cell in cells
        ],
    }
    return cells, detail


def recognize_item_grid(
    image: np.ndarray,
    index: AgentIndex,
    roi: tuple[int, int, int, int],
    grid: tuple[int, int],
    params: dict,
    context: Context | None,
    layout_cells: list[LayoutCell] | None = None,
) -> tuple[list[dict], list[dict]]:
    min_roi_bottom_distance = float(params.get("count_min_roi_bottom_distance", 0))
    if not math.isfinite(min_roi_bottom_distance) or min_roi_bottom_distance < 0:
        raise ValueError("count_min_roi_bottom_distance 必须是非负有限数")
    roi_bottom = roi[1] + roi[3]
    recognized: list[dict] = []
    rejected: list[dict] = []
    cells = layout_cells if layout_cells is not None else _fixed_layout_cells(roi, grid)
    for cell in cells:
        slot, row, column = cell.slot, cell.row, cell.column
        x, y, width, height = cell.box
        center = cell.center
        matched, diagnostics = recognize_agent_in_cell(image, center, index, params)
        if matched is None:
            rejected.append(
                {
                    "slot": slot,
                    "row": row,
                    "column": column,
                    "cell_box": [x, y, width, height],
                    **diagnostics,
                }
            )
            continue

        count: Optional[int] = None
        count_score = 0.0
        count_raw = ""
        count_box = (0, 0, 0, 0)
        recognize_count_enabled = bool(params.get("recognize_count", True))
        if recognize_count_enabled:
            roi_bottom_distance = roi_bottom - matched.item_center[1]
            if roi_bottom_distance < min_roi_bottom_distance:
                rejected.append(
                    {
                        "slot": slot,
                        "row": row,
                        "column": column,
                        "cell_box": [x, y, width, height],
                        "entity_type": matched.entity_type,
                        "item_id": matched.agent_id,
                        "item_name": matched.operator_name,
                        "best_agent_id": matched.agent_id,
                        "operator_id": matched.operator_id,
                        "operator_name": matched.operator_name,
                        "coarse_score": matched.coarse_score,
                        "match_score": matched.match_score,
                        "match_scale": matched.match_scale,
                        "refined": matched.refined,
                        "refine_group": matched.refine_group,
                        "refine_score": matched.refine_score,
                        "refine_margin": matched.refine_margin,
                        "match_box": list(matched.match_box),
                        "item_center": list(matched.item_center),
                        "count_box": list(count_box),
                        "count_score": count_score,
                        "count_raw": count_raw,
                        "roi_bottom_distance": roi_bottom_distance,
                        "reason": "count-too-close-to-roi-bottom",
                    }
                )
                continue
            count, count_score, count_raw, count_box = recognize_count(
                context, image, matched.item_center, params
            )
            min_count = max(0, int(params.get("count_min_value", 1)))
            if bool(params.get("count_required", True)) and (
                count is None or count < min_count
            ):
                rejected.append(
                    {
                        "slot": slot,
                        "row": row,
                        "column": column,
                        "cell_box": [x, y, width, height],
                        "entity_type": matched.entity_type,
                        "item_id": matched.agent_id,
                        "item_name": matched.operator_name,
                        "best_agent_id": matched.agent_id,
                        "operator_id": matched.operator_id,
                        "operator_name": matched.operator_name,
                        "coarse_score": matched.coarse_score,
                        "match_score": matched.match_score,
                        "match_scale": matched.match_scale,
                        "refined": matched.refined,
                        "refine_group": matched.refine_group,
                        "refine_score": matched.refine_score,
                        "refine_margin": matched.refine_margin,
                        "match_box": list(matched.match_box),
                        "item_center": list(matched.item_center),
                        "count_box": list(count_box),
                        "count_score": count_score,
                        "count_raw": count_raw,
                        "reason": "count-not-recognized",
                    }
                )
                continue

        recognized.append(
            {
                "slot": slot,
                "row": row,
                "column": column,
                "entity_type": matched.entity_type,
                "item_id": matched.agent_id,
                "item_name": matched.operator_name,
                "agent_id": matched.agent_id,
                "operator_id": matched.operator_id,
                "operator_name": matched.operator_name,
                "count": count,
                "count_score": count_score,
                "count_raw": count_raw,
                "coarse_score": matched.coarse_score,
                "match_score": matched.match_score,
                "match_scale": matched.match_scale,
                "refined": matched.refined,
                "refine_group": matched.refine_group,
                "refine_score": matched.refine_score,
                "refine_margin": matched.refine_margin,
                "cell_box": [x, y, width, height],
                "match_box": list(matched.match_box),
                "item_center": list(matched.item_center),
                "count_box": list(count_box),
            }
        )
    return recognized, rejected


def recognize_item_auto(
    image: np.ndarray,
    index: AgentIndex,
    roi: tuple[int, int, int, int],
    grid_hint: tuple[int | None, int | None],
    params: dict,
    context: Context | None,
) -> tuple[list[dict], list[dict], dict]:
    cells, layout = detect_auto_layout(image, index, roi, grid_hint, params)
    if not cells:
        return [], [layout], layout
    auto_params = auto_recognition_params(params)
    recognized, rejected = recognize_item_grid(
        image,
        index,
        roi,
        (1, 1),
        auto_params,
        context,
        layout_cells=cells,
    )
    return recognized, rejected, layout


def auto_recognition_params(params: dict) -> dict:
    output = dict(params)
    output.setdefault("top_k", 20)
    output.setdefault("feature_y_offsets", [-4, 0, 4])
    output.setdefault("match_search_box", [-45, -43, 90, 86])
    return output


def _record_results(
    path: Path,
    timestamp: str,
    invocation_id: str,
    mode: str,
    index_path: Path,
    results: list[dict],
) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp": timestamp,
            "invocation_id": invocation_id,
            "mode": mode,
            "acquisition_channel": result.get("acquisition_channel", ""),
            **result,
            "cell_box": json.dumps(result["cell_box"], ensure_ascii=False),
            "match_box": json.dumps(result["match_box"], ensure_ascii=False),
            "item_center": json.dumps(result["item_center"], ensure_ascii=False),
            "count_box": json.dumps(result["count_box"], ensure_ascii=False),
            "index_path": str(index_path),
        }
        for result in results
    ]

    with _RECORD_LOCK:
        if path.suffix.lower() in {".txt", ".jsonl"}:
            with path.open("a", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
            return

        if path.exists() and path.stat().st_size:
            _upgrade_csv_fields(path)
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerows(rows)


def _upgrade_csv_fields(path: Path) -> None:
    """Add newly introduced report columns to an existing compatible CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        existing_fields = reader.fieldnames
        if existing_fields == CSV_FIELDS:
            return
        if not existing_fields:
            raise ValueError(f"报告 CSV 缺少表头: {path}")
        unknown_fields = [field for field in existing_fields if field not in CSV_FIELDS]
        if unknown_fields:
            raise ValueError(f"报告 CSV 含有无法迁移的字段 {unknown_fields}: {path}")
        rows = list(reader)

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@AgentServer.custom_recognition("AgentItemRecognition")
class AgentItemRecognition(CustomRecognition):
    """Recognize agent-associated items and quantities, then append a log.

    Required custom_recognition_param:
        roi: [x, y, width, height].

    Common optional parameters:
        index_path / npz_path / npx_path:
            Recognition NPZ.  Relative paths are resolved from the repo root.
        record_path / output_path:
            .csv appends CSV rows; .txt/.jsonl appends one JSON object per row.
        layout_mode:
            fixed (default) evenly divides the ROI by ``grid``; auto detects
            the circular item backgrounds and does not require ``grid``.
        grid:
            In fixed mode, 1 or [1, 1] selects single mode and
            [columns, rows] selects batch mode.  In auto mode it may be
            omitted, be an expected column count, or be [columns, rows].
        auto_circle_radius:
            Optional [minimum, maximum] circle radius.  By default it is
            inferred from the loaded template scale.
        top_k: coarse candidates to verify (fixed default 5; auto default 20).
        match_threshold: full-template threshold (default 0.90).
        recognize_count: enable quantity recognition (default true).
        count_required: reject a matched portrait when quantity recognition
            fails (default true).
        count_min_roi_bottom_distance: reject a matched portrait whose center
            is closer than this many pixels to the ROI bottom (default 0).
        count_mode: digit_template (default) or ocr.
        digit_index_path: 0-9 glyph NPZ (default agent/agent-item-digit-index.npz).
        count_max_digits: maximum accepted quantity length (default 6).
        count_digit_threshold: per-digit template threshold (default 0.45).
        count_ocr_model/count_ocr_threshold: used only in legacy OCR mode.
        feature_y_offsets: optional vertical offsets for a scrolling grid.
        enable_refine: enable index-defined conditional color refinement
            (default true).
        refine_threshold/refine_min_margin: optionally override the index's
            refinement acceptance thresholds.

    Pixel boxes below are relative to each grid cell's center and can be
    overridden for another UI layout:
        feature_box: [-24, -27, 48, 44]
        match_search_box: [-41, -38, 82, 71]
        count_box: loaded from the digit index; currently [-35, 20, 95, 44].
        count_tight_box: [25, 26, 27, 24] (legacy OCR mode only).
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        try:
            params = json.loads(argv.custom_recognition_param or "{}")
            if not isinstance(params, dict):
                raise ValueError("custom_recognition_param 必须是对象")

            image = argv.image
            roi = _parse_rect(params.get("roi"), "roi")
            clipped_roi = _clip_rect(roi, image)
            if clipped_roi != roi:
                raise ValueError(f"roi 超出截图范围: {roi}, image={image.shape}")
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
            layout_mode = str(params.get("layout_mode", "fixed")).strip().lower()
            layout_detail: dict = {}
            if layout_mode == "auto":
                grid_hint = _parse_auto_grid_hint(params.get("grid"))
                results, rejected, layout_detail = recognize_item_auto(
                    image, index, roi, grid_hint, params, context
                )
                grid: tuple[int, int] | None = None
                grid_log: Any = grid_hint
                mode = "auto"
            elif layout_mode == "fixed":
                grid = _parse_grid(params.get("grid", 1))
                results, rejected = recognize_item_grid(
                    image, index, roi, grid, params, context
                )
                grid_log = grid
                mode = "single" if grid == (1, 1) else "batch"
            else:
                raise ValueError("layout_mode 必须为 fixed 或 auto")
            if not results:
                logger.info(
                    f"角色物品识别：未识别到有效条目，layout={layout_mode}, "
                    f"roi={roi}, grid={grid_log}, "
                    f"rejected={rejected}"
                )
                return None

            timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            invocation_id = uuid.uuid4().hex
            _record_results(
                record_path,
                timestamp,
                invocation_id,
                mode,
                index_path,
                results,
            )

            detail = {
                "timestamp": timestamp,
                "invocation_id": invocation_id,
                "mode": mode,
                "layout_mode": layout_mode,
                "roi": list(roi),
                "grid": list(grid) if grid is not None else params.get("grid"),
                "layout": layout_detail,
                "index_path": str(index_path),
                "record_path": str(record_path),
                "results": results,
                "rejected": rejected,
            }
            logger.info(
                f"角色物品识别：识别到 {len(results)} 个条目，"
                f"已记录至 {record_path}"
            )
            return CustomRecognition.AnalyzeResult(
                box=roi,
                detail=json.dumps(detail, ensure_ascii=False),
            )
        except Exception as exc:
            logger.exception(f"角色物品识别失败: {exc}")
            return None
