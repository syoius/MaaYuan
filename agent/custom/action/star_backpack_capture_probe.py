from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_ONLY_MODE = "capture_only"
PAIR_PROBE_MODE = "pair_probe"
FEEDBACK_PROBE_MODE = "feedback_probe"
CONTINUOUS_CAPTURE_MODE = "continuous_capture"
NORMAL_MOTION_MIN_PX = 560.0
NORMAL_MOTION_MAX_PX = 660.0
NORMAL_ACCEPTANCE_EPSILON_PX = 1.0
MIN_DIRECT_OVERLAP_PX = 80
DIRECT_OVERLAP_MARGIN_PX = 6
DIRECT_NORMAL_MIN_SCORE = 0.70
DIRECT_NORMAL_LOCAL_RADIUS_PX = 2
DIRECT_HYPOTHESIS_DEDUPE_PX = 12.0
# YuanStar's phone_9_16 profile uses a 0.079 viewport-width card radius.  In
# the calibrated 720-wide phone captures that is about 56.9 px, or 0.34 of the
# observed 166.5 px row cadence.  Its complete-card geometry is defined in
# card-completeness.ts as level y-r*1.06 through name y+r*1.70.
YUANSTAR_ROW_RADIUS_TO_PITCH_RATIO = 0.34
YUANSTAR_ENVELOPE_TOP_RADIUS = 1.06
YUANSTAR_ENVELOPE_BOTTOM_RADIUS = 1.70
SEMANTIC_ENVELOPE_SAFETY_MARGIN_PX = 10.0
SUPPORTED_MODES = frozenset(
    {
        CAPTURE_ONLY_MODE,
        PAIR_PROBE_MODE,
        FEEDBACK_PROBE_MODE,
        CONTINUOUS_CAPTURE_MODE,
    }
)


def _parse_params(raw: Any) -> dict[str, Any]:
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


def _parse_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} 必须不小于 {minimum}")
    return result


def _parse_rect(value: Any, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} 必须是 [x, y, width, height]")
    x, y, width, height = (
        _parse_int(value[0], f"{name}[0]", minimum=0),
        _parse_int(value[1], f"{name}[1]", minimum=0),
        _parse_int(value[2], f"{name}[2]", minimum=1),
        _parse_int(value[3], f"{name}[3]", minimum=1),
    )
    return x, y, width, height


def _parse_swipe(value: Any, name: str = "swipe") -> tuple[int, int, int, int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"必须显式提供 {name}")
    start = value.get("start")
    end = value.get("end")
    if not isinstance(start, (list, tuple)) or len(start) != 2:
        raise ValueError(f"{name}.start 必须是 [x, y]")
    if not isinstance(end, (list, tuple)) or len(end) != 2:
        raise ValueError(f"{name}.end 必须是 [x, y]")
    return (
        _parse_int(start[0], f"{name}.start[0]", minimum=0),
        _parse_int(start[1], f"{name}.start[1]", minimum=0),
        _parse_int(end[0], f"{name}.end[0]", minimum=0),
        _parse_int(end[1], f"{name}.end[1]", minimum=0),
        _parse_int(value.get("duration_ms"), f"{name}.duration_ms", minimum=1),
    )


def _parse_unit_interval(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 0 到 1 的数字") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} 必须是 0 到 1 的数字")
    return result


def _parse_int_range(value: Any, name: str, minimum: int = 0) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须是 [min, max]")
    lower = _parse_int(value[0], f"{name}[0]", minimum=minimum)
    upper = _parse_int(value[1], f"{name}[1]", minimum=lower)
    return lower, upper


def _parse_positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正数") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} 必须是正数")
    return result


def _parse_bool(value: Any, name: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值")
    return value


def _parse_diagnostic_feedback(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("feedback 必须是对象")
    safe_min, safe_max = _parse_int_range(
        value.get("diagnostic_safe_shift_px"), "feedback.diagnostic_safe_shift_px", 1
    )
    expected_min, expected_max = _parse_int_range(
        value.get("diagnostic_expected_shift_px"), "feedback.diagnostic_expected_shift_px", 1
    )
    physical_min, physical_max = _parse_int_range(
        value.get("diagnostic_physical_shift_px"), "feedback.diagnostic_physical_shift_px", 1
    )
    target_min, target_max = _parse_int_range(
        value.get("diagnostic_target_shift_px"), "feedback.diagnostic_target_shift_px", 1
    )
    normal_safe_min, normal_safe_max = _parse_int_range(
        value.get("diagnostic_normal_safe_shift_px"),
        "feedback.diagnostic_normal_safe_shift_px",
        1,
    )
    max_micro_attempts = _parse_int(
        value.get("max_micro_attempts"), "feedback.max_micro_attempts", minimum=0
    )
    if max_micro_attempts > 2:
        raise ValueError("feedback.max_micro_attempts 不能超过 2")
    return {
        "max_micro_attempts": max_micro_attempts,
        "local_search_radius_px": _parse_int(
            value.get("local_search_radius_px"),
            "feedback.local_search_radius_px",
            minimum=1,
        ),
        "diagnostic_safe_shift_px": (safe_min, safe_max),
        "diagnostic_expected_shift_px": (expected_min, expected_max),
        "diagnostic_physical_shift_px": (physical_min, physical_max),
        "diagnostic_target_shift_px": (target_min, target_max),
        "diagnostic_normal_safe_shift_px": (normal_safe_min, normal_safe_max),
        "diagnostic_row_pitch_px": _parse_positive_float(
            value.get("diagnostic_row_pitch_px"), "feedback.diagnostic_row_pitch_px"
        ),
        "diagnostic_micro_trigger_shift_px": _parse_int(
            value.get("diagnostic_micro_trigger_shift_px"),
            "feedback.diagnostic_micro_trigger_shift_px",
            minimum=1,
        ),
        "diagnostic_min_confidence": _parse_unit_interval(
            value.get("diagnostic_min_confidence"),
            "feedback.diagnostic_min_confidence",
        ),
        "diagnostic_min_local_overlap_score": _parse_unit_interval(
            value.get("diagnostic_min_local_overlap_score"),
            "feedback.diagnostic_min_local_overlap_score",
        ),
        "diagnostic_no_move_similarity": _parse_unit_interval(
            value.get("diagnostic_no_move_similarity"),
            "feedback.diagnostic_no_move_similarity",
        ),
        "diagnostic_no_move_shift_px": _parse_int(
            value.get("diagnostic_no_move_shift_px"),
            "feedback.diagnostic_no_move_shift_px",
            minimum=0,
        ),
    }


def parse_capture_probe_params(raw: Any) -> dict[str, Any]:
    """Validate probe input without deriving any device-specific coordinates."""
    params = _parse_params(raw)
    mode = params.get("mode", CAPTURE_ONLY_MODE)
    if not isinstance(mode, str) or mode not in SUPPORTED_MODES:
        raise ValueError(
            "mode 必须是 capture_only、pair_probe、feedback_probe 或 continuous_capture"
        )

    debug_dir = params.get("debug_dir", "debug/star-backpack-probe")
    if not isinstance(debug_dir, str) or not debug_dir.strip():
        raise ValueError("debug_dir 必须是非空路径字符串")

    parsed: dict[str, Any] = {"mode": mode, "debug_dir": debug_dir.strip()}
    if mode in {PAIR_PROBE_MODE, FEEDBACK_PROBE_MODE, CONTINUOUS_CAPTURE_MODE}:
        if "compare_roi" not in params:
            raise ValueError(f"{mode} 必须显式提供 compare_roi")
        parsed["compare_roi"] = _parse_rect(params["compare_roi"], "compare_roi")

    if mode == PAIR_PROBE_MODE:
        parsed["swipe"] = _parse_swipe(params.get("swipe"))
        parsed["settle_ms"] = _parse_int(
            params.get("settle_ms"), "settle_ms", minimum=0
        )

        compare = params.get("compare", {})
        if not isinstance(compare, dict):
            raise ValueError("compare 必须是对象")
        min_ratio = float(compare.get("min_overlap_ratio", 0.20))
        max_ratio = float(compare.get("max_overlap_ratio", 0.90))
        if (
            not math.isfinite(min_ratio)
            or not math.isfinite(max_ratio)
            or not 0 < min_ratio <= max_ratio <= 1
        ):
            raise ValueError(
                "compare.min_overlap_ratio 和 max_overlap_ratio 必须满足 0 < min <= max <= 1"
            )
        parsed["compare"] = {
            "min_overlap_ratio": min_ratio,
            "max_overlap_ratio": max_ratio,
        }
    if mode == FEEDBACK_PROBE_MODE:
        parsed["settle_ms"] = _parse_int(
            params.get("settle_ms"), "settle_ms", minimum=0
        )
        parsed["coarse_swipe"] = _parse_swipe(
            params.get("coarse_swipe"), "coarse_swipe"
        )
        parsed["micro_swipe"] = _parse_swipe(
            params.get("micro_swipe"), "micro_swipe"
        )
        # Calibration is opt-in so the B1d terminal-partial behaviour remains
        # available for non-calibration feedback callers.
        parsed["single_swipe_calibration"] = _parse_bool(
            params.get("single_swipe_calibration"),
            "single_swipe_calibration",
        )
        parsed["feedback"] = _parse_diagnostic_feedback(params.get("feedback"))
    if mode == CONTINUOUS_CAPTURE_MODE:
        parsed["settle_ms"] = _parse_int(
            params.get("settle_ms"), "settle_ms", minimum=0
        )
        parsed["swipe"] = _parse_swipe(params.get("swipe"))
        max_transitions = _parse_int(
            params.get("max_transitions"), "max_transitions", minimum=1
        )
        if max_transitions > 50:
            raise ValueError("max_transitions 不能超过 50")
        parsed["max_transitions"] = max_transitions
        parsed["feedback"] = _parse_diagnostic_feedback(params.get("feedback"))
        game_version = params.get("game_version", "如鸢")
        if game_version not in {"如鸢", "代号鸢"}:
            raise ValueError("game_version 必须是 如鸢 或 代号鸢")
        parsed["game_version"] = game_version
    return parsed


def _resolve_debug_base(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _prepare_run_directory(debug_dir: str) -> Path:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    run_dir = _resolve_debug_base(debug_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"无法编码 PNG: {path}")
    path.write_bytes(encoded.tobytes())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _image_metadata(image: np.ndarray, mode: str, file_name: str) -> dict[str, Any]:
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise ValueError("controller screenshot 不是有效图像数组")
    channels = int(image.shape[2]) if image.ndim >= 3 else 1
    return {
        "mode": mode,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": channels,
        "dtype": str(image.dtype),
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "file": file_name,
    }


def _crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    image_height, image_width = image.shape[:2]
    if x + width > image_width or y + height > image_height:
        raise ValueError(f"compare_roi 超出截图范围: {roi}, image={image.shape}")
    return image[y : y + height, x : x + width].copy()


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"不支持的截图通道数: {image.shape}")


def _prepare_compare_image(image: np.ndarray) -> np.ndarray:
    gray = _grayscale(image).astype(np.float32)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.size == 0:
        return 0.0
    first_vector = first.reshape(-1).astype(np.float32)
    second_vector = second.reshape(-1).astype(np.float32)
    first_vector -= float(first_vector.mean())
    second_vector -= float(second_vector.mean())
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    if denominator <= 1e-6:
        return 0.0
    return float(np.clip(first_vector @ second_vector / denominator, -1.0, 1.0))


def compute_visual_overlap(
    before_roi: np.ndarray,
    after_roi: np.ndarray,
    min_overlap_ratio: float,
    max_overlap_ratio: float,
) -> dict[str, float | int | None | str]:
    """Compare page pixels only; values are diagnostic and never business decisions."""
    if before_roi.shape[:2] != after_roi.shape[:2]:
        raise ValueError("before/after compare ROI 的尺寸必须一致")
    height = before_roi.shape[0]
    if height < 2:
        raise ValueError("compare_roi 高度至少为 2")

    before = _prepare_compare_image(before_roi)
    after = _prepare_compare_image(after_roi)
    same_position_score = _normalized_correlation(before, after)
    minimum = max(1, math.ceil(height * min_overlap_ratio))
    maximum = min(height, math.floor(height * max_overlap_ratio))
    if minimum > maximum:
        raise ValueError("compare overlap 范围在当前 ROI 高度下为空")

    best_overlap_px: int | None = None
    best_overlap_score = -1.0
    for overlap_px in range(minimum, maximum + 1):
        score = _normalized_correlation(before[-overlap_px:, :], after[:overlap_px, :])
        if score > best_overlap_score:
            best_overlap_score = score
            best_overlap_px = overlap_px

    return {
        "same_position_score": same_position_score,
        "best_overlap_score": best_overlap_score,
        "best_overlap_px": best_overlap_px,
        "best_shift_px": height - best_overlap_px if best_overlap_px else None,
        "classification": "diagnostic_only",
    }


@dataclass(frozen=True)
class MotionEstimate:
    """Image-only vertical motion evidence, with positive values meaning content moved up."""

    shift_y: float = 0.0
    inlier_count: int = 0
    match_count: int = 0
    inlier_ratio: float = 0.0
    median_abs_deviation: float = 0.0
    confidence: float = 0.0
    motion_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    selected_hypothesis: dict[str, Any] | None = None
    direction_rejected_match_count: int = 0
    direct_normal_search: dict[str, Any] = field(default_factory=dict)


@dataclass
class _MotionHypothesis:
    matches: list[cv2.DMatch]
    shift_y: float
    median_abs_deviation: float
    x_coverage: float
    y_coverage: float
    mean_descriptor_distance: float
    anchor_score: float | None = None
    anchor_height_px: int = 0
    full_overlap_score: float | None = None
    full_overlap_height_px: int = 0
    full_overlap_gray_score: float | None = None
    full_overlap_gradient_score: float | None = None
    proposal_source: str = "orb"

    def debug_dict(
        self,
        evidence_score: float | None = None,
        legacy_inside_expected_range: bool | None = None,
        inside_normal_motion_band: bool | None = None,
        inside_physical_motion_range: bool | None = None,
        selection_score: float | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "shift_y": self.shift_y,
            "match_count": len(self.matches),
            "median_abs_deviation": self.median_abs_deviation,
            "x_coverage": self.x_coverage,
            "y_coverage": self.y_coverage,
            "mean_descriptor_distance": self.mean_descriptor_distance,
            "anchor_score": self.anchor_score,
            "anchor_height_px": self.anchor_height_px,
            "full_overlap_score": self.full_overlap_score,
            "full_overlap_height_px": self.full_overlap_height_px,
            "full_overlap_gray_score": self.full_overlap_gray_score,
            "full_overlap_gradient_score": self.full_overlap_gradient_score,
            "full_overlap_valid": self.full_overlap_score is not None,
            "proposal_source": self.proposal_source,
        }
        if evidence_score is not None:
            result["evidence_score"] = evidence_score
        if legacy_inside_expected_range is not None:
            result["inside_legacy_expected_range"] = legacy_inside_expected_range
        if inside_normal_motion_band is not None:
            result["inside_normal_motion_band"] = inside_normal_motion_band
        if inside_physical_motion_range is not None:
            result["inside_physical_motion_range"] = inside_physical_motion_range
        if selection_score is not None:
            result["selection_score"] = selection_score
        return result


def _cluster_positive_vertical_matches(
    matches: list[cv2.DMatch],
    before_keypoints: list[cv2.KeyPoint],
    after_keypoints: list[cv2.KeyPoint],
    width: int,
    height: int,
    tolerance_px: float = 10.0,
) -> list[_MotionHypothesis]:
    ordered = sorted(
        matches,
        key=lambda match: before_keypoints[match.queryIdx].pt[1]
        - after_keypoints[match.trainIdx].pt[1],
    )
    clusters: list[list[cv2.DMatch]] = []
    for match in ordered:
        motion = before_keypoints[match.queryIdx].pt[1] - after_keypoints[match.trainIdx].pt[1]
        if not clusters:
            clusters.append([match])
            continue
        previous_motions = np.asarray(
            [
                before_keypoints[item.queryIdx].pt[1]
                - after_keypoints[item.trainIdx].pt[1]
                for item in clusters[-1]
            ],
            dtype=np.float32,
        )
        if abs(motion - float(np.median(previous_motions))) <= tolerance_px:
            clusters[-1].append(match)
        else:
            clusters.append([match])

    hypotheses: list[_MotionHypothesis] = []
    for cluster in clusters:
        motions = np.asarray(
            [
                before_keypoints[match.queryIdx].pt[1]
                - after_keypoints[match.trainIdx].pt[1]
                for match in cluster
            ],
            dtype=np.float32,
        )
        x_positions = np.asarray(
            [before_keypoints[match.queryIdx].pt[0] for match in cluster], dtype=np.float32
        )
        y_positions = np.asarray(
            [before_keypoints[match.queryIdx].pt[1] for match in cluster], dtype=np.float32
        )
        hypotheses.append(
            _MotionHypothesis(
                matches=cluster,
                shift_y=float(np.median(motions)),
                median_abs_deviation=float(np.median(np.abs(motions - np.median(motions)))),
                x_coverage=float(np.clip((x_positions.max() - x_positions.min()) / width, 0.0, 1.0)),
                y_coverage=float(np.clip((y_positions.max() - y_positions.min()) / height, 0.0, 1.0)),
                mean_descriptor_distance=float(np.mean([match.distance for match in cluster])),
            )
        )
    return hypotheses


def _hypothesis_score(hypothesis: _MotionHypothesis, tolerance_px: float = 10.0) -> float:
    coverage = 0.5 * hypothesis.x_coverage + 0.5 * hypothesis.y_coverage
    consistency = math.exp(-hypothesis.median_abs_deviation / tolerance_px)
    descriptor_quality = (
        0.0
        if hypothesis.mean_descriptor_distance is None
        else max(0.0, 1.0 - hypothesis.mean_descriptor_distance / 256.0)
    )
    return len(hypothesis.matches) * (0.4 + 0.6 * coverage) * consistency * descriptor_quality


def _direct_overlap_anchor(
    before_roi: np.ndarray, after_roi: np.ndarray, shift_y: float
) -> tuple[float | None, int]:
    """Score the actual shared top strip for one shift hypothesis, without OCR."""
    height = before_roi.shape[0]
    shift = int(round(shift_y))
    physical_overlap = height - shift
    anchor_height = min(120, physical_overlap - 12)
    if shift < 0 or anchor_height < 64 or shift + anchor_height > height:
        return None, 0
    before_anchor = before_roi[shift : shift + anchor_height, :]
    after_anchor = after_roi[:anchor_height, :]
    color_score = _normalized_correlation(
        before_anchor.astype(np.float32), after_anchor.astype(np.float32)
    )
    before_gradient = cv2.Sobel(_grayscale(before_anchor), cv2.CV_32F, 0, 1)
    after_gradient = cv2.Sobel(_grayscale(after_anchor), cv2.CV_32F, 0, 1)
    gradient_score = _normalized_correlation(before_gradient, after_gradient)
    return 0.65 * color_score + 0.35 * gradient_score, anchor_height


def _apply_direct_overlap_anchors(
    hypotheses: list[_MotionHypothesis],
    before_roi: np.ndarray,
    after_roi: np.ndarray,
) -> None:
    for hypothesis in hypotheses:
        hypothesis.anchor_score, hypothesis.anchor_height_px = _direct_overlap_anchor(
            before_roi, after_roi, hypothesis.shift_y
        )
        (
            hypothesis.full_overlap_score,
            hypothesis.full_overlap_height_px,
            hypothesis.full_overlap_gray_score,
            hypothesis.full_overlap_gradient_score,
        ) = _full_overlap_direct_validation(before_roi, after_roi, hypothesis.shift_y)


def _full_overlap_direct_validation(
    before_roi: np.ndarray,
    after_roi: np.ndarray,
    shift_y: float,
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[float | None, int, float | None, float | None]:
    """Validate the complete physically shared region for one motion hypothesis.

    A fixed strip can sit entirely inside one repeated card row.  This compares
    every available shared pixel (except a small capture-edge margin), without
    interpreting card content or invoking OCR.
    """
    height = before_roi.shape[0]
    shift = int(round(shift_y))
    available_overlap_height = height - shift
    overlap_height = available_overlap_height - (2 * DIRECT_OVERLAP_MARGIN_PX)
    if (
        shift < 0
        or overlap_height < MIN_DIRECT_OVERLAP_PX
        or shift + DIRECT_OVERLAP_MARGIN_PX + overlap_height > height
    ):
        return None, 0, None, None
    if prepared is None:
        before_gray = _grayscale(before_roi).astype(np.float32)
        after_gray = _grayscale(after_roi).astype(np.float32)
        before_gradient = cv2.Sobel(before_gray, cv2.CV_32F, 0, 1)
        after_gradient = cv2.Sobel(after_gray, cv2.CV_32F, 0, 1)
    else:
        before_gray, after_gray, before_gradient, after_gradient = prepared
    before_slice = slice(
        shift + DIRECT_OVERLAP_MARGIN_PX, height - DIRECT_OVERLAP_MARGIN_PX
    )
    after_slice = slice(
        DIRECT_OVERLAP_MARGIN_PX, available_overlap_height - DIRECT_OVERLAP_MARGIN_PX
    )
    gray_score = _normalized_correlation(
        before_gray[before_slice, :], after_gray[after_slice, :]
    )
    gradient_score = _normalized_correlation(
        before_gradient[before_slice, :], after_gradient[after_slice, :]
    )
    return (
        0.45 * gray_score + 0.55 * gradient_score,
        overlap_height,
        gray_score,
        gradient_score,
    )


def _find_direct_normal_band_candidate(
    before_roi: np.ndarray, after_roi: np.ndarray
) -> tuple[_MotionHypothesis | None, dict[str, Any]]:
    """Propose one robust direct-motion branch inside the calibrated normal band."""
    started = time.perf_counter()
    minimum_shift = int(NORMAL_MOTION_MIN_PX)
    maximum_shift = min(int(NORMAL_MOTION_MAX_PX), before_roi.shape[0] - 1)
    before_gray = _grayscale(before_roi).astype(np.float32)
    after_gray = _grayscale(after_roi).astype(np.float32)
    prepared = (
        before_gray,
        after_gray,
        cv2.Sobel(before_gray, cv2.CV_32F, 0, 1),
        cv2.Sobel(after_gray, cv2.CV_32F, 0, 1),
    )
    scores: dict[int, tuple[float, int, float, float]] = {}
    for shift in range(minimum_shift, maximum_shift + 1):
        score, overlap_height, gray_score, gradient_score = _full_overlap_direct_validation(
            before_roi, after_roi, float(shift), prepared
        )
        if (
            score is not None
            and gray_score is not None
            and gradient_score is not None
            and overlap_height >= MIN_DIRECT_OVERLAP_PX
        ):
            scores[shift] = (score, overlap_height, gray_score, gradient_score)
    elapsed_ms = (time.perf_counter() - started) * 1000
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "min_shift_px": minimum_shift,
        "max_shift_px": maximum_shift,
        "best_shift_px": None,
        "best_score": None,
        "best_rank_score": None,
        "candidate_injected": False,
        "elapsed_ms": elapsed_ms,
    }
    if not scores:
        return None, diagnostics

    def rank(shift: int) -> float:
        score = scores[shift][0]
        neighborhood = [
            scores[nearby][0]
            for nearby in range(
                shift - DIRECT_NORMAL_LOCAL_RADIUS_PX,
                shift + DIRECT_NORMAL_LOCAL_RADIUS_PX + 1,
            )
            if nearby in scores
        ]
        return 0.70 * score + 0.30 * float(np.mean(neighborhood))

    best_shift = max(scores, key=rank)
    best_score, overlap_height, gray_score, gradient_score = scores[best_shift]
    best_rank_score = rank(best_shift)
    diagnostics.update(
        {
            "best_shift_px": best_shift,
            "best_score": best_score,
            "best_rank_score": best_rank_score,
        }
    )
    if best_score < DIRECT_NORMAL_MIN_SCORE:
        return None, diagnostics
    anchor_score, anchor_height_px = _direct_overlap_anchor(
        before_roi, after_roi, float(best_shift)
    )
    if anchor_score is None:
        return None, diagnostics
    diagnostics["candidate_injected"] = True
    return (
        _MotionHypothesis(
            matches=[],
            shift_y=float(best_shift),
            median_abs_deviation=0.0,
            x_coverage=1.0,
            y_coverage=1.0,
            mean_descriptor_distance=None,
            anchor_score=anchor_score,
            anchor_height_px=anchor_height_px,
            full_overlap_score=best_score,
            full_overlap_height_px=overlap_height,
            full_overlap_gray_score=gray_score,
            full_overlap_gradient_score=gradient_score,
            proposal_source="direct_normal_band",
        ),
        diagnostics,
    )


def _merge_direct_normal_candidate(
    orb_hypotheses: list[_MotionHypothesis], direct_candidate: _MotionHypothesis | None
) -> list[_MotionHypothesis]:
    """Avoid duplicate normal branches while preserving real ORB support when present."""
    if direct_candidate is None:
        return orb_hypotheses
    for index, hypothesis in enumerate(orb_hypotheses):
        if abs(hypothesis.shift_y - direct_candidate.shift_y) <= DIRECT_HYPOTHESIS_DEDUPE_PX:
            orb_hypotheses[index] = _MotionHypothesis(
                matches=hypothesis.matches,
                shift_y=direct_candidate.shift_y,
                median_abs_deviation=hypothesis.median_abs_deviation,
                x_coverage=hypothesis.x_coverage,
                y_coverage=hypothesis.y_coverage,
                mean_descriptor_distance=hypothesis.mean_descriptor_distance,
                anchor_score=direct_candidate.anchor_score,
                anchor_height_px=direct_candidate.anchor_height_px,
                full_overlap_score=direct_candidate.full_overlap_score,
                full_overlap_height_px=direct_candidate.full_overlap_height_px,
                full_overlap_gray_score=direct_candidate.full_overlap_gray_score,
                full_overlap_gradient_score=direct_candidate.full_overlap_gradient_score,
                proposal_source="merged",
            )
            return orb_hypotheses
    return [*orb_hypotheses, direct_candidate]


def _has_validated_direct_normal_support(
    hypothesis: _MotionHypothesis,
    direct_normal_search: dict[str, Any] | None,
) -> bool:
    """Keep strict direct evidence when it is merged with sparse ORB support."""
    if hypothesis.proposal_source == "direct_normal_band":
        return True
    if hypothesis.proposal_source != "merged" or not direct_normal_search:
        return False
    if direct_normal_search.get("candidate_injected") is not True:
        return False
    best_shift_px = direct_normal_search.get("best_shift_px")
    best_score = direct_normal_search.get("best_score")
    if (
        not isinstance(best_shift_px, (int, float))
        or abs(float(best_shift_px) - hypothesis.shift_y) > DIRECT_HYPOTHESIS_DEDUPE_PX
        or not isinstance(best_score, (int, float))
        or best_score < DIRECT_NORMAL_MIN_SCORE
    ):
        return False
    return True


def _select_motion_hypothesis(
    hypotheses: list[_MotionHypothesis],
    expected_shift_px: tuple[int, int] | None,
    physical_shift_px: tuple[int, int] | None,
    direct_normal_search: dict[str, Any] | None = None,
    *,
    terminal_confirmed: bool = False,
    terminal_max_shift_px: float | None = None,
) -> tuple[_MotionHypothesis | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Prefer a verified normal-motion branch before terminal fallback.

    ``expected_shift_px`` remains a legacy diagnostic field only.  It cannot
    overrule full shared-pixel evidence or turn a low periodic alias into the
    selected normal page motion.
    """
    evidence_scores = [_hypothesis_score(hypothesis) for hypothesis in hypotheses]
    maximum_evidence = max(evidence_scores, default=1.0)
    scored_hypotheses: list[dict[str, Any]] = []
    for hypothesis, evidence_score in zip(hypotheses, evidence_scores):
        legacy_inside_expected_range = (
            expected_shift_px is not None
            and expected_shift_px[0] <= hypothesis.shift_y <= expected_shift_px[1]
        )
        inside_normal_motion_band = (
            NORMAL_MOTION_MIN_PX <= hypothesis.shift_y <= NORMAL_MOTION_MAX_PX
        )
        inside_physical_motion_range = (
            physical_shift_px is None
            or physical_shift_px[0] <= hypothesis.shift_y <= physical_shift_px[1]
        )
        evidence_quality = evidence_score / maximum_evidence
        anchor_quality = hypothesis.anchor_score if hypothesis.anchor_score is not None else 0.0
        full_overlap_quality = (
            hypothesis.full_overlap_score
            if hypothesis.full_overlap_score is not None
            else 0.0
        )
        robust_feature_support = min(1.0, len(hypothesis.matches) / 20.0)
        robust_feature_support *= math.exp(-hypothesis.median_abs_deviation / 12.0)
        robust_feature_support *= (
            0.0
            if hypothesis.mean_descriptor_distance is None
            else max(0.0, 1.0 - hypothesis.mean_descriptor_distance / 256.0)
        )
        normal_selection_score = (
            0.55 * full_overlap_quality
            + 0.25 * anchor_quality
            + 0.10 * hypothesis.x_coverage
            + 0.10 * robust_feature_support
        )
        fallback_selection_score = (
            0.40 * evidence_quality
            + 0.35 * anchor_quality
            + 0.25 * full_overlap_quality
        )
        scored_hypotheses.append(
            {
                "hypothesis": hypothesis,
                "evidence_score": evidence_score,
                "legacy_inside_expected_range": legacy_inside_expected_range,
                "inside_normal_motion_band": inside_normal_motion_band,
                "inside_physical_motion_range": inside_physical_motion_range,
                "normal_selection_score": normal_selection_score,
                "fallback_selection_score": fallback_selection_score,
            }
        )
    debug_hypotheses = [
        item["hypothesis"].debug_dict(
            item["evidence_score"],
            item["legacy_inside_expected_range"],
            item["inside_normal_motion_band"],
            item["inside_physical_motion_range"],
        )
        for item in scored_hypotheses
    ]
    physical_candidates = [
        item for item in scored_hypotheses if item["inside_physical_motion_range"]
    ]
    if not physical_candidates:
        return None, None, debug_hypotheses
    terminal_candidates = [
        item
        for item in physical_candidates
        if (
            terminal_confirmed
            and terminal_max_shift_px is not None
            and 0 < item["hypothesis"].shift_y <= terminal_max_shift_px
            and len(item["hypothesis"].matches) >= 4
            and item["hypothesis"].median_abs_deviation <= 12.0
            and item["hypothesis"].full_overlap_score is not None
            and item["hypothesis"].full_overlap_score >= DIRECT_NORMAL_MIN_SCORE
            and item["hypothesis"].full_overlap_height_px >= MIN_DIRECT_OVERLAP_PX
            and item["hypothesis"].anchor_score is not None
            and item["hypothesis"].anchor_score >= DIRECT_NORMAL_MIN_SCORE
        )
    ]
    normal_candidates = [
        item
        for item in physical_candidates
        if (
            item["inside_normal_motion_band"]
            and item["hypothesis"].full_overlap_score is not None
            and item["hypothesis"].full_overlap_height_px >= MIN_DIRECT_OVERLAP_PX
            and item["hypothesis"].anchor_score is not None
            and item["hypothesis"].median_abs_deviation <= 12.0
            and (
                len(item["hypothesis"].matches) >= 3
                or _has_validated_direct_normal_support(
                    item["hypothesis"], direct_normal_search
                )
            )
        )
    ]
    if terminal_confirmed:
        if not terminal_candidates:
            return None, None, debug_hypotheses
        selected_item = max(
            terminal_candidates, key=lambda item: item["fallback_selection_score"]
        )
        selection_mode = "terminal_confirmed_short_motion"
        selected_by = "terminal_confirmed_evidence"
        selection_score = selected_item["fallback_selection_score"]
    elif normal_candidates:
        selected_item = max(normal_candidates, key=lambda item: item["normal_selection_score"])
        selection_mode = "normal_motion_full_overlap"
        selected_by = "full_overlap_score"
        selection_score = selected_item["normal_selection_score"]
    else:
        selected_item = max(
            physical_candidates, key=lambda item: item["fallback_selection_score"]
        )
        selection_mode = "terminal_fallback"
        selected_by = "fallback_evidence"
        selection_score = selected_item["fallback_selection_score"]
    selected = selected_item["hypothesis"]
    selected_reason = selected.debug_dict(
        selected_item["evidence_score"],
        selected_item["legacy_inside_expected_range"],
        selected_item["inside_normal_motion_band"],
        selected_item["inside_physical_motion_range"],
        selection_score,
    )
    selected_reason.update(
        {
            "selection_mode": selection_mode,
            "normal_motion_candidate_count": len(normal_candidates),
            "selected_by": selected_by,
            "fallback_used": selection_mode != "normal_motion_full_overlap",
        }
    )
    return selected, selected_reason, debug_hypotheses


def _direct_candidate_motion_estimate(
    selected: _MotionHypothesis,
    selected_reason: dict[str, Any],
    debug_hypotheses: list[dict[str, Any]],
    direction_rejected_match_count: int,
    direct_normal_search: dict[str, Any],
) -> MotionEstimate:
    """Expose a direct-only proposal without inventing ORB/RANSAC evidence."""
    return MotionEstimate(
        shift_y=selected.shift_y,
        confidence=float(selected.full_overlap_score or 0.0),
        motion_hypotheses=debug_hypotheses,
        selected_hypothesis=selected_reason,
        direction_rejected_match_count=direction_rejected_match_count,
        direct_normal_search=direct_normal_search,
    )


def estimate_vertical_motion(
    before_roi: np.ndarray,
    after_roi: np.ndarray,
    expected_shift_px: tuple[int, int] | None = None,
    physical_shift_px: tuple[int, int] | None = None,
    *,
    terminal_confirmed: bool = False,
    terminal_max_shift_px: float | None = None,
) -> MotionEstimate:
    """Estimate one vertical page shift with hypothesis clustering then cluster RANSAC.

    ``shift_y`` is positive when content in ``after_roi`` moved upward relative to
    ``before_roi``. ORB remains the feature proposer; a narrow direct search
    supplements only the calibrated normal band when ORB misses that branch.
    """
    if before_roi.shape[:2] != after_roi.shape[:2]:
        raise ValueError("before/after compare ROI 的尺寸必须一致")
    height, width = before_roi.shape[:2]
    if height < 32 or width < 32:
        return MotionEstimate()
    direct_candidate, direct_normal_search = _find_direct_normal_band_candidate(
        before_roi, after_roi
    )

    before = cv2.GaussianBlur(_grayscale(before_roi), (3, 3), 0)
    after = cv2.GaussianBlur(_grayscale(after_roi), (3, 3), 0)
    orb = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
    before_keypoints, before_descriptors = orb.detectAndCompute(before, None)
    after_keypoints, after_descriptors = orb.detectAndCompute(after, None)
    if before_descriptors is None or after_descriptors is None:
        hypotheses = _merge_direct_normal_candidate([], direct_candidate)
        selected, selected_reason, debug_hypotheses = _select_motion_hypothesis(
            hypotheses,
            expected_shift_px,
            physical_shift_px,
            direct_normal_search,
            terminal_confirmed=terminal_confirmed,
            terminal_max_shift_px=terminal_max_shift_px,
        )
        if selected is not None and selected_reason is not None:
            return _direct_candidate_motion_estimate(
                selected,
                selected_reason,
                debug_hypotheses,
                0,
                direct_normal_search,
            )
        return MotionEstimate(direct_normal_search=direct_normal_search)

    raw_matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        before_descriptors, after_descriptors, k=2
    )
    maximum_dx = max(12.0, width * 0.08)
    maximum_dy = height * 0.95
    positive_matches: list[cv2.DMatch] = []
    direction_rejected_match_count = 0
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance >= 0.72 * second.distance:
            continue
        source = before_keypoints[first.queryIdx].pt
        target = after_keypoints[first.trainIdx].pt
        dx = target[0] - source[0]
        motion_y = source[1] - target[1]
        if abs(dx) > maximum_dx or abs(motion_y) > maximum_dy:
            continue
        # The feedback probe only scrolls the list upward. Near-zero evidence is
        # intentionally left to the independent same-position no_move check.
        if motion_y <= 0:
            direction_rejected_match_count += 1
            continue
        positive_matches.append(first)
    if len(positive_matches) < 3:
        hypotheses = _merge_direct_normal_candidate([], direct_candidate)
        selected, selected_reason, debug_hypotheses = _select_motion_hypothesis(
            hypotheses,
            expected_shift_px,
            physical_shift_px,
            direct_normal_search,
            terminal_confirmed=terminal_confirmed,
            terminal_max_shift_px=terminal_max_shift_px,
        )
        if selected is not None and selected_reason is not None:
            return _direct_candidate_motion_estimate(
                selected,
                selected_reason,
                debug_hypotheses,
                direction_rejected_match_count,
                direct_normal_search,
            )
        return MotionEstimate(
            match_count=len(positive_matches),
            direction_rejected_match_count=direction_rejected_match_count,
            direct_normal_search=direct_normal_search,
        )

    hypotheses = _cluster_positive_vertical_matches(
        positive_matches, before_keypoints, after_keypoints, width, height
    )
    _apply_direct_overlap_anchors(hypotheses, before_roi, after_roi)
    hypotheses = _merge_direct_normal_candidate(hypotheses, direct_candidate)
    selected, selected_reason, debug_hypotheses = _select_motion_hypothesis(
        hypotheses,
        expected_shift_px,
        physical_shift_px,
        direct_normal_search,
        terminal_confirmed=terminal_confirmed,
        terminal_max_shift_px=terminal_max_shift_px,
    )
    if selected is None or selected_reason is None:
        return MotionEstimate(
            match_count=len(positive_matches),
            motion_hypotheses=debug_hypotheses,
            direction_rejected_match_count=direction_rejected_match_count,
            direct_normal_search=direct_normal_search,
        )
    if (
        not selected.matches
        or (
            len(selected.matches) < 4
            and _has_validated_direct_normal_support(selected, direct_normal_search)
        )
    ):
        return _direct_candidate_motion_estimate(
            selected,
            selected_reason,
            debug_hypotheses,
            direction_rejected_match_count,
            direct_normal_search,
        )
    source_points = np.float32(
        [before_keypoints[match.queryIdx].pt for match in selected.matches]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [after_keypoints[match.trainIdx].pt for match in selected.matches]
    ).reshape(-1, 1, 2)
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    if transform is None or inlier_mask is None:
        return MotionEstimate(
            match_count=len(selected.matches),
            motion_hypotheses=debug_hypotheses,
            selected_hypothesis=selected_reason,
            direction_rejected_match_count=direction_rejected_match_count,
            direct_normal_search=direct_normal_search,
        )

    mask = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(mask.sum())
    if not inlier_count:
        return MotionEstimate(
            match_count=len(selected.matches),
            motion_hypotheses=debug_hypotheses,
            selected_hypothesis=selected_reason,
            direction_rejected_match_count=direction_rejected_match_count,
            direct_normal_search=direct_normal_search,
        )
    motions = np.asarray(
        [
            before_keypoints[match.queryIdx].pt[1]
            - after_keypoints[match.trainIdx].pt[1]
            for match, is_inlier in zip(selected.matches, mask)
            if is_inlier
        ],
        dtype=np.float32,
    )
    median_motion = float(np.median(motions))
    mad = float(np.median(np.abs(motions - median_motion)))
    translation_x = float(transform[0, 2])
    translation_y = float(transform[1, 2])
    inlier_ratio = inlier_count / len(selected.matches)
    count_evidence = min(1.0, inlier_count / 30.0)
    consistency = math.exp(-mad / 12.0)
    horizontal_evidence = math.exp(-abs(translation_x) / maximum_dx)
    confidence = float(
        np.clip(count_evidence * inlier_ratio * consistency * horizontal_evidence, 0.0, 1.0)
    )
    shift_y = -translation_y
    if shift_y <= 0:
        confidence = 0.0
    return MotionEstimate(
        shift_y=shift_y,
        inlier_count=inlier_count,
        match_count=len(selected.matches),
        inlier_ratio=inlier_ratio,
        median_abs_deviation=mad,
        confidence=confidence,
        motion_hypotheses=debug_hypotheses,
        selected_hypothesis=selected_reason,
        direction_rejected_match_count=direction_rejected_match_count,
        direct_normal_search=direct_normal_search,
    )


def _local_overlap_confirmation(
    before_roi: np.ndarray,
    after_roi: np.ndarray,
    predicted_shift_y: float,
    search_radius_px: int,
) -> tuple[float | None, int | None, bool]:
    """Confirm only around ORB's estimate; never re-scan the complete offset range."""
    before = _prepare_compare_image(before_roi)
    after = _prepare_compare_image(after_roi)
    height = before.shape[0]
    center = int(round(predicted_shift_y))
    if center < 0 or center >= height:
        return None, None, False
    candidates = range(
        max(0, center - search_radius_px),
        min(height - 1, center + search_radius_px) + 1,
    )
    best_score = -1.0
    best_shift: int | None = None
    for shift in candidates:
        score = _normalized_correlation(before[shift:, :], after[: height - shift, :])
        if score > best_score:
            best_score = score
            best_shift = shift
    if best_shift is None:
        return None, None, False
    return best_score, best_shift, True


def _estimate_row_lattice(roi: np.ndarray, row_pitch_px: float) -> dict[str, Any]:
    """Estimate the current frame's four-column row phase without OCR.

    The phase score samples the expected disc and text-edge heights around each
    row centre.  Each frame is estimated independently, avoiding any
    accumulation of historical swipe error.
    """
    height = roi.shape[0]
    pitch = float(row_pitch_px)
    radius = pitch * YUANSTAR_ROW_RADIUS_TO_PITCH_RATIO
    gray = _grayscale(roi).astype(np.float32)
    energy = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1)), axis=1)
    phase_limit = max(1, int(round(pitch)))

    def sample(position: float) -> float:
        center = int(round(position))
        start = max(0, center - 2)
        end = min(height, center + 3)
        return float(np.mean(energy[start:end])) if end > start else 0.0

    def phase_score(phase: int) -> float:
        values: list[float] = []
        center = float(phase)
        while center < height + pitch:
            if -radius <= center <= height + radius:
                # YuanStar's disc, level and name ROI boundaries are all
                # visible geometric evidence even though no text is read.
                values.append(
                    0.35 * sample(center - radius)
                    + 0.35 * sample(center + radius)
                    + 0.15 * sample(center - radius * 0.77)
                    + 0.15 * sample(center + radius * 1.39)
                )
            center += pitch
        return float(np.mean(values)) if values else 0.0

    best_phase = max(range(phase_limit), key=phase_score)
    phase_evidence = phase_score(best_phase)
    first_center = float(best_phase)
    while first_center - pitch > -pitch:
        first_center -= pitch
    centers: list[float] = []
    center = first_center
    while center <= height + pitch:
        if -pitch <= center <= height + pitch:
            centers.append(round(center, 2))
        center += pitch
    return {
        "row_centers": centers,
        "row_pitch_px": pitch,
        "radius_px": radius,
        "phase_px": float(best_phase),
        "phase_evidence": phase_evidence,
    }


def evaluate_semantic_row_overlap(
    prev_roi: np.ndarray,
    candidate_roi: np.ndarray,
    actual_shift_px: float,
    row_pitch_px: float,
    *,
    prev_row_centers: list[float] | None = None,
    candidate_row_centers: list[float] | None = None,
) -> dict[str, Any]:
    """Decide whether a complete YuanStar OCR row exists in both shared areas."""
    started = time.perf_counter()
    height = prev_roi.shape[0]
    shift = float(np.clip(actual_shift_px, 0.0, float(height)))
    physical_overlap_px = max(0.0, float(height) - shift)
    prev_lattice = _estimate_row_lattice(prev_roi, row_pitch_px)
    candidate_lattice = _estimate_row_lattice(candidate_roi, row_pitch_px)
    prev_centers = prev_row_centers if prev_row_centers is not None else prev_lattice["row_centers"]
    candidate_centers = (
        candidate_row_centers
        if candidate_row_centers is not None
        else candidate_lattice["row_centers"]
    )
    radius = float(prev_lattice["radius_px"])
    top_offset = -YUANSTAR_ENVELOPE_TOP_RADIUS * radius
    bottom_offset = YUANSTAR_ENVELOPE_BOTTOM_RADIUS * radius
    safety_margin = SEMANTIC_ENVELOPE_SAFETY_MARGIN_PX
    prev_shared = (shift, float(height))
    candidate_shared = (0.0, physical_overlap_px)
    phase_tolerance = max(12.0, row_pitch_px * 0.12)
    full_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []
    for prev_center in prev_centers:
        expected_candidate_center = prev_center - shift
        if not candidate_centers:
            continue
        candidate_center = min(
            candidate_centers, key=lambda center: abs(center - expected_candidate_center)
        )
        phase_delta = abs(candidate_center - expected_candidate_center)
        if phase_delta > phase_tolerance:
            continue
        prev_envelope = (prev_center + top_offset, prev_center + bottom_offset)
        candidate_envelope = (
            candidate_center + top_offset,
            candidate_center + bottom_offset,
        )
        clearance = min(
            prev_envelope[0] - prev_shared[0],
            prev_shared[1] - prev_envelope[1],
            candidate_envelope[0] - candidate_shared[0],
            candidate_shared[1] - candidate_envelope[1],
        )
        row_evidence = {
            "prev_center_y": round(prev_center, 2),
            "candidate_center_y": round(candidate_center, 2),
            "phase_delta_px": round(phase_delta, 2),
            "clearance_px": round(clearance, 2),
        }
        if clearance >= safety_margin:
            full_rows.append(row_evidence)
        elif clearance >= 0:
            ambiguous_rows.append(row_evidence)
    if full_rows:
        state = "definitely_full_row"
        required = True
        reason = "complete_envelope_in_both_shared_regions"
        candidates = full_rows
    elif ambiguous_rows:
        state = "ambiguous"
        required = True
        reason = "envelope_near_shared_region_boundary"
        candidates = ambiguous_rows
    else:
        state = "definitely_no_full_row"
        required = False
        reason = "no_complete_envelope_in_both_shared_regions"
        candidates = []
    return {
        "visual_overlap": physical_overlap_px > 0,
        "physical_overlap_px": physical_overlap_px,
        "actual_shift_px": shift,
        "row_pitch_px": row_pitch_px,
        "prev_row_centers": prev_centers,
        "candidate_row_centers": candidate_centers,
        "ocr_row_envelope": {
            "top_offset": top_offset,
            "bottom_offset": bottom_offset,
            "safety_margin_px": safety_margin,
            "source": "yuanstar_runtime_geometry",
        },
        "semantic_overlap_state": state,
        "ocr_overlap_pair_required": required,
        "full_row_candidates": candidates,
        "reason": reason,
        "runtime_ms": (time.perf_counter() - started) * 1000,
        "row_lattice": {
            "prev": prev_lattice,
            "candidate": candidate_lattice,
        },
    }


def _has_reliable_direct_normal_motion(estimate: MotionEstimate) -> bool:
    """Accept a selected direct-normal branch only after every direct check agrees.

    This is intentionally independent from ORB/RANSAC confidence.  It is not a
    confidence bypass: the selected hypothesis, full-overlap validation,
    anchor, direct-search injection, and the two local-shift values must all
    describe the same normal-band motion.
    """
    selected = estimate.selected_hypothesis
    direct_search = estimate.direct_normal_search
    if not selected or not direct_search:
        return False
    if selected.get("selection_mode") != "normal_motion_full_overlap":
        return False
    if selected.get("fallback_used") is not False:
        return False
    if selected.get("inside_normal_motion_band") is not True:
        return False
    if selected.get("inside_physical_motion_range") is not True:
        return False
    if selected.get("proposal_source") not in {"direct_normal_band", "merged"}:
        return False
    if selected.get("full_overlap_valid") is not True:
        return False
    full_overlap_score = selected.get("full_overlap_score")
    anchor_score = selected.get("anchor_score")
    if (
        not isinstance(full_overlap_score, (int, float))
        or full_overlap_score < DIRECT_NORMAL_MIN_SCORE
        or not isinstance(anchor_score, (int, float))
        or anchor_score < DIRECT_NORMAL_MIN_SCORE
    ):
        return False
    if direct_search.get("candidate_injected") is not True:
        return False
    best_shift_px = direct_search.get("best_shift_px")
    if not isinstance(best_shift_px, (int, float)):
        return False
    return abs(float(best_shift_px) - estimate.shift_y) <= DIRECT_HYPOTHESIS_DEDUPE_PX


def evaluate_feedback_candidate(
    prev_roi: np.ndarray,
    candidate_roi: np.ndarray,
    feedback: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one candidate against the same accepted previous screenshot.

    The diagnostic window is supplied by the caller. It is intentionally not a
    production threshold: a later calibrated collector may choose different
    values without changing the image-estimation semantics.
    """
    estimate = estimate_vertical_motion(
        prev_roi,
        candidate_roi,
        feedback["diagnostic_expected_shift_px"],
        feedback["diagnostic_physical_shift_px"],
    )
    same_position_score = _normalized_correlation(
        _prepare_compare_image(prev_roi), _prepare_compare_image(candidate_roi)
    )
    local_overlap_score, local_overlap_shift_px, local_overlap_valid = _local_overlap_confirmation(
        prev_roi,
        candidate_roi,
        estimate.shift_y,
        feedback["local_search_radius_px"],
    )
    safe_min, safe_max = feedback["diagnostic_safe_shift_px"]
    normal_safe_min, normal_safe_max = feedback["diagnostic_normal_safe_shift_px"]
    target_min, target_max = feedback["diagnostic_target_shift_px"]
    # The bottom of the configured normal window is deliberately tolerant of
    # estimation noise.  It is not the efficiency floor: 520--600 px remains
    # a one-shot correction band, and 600 px is the lower acceptable edge.
    acceptable_normal_min = max(normal_safe_min, target_min - 10)
    physical_min, physical_max = feedback["diagnostic_physical_shift_px"]
    row_pitch_px = feedback["diagnostic_row_pitch_px"]
    semantic_overlap = evaluate_semantic_row_overlap(
        prev_roi, candidate_roi, estimate.shift_y, row_pitch_px
    )
    visual_overlap_rows = semantic_overlap["physical_overlap_px"] / row_pitch_px
    orb_confidence_passed = (
        estimate.confidence >= feedback["diagnostic_min_confidence"]
    )
    direct_normal_evidence_passed = _has_reliable_direct_normal_motion(estimate)
    has_reliable_motion = orb_confidence_passed or direct_normal_evidence_passed
    if orb_confidence_passed and direct_normal_evidence_passed:
        motion_reliability_mode = "orb_and_direct"
    elif orb_confidence_passed:
        motion_reliability_mode = "orb_confidence"
    elif direct_normal_evidence_passed:
        motion_reliability_mode = "direct_normal_full_overlap"
    else:
        motion_reliability_mode = "unreliable"
    motion_reliability = {
        "reliable": has_reliable_motion,
        "mode": motion_reliability_mode,
        "orb_confidence_passed": orb_confidence_passed,
        "direct_normal_evidence_passed": direct_normal_evidence_passed,
    }
    has_reliable_overlap = (
        has_reliable_motion
        and local_overlap_valid
        and local_overlap_score is not None
        and local_overlap_score >= feedback["diagnostic_min_local_overlap_score"]
    )
    no_move = (
        same_position_score >= feedback["diagnostic_no_move_similarity"]
        and (
            # Preserve the original no-move fail-safe: even a periodic direct
            # candidate cannot overrule strong same-position evidence when
            # ORB/RANSAC did not independently confirm movement.
            not orb_confidence_passed
            or abs(estimate.shift_y) <= feedback["diagnostic_no_move_shift_px"]
        )
    )
    if no_move:
        # Two identical frames are not an adjacent image pair, regardless of
        # how many complete rows happen to be visible in their shared region.
        semantic_overlap = {
            **semantic_overlap,
            "semantic_overlap_state": "not_applicable_no_move",
            "ocr_overlap_pair_required": False,
            "reason": "no_move_not_an_adjacent_pair",
        }
    elif not has_reliable_overlap:
        # Without trustworthy motion and local-overlap evidence, the two
        # screenshots cannot be treated as an adjacent pair. Keep the raw
        # physical and lattice diagnostics for review, but do not infer OCR
        # pair semantics from a coincidental geometric envelope.
        semantic_overlap = {
            **semantic_overlap,
            "semantic_overlap_state": "not_applicable_unreliable_motion",
            "ocr_overlap_pair_required": False,
            "full_row_candidates": [],
            "reason": "unreliable_motion_not_an_adjacent_pair",
        }
    ocr_overlap_pair_required = semantic_overlap["ocr_overlap_pair_required"]
    accepted = (
        has_reliable_overlap
        and (
            acceptable_normal_min - NORMAL_ACCEPTANCE_EPSILON_PX
            <= estimate.shift_y
            <= normal_safe_max
        )
    )
    if accepted:
        reason = "diagnostic_only"
        relation: str | None = "overlap" if ocr_overlap_pair_required else None
        if target_min <= estimate.shift_y <= target_max:
            efficiency_status = "semantic_zero_overlap_target"
        else:
            efficiency_status = "semantic_zero_overlap_acceptable"
    elif no_move:
        reason = "bottom_no_move"
        relation = None
        efficiency_status = "terminal_partial"
    elif (
        has_reliable_overlap
        and physical_min <= estimate.shift_y < feedback["diagnostic_micro_trigger_shift_px"]
    ):
        reason = "terminal_partial_candidate"
        relation = None
        efficiency_status = "terminal_partial"
    elif (
        has_reliable_overlap
        and feedback["diagnostic_micro_trigger_shift_px"]
        <= estimate.shift_y < acceptable_normal_min
    ):
        reason = "efficiency_correction_required"
        relation = None
        efficiency_status = "conservative"
    else:
        reason = "unsafe_gap_risk"
        relation = None
        efficiency_status = "aggressive" if estimate.shift_y > normal_safe_max else "conservative"
    return {
        "shift_estimate": asdict(estimate),
        "same_position_score": same_position_score,
        "local_overlap_score": local_overlap_score,
        "local_overlap_shift_px": local_overlap_shift_px,
        "local_overlap_valid": local_overlap_valid,
        "motion_reliability": motion_reliability,
        "target_shift_range": list(feedback["diagnostic_target_shift_px"]),
        "safe_shift_range": list(feedback["diagnostic_normal_safe_shift_px"]),
        "physical_shift_range": list(feedback["diagnostic_physical_shift_px"]),
        "efficiency_status": efficiency_status,
        "row_pitch_px": row_pitch_px,
        "visual_overlap_rows": visual_overlap_rows,
        "visual_overlap": semantic_overlap["visual_overlap"],
        "physical_overlap_px": semantic_overlap["physical_overlap_px"],
        "actual_shift_px": semantic_overlap["actual_shift_px"],
        "ocr_row_envelope": semantic_overlap["ocr_row_envelope"],
        "semantic_overlap_state": semantic_overlap["semantic_overlap_state"],
        "ocr_overlap_pair_required": ocr_overlap_pair_required,
        "full_row_candidates": semantic_overlap["full_row_candidates"],
        "semantic_overlap_reason": semantic_overlap["reason"],
        "semantic_overlap_runtime_ms": semantic_overlap["runtime_ms"],
        "row_lattice": semantic_overlap["row_lattice"],
        "true_shift_rows": estimate.shift_y / row_pitch_px,
        "accepted": accepted,
        "relation": relation,
        "reason": reason,
        "classification": "diagnostic_only",
    }


def _feedback_metrics_contract(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Project an internal feedback evaluation onto the stable JSON contract.

    Candidate selection, row-lattice geometry, thresholds, and retry reasons
    remain local implementation details.  B2 only needs the final transition
    decision plus a small amount of non-duplicated diagnostic evidence.
    """
    return {
        "accepted": evaluation["accepted"],
        "relation": evaluation["relation"],
        "ocr_overlap_pair_required": evaluation["ocr_overlap_pair_required"],
        "semantic_overlap_state": evaluation["semantic_overlap_state"],
        "section_complete": evaluation["section_complete"],
        "image_pair": evaluation["image_pair"],
        "diagnostics": {
            "actual_shift_px": evaluation["actual_shift_px"],
            "motion_reliability": evaluation["motion_reliability"]["mode"],
            "physical_overlap_px": evaluation["physical_overlap_px"],
        },
    }


def _evaluate_continuous_transition(
    prev_roi: np.ndarray,
    candidate_roi: np.ndarray,
    feedback: dict[str, Any],
) -> dict[str, Any]:
    """Reuse the B1 decision and expose only its stable transition contract."""
    evaluation = evaluate_feedback_candidate(prev_roi, candidate_roi, feedback)
    semantic_state = evaluation["semantic_overlap_state"]
    return _feedback_metrics_contract(
        {
            **evaluation,
            "section_complete": semantic_state == "not_applicable_no_move",
            "image_pair": None,
        }
    )


def _is_capture_safe_progress(
    transition: dict[str, Any], feedback: dict[str, Any]
) -> bool:
    """Accept reliable forward progress even when B1 marks it inefficient.

    B1's ``accepted`` flag deliberately includes its target-stride efficiency
    window. B2 instead needs a conservative capture rule: retain any reliable
    positive movement that cannot have skipped a row according to B1's
    already-configured physical and normal-safe bounds.
    """
    if transition["diagnostics"]["motion_reliability"] == "unreliable":
        return False
    actual_shift_px = transition["diagnostics"]["actual_shift_px"]
    if not isinstance(actual_shift_px, (int, float)):
        return False
    physical_min, _ = feedback["diagnostic_physical_shift_px"]
    _, normal_safe_max = feedback["diagnostic_normal_safe_shift_px"]
    return (
        physical_min
        <= actual_shift_px
        <= normal_safe_max + NORMAL_ACCEPTANCE_EPSILON_PX
    )


def _terminal_confirmed_overlap_pair_required(
    prev_roi: np.ndarray,
    candidate_roi: np.ndarray,
    feedback: dict[str, Any],
) -> bool:
    """Recheck only the final retained pair after the next frame confirms bottom.

    Normal continuous transitions keep their normal-band protection. This narrow
    path can select a shorter ORB-supported movement only after the following
    transition has independently established ``bottom_no_move``.
    """
    physical_min, _ = feedback["diagnostic_physical_shift_px"]
    terminal_max_shift_px = feedback["diagnostic_micro_trigger_shift_px"]
    estimate = estimate_vertical_motion(
        prev_roi,
        candidate_roi,
        feedback["diagnostic_expected_shift_px"],
        feedback["diagnostic_physical_shift_px"],
        terminal_confirmed=True,
        terminal_max_shift_px=float(terminal_max_shift_px),
    )
    selected = estimate.selected_hypothesis
    if (
        selected is None
        or not physical_min <= estimate.shift_y <= terminal_max_shift_px
        or estimate.match_count < 4
        or estimate.inlier_count < 4
        or estimate.confidence < feedback["diagnostic_min_confidence"]
        or selected.get("proposal_source") == "direct_normal_band"
        or selected.get("full_overlap_score", 0.0) < DIRECT_NORMAL_MIN_SCORE
        or selected.get("anchor_score", 0.0) < DIRECT_NORMAL_MIN_SCORE
    ):
        return False
    local_score, _, local_valid = _local_overlap_confirmation(
        prev_roi,
        candidate_roi,
        estimate.shift_y,
        feedback["local_search_radius_px"],
    )
    if (
        not local_valid
        or local_score is None
        or local_score < feedback["diagnostic_min_local_overlap_score"]
    ):
        return False
    semantic_overlap = evaluate_semantic_row_overlap(
        prev_roi,
        candidate_roi,
        estimate.shift_y,
        feedback["diagnostic_row_pitch_px"],
    )
    if (
        semantic_overlap["semantic_overlap_state"]
        in {"definitely_full_row", "ambiguous"}
        and semantic_overlap["ocr_overlap_pair_required"] is True
    ):
        return True

    # The terminal pair can lose the prior frame's lattice phase when the list
    # header leaves the compare ROI. The selected ORB/RANSAC transform and the
    # full/local-overlap checks above already establish the two frames' pixel
    # correspondence. Re-anchor only the independently detected candidate rows
    # through that transform, then reuse the unchanged row-envelope decision.
    candidate_row_centers = semantic_overlap["candidate_row_centers"]
    if not candidate_row_centers:
        return False
    reanchored_semantic_overlap = evaluate_semantic_row_overlap(
        prev_roi,
        candidate_roi,
        estimate.shift_y,
        feedback["diagnostic_row_pitch_px"],
        prev_row_centers=[center + estimate.shift_y for center in candidate_row_centers],
        candidate_row_centers=candidate_row_centers,
    )
    return (
        reanchored_semantic_overlap["semantic_overlap_state"]
        in {"definitely_full_row", "ambiguous"}
        and reanchored_semantic_overlap["ocr_overlap_pair_required"] is True
    )


def _gesture_payload(gesture: tuple[int, int, int, int, int]) -> dict[str, Any]:
    return {
        "start": [gesture[0], gesture[1]],
        "end": [gesture[2], gesture[3]],
        "duration_ms": gesture[4],
    }


@AgentServer.custom_action("StarBackpackCaptureProbe")
class StarBackpackCaptureProbe(CustomAction):
    """Capture a star-backpack screen with opt-in visual-only diagnostic probes."""

    def _capture(self, context: Context) -> np.ndarray:
        image = context.tasker.controller.post_screencap().wait().get()
        if image is None:
            raise RuntimeError("星石背包截图失败")
        if not isinstance(image, np.ndarray):
            raise RuntimeError("controller screenshot 不是 NumPy 图像")
        return image

    def _run_continuous_capture(
        self,
        context: Context,
        params: dict[str, Any],
        run_dir: Path,
        initial: np.ndarray,
    ) -> dict[str, Any]:
        """Capture adjacent main-star pages using one B1-evaluated swipe at a time."""
        retained_images = ["capture-00.png"]
        adjacent_relations: list[dict[str, str]] = []
        _write_png(run_dir / retained_images[0], initial)
        prev = initial
        prev_roi = _crop_roi(prev, params["compare_roi"])
        transition_count = 0
        failed_transition: int | None = None
        stop_reason = "transition_limit_reached"
        success = False
        last_semantic_overlap_state: str | None = None
        last_motion_reliability: str | None = None
        last_actual_shift_px: float | None = None
        last_retained_pair_rois: tuple[np.ndarray, np.ndarray] | None = None

        for transition_index in range(1, params["max_transitions"] + 1):
            # The B2 loop is deliberately a single gesture.  Retry and motion
            # policy remain B1 concerns; this layer never adds a second swipe.
            context.tasker.controller.post_swipe(*params["swipe"]).wait()
            transition_count = transition_index
            if params["settle_ms"]:
                time.sleep(params["settle_ms"] / 1000)
            candidate = self._capture(context)
            candidate_roi = _crop_roi(candidate, params["compare_roi"])
            transition = _evaluate_continuous_transition(
                prev_roi, candidate_roi, params["feedback"]
            )
            last_semantic_overlap_state = transition["semantic_overlap_state"]
            last_motion_reliability = transition["diagnostics"][
                "motion_reliability"
            ]
            last_actual_shift_px = transition["diagnostics"]["actual_shift_px"]

            if _is_capture_safe_progress(transition, params["feedback"]):
                previous_image = retained_images[-1]
                image_name = f"capture-{len(retained_images):02d}.png"
                _write_png(run_dir / image_name, candidate)
                retained_images.append(image_name)
                last_retained_pair_rois = (prev_roi, candidate_roi)
                if transition["ocr_overlap_pair_required"] is True:
                    adjacent_relations.append(
                        {
                            "previous_image": previous_image,
                            "current_image": image_name,
                            "relation": "overlap",
                        }
                    )
                prev = candidate
                prev_roi = candidate_roi
                continue

            if last_semantic_overlap_state == "not_applicable_no_move":
                if len(retained_images) > 1:
                    success = True
                    stop_reason = "bottom_no_move"
                    final_pair = (retained_images[-2], retained_images[-1])
                    final_pair_has_relation = any(
                        relation["previous_image"] == final_pair[0]
                        and relation["current_image"] == final_pair[1]
                        for relation in adjacent_relations
                    )
                    if (
                        not final_pair_has_relation
                        and last_retained_pair_rois is not None
                        and _terminal_confirmed_overlap_pair_required(
                            last_retained_pair_rois[0],
                            last_retained_pair_rois[1],
                            params["feedback"],
                        )
                    ):
                        adjacent_relations.append(
                            {
                                "previous_image": final_pair[0],
                                "current_image": final_pair[1],
                                "relation": "overlap",
                            }
                        )
                else:
                    stop_reason = "no_move_before_progress"
                    failed_transition = transition_index
            elif last_semantic_overlap_state == "not_applicable_unreliable_motion":
                stop_reason = "unreliable_transition"
                failed_transition = transition_index
            else:
                stop_reason = "rejected_transition"
                failed_transition = transition_index
            if stop_reason in {"unreliable_transition", "rejected_transition"}:
                _write_png(
                    run_dir / f"failed-candidate-{transition_index:02d}.png",
                    candidate,
                )
            break
        else:
            failed_transition = transition_count

        session = {
            "success": success,
            "stop_reason": stop_reason,
            "retained_images": retained_images,
            "retained_image_count": len(retained_images),
            "adjacent_relations": adjacent_relations,
            "transition_count": transition_count,
            "failed_transition": failed_transition,
            "diagnostics": {
                "last_semantic_overlap_state": last_semantic_overlap_state,
                "last_motion_reliability": last_motion_reliability,
                "last_actual_shift_px": last_actual_shift_px,
            },
        }
        _write_json(run_dir / "session.json", session)
        return session

    def _run_feedback_probe(
        self,
        context: Context,
        params: dict[str, Any],
        run_dir: Path,
        prev: np.ndarray,
    ) -> dict[str, Any]:
        """Keep every retry anchored to ``prev`` so gesture error cannot accumulate.

        Incorrect: assume an expected X-pixel stride and chain page1+X, page2+X.
        Correct: compare the accepted real prev image with each real candidate,
        accepting a candidate only after its direct image evidence is safe.
        """
        _write_png(run_dir / "prev.png", prev)
        prev_roi = _crop_roi(prev, params["compare_roi"])
        _write_png(run_dir / "prev-roi.png", prev_roi)

        attempts: list[dict[str, Any]] = []
        final_candidate: np.ndarray | None = None
        final_candidate_roi: np.ndarray | None = None
        final_evaluation: dict[str, Any] | None = None
        candidate_index = 0

        def capture_candidate(
            gesture: tuple[int, int, int, int, int], stage: str
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            nonlocal candidate_index
            context.tasker.controller.post_swipe(*gesture).wait()
            if params["settle_ms"]:
                time.sleep(params["settle_ms"] / 1000)
            candidate = self._capture(context)
            candidate_roi = _crop_roi(candidate, params["compare_roi"])
            evaluation = evaluate_feedback_candidate(
                prev_roi, candidate_roi, params["feedback"]
            )
            estimate = evaluation["shift_estimate"]
            attempts.append(
                {
                    "stage": stage,
                    "gesture": _gesture_payload(gesture),
                    "shift_y": estimate["shift_y"],
                    "confidence": estimate["confidence"],
                    "local_overlap_score": evaluation["local_overlap_score"],
                    "reason": evaluation["reason"],
                }
            )
            # A no-move confirmation is evidence only. It is never promoted to
            # a business capture, while a positive candidate is retained for
            # diagnostic review and potential terminal-partial acceptance.
            if evaluation["reason"] != "bottom_no_move":
                candidate_index += 1
                attempt_name = f"candidate-{candidate_index:02d}"
                _write_png(run_dir / f"{attempt_name}.png", candidate)
                _write_png(run_dir / f"{attempt_name}-roi.png", candidate_roi)
            return candidate, candidate_roi, evaluation

        candidate, candidate_roi, evaluation = capture_candidate(
            params["coarse_swipe"], "coarse"
        )
        if (
            not params["single_swipe_calibration"]
            and evaluation["reason"] == "terminal_partial_candidate"
        ):
            terminal_candidate = candidate
            terminal_candidate_roi = candidate_roi
            terminal_evaluation = evaluation
            # The confirmation deliberately compares against the temporary
            # terminal candidate, not ``prev``. If it no longer moves, the
            # temporary candidate is the final page evidence to accept.
            confirmation_gesture = params["micro_swipe"]
            context.tasker.controller.post_swipe(*confirmation_gesture).wait()
            if params["settle_ms"]:
                time.sleep(params["settle_ms"] / 1000)
            confirmation = self._capture(context)
            confirmation_roi = _crop_roi(confirmation, params["compare_roi"])
            confirmation_evaluation = evaluate_feedback_candidate(
                terminal_candidate_roi, confirmation_roi, params["feedback"]
            )
            confirmation_estimate = confirmation_evaluation["shift_estimate"]
            attempts.append(
                {
                    "stage": "terminal_confirmation",
                    "gesture": _gesture_payload(confirmation_gesture),
                    "shift_y": confirmation_estimate["shift_y"],
                    "confidence": confirmation_estimate["confidence"],
                    "local_overlap_score": confirmation_evaluation["local_overlap_score"],
                    "reason": confirmation_evaluation["reason"],
                }
            )
            if confirmation_evaluation["reason"] == "bottom_no_move":
                final_candidate = terminal_candidate
                final_candidate_roi = terminal_candidate_roi
                final_evaluation = dict(terminal_evaluation)
                final_evaluation.update(
                    {
                        "accepted": True,
                        "relation": (
                            "overlap"
                            if terminal_evaluation["ocr_overlap_pair_required"]
                            else None
                        ),
                        "reason": "diagnostic_only",
                        "efficiency_status": "terminal_partial",
                        "terminal_partial_confirmed": True,
                        "section_complete": True,
                    }
                )
            else:
                # The confirmation did move, so it becomes a fresh candidate
                # against the original accepted prev. No assumed stride is
                # carried forward from the temporary terminal frame.
                candidate = confirmation
                candidate_roi = confirmation_roi
                evaluation = evaluate_feedback_candidate(
                    prev_roi, candidate_roi, params["feedback"]
                )
                if evaluation["reason"] != "bottom_no_move":
                    candidate_index += 1
                    attempt_name = f"candidate-{candidate_index:02d}"
                    _write_png(run_dir / f"{attempt_name}.png", candidate)
                    _write_png(run_dir / f"{attempt_name}-roi.png", candidate_roi)
                final_candidate = candidate
                final_candidate_roi = candidate_roi
                final_evaluation = evaluation

        if final_evaluation is None:
            final_candidate = candidate
            final_candidate_roi = candidate_roi
            final_evaluation = evaluation
        # A normal page with only 3-ish visual rows advanced needs exactly one
        # corrective micro swipe.  Each attempt remains measured from ``prev``;
        # no accumulated page-to-page assumption is allowed here.
        if (
            not params["single_swipe_calibration"]
            and final_evaluation["reason"] == "efficiency_correction_required"
        ):
            final_candidate, final_candidate_roi, final_evaluation = capture_candidate(
                params["micro_swipe"], "efficiency_micro"
            )

        if (
            not params["single_swipe_calibration"]
            and final_evaluation["reason"]
            in {"move_too_small", "efficiency_correction_required"}
        ):
            final_evaluation["reason"] = "unsafe_gap_risk"
        if final_evaluation["reason"] == "bottom_no_move":
            if not params["single_swipe_calibration"]:
                final_candidate = None
                final_candidate_roi = None
            final_evaluation["section_complete"] = True
        else:
            final_evaluation.setdefault("section_complete", False)
        if final_candidate is not None and final_candidate_roi is not None:
            _write_png(run_dir / "candidate.png", final_candidate)
            _write_png(run_dir / "candidate-roi.png", final_candidate_roi)
        final_evaluation["attempts"] = attempts
        final_evaluation["gesture_count"] = len(attempts)
        final_evaluation["single_swipe_calibration"] = params[
            "single_swipe_calibration"
        ]
        final_evaluation["image_pair"] = (
            {
                "previous_image_id": "prev.png",
                "current_image_id": "candidate.png",
            }
            if (
                final_evaluation["accepted"]
                and final_evaluation["ocr_overlap_pair_required"]
                and final_candidate is not None
            )
            else None
        )
        metrics = _feedback_metrics_contract(final_evaluation)
        _write_json(run_dir / "feedback-metrics.json", metrics)
        return metrics

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        try:
            params = parse_capture_probe_params(argv.custom_action_param)
            run_dir = _prepare_run_directory(params["debug_dir"])
            before = self._capture(context)

            if params["mode"] == CAPTURE_ONLY_MODE:
                _write_png(run_dir / "capture.png", before)
                metadata = _image_metadata(before, CAPTURE_ONLY_MODE, "capture.png")
                _write_json(run_dir / "metadata.json", metadata)
                logger.info(
                    "星石背包截图探针已保存: "
                    f"size={metadata['width']}x{metadata['height']}, "
                    f"channels={metadata['channels']}, dtype={metadata['dtype']}, "
                    f"dir={run_dir}"
                )
                return CustomAction.RunResult(success=True)

            if params["mode"] == PAIR_PROBE_MODE:
                _write_png(run_dir / "before.png", before)
                context.tasker.controller.post_swipe(*params["swipe"]).wait()
                if params["settle_ms"]:
                    time.sleep(params["settle_ms"] / 1000)
                after = self._capture(context)
                _write_png(run_dir / "after.png", after)

                before_roi = _crop_roi(before, params["compare_roi"])
                after_roi = _crop_roi(after, params["compare_roi"])
                _write_png(run_dir / "before-roi.png", before_roi)
                _write_png(run_dir / "after-roi.png", after_roi)
                metrics = compute_visual_overlap(
                    before_roi,
                    after_roi,
                    params["compare"]["min_overlap_ratio"],
                    params["compare"]["max_overlap_ratio"],
                )
                _write_json(run_dir / "metrics.json", metrics)
                logger.info(
                    "星石背包 pair_probe 完成（仅诊断）: "
                    f"same={metrics['same_position_score']:.4f}, "
                    f"overlap={metrics['best_overlap_score']:.4f}, "
                    f"overlap_px={metrics['best_overlap_px']}, dir={run_dir}"
                )
                return CustomAction.RunResult(success=True)
            elif params["mode"] == FEEDBACK_PROBE_MODE:
                metrics = self._run_feedback_probe(context, params, run_dir, before)
                logger.info(
                    "星石背包 feedback_probe 完成（仅诊断）: "
                    f"accepted={metrics['accepted']}, "
                    f"section_complete={metrics['section_complete']}, dir={run_dir}"
                )
                return CustomAction.RunResult(success=True)

            session = self._run_continuous_capture(context, params, run_dir, before)
            logger.info(
                "星石背包连续采集完成: "
                f"success={session['success']}, stop={session['stop_reason']}, "
                f"images={session['retained_image_count']}, dir={run_dir}"
            )
            return CustomAction.RunResult(success=session["success"])
        except Exception as exc:
            logger.exception(f"星石背包截图探针失败: {exc}")
            return CustomAction.RunResult(success=False)
