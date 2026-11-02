"""Motion analysis and opt-in calibration diagnostics; no screenshot upload or capture task."""

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

NORMAL_MOTION_MIN_PX = 560.0

NORMAL_MOTION_MAX_PX = 660.0

MIN_DIRECT_OVERLAP_PX = 80

DIRECT_OVERLAP_MARGIN_PX = 6

DIRECT_NORMAL_MIN_SCORE = 0.70

DIRECT_NORMAL_LOCAL_RADIUS_PX = 2

DIRECT_HYPOTHESIS_DEDUPE_PX = 12.0

YUANSTAR_ROW_RADIUS_TO_PITCH_RATIO = 0.34

YUANSTAR_ENVELOPE_TOP_RADIUS = 1.06

YUANSTAR_ENVELOPE_BOTTOM_RADIUS = 1.70

SEMANTIC_ENVELOPE_SAFETY_MARGIN_PX = 10.0

def _resolve_debug_base(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path

def _prepare_run_directory(debug_dir: str) -> Path:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    run_dir = _resolve_debug_base(debug_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

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

@AgentServer.custom_action("StarBackpackCaptureProbe")
class StarBackpackCaptureProbe(CustomAction):
    """Private one-gesture calibration entry retained for the existing tool."""
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        try:
            params = json.loads(argv.custom_action_param) if isinstance(argv.custom_action_param, str) else argv.custom_action_param
            if params.get('mode') != 'swipe_stability_probe':
                raise ValueError('only explicit swipe calibration is supported')
            from utils.star_swipe_calibration import parse_calibration_params, collect_trial
            parsed = {**parse_calibration_params(params), 'debug_dir': params.get('debug_dir', 'debug/star-backpack-probe')}
            result = collect_trial(context, parsed, _prepare_run_directory(parsed['debug_dir']))
            return CustomAction.RunResult(success=result['status'] == 'captured')
        except Exception:
            logger.error('星石滑动诊断未完成')
            return CustomAction.RunResult(success=False)
