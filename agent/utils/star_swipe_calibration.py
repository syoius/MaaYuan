"""Opt-in, one-gesture diagnostics. No OCR, upload, tab switching or touch primitives."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

GROUPS = {"A": (700, 0), "B": (250, 0), "C": (700, 250), "D": (250, 250)}
SAMPLE_OFFSETS_MS = (0, 100, 250, 500, 1000, 2500)
EXTENDED_SAMPLE_OFFSETS_MS = (*SAMPLE_OFFSETS_MS, 3000, 4000)
SWIPE_NODE = "StarSwipeEndHoldCalibrationNative"


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def parse_calibration_params(raw: dict) -> dict:
    """Keep calibration separate from production swipe/settle parameters."""
    group = raw.get("group")
    if group not in {*GROUPS, "E", "F"}:
        raise ValueError("group must be A, B, C, D or independent candidate E/F")
    extended = raw.get("extended_observation", False)
    if type(extended) is not bool:
        raise ValueError("extended_observation must be a boolean")
    if group == "E" and (not extended or type(raw.get("trial")) is not int or raw["trial"] != 1):
        raise ValueError("independent E1 requires extended observation and trial=1")
    if group == "F" and (extended or type(raw.get("trial")) is not int or raw["trial"] != 1 or raw.get("section") != "main"):
        raise ValueError("independent F1 requires main/trial=1 and the 2500ms observation schedule")
    duration, hold = (700, 250) if group in {"E", "F"} else GROUPS[group]
    end = {"E": [360, 480], "F": [360, 435]}.get(group, [360, 540])
    for name, expected in (("swipe_begin", [360, 930]), ("swipe_end", end)):
        point = raw.get(name)
        if not isinstance(point, (list, tuple)) or list(point) != expected or any(type(v) is not int for v in point):
            raise ValueError(f"first-round {name} must equal {expected}")
    if raw.get("swipe_duration_ms", duration) != duration or type(raw.get("swipe_duration_ms", duration)) is not int:
        raise ValueError("duration must match the selected group")
    if raw.get("swipe_end_hold_ms", hold) != hold or type(raw.get("swipe_end_hold_ms", hold)) is not int:
        raise ValueError("end_hold must match the selected group")
    if raw.get("post_release_settle_ms", 2500) != 2500 or type(raw.get("post_release_settle_ms", 2500)) is not int:
        raise ValueError("first-round observation horizon must remain 2500ms")
    for flag in ("manual_start_authorized", "manual_reset_confirmed", "backend_touch_hold_confirmed"):
        if raw.get(flag) is not True:
            raise ValueError(f"{flag}=true is required for a single real gesture")
    section = raw.get("section")
    if section not in {"main", "support"}:
        raise ValueError("section must be main or support; experience is excluded")
    return {
        "group": group, "trial": _integer(raw.get("trial"), "trial", 1, 3), "section": section,
        "swipe_begin": [360, 930], "swipe_end": end,
        "swipe_duration_ms": duration, "swipe_end_hold_ms": hold,
        "post_release_settle_ms": 2500, "tab_settle_ms": 2500,
        "compare_roi": [40, 280, 640, 760],
        "sample_offsets_ms": list(EXTENDED_SAMPLE_OFFSETS_MS if extended else SAMPLE_OFFSETS_MS),
        "observation_horizon_ms": 4000 if extended else 2500,
        "manual_start_authorized": True, "manual_reset_confirmed": True,
        "backend_touch_hold_confirmed": True,
    }


def swipe_pipeline(params: dict) -> dict:
    x1, y1 = params["swipe_begin"]
    x2, y2 = params["swipe_end"]
    return {SWIPE_NODE: {
        "recognition": "DirectHit",
        "action": {"type": "Swipe", "param": {
            # 1x1 ROIs avoid random rectangle targets and zero-area fallback.
            "begin": [x1, y1, 1, 1], "end": [x2, y2, 1, 1],
            "duration": params["swipe_duration_ms"], "end_hold": params["swipe_end_hold_ms"],
            "only_hover": False, "contact": 0,
        }},
        # Explicit zeroes avoid the native default post-action delay in this timing experiment.
        "pre_delay": 0, "post_delay": 0, "rate_limit": 0, "next": [], "on_error": [],
    }}


def _cancelled(context) -> bool:
    return bool(getattr(context.tasker, "stopping", False))


def _check_cancel(context) -> None:
    if _cancelled(context):
        raise InterruptedError("calibration stopped")


def _stamp(clock=time) -> dict:
    return {"monotonic_ns": clock.perf_counter_ns(), "utc_ns": clock.time_ns()}


def _capture(context, clock=time) -> tuple[np.ndarray, dict]:
    _check_cancel(context)
    start = _stamp(clock)
    job = context.tasker.controller.post_screencap().wait()
    end = _stamp(clock)
    if not job.succeeded:
        # A failed screencap can leave a cached previous image: never accept it.
        raise RuntimeError("screencap job failed; cached image rejected")
    image = job.get()
    if not isinstance(image, np.ndarray) or image.shape != (1280, 720, 3) or image.dtype != np.uint8:
        raise ValueError("calibration requires an actual 720x1280 BGR uint8 frame")
    frozen = np.ascontiguousarray(image).copy()
    frozen.flags.writeable = False
    return frozen, {"request": start, "completion": end, "frame_exposure_time": None,
                    "screencap_duration_ms": (end["monotonic_ns"] - start["monotonic_ns"]) / 1e6}


def _task_success(detail) -> bool:
    return bool(detail and getattr(getattr(detail, "status", None), "succeeded", False))


def collect_trial(context, params: dict, run_dir: Path, *, clock=time) -> dict:
    """Exactly one gesture. Buffers first, disk writes after the sampling window."""
    offsets = tuple(params["sample_offsets_ms"])
    if offsets not in (SAMPLE_OFFSETS_MS, EXTENDED_SAMPLE_OFFSETS_MS):
        raise ValueError("unsupported diagnostic sample schedule")
    if any((run_dir / name).exists() for name in ("trial.json", "before.png", *(f"after-{v:04d}ms.png" for v in offsets))):
        raise FileExistsError("refusing to overwrite an existing trial")
    manifest = {"schemaVersion": 1, "status": "started", "scope": "device_calibration",
                "parameters": params, "started": _stamp(clock), "samples": [],
                "release_physical_time": None, "release_time_note": "not exposed by the public API"}
    buffers = []
    try:
        before, capture = _capture(context, clock)
        buffers.append(("before.png", before))
        manifest["before"] = {"fileName": "before.png", **capture}
        _check_cancel(context)
        manifest["gesture_request"] = _stamp(clock)
        detail = context.run_task(SWIPE_NODE, swipe_pipeline(params))
        returned = _stamp(clock)
        manifest["gesture_return"] = returned
        # No public touch-up timestamp: this is an explicitly labelled host-side upper bound.
        manifest["release_reference"] = {**returned, "source": "zero-delay native Swipe task return",
                                          "is_physical_touch_up_timestamp": False}
        _check_cancel(context)
        if not _task_success(detail):
            raise RuntimeError("native Pipeline Swipe failed or was cancelled")
        actions = [n.action for n in getattr(detail, "nodes", []) if getattr(n, "action", None)]
        if not actions or any(not a.success for a in actions):
            raise RuntimeError("native Swipe did not return a successful ActionDetail")
        manifest["native_action_results"] = [a.raw_detail for a in actions]
        origin = returned["monotonic_ns"]
        for offset in offsets:
            target = origin + offset * 1_000_000
            while clock.perf_counter_ns() < target:
                _check_cancel(context)
                clock.sleep(min(.02, (target - clock.perf_counter_ns()) / 1e9))
            image, sample = _capture(context, clock)
            filename = f"after-{offset:04d}ms.png"
            sample.update(fileName=filename, requested_offset_ms=offset,
                request_offset_ms=(sample["request"]["monotonic_ns"]-origin)/1e6,
                completion_offset_ms=(sample["completion"]["monotonic_ns"]-origin)/1e6)
            sample["lateness_ms"] = max(0., sample["request_offset_ms"] - offset)
            manifest["samples"].append(sample)
            buffers.append((filename, image))
        manifest["status"] = "captured"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "cancelled" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyboardInterrupt) and hasattr(context, "abort_and_wait"):
            context.abort_and_wait()
    finally:
        # Preserve partial evidence on failure/cancel; never silently drop a captured buffer.
        manifest["finished"] = _stamp(clock)
        for name, image in buffers:
            ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3,
                                                       cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE])
            if not ok:
                raise RuntimeError(f"PNG encoding failed: {name}")
            data = encoded.tobytes()
            (run_dir / name).write_bytes(data)
            record = manifest["before"] if name == "before.png" else next(s for s in manifest["samples"] if s["fileName"] == name)
            record["sha256"] = hashlib.sha256(data).hexdigest()
        write_json(run_dir / "trial.json", manifest)
    return manifest


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"invalid image: {path}")
    return image


def residual_motion(first: np.ndarray, second: np.ndarray, probe) -> dict:
    """Diagnostic signed small motion, deliberately separate from the production normal band."""
    a = probe._prepare_compare_image(first)
    b = probe._prepare_compare_image(second)
    same = probe._normalized_correlation(a, b)
    limit = min(120, (a.shape[0]-32)//2)
    template = a[limit:-limit]
    scores = cv2.matchTemplate(b, template, cv2.TM_CCOEFF_NORMED).reshape(-1)
    position = int(np.argmax(scores))
    shift = limit-position  # upward is positive; downward is negative
    score = float(scores[position])
    alternatives = scores.copy()
    alternatives[max(0, position-4):position+5] = -1
    margin = score-float(alternatives.max())
    # Signed ORB supports downward drift; the production helper intentionally rejects it.
    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
    ka, da = orb.detectAndCompute(a.astype(np.uint8), None)
    kb, db = orb.detectAndCompute(b.astype(np.uint8), None)
    dys = []
    if da is not None and db is not None and len(db) >= 2:
        for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2):
            if len(pair) != 2 or pair[0].distance >= .72*pair[1].distance:
                continue
            source, target = ka[pair[0].queryIdx].pt, kb[pair[0].trainIdx].pt
            dy = source[1]-target[1]
            if abs(source[0]-target[0]) <= 12 and abs(dy) < limit:
                dys.append(dy)
    orb_shift = float(np.median(dys)) if len(dys) >= 8 else None
    orb_mad = float(np.median(np.abs(np.asarray(dys)-orb_shift))) if orb_shift is not None else None
    supported = orb_shift is not None and orb_mad <= 2 and abs(orb_shift-shift) <= 2
    valid = abs(shift) < limit-2 and score >= .90 and (margin >= .01 or supported or (shift == 0 and same >= .97))
    return {"signed_shift_y_px": float(shift) if valid else None, "measured": bool(valid),
            "same_position_score": same, "direct_score": score, "direct_margin": margin,
            "orb_shift_y_px": orb_shift, "orb_mad_px": orb_mad, "orb_matches": len(dys),
            "search_range_px": [-limit, limit], "diagnostic_stable": bool(valid and abs(shift) <= 2 and same >= .97)}


def analyse_trial(run_dir: Path, probe) -> dict:
    manifest = json.loads((run_dir / "trial.json").read_text(encoding="utf-8"))
    if manifest["status"] != "captured":
        raise ValueError("only captured trials can be analysed")
    before = read_image(run_dir / manifest["before"]["fileName"])
    samples = manifest["samples"]
    expected = manifest["parameters"].get("sample_offsets_ms", list(SAMPLE_OFFSETS_MS))
    if tuple(expected) not in (SAMPLE_OFFSETS_MS, EXTENDED_SAMPLE_OFFSETS_MS):
        raise ValueError("unsupported diagnostic sample schedule")
    if [s["requested_offset_ms"] for s in samples] != expected:
        raise ValueError("trial is missing required timepoints")
    frames = [read_image(run_dir / s["fileName"]) for s in samples]
    for record, image in [(manifest["before"], before), *zip(samples, frames)]:
        if hashlib.sha256((run_dir/record["fileName"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("image hash mismatch")
        if image.shape != (1280, 720, 3):
            raise ValueError("frame shape changed")
    crop = lambda img: img[280:1040, 40:680]
    total = probe.estimate_vertical_motion(crop(before), crop(frames[-1]), (250, 700), (30, 700))
    # The production selector prefers its calibrated normal band. During calibration,
    # a different duration can change the distribution: do not silently turn that prior
    # into a measured distance if a periodic image supports another strong hypothesis.
    selected_score = (total.selected_hypothesis or {}).get("full_overlap_score")
    selected_score = selected_score if selected_score is not None else total.confidence
    aliases = [candidate for candidate in total.motion_hypotheses
               if candidate.get("full_overlap_valid") and candidate.get("full_overlap_score") is not None
               and abs(candidate["shift_y"]-total.shift_y)>12
               and candidate["full_overlap_score"] >= max(.70, selected_score-.03)]
    reliable = bool((total.confidence >= .2 or probe._has_reliable_direct_normal_motion(total)) and not aliases)
    semantic = probe.evaluate_semantic_row_overlap(crop(before), crop(frames[-1]), total.shift_y, 166.5) if reliable else None
    adjacent = [residual_motion(crop(a), crop(b), probe) for a, b in zip(frames, frames[1:])]
    to_final = [residual_motion(crop(a), crop(frames[-1]), probe) for a in frames]
    for i, record in enumerate(adjacent):
        left = (samples[i]["request_offset_ms"] + samples[i]["completion_offset_ms"])/2
        right = (samples[i+1]["request_offset_ms"] + samples[i+1]["completion_offset_ms"])/2
        record["host_midpoint_interval_ms"] = right-left
        record["velocity_proxy_px_per_second"] = record["signed_shift_y_px"]*1000/(right-left) if record["measured"] and right>left else None
    velocities = [r["velocity_proxy_px_per_second"] for r in adjacent]
    known_residuals = [r["signed_shift_y_px"] for r in to_final if r["measured"]]
    directions = {1 if v>2 else -1 for v in known_residuals if abs(v)>2}
    stable_by = None
    for i in range(len(samples)-1):
        enough_span = samples[-1]["request_offset_ms"]-samples[i]["completion_offset_ms"] >= 500
        if enough_span and all(r["diagnostic_stable"] for r in to_final[i:]) and all(r["diagnostic_stable"] for r in adjacent[i:]):
            stable_by = samples[i]["completion_offset_ms"]
            break
    result = {"schemaVersion": 1, "scope": "observed_device_frames", "parameters": manifest["parameters"],
        "actual_content_shift_y_px": total.shift_y if reliable else None,
        "motion_reliable": bool(reliable), "motion_confidence": total.confidence,
        "reference_band_estimate_y_px": total.shift_y,
        "periodic_alias_candidates": aliases,
        "measurement_limit": "normal-band production estimator is reference evidence only; competing strong periodic hypotheses invalidate actual distance; loss of shared pixels cannot establish arbitrary large motion",
        "selected_hypothesis": total.selected_hypothesis, "semantic_overlap": semantic,
        "residual_adjacent": adjacent, "residual_to_final": to_final,
        "maximum_observed_residual_to_final_px": max(map(abs,known_residuals)) if known_residuals else None,
        "residual_measurements_complete": len(known_residuals)==len(to_final),
        "residual_direction_consistent": len(directions)<=1 if len(known_residuals)==len(to_final) else None,
        "residual_velocity_magnitude_nonincreasing": all(abs(a)>=abs(b) for a,b in zip(velocities,velocities[1:])) if all(v is not None for v in velocities) else None,
        "sample_intervals": samples, "stable_by_observed_ms": stable_by,
        "stability_rule": "<=2px and NCC>=0.97 at every remaining observed interval through final; >=500ms observation span; no continuous-time guarantee",
        "physical_release_time_known": False,
        "release_reference": manifest["release_reference"],
        "time_reference_limit": "host zero-delay task-return proxy, not touch-up/exposure timestamp; late samples cannot establish earlier stability"}
    write_json(run_dir / "analysis.json", result)
    return result


def summarize_trials(run_dirs: list[Path], probe) -> dict:
    """No ranking without 3 valid unique repeats per group and matching initial images."""
    summaries = {}
    excluded = []
    baseline = None
    for directory in run_dirs:
        m = json.loads((directory/"trial.json").read_text(encoding="utf-8"))
        if m["status"] != "captured":
            excluded.append({"directory": str(directory), "status": m["status"], "reason": m.get("error")})
            continue
        p = m["parameters"]
        key = (p["section"], p["group"])
        bucket = summaries.setdefault(key, {"trials": [], "seen": set()})
        if p["trial"] in bucket["seen"]:
            raise ValueError(f"duplicate trial number: {key}/{p['trial']}; choose a single run explicitly")
        bucket["seen"].add(p["trial"])
        analysis = analyse_trial(directory, probe)
        initial = read_image(directory/"before.png")[280:1040, 40:680]
        if baseline is None:
            baseline = {}
        comparable = True
        if p["section"] in baseline:
            comparable = residual_motion(baseline[p["section"]], initial, probe)["diagnostic_stable"]
        else:
            baseline[p["section"]] = initial
        backend = m.get("backend")
        bucket["trials"].append({"directory": str(directory), "trial": p["trial"], "initial_position_comparable": comparable,
            "backend": backend,
            "actual_shift_y_px": analysis["actual_content_shift_y_px"], "stable_by_observed_ms": analysis["stable_by_observed_ms"],
            "residual_to_final_px": analysis["residual_to_final"][0]["signed_shift_y_px"]})
    groups = []
    for (section, group), bucket in sorted(summaries.items()):
        trials = bucket["trials"]
        values = [r["actual_shift_y_px"] for r in trials if r["actual_shift_y_px"] is not None]
        stable = [r["stable_by_observed_ms"] for r in trials if r["stable_by_observed_ms"] is not None]
        backends = {json.dumps(r["backend"], sort_keys=True) for r in trials}
        valid = len(trials) == 3 and bucket["seen"] == {1, 2, 3} and len(values) == 3 and len(stable) == 3 and all(r["initial_position_comparable"] and r["backend"] for r in trials) and len(backends) == 1
        groups.append({"section": section, "group": group, "complete_comparable_group": valid, "trials": trials,
            "shift_mean_px": float(np.mean(values)) if values else None,
            "shift_min_px": min(values) if values else None, "shift_max_px": max(values) if values else None,
            "shift_range_px": max(values)-min(values) if values else None,
            "shift_population_std_px": float(np.std(values)) if values else None,
            "stable_by_worst_observed_ms": max(stable) if len(stable) == 3 else None})
    return {"groups": groups, "excluded_trials": excluded, "winner": None, "settle_recommendation_ms": None,
            "ranking_note": "Compare only complete groups of the same section/backend/reset position; physical release and exposure timestamps remain unavailable; no automatic production parameter changes."}
