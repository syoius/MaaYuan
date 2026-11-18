"""Backpack swipes that release only after an end hold and wait for a still list."""
from __future__ import annotations

import math
import time
from collections.abc import Callable

import cv2
import numpy as np


SWIPE_NODE = "BackpackSwipeEndHold"
END_HOLD_MS = 600
STABILITY_TIMEOUT_SECONDS = 8.0
SAMPLE_INTERVAL_SECONDS = 0.2


def _check_stopped(stopped: Callable[[], bool]) -> None:
    if stopped():
        raise InterruptedError("背包扫描已停止，已保留原数量")


def _capture(context, stopped: Callable[[], bool]) -> np.ndarray:
    _check_stopped(stopped)
    job = context.tasker.controller.post_screencap().wait()
    if not job.succeeded:
        raise RuntimeError("背包扫描截图失败，拒绝使用缓存截图，已保留原数量")
    image = job.get()
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError("背包扫描截图无效，已保留原数量")
    return image


def _content_frame(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    if min(x, y) < 0 or min(width, height) <= 100 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("背包滑动截图区域变化，已保留原数量")
    # Exclude the fixed tabs/footer, leaving the moving icons and item names.
    gray = cv2.cvtColor(image[y + 50:y + height - 50, x:x + width], cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)


def frames_are_still(before: np.ndarray, after: np.ndarray) -> bool:
    if before.shape != after.shape:
        return False
    if np.array_equal(before, after):
        return True
    # Reuse the backpack's historical 4x4 block-motion rule (dbbce96a), rather
    # than treating animated icons as scrolling because their pixels changed.
    shifts = []
    height, width = before.shape
    for row in range(4):
        top, bottom = row * height // 4, (row + 1) * height // 4
        for column in range(4):
            left, right = column * width // 4, (column + 1) * width // 4
            pair = []
            for image in (before, after):
                block = image[top:bottom, left:right].copy()
                block -= float(block.mean())
                h, w = block.shape
                padded_h = 2 * cv2.getOptimalDFTSize((h + 1) // 2)
                padded_w = 2 * cv2.getOptimalDFTSize((w + 1) // 2)
                pair.append(cv2.copyMakeBorder(block, 0, padded_h - h, 0, padded_w - w,
                                               cv2.BORDER_CONSTANT))
            window = cv2.createHanningWindow((padded_w, padded_h), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(*pair, window)
            if np.isfinite([dx, dy, response]).all() and response >= .5:
                shifts.append((dx, dy))
    if len(shifts) < 9:
        return False
    shifts = np.asarray(shifts)
    center = np.median(shifts, axis=0)
    consistent = np.max(np.abs(shifts - center), axis=1) <= 1
    if int(consistent.sum()) < 9:
        return False
    return bool(np.max(np.abs(np.median(shifts[consistent], axis=0))) <= 3)


def wait_backpack_still(context, roi: tuple[int, int, int, int], *,
                        stopped: Callable[[], bool], minimum_wait_seconds: float) -> None:
    started = time.monotonic()
    previous = _content_frame(_capture(context, stopped), roi)
    anchor = previous
    stable_comparisons = 0
    # A configured minimum wait must not consume the time allowed to settle.
    deadline = started + minimum_wait_seconds + STABILITY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        _check_stopped(stopped)
        time.sleep(SAMPLE_INTERVAL_SECONDS)
        current = _content_frame(_capture(context, stopped), roi)
        if frames_are_still(previous, current) and frames_are_still(anchor, current):
            stable_comparisons += 1
        else:
            stable_comparisons = 0
            anchor = current
        if stable_comparisons >= 2 and time.monotonic() - started >= minimum_wait_seconds:
            return
        previous = current
    raise TimeoutError("背包列表持续移动，停止本分区，已保留原数量")


def scroll_backpack(context, swipe: tuple[int, int, int, int, int],
                    roi: tuple[int, int, int, int], *,
                    stopped: Callable[[], bool], minimum_wait_seconds: float = .7) -> None:
    _check_stopped(stopped)
    if not math.isfinite(minimum_wait_seconds):
        raise ValueError("背包滑动等待时间必须是有限数值")
    x1, y1, x2, y2, duration = swipe
    if (duration <= 0 or (x1, y1) == (x2, y2)
            or any(not (roi[0] < x < roi[0] + roi[2] and roi[1] < y < roi[1] + roi[3])
                   for x, y in ((x1, y1), (x2, y2)))):
        raise ValueError("背包滑动必须位于列表内容区内")
    pipeline = {SWIPE_NODE: {
        "recognition": "DirectHit",
        "action": {"type": "Swipe", "param": {
            "begin": [x1, y1, 1, 1], "end": [x2, y2, 1, 1],
            "duration": duration, "end_hold": END_HOLD_MS,
            "only_hover": False, "contact": 0,
        }},
        "pre_delay": 0, "post_delay": 0, "rate_limit": 0, "next": [], "on_error": [],
    }}
    detail = context.run_task(SWIPE_NODE, pipeline)
    _check_stopped(stopped)
    if not detail or not detail.status.succeeded:
        raise RuntimeError("背包滑动失败，已保留原数量")
    actions = [node.action for node in detail.nodes if node.action]
    if len(actions) != 1 or not actions[0].success:
        raise RuntimeError("背包滑动未返回唯一成功动作，已保留原数量")
    wait_backpack_still(context, roi, stopped=stopped, minimum_wait_seconds=max(.2, minimum_wait_seconds))
