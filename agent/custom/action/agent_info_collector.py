from __future__ import annotations

import copy
import json
import re
import time
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from zhconv import convert

from utils import logger
from custom.action.operator_catalog_sync import refresh_operator_catalog
from custom.action.inventory_reporting import (
    bound_account_report_filename,
    get_bound_account,
    read_upload_settings,
)
from custom.action.operator_growth_exchange import (
    DEFAULT_GAME,
    _disc_loadouts,
    build_v3_document,
    commit_v3_document,
    discover_v3_schema,
    preview_v3_document,
    read_growth_states,
    scrub_v3_equipment,
    set_operator_catalog,
    stable_scan_id,
    summarize_preview,
    validate_v3_document,
    write_v3_document,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OCR_NODE = "密探信息采集-OCR"
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 1280

# Each page transition is confirmed by a pipeline node whose recognition uses
# a page-specific marker and whose pre_delay is 500 ms.
PAGE_NODES = {
    "main": "密探信息采集-主界面就绪",
    "detail": "密探信息采集-详情界面就绪",
    "huaji": "密探信息采集-化极界面就绪",
    "disc": "密探信息采集-命盘界面就绪",
}
PAGE_PRE_DELAY_MS = 500

MAIN_NAME_ROI = (32, 221, 106, 226)
MAIN_STAT_ROIS = {
    "life": (133, 833, 86, 31),
    "attack": (140, 875, 80, 37),
    "level": (67, 1080, 93, 65),
    "cultivation": (244, 1079, 76, 61),
}
ODDITY_ROWS = (
    {
        "label": "攻击力",
        "label_roi": (120, 818, 150, 45),
        "value_roi": (530, 808, 98, 34),
    },
    {
        "label": "生命值",
        "label_roi": (120, 870, 150, 45),
        "value_roi": (530, 860, 98, 34),
    },
    {
        "label": None,
        "label_roi": (120, 920, 160, 45),
        "value_roi": (530, 910, 98, 34),
    },
)
ODDITY_THIRD_LABELS = ("治疗加成", "增伤值", "免伤值")

HUAJI_STAR_CENTERS = ((254, 862), (306, 862), (359, 862), (412, 862), (465, 862))
HUAJI_SP_STAR_CENTERS_LEFT = (
    (101, 862),
    (154, 862),
    (207, 862),
    (260, 862),
    (313, 862),
)
HUAJI_NODE_CENTERS = (
    (185, 997),
    (274, 958),
    (365, 997),
    (451, 958),
    (545, 997),
)
HUAJI_STATUS_ROI = (40, 900, 640, 150)
HUAJI_ACTION_ROI = (430, 1070, 260, 100)
HUAJI_PENDING_ROI = (150, 800, 420, 110)
HUAJI_MAX_ROI = (180, 900, 420, 180)

# The central command paths are not cells. These are the twelve visible outer cells.
DISC_CELLS = (
    ("r1c1", (14, 203, 172, 170)),
    ("r1c2", (188, 203, 171, 170)),
    ("r1c3", (361, 203, 171, 170)),
    ("r1c4", (535, 203, 171, 170)),
    ("r2c1", (14, 377, 172, 171)),
    ("r2c4", (535, 377, 171, 171)),
    ("r3c1", (14, 551, 172, 171)),
    ("r3c4", (535, 551, 171, 171)),
    ("r4c1", (14, 724, 172, 170)),
    ("r4c2", (188, 724, 171, 170)),
    ("r4c3", (361, 724, 172, 170)),
    ("r4c4", (535, 724, 171, 170)),
)

# Relative to a disc cell. The middle band excludes top decorations and
# the bottom "生效中" badge.
DISC_NAME_ROI = (4, 55, 164, 70)

# Game-specific screen coordinates.  The two dictionaries intentionally stay
# independent so zh_tw layouts can be tuned without changing base.
ROI_CONFIGS = {
    "代号鸢": {
        "clicks": {
            "detail_entry": (71, 762),
            "detail_close": (667, 214),
            "huaji_entry": (451, 1040),
            "huaji_back": (360, 110),
            "disc_entry": (626, 1040),
            "disc_back": (68, 68),
            "disc_switch": (91, 165),
            "next_operator": (679, 638),
        },
        "main_name": MAIN_NAME_ROI,
        "main_stats": MAIN_STAT_ROIS,
        "oddity_rows": ODDITY_ROWS,
        "huaji_stars": HUAJI_STAR_CENTERS,
        "huaji_sp_stars": HUAJI_SP_STAR_CENTERS_LEFT,
        "huaji_nodes": HUAJI_NODE_CENTERS,
        "huaji_status": HUAJI_STATUS_ROI,
        "huaji_action": HUAJI_ACTION_ROI,
        "huaji_pending": HUAJI_PENDING_ROI,
        "huaji_max": HUAJI_MAX_ROI,
        "disc_cells": DISC_CELLS,
    },
    "如鸢": {
        "clicks": {
            "detail_entry": (68, 864),
            "detail_close": (661, 221),
            "huaji_entry": (418, 1086),
            "huaji_back": (360, 110),
            "disc_entry": (577, 1066),
            "disc_back": (68, 68),
            "disc_switch": (91, 165),
            "next_operator": (679, 638),
        },
        "main_name": (32, 221, 106, 226),
        "main_stats": {
            "life": (170, 828, 105, 30),
            "attack": (183, 868, 88, 28),
            "level": (82, 1077, 76, 65),
            "cultivation": (230, 1081, 77, 61),
        },
        "oddity_rows": (
            {
                "label": "攻击力",
                "label_roi": (120, 818, 150, 45),
                "value_roi": (530, 808, 98, 34),
            },
            {
                "label": "生命值",
                "label_roi": (120, 870, 150, 45),
                "value_roi": (530, 860, 98, 34),
            },
            {
                "label": None,
                "label_roi": (120, 920, 160, 45),
                "value_roi": (530, 910, 98, 34),
            },
        ),
        "huaji_stars": ((254, 862), (306, 862), (359, 862), (412, 862), (465, 862)),
        "huaji_sp_stars": ((101, 862), (154, 862), (207, 862), (260, 862), (313, 862)),
        "huaji_nodes": ((185, 997), (274, 958), (365, 997), (451, 958), (545, 997)),
        "huaji_status": (40, 900, 640, 150),
        "huaji_action": (430, 1070, 260, 110),
        "huaji_pending": (150, 800, 420, 110),
        "huaji_max": (180, 900, 420, 180),
        # These are deliberately separate containers.  Edit this block when
        # the 如鸢 layout changes; it must not mutate the 代号鸢 coordinates.
        "disc_cells": tuple((position, tuple(roi)) for position, roi in DISC_CELLS),
    },
}


def _parse_params(raw: Any) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("AgentInfoCollector: custom_action_param 不是有效 JSON")
            return {}
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _normalise(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\n", "").replace("\r", "")
    value = re.sub(r"\s+", "", value)
    value = value.replace("：", ":").replace("／", "/").replace("％", "%")
    try:
        return convert(value, "zh-cn")
    except Exception:
        return value


def _clean_operator_name(text: Any) -> str:
    """Remove the awakened-state badge captured beside the operator name."""
    value = _normalise(text)
    return re.sub(r"(?:已)?觉醒", "", value).strip()


def _operator_name_key(text: Any) -> str:
    """Canonical name used for matching; OCR commonly omits the SP middle dot."""
    return re.sub(r"[·•・]", "", _clean_operator_name(text))


def _disc_ocr_name_matches(raw_name: Any, catalog_name: Any) -> bool:
    """Accept the one-letter suffix OCR sometimes appends to a disc name."""
    raw = _normalise(raw_name)
    catalog = _normalise(catalog_name)
    if not raw or not catalog:
        return False
    return raw == catalog or bool(re.fullmatch(re.escape(catalog) + r"[A-Za-z]", raw))


def _disc_description_key(value: Any) -> str:
    """Fold OCR punctuation differences before comparing unlock descriptions."""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", _normalise(value))


def _has_awakened_badge(text: Any) -> bool:
    return "觉醒" in _normalise(text)


def _is_pending_awaken_layout(text: Any) -> bool:
    return "待觉醒" in _normalise(text)


def _is_max_huaji_layout(text: Any) -> bool:
    return "最高等级" in _normalise(text)


def _box_values(box: Any) -> Optional[tuple[int, int, int, int]]:
    if box is None:
        return None
    try:
        values = tuple(int(v) for v in box)
    except (TypeError, ValueError):
        return None
    return values if len(values) == 4 else None


def _result_items(detail: Any) -> list[dict[str, Any]]:
    if detail is None:
        return []
    results: Iterable[Any] = ()
    for attr in ("filtered_results", "filterd_results", "all_results"):
        candidate = getattr(detail, attr, None)
        if candidate:
            results = candidate
            break
    if not results:
        best = getattr(detail, "best_result", None)
        results = (best,) if best is not None else ()

    items: list[dict[str, Any]] = []
    for result in results:
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            continue
        items.append({"text": text, "box": _box_values(getattr(result, "box", None))})
    return items


def _joined_text(detail: Any) -> str:
    return "".join(item["text"] for item in _result_items(detail))


def _clip_roi(roi: tuple[int, int, int, int], width: int, height: int) -> list[int]:
    x, y, w, h = roi
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return [x, y, w, h]


def _scale_point(point: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    return (
        round(point[0] * width / SCREEN_WIDTH),
        round(point[1] * height / SCREEN_HEIGHT),
    )


def _scale_roi(
    roi: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    sx = width / SCREEN_WIDTH
    sy = height / SCREEN_HEIGHT
    return (round(x * sx), round(y * sy), round(w * sx), round(h * sy))


def _number(text: str, signed: bool = False) -> Optional[int]:
    pattern = r"[+-]?\d[\d,]*" if signed else r"\d[\d,]*"
    match = re.search(pattern, text or "")
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _ratio(text: str) -> tuple[Optional[int], Optional[int]]:
    match = re.search(r"([\d,]+)\s*%?\s*/\s*([\d,]+)", text or "")
    if not match:
        return None, None
    try:
        return (
            int(match.group(1).replace(",", "")),
            int(match.group(2).replace(",", "")),
        )
    except ValueError:
        return None, None


def _remove_disc_state(text: str) -> str:
    value = _normalise(text)
    for token in ("生效中", "可解锁", "锁定", "解锁"):
        value = value.replace(token, "")
    return value.strip(" :：|-")


def _unlock_description(text: str) -> str:
    value = _normalise(text)
    value = re.sub(r"消耗材料可以解锁.*$", "", value)
    return value.strip(" :：|-")


def _is_sp_huaji_layout(
    text: str,
    image: Optional[np.ndarray] = None,
    sp_centers: tuple[tuple[int, int], ...] = HUAJI_SP_STAR_CENTERS_LEFT,
    regular_centers: tuple[tuple[int, int], ...] = HUAJI_STAR_CENTERS,
) -> bool:
    """SP 化极显示当前/下一星级两组星图，不显示普通节点图。"""
    normalised = _normalise(text)
    if len(re.findall(r"\d+级", normalised)) >= 2:
        return True
    if image is None:
        return False
    # OCR may miss the two level labels. SP's leftmost current-star group is
    # still visually distinct from the regular layout's centered first star.
    return (
        _gold_ratio(image, sp_centers[0]) > 0.035
        and _gold_ratio(image, regular_centers[0]) <= 0.035
    )


def _gold_ratio(image: np.ndarray, center: tuple[int, int], radius: int = 22) -> float:
    height, width = image.shape[:2]
    x, y = _scale_point(center, width, height)
    left = max(0, x - radius)
    top = max(0, y - radius)
    right = min(width, x + radius + 1)
    bottom = min(height, y + radius + 1)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    # Screenshot arrays are BGR. Gold active elements are bright and red/yellow-heavy.
    b, g, r = cv2.split(crop)
    mask = (r > 145) & (g > 105) & (b < 155) & ((r.astype(np.int16) - b) > 35)
    return float(np.count_nonzero(mask)) / float(mask.size)


def _huaji_nodes(
    image: np.ndarray,
    centers: tuple[tuple[int, int], ...] = HUAJI_NODE_CENTERS,
) -> list[dict[str, Any]]:
    return [
        {
            "index": index + 1,
            "active": _gold_ratio(image, center, 18) > 0.02,
        }
        for index, center in enumerate(centers)
    ]


def _regular_huaji_advance_state(
    image: np.ndarray,
    star_centers: tuple[tuple[int, int], ...] = HUAJI_SP_STAR_CENTERS_LEFT,
    node_centers: tuple[tuple[int, int], ...] = HUAJI_NODE_CENTERS,
) -> Optional[dict[str, Any]]:
    """Read the left/current star group shown after all five nodes are active."""
    if _gold_ratio(image, star_centers[0]) <= 0.035:
        return None
    stars = sum(_gold_ratio(image, center) > 0.035 for center in star_centers)
    if not 1 <= stars <= 4:
        return None
    return {
        "stars": stars,
        "nodes": [
            {"index": index, "active": True}
            for index in range(1, len(node_centers) + 1)
        ],
    }


class _AgentInfoReader:
    def __init__(self, context: Context, params: dict):
        self.context = context
        self.params = params
        resource = str(params.get("resource", "") or "").lower()
        self.language = (
            "zh-tw" if "zh_tw" in resource or "zh-tw" in resource else "zh-cn"
        )
        self.game = str(
            params.get("game") or ("如鸢" if self.language == "zh-tw" else DEFAULT_GAME)
        )
        self.roi = ROI_CONFIGS.get(self.game, ROI_CONFIGS[DEFAULT_GAME])
        self.wait_ms = max(100, min(5000, int(params.get("wait_ms", 900))))
        self.transition_timeout_ms = max(
            2000,
            min(30000, int(params.get("transition_timeout_ms", 10000))),
        )
        self.poll_interval_ms = max(
            100,
            min(1000, int(params.get("poll_interval_ms", 250))),
        )
        self.current_page: Optional[str] = None
        self.max_operators = max(1, min(300, int(params.get("max_operators", 200))))
        self.operators = self._load_operators()
        self.scan_id = stable_scan_id(params.get("scan_id"))
        self.publish_sequence = 0
        self.scan_started_at = (
            datetime.now().astimezone().isoformat(timespec="milliseconds")
        )

    @property
    def _roi_config(self) -> dict[str, Any]:
        # Keep parsing helpers usable in isolated tests and diagnostics that
        # construct a reader without running __init__.
        return getattr(self, "roi", ROI_CONFIGS[DEFAULT_GAME])

    def _should_stop(self) -> bool:
        try:
            if bool(getattr(self.context, "stop", False)):
                return True
            tasker = getattr(self.context, "tasker", None)
            return bool(
                tasker is not None
                and (
                    getattr(tasker, "stopping", False)
                    or getattr(tasker, "running", None) is False
                )
            )
        except Exception:
            return False

    def _ensure_running(self) -> None:
        if self._should_stop():
            logger.info("AgentInfoCollector: 收到停止请求，终止采集")
            raise InterruptedError("AgentInfoCollector stopped")

    def _sleep_checked(self, milliseconds: int) -> None:
        deadline = time.monotonic() + max(0, milliseconds) / 1000.0
        while time.monotonic() < deadline:
            self._ensure_running()
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    def _load_operators(self) -> dict[str, dict]:
        self._ensure_running()
        data = refresh_operator_catalog(REPO_ROOT / "agent" / "operators.json")
        self._ensure_running()
        set_operator_catalog(data)
        return {
            _operator_name_key(operator["name"]): operator
            for operator in data["OPERATORS"]
            if isinstance(operator, dict) and operator.get("name")
        }

    def screenshot(self) -> Optional[np.ndarray]:
        self._ensure_running()
        try:
            return self.context.tasker.controller.post_screencap().wait().get()
        except Exception:
            logger.exception("AgentInfoCollector: 截图失败")
            return None

    def click(
        self,
        point: tuple[int, int],
        image: Optional[np.ndarray] = None,
        settle_ms: Optional[int] = None,
    ) -> None:
        self._ensure_running()
        if image is None:
            image = self.screenshot()
        height, width = (
            image.shape[:2] if image is not None else (SCREEN_HEIGHT, SCREEN_WIDTH)
        )
        x, y = _scale_point(point, width, height)
        self.context.tasker.controller.post_click(x, y).wait()
        delay = self.wait_ms if settle_ms is None else max(0, settle_ms)
        deadline = time.monotonic() + delay / 1000.0
        while time.monotonic() < deadline:
            self._ensure_running()
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    def _wait_for_page(self, page: str) -> Optional[np.ndarray]:
        self._ensure_running()
        if self.current_page != page:
            node = PAGE_NODES[page]
            if self._run_task(node) is not True:
                diagnostics = ""
                if page == "huaji":
                    image = self.screenshot()
                    if image is not None:
                        action_text = _normalise(
                            self.ocr_text(image, self._roi_config["huaji_action"])
                        )
                        max_text = _normalise(
                            self.ocr_text(image, self._roi_config["huaji_max"])
                        )
                        main_text = _normalise(
                            self.ocr_text(image, (561, 759, 158, 116))
                        )
                        awakened_text = _normalise(
                            self.ocr_text(image, (0, 800, 720, 480))
                        )
                        pending_text = _normalise(
                            self.ocr_text(image, self._roi_config["huaji_pending"])
                        )
                        diagnostics = (
                            f", action_text={action_text!r}, "
                            f"awakened_text={awakened_text!r}, "
                            f"pending_text={pending_text!r}, "
                            f"max_text={max_text!r}, "
                            f"main_text={main_text!r}"
                        )
                logger.error(
                    f"AgentInfoCollector: 等待页面超时 page={page!r}, "
                    f"node={node!r}{diagnostics}"
                )
                return None
            self.current_page = page
            # logger.info(
            #     f"AgentInfoCollector: 页面已就绪 page={page!r}, node={node!r}"
            # )
        return self.screenshot()

    def _require_page(self, page: str) -> np.ndarray:
        image = self._wait_for_page(page)
        if image is None:
            raise RuntimeError(f"等待 {page} 页面超时")
        return image

    def _wait_for_text_change(
        self,
        roi: tuple[int, int, int, int],
        previous: str,
        description: str,
    ) -> Optional[str]:
        previous = _normalise(previous)
        deadline = time.monotonic() + self.transition_timeout_ms / 1000.0
        last_text = previous
        while True:
            self._ensure_running()
            image = self.screenshot()
            if image is not None:
                last_text = _normalise(self.ocr_text(image, roi))
                if last_text and last_text != previous:
                    self._sleep_checked(PAGE_PRE_DELAY_MS)
                    # logger.info(
                    #     f"AgentInfoCollector: {description}已更新 "
                    #     f"from={previous!r}, to={last_text!r}"
                    # )
                    return last_text
            if time.monotonic() >= deadline:
                logger.error(
                    f"AgentInfoCollector: 等待{description}更新超时 "
                    f"previous={previous!r}, last_text={last_text!r}"
                )
                return None
            self._sleep_checked(self.poll_interval_ms)

    def _run_task(self, name: str) -> Optional[bool]:
        self._ensure_running()
        try:
            result = self.context.run_task(name)
        except Exception:
            return None
        self._ensure_running()
        status = getattr(result, "status", None)
        return bool(status and getattr(status, "succeeded", False))

    def _toggle_disc(self, previous_label: str = "") -> bool:
        """Toggle the existing command and distinguish a one-config prompt."""
        task_result = self._run_task("自动编队-尝试切换命盘")
        # logger.info(f"AgentInfoCollector: 执行命盘切换任务 result={task_result}")
        if task_result is not True:
            image = self.screenshot()
            self.click(self._roi_config["clicks"]["disc_switch"], image, settle_ms=0)
            logger.info("AgentInfoCollector: 任务切换未确认，使用坐标点击命盘切换")
            self._require_page("disc")
            if previous_label:
                self._wait_for_text_change(
                    (180, 142, 101, 42), previous_label, "命盘方案"
                )
            return True

        prompt_result = self._run_task("自动编队-关闭命盘提示")
        # logger.info(f"AgentInfoCollector: 命盘单套提示 result={prompt_result}")
        if prompt_result:
            self._require_page("disc")
            return False
        self._require_page("disc")
        if previous_label:
            self._wait_for_text_change((180, 142, 101, 42), previous_label, "命盘方案")
        return True

    def ocr(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> Any:
        self._ensure_running()
        height, width = image.shape[:2]
        scaled = _scale_roi(roi, width, height)
        override = {
            OCR_NODE: {
                "recognition": {
                    "type": "OCR",
                    "param": {
                        "roi": _clip_roi(scaled, width, height),
                        "expected": "",
                    },
                }
            }
        }
        try:
            result = self.context.run_recognition(OCR_NODE, image, override)
            self._ensure_running()
            return result
        except InterruptedError:
            raise
        except Exception:
            logger.exception("AgentInfoCollector: OCR 失败，roi=%s", roi)
            return None

    def ocr_text(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> str:
        return _joined_text(self.ocr(image, roi))

    def _horizontal_name_readings(self, image: np.ndarray) -> list[str]:
        """Retry a vertical name as two/three upright glyphs in a horizontal line."""
        self._ensure_running()
        height, width = image.shape[:2]
        x, y, w, h = _clip_roi(
            _scale_roi(self._roi_config["main_name"], width, height), width, height
        )
        crop = cv2.resize(image[y : y + h, x : x + w], (106, 226))
        # The red awakened badge occupies the right side of this ROI and moves
        # the gold name to the left. The regular layout keeps it on the right.
        lower = crop[150:215]
        blue, green, red = cv2.split(lower)
        red_pixels = (
            (red > 140)
            & (green > 45)
            & (red.astype(np.int16) > green.astype(np.int16) * 1.25)
            & (red.astype(np.int16) > blue.astype(np.int16) * 1.8)
        )
        badge_ratio = np.mean(red_pixels[:, 72:106])
        left_ratio = np.mean(red_pixels[:, :34])
        glyph_left = 0 if badge_ratio > 0.25 and badge_ratio > left_ratio * 2 else 35
        readings = []
        node = "密探信息采集-横排姓名OCR"
        for bounds in ((88, 149, 215), (28, 88, 149, 215)):
            self._ensure_running()
            parts = [
                cv2.resize(crop[top:bottom, glyph_left:glyph_left + 71], (71, 66))
                for top, bottom in zip(bounds, bounds[1:])
            ]
            horizontal = np.concatenate(parts, axis=1)
            override = {
                node: {
                    "recognition": {
                        "type": "OCR",
                        "param": {
                            "roi": [0, 0, horizontal.shape[1], horizontal.shape[0]],
                            "expected": "",
                            "only_rec": True,
                            "replace": [],
                        },
                    }
                }
            }
            try:
                result = self.context.run_recognition(node, horizontal, override)
                self._ensure_running()
                readings.append(_joined_text(result))
            except InterruptedError:
                raise
            except Exception:
                logger.exception("AgentInfoCollector: 横排姓名 OCR 失败")
                # Both hypotheses must finish before accepting a unique candidate.
                return []
        return readings

    def _name_crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        x, y, w, h = _clip_roi(
            _scale_roi(self._roi_config["main_name"], width, height), width, height
        )
        return image[y : y + h, x : x + w]

    def _name_signature(self, image: np.ndarray) -> np.ndarray:
        crop = self._name_crop(image)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    def _catalog_match(self, raw_name: str) -> Optional[str]:
        operator = self.operators.get(_operator_name_key(raw_name))
        return str(operator["name"]) if operator else None

    def run_name_comparison(self) -> bool:
        """Compare baseline OCR with horizontally rearranged name glyphs."""
        self.context.run_task("进入界面-密探")
        self._require_page("main")
        image = self.screenshot()
        if image is None:
            return False

        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = REPO_ROOT / f"AgentNameOcrComparison-{stamp}"
        image_dir = output_dir / "names"
        image_dir.mkdir(parents=True)
        report_path = output_dir / "report.json"
        rows: list[dict[str, Any]] = []
        first_signature: Optional[np.ndarray] = None
        previous_signature: Optional[np.ndarray] = None

        for index in range(self.max_operators):
            self._ensure_running()
            signature = self._name_signature(image)
            if first_signature is not None and np.mean(cv2.absdiff(signature, first_signature)) < 2:
                logger.info("密探姓名 OCR 对比：已绕行一圈")
                break
            if first_signature is None:
                first_signature = signature

            maa_raw = self.ocr_text(image, self._roi_config["main_name"])
            fallback_readings = self._horizontal_name_readings(image)
            filename = f"{index + 1:03d}.png"
            encoded_ok, encoded = cv2.imencode(".png", self._name_crop(image))
            if not encoded_ok:
                raise OSError(f"无法保存密探姓名裁图: {filename}")
            (image_dir / filename).write_bytes(encoded.tobytes())
            maa_match = self._catalog_match(maa_raw)
            fallback_candidates = sorted({
                match
                for reading in fallback_readings
                if (match := self._catalog_match(reading))
            })
            fallback_resolved = fallback_candidates[0] if len(fallback_candidates) == 1 else None
            rows.append(
                {
                    "index": index + 1,
                    "image": f"names/{filename}",
                    "maa": {"raw": maa_raw, "catalog_match": maa_match},
                    "horizontal_fallback": {
                        "readings": fallback_readings,
                        "catalog_candidates": fallback_candidates,
                        "resolved": fallback_resolved,
                    },
                    "same_text": bool(maa_raw and fallback_resolved)
                    and _operator_name_key(maa_raw) == _operator_name_key(fallback_resolved),
                }
            )
            count = len(rows)
            summary = {
                "samples": count,
                "maa_catalog_matches": sum(bool(row["maa"]["catalog_match"]) for row in rows),
                "horizontal_catalog_matches": sum(
                    bool(row["horizontal_fallback"]["resolved"]) for row in rows
                ),
                "different_text": sum(not row["same_text"] for row in rows),
            }
            summary["maa_catalog_match_rate"] = round(
                summary["maa_catalog_matches"] / count, 3
            )
            summary["horizontal_catalog_match_rate"] = round(
                summary["horizontal_catalog_matches"] / count, 3
            )
            report = {
                "note": "名册精确匹配率仅供快速筛查；真实正确率请对照 names 裁图核对。",
                "summary": summary,
                "records": rows,
            }
            pending = report_path.with_name("report.json.tmp")
            pending.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            pending.replace(report_path)
            logger.info(
                f"密探姓名 OCR 对比 {count}: 内置={maa_raw!r} ({maa_match or '未匹配'}), "
                f"横排={fallback_readings!r} ({fallback_resolved or '未匹配'})"
            )

            if self.params.get("scan_single", False):
                break
            previous_signature = signature
            self.click(self._roi_config["clicks"]["next_operator"], image)
            image = self.screenshot()
            if image is None:
                break
            if np.mean(cv2.absdiff(self._name_signature(image), previous_signature)) < 2:
                logger.warning("密探姓名 OCR 对比：切换密探后姓名区域未变化，停止扫描")
                break

        logger.info(f"密探姓名 OCR 对比报告：{report_path}")
        return bool(rows)

    def run_disc_comparison(self) -> bool:
        """Compare the two OCR inputs through the actual disc-loadout pipeline."""
        self.context.run_task("进入界面-密探")
        self._require_page("main")
        image = self.screenshot()
        if image is None:
            return False

        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = REPO_ROOT / f"AgentDiscOcrComparison-{stamp}"
        screen_dir = output_dir / "screens"
        cell_dir = output_dir / "cells"
        screen_dir.mkdir(parents=True)
        cell_dir.mkdir()
        report_path = output_dir / "report.json"
        rows: list[dict[str, Any]] = []
        first_signature: Optional[np.ndarray] = None

        for index in range(self.max_operators):
            self._ensure_running()
            signature = self._name_signature(image)
            if first_signature is not None and np.mean(cv2.absdiff(signature, first_signature)) < 2:
                logger.info("命盘 OCR 对比：已绕行一圈")
                break
            if first_signature is None:
                first_signature = signature

            main = self._read_main(image)
            comparison = {
                "screen_dir": screen_dir,
                "cell_dir": cell_dir,
                "operator_index": index + 1,
                "captures": [],
            }
            original_configs = self._collect_discs(main, comparison=comparison)
            enhanced_configs = comparison["enhanced_configs"]
            original = self._disc_comparison_loadouts(main, original_configs)
            enhanced = self._disc_comparison_loadouts(main, enhanced_configs)
            rows.append({
                "index": index + 1,
                "name_raw": main["name_raw"],
                "configs": comparison["captures"],
                "navigation": comparison["navigation"],
                "original": original,
                "enhanced": enhanced,
                "same_loadouts": original["disc_loadouts"] == enhanced["disc_loadouts"],
            })
            all_cells = [cell for row in rows for config in row["configs"] for cell in config["cells"]]
            summary = {
                "samples": len(rows),
                "cells": len(all_cells),
                "different_text": sum(not cell["same_text"] for cell in all_cells),
                "different_loadouts": sum(not row["same_loadouts"] for row in rows),
            }
            report = {
                "note": "两套命盘均按采集流程识别，original 是正式采集结果；请对照截图核实两种 OCR 的差异。此任务不上传。",
                "enhancement": "LAB 亮度 CLAHE(clipLimit=2.0, tileGridSize=4x4) + 2x cubic resize",
                "summary": summary,
                "records": rows,
            }
            pending = report_path.with_name("report.json.tmp")
            pending.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            pending.replace(report_path)
            logger.info(
                f"命盘 OCR 对比 {len(rows)}: name={main['name_raw']!r}, "
                f"original={original['disc_loadouts']!r}, enhanced={enhanced['disc_loadouts']!r}"
            )
            if self.params.get("scan_single", False):
                break
            image = self.screenshot()
            if image is None:
                break
            self.click(self._roi_config["clicks"]["next_operator"], image)
            image = self.screenshot()
            if image is None:
                break
            if np.mean(cv2.absdiff(self._name_signature(image), signature)) < 2:
                logger.warning("命盘 OCR 对比：切换密探后姓名区域未变化，停止扫描")
                break

        logger.info(f"命盘 OCR 对比报告：{report_path}")
        return bool(rows)

    def _disc_comparison_loadouts(self, main: dict, configs: list[dict]) -> dict:
        configs = copy.deepcopy(configs)
        operator_id = main.get("operator_id")
        operator = self._confirm_operator_from_discs(
            main, configs, log_failure=not bool(operator_id)
        )
        if operator and operator.get("id") != operator_id:
            operator_id = operator["id"]
            self._resolve_locked_disc_names(configs, operator)
        status, loadouts, diagnostics = _disc_loadouts({
            "operator_id": operator_id, "disc_configs": configs
        })
        return {
            "operator_id": operator_id,
            "disc_configs": configs,
            "section_status": status,
            "disc_loadouts": loadouts,
            "diagnostics": diagnostics,
        }

    def _operator_for_name(self, raw_name: str) -> Optional[dict]:
        normalised = _operator_name_key(raw_name)
        if normalised in self.operators:
            return self.operators[normalised]
        candidates = [
            operator
            for key, operator in self.operators.items()
            if key
            and min(len(key), len(normalised)) >= 2
            and (key in normalised or normalised in key)
        ]
        candidate_ids = {operator.get("id") for operator in candidates}
        if len(candidate_ids) == 1:
            return candidates[0]
        return None

    def _fuzzy_operator_candidates(self, raw_name: str) -> list[dict[str, Any]]:
        normalised = _operator_name_key(raw_name)
        candidates = []
        for key, operator in self.operators.items():
            score = SequenceMatcher(None, normalised, key).ratio()
            if score >= 0.5:
                candidates.append(
                    {
                        "operator_id": operator.get("id"),
                        "name": operator.get("name"),
                        "score": score,
                    }
                )
        candidates.sort(key=lambda item: (-item["score"], str(item["name"])))
        return candidates

    def _operator_by_id(self, operator_id: Any) -> Optional[dict]:
        if not operator_id:
            return None
        return next(
            (
                operator
                for operator in self.operators.values()
                if operator.get("id") == operator_id
            ),
            None,
        )

    def _log_unconfirmed_name(
        self,
        raw_name: str,
        candidates: list[dict[str, Any]],
    ) -> None:
        if candidates:
            logger.warning(
                f"AgentInfoCollector: 密探名称 OCR 仅能模糊匹配 {raw_name!r}，"
                "等待命盘专属名称二次确认 "
                f"candidates={[(item['name'], round(item['score'], 3)) for item in candidates]}"
            )
        else:
            logger.warning(
                f"AgentInfoCollector: 密探名称无法匹配 operators.json: {raw_name!r}"
            )

    def _read_main(self, image: np.ndarray) -> dict:
        raw_name = self.ocr_text(image, self._roi_config["main_name"])
        cleaned_name = _clean_operator_name(raw_name)
        operator = self.operators.get(_operator_name_key(raw_name))
        fallback_readings = []
        if operator is None:
            fallback_readings = self._horizontal_name_readings(image)
            candidates = {
                _operator_name_key(text)
                for text in fallback_readings
                if _operator_name_key(text) in self.operators
            }
            if len(candidates) == 1:
                operator = self.operators[next(iter(candidates))]
            elif len(candidates) > 1:
                logger.warning(
                    "AgentInfoCollector: 横排姓名候选冲突，等待命盘确认 "
                    f"name={raw_name!r}, readings={fallback_readings!r}"
                )
        fuzzy_candidates = [] if operator else self._fuzzy_operator_candidates(raw_name)
        if not operator:
            self._log_unconfirmed_name(raw_name, fuzzy_candidates)
        name = str(operator.get("name")) if operator else cleaned_name
        stats: dict[str, Optional[int]] = {}
        for key, roi in self._roi_config["main_stats"].items():
            value = _number(self.ocr_text(image, roi), signed=key == "cultivation")
            stats[key] = value
        return {
            "operator_id": operator.get("id") if operator else None,
            "name": name,
            "name_raw": raw_name,
            "name_cleaned": cleaned_name,
            "operator_lookup": bool(operator),
            "_operator_match": "name" if operator else "unconfirmed",
            "_name_exact": bool(operator),
            "_name_ocr_fallback": fallback_readings,
            "_operator_candidates": fuzzy_candidates,
            "stats": stats,
        }

    def _read_oddities(self, image: np.ndarray) -> dict[str, dict[str, Any]]:
        oddities: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(self._roi_config["oddity_rows"]):
            value_raw = self.ocr_text(image, row["value_roi"])
            current, maximum = _ratio(value_raw)
            if current is None or maximum is None:
                continue
            label = row["label"]
            if label is None:
                label_raw = _normalise(self.ocr_text(image, row["label_roi"]))
                label = next(
                    (option for option in ODDITY_THIRD_LABELS if option in label_raw),
                    None,
                )
            label = label or f"field_{index + 1}"
            oddities[label] = {"current": current, "max": maximum}
        return oddities

    def _collect_details(self, main: dict) -> dict:
        image = self._require_page("main")
        self.click(self._roi_config["clicks"]["detail_entry"], image, settle_ms=0)
        image = self._require_page("detail")
        oddities = self._read_oddities(image)
        self.click(self._roi_config["clicks"]["detail_close"], image, settle_ms=0)
        self._require_page("main")
        return oddities

    def _collect_huaji(self, main: dict) -> dict:
        # The awakened badge is shown beside the operator name on the main
        # page. It is authoritative, so an awakened operator has no need to
        # open the huaji page (which has a different layout).
        if _has_awakened_badge(main.get("name_raw")):
            # logger.info(
            #     "AgentInfoCollector: 主界面名称已标记觉醒，跳过化极页面识别 "
            #     f"name={main.get('name_raw')!r}"
            # )
            return {
                "layout": "awakened",
                "stars": 5,
                "nodes": [],
                "awakened": True,
            }
        image = self._require_page("main")
        self.click(self._roi_config["clicks"]["huaji_entry"], image, settle_ms=0)
        image = self._require_page("huaji")

        status_detail = self.ocr(image, self._roi_config["huaji_status"])
        status_text = _joined_text(status_detail)
        action_text = _normalise(self.ocr_text(image, self._roi_config["huaji_action"]))
        pending_text = _normalise(
            self.ocr_text(image, self._roi_config["huaji_pending"])
        )
        max_text = _normalise(self.ocr_text(image, self._roi_config["huaji_max"]))
        awakened = _has_awakened_badge(main.get("name_raw")) or "觉醒" in action_text
        if _is_max_huaji_layout(max_text):
            self.click(self._roi_config["clicks"]["huaji_back"], image, settle_ms=0)
            self._require_page("main")
            return {
                "layout": "awakened",
                "stars": 5,
                "nodes": [],
                "awakened": True,
            }
        if _is_pending_awaken_layout(pending_text):
            self.click(self._roi_config["clicks"]["huaji_back"], image, settle_ms=0)
            self._require_page("main")
            return {
                "layout": "pending_awaken",
                "stars": 5,
                "nodes": [],
                "awakened": False,
            }
        operator_id = str(main.get("operator_id") or "").lower()
        is_sp = (
            operator_id.endswith("sp")
            if operator_id
            else _is_sp_huaji_layout(
                status_text,
                image,
                self._roi_config["huaji_sp_stars"],
                self._roi_config["huaji_stars"],
            )
        )
        if is_sp:
            stars = sum(
                _gold_ratio(image, center) > 0.035
                for center in self._roi_config["huaji_sp_stars"]
            )
            self.click(self._roi_config["clicks"]["huaji_back"], image, settle_ms=0)
            self._require_page("main")
            return {
                "layout": "sp",
                "stars": stars,
                "nodes": [],
                "awakened": awakened,
            }

        advance_state = _regular_huaji_advance_state(
            image,
            self._roi_config["huaji_sp_stars"],
            self._roi_config["huaji_nodes"],
        )
        if advance_state is not None:
            self.click(self._roi_config["clicks"]["huaji_back"], image, settle_ms=0)
            self._require_page("main")
            return {
                "layout": "regular",
                **advance_state,
                "awakened": awakened,
            }

        stars = sum(
            _gold_ratio(image, center) > 0.035
            for center in self._roi_config["huaji_stars"]
        )
        nodes = _huaji_nodes(image, self._roi_config["huaji_nodes"])

        self.click(self._roi_config["clicks"]["huaji_back"], image, settle_ms=0)
        self._require_page("main")
        return {
            "layout": "regular",
            "stars": stars,
            "nodes": nodes,
            "awakened": awakened,
        }

    def _lookup_disc(self, operator: Optional[dict], description: str) -> Optional[str]:
        if not operator:
            return None
        needle = _normalise(description)
        if not needle:
            return None
        needle_key = _disc_description_key(needle)
        for disc in operator.get("discs", []):
            candidate = _normalise(disc.get("desp", ""))
            candidate_key = _disc_description_key(candidate)
            if candidate_key and (
                candidate_key == needle_key
                or candidate_key in needle_key
                or needle_key in candidate_key
            ):
                return disc.get("ot_name")
        return None

    def _unique_operator_for_disc_name(self, name: Any) -> Optional[dict]:
        """Return the sole operator owning an exact disc name."""
        needle = _normalise(name)
        if not needle:
            return None
        owners = {
            str(operator.get("id")): operator
            for operator in self.operators.values()
            if operator.get("id")
            and any(
                _disc_ocr_name_matches(name, disc.get("ot_name"))
                for disc in operator.get("discs", [])
            )
        }
        return next(iter(owners.values())) if len(owners) == 1 else None

    def _known_disc_name(self, name: Any) -> Optional[str]:
        """Resolve an OCR cell name to an exact name present in operators.json."""
        needle = _normalise(name)
        if not needle:
            return None
        matches = []
        for operator in self.operators.values():
            for disc in operator.get("discs", []):
                if _disc_ocr_name_matches(name, disc.get("ot_name")):
                    matches.append(str(disc.get("ot_name")))
        unique = set(matches)
        return next(iter(unique)) if len(unique) == 1 else None

    def _confirm_operator_from_discs(
        self,
        main: dict,
        configs: list[dict],
        *,
        log_failure: bool = True,
    ) -> Optional[dict]:
        candidate_ids = {
            str(item.get("operator_id"))
            for item in main.get("_operator_candidates", [])
            if item.get("operator_id")
        }
        evidence: dict[str, set[str]] = {}
        for config in configs:
            for slot in config.get("slots", []):
                if slot.get("state") not in {"active", "inactive", "locked"}:
                    continue
                source = slot.get("name")
                operator = self._unique_operator_for_disc_name(source)
                operator_id = str(operator.get("id")) if operator else ""
                if operator_id:
                    evidence.setdefault(operator_id, set()).add(str(source))

        if len(evidence) != 1:
            if not log_failure:
                return None
            logger.warning(
                "AgentInfoCollector: 命盘未能唯一确认模糊名称 "
                f"name={main.get('name_raw')!r}, evidence={evidence!r}"
            )
            return None
        operator_id, sources = next(iter(evidence.items()))
        if operator_id not in candidate_ids and len(sources) < 2:
            if not log_failure:
                return None
            logger.warning(
                "AgentInfoCollector: 名称无匹配，仅有一条专属命盘证据，"
                "暂不确认身份 "
                f"name={main.get('name_raw')!r}, evidence={evidence!r}"
            )
            return None

        operator = self._operator_by_id(operator_id)
        if operator:
            logger.info(
                "AgentInfoCollector: 命盘专属名称确认密探 "
                f"{main.get('name_raw')!r} -> {operator.get('name')!r}, "
                f"evidence={sorted(sources)!r}"
            )
        return operator

    def _resolve_locked_disc_names(
        self,
        configs: list[dict],
        operator: Optional[dict],
    ) -> None:
        if not operator:
            return
        for config in configs:
            for slot in config.get("slots", []):
                if slot.get("state") == "locked" or slot.get("locked") is True:
                    slot["name"] = self._lookup_disc(
                        operator,
                        slot.get("unlock_description", ""),
                    )
            config["signature"] = tuple(
                sorted(
                    (slot["position"], slot["state"], slot.get("name") or "")
                    for slot in config.get("slots", [])
                )
            )

    def _disc_slot_from_text(
        self, position: str, text: str
    ) -> Optional[dict[str, Any]]:
        normalised = _normalise(text)
        is_locked = "可解锁" in normalised or "未解锁" in normalised
        if "生效中" in normalised or is_locked:
            return {
                "position": position,
                "state": "active",
                **({"locked": True} if is_locked else {}),
                "name": None if is_locked else _remove_disc_state(text),
            }
        name = self._known_disc_name(_remove_disc_state(text))
        return {"position": position, "state": "inactive", "name": name} if name else None

    def _disc_name_roi(self, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        left, top, name_width, name_height = DISC_NAME_ROI
        x, y, width, height = roi
        return (
            x + left, y + top,
            min(name_width, width - left), min(name_height, height - top),
        )

    def _scan_disc_config(self, operator: Optional[dict], comparison: Optional[dict] = None) -> dict:
        image = self.screenshot()
        if image is None:
            return {"available": False, "slots": []}
        label = self.ocr_text(image, (180, 142, 101, 42))
        slots = []
        enhanced_slots = []
        capture = None
        if comparison is not None:
            config_index = len(comparison["captures"]) + 1
            prefix = f"{comparison['operator_index']:03d}-{config_index}"
            screen_path = comparison["screen_dir"] / f"{prefix}.png"
            if not cv2.imwrite(str(screen_path), image):
                raise OSError(f"无法保存命盘截图: {screen_path}")
            capture = {"index": config_index, "screen": f"screens/{screen_path.name}", "cells": []}
            comparison["captures"].append(capture)
        for position, roi in self._roi_config["disc_cells"]:
            cell_text = self.ocr_text(image, roi)
            slot = self._disc_slot_from_text(position, cell_text)
            name_raw = None
            if slot and slot["state"] == "active" and not slot.get("locked"):
                name_raw = self.ocr_text(image, self._disc_name_roi(roi))
                if name_raw:
                    slot["name"] = _remove_disc_state(name_raw)
            elif slot is None and cell_text.strip():
                name_raw = self.ocr_text(image, self._disc_name_roi(roi))
                slot = self._disc_slot_from_text(position, name_raw)
            enhanced_slot = None
            if capture is not None:
                height, width = image.shape[:2]
                x, y, w, h = _clip_roi(_scale_roi(roi, width, height), width, height)
                crop = image[y : y + h, x : x + w]
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
                enhanced = cv2.resize(
                    cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), None,
                    fx=2, fy=2, interpolation=cv2.INTER_CUBIC,
                )
                original_path = comparison["cell_dir"] / f"{prefix}-{position}.png"
                enhanced_path = comparison["cell_dir"] / f"{prefix}-{position}-enhanced.png"
                if not cv2.imwrite(str(original_path), crop) or not cv2.imwrite(str(enhanced_path), enhanced):
                    raise OSError(f"无法保存命盘格子裁图: {position}")
                enhanced_text = self.ocr_text(enhanced, (0, 0, 720, 1280))
                enhanced_slot = self._disc_slot_from_text(position, enhanced_text)
                enhanced_name_raw = None
                if (enhanced_slot and enhanced_slot["state"] == "active"
                        and not enhanced_slot.get("locked")) or (
                            enhanced_slot is None and enhanced_text.strip()
                        ):
                    left, top, name_width, name_height = DISC_NAME_ROI
                    name_image = enhanced[
                        top * 2 : (top + min(name_height, roi[3] - top)) * 2,
                        left * 2 : (left + min(name_width, roi[2] - left)) * 2,
                    ]
                    enhanced_name_raw = self.ocr_text(name_image, (0, 0, 720, 1280))
                    if enhanced_slot and enhanced_slot["state"] == "active":
                        if enhanced_name_raw:
                            enhanced_slot["name"] = _remove_disc_state(enhanced_name_raw)
                    elif enhanced_name_raw:
                        enhanced_slot = self._disc_slot_from_text(position, enhanced_name_raw)
                capture["cells"].append({
                    "position": position, "roi": list(roi), "pixel_roi": [x, y, w, h],
                    "image": f"cells/{original_path.name}",
                    "enhanced_image": f"cells/{enhanced_path.name}",
                    "original": {"raw": cell_text, "name_raw": name_raw, "slot": slot},
                    "enhanced": {"raw": enhanced_text, "name_raw": enhanced_name_raw, "slot": enhanced_slot},
                    "same_text": _normalise(cell_text) == _normalise(enhanced_text),
                })
            if (slot and slot.get("locked")) or (enhanced_slot and enhanced_slot.get("locked")):
                self.click(
                    (roi[0] + roi[2] // 2, roi[1] + roi[3] // 2),
                    image,
                    settle_ms=220,
                )
                detail_image = self.screenshot()
                detail_text = (
                    self.ocr_text(detail_image, (100, 900, 600, 180))
                    if detail_image is not None
                    else ""
                )
                match = re.search(r"解锁即获得[:：]?(.*)", _normalise(detail_text))
                description = _unlock_description(
                    match.group(1) if match else detail_text
                )
                for candidate in (slot, enhanced_slot):
                    if candidate and candidate.get("locked"):
                        candidate["unlock_description"] = description
                        candidate["name"] = self._lookup_disc(operator, description)
                if capture is not None:
                    capture["cells"][-1]["detail_raw"] = detail_text
            if slot:
                slots.append(slot)
            if enhanced_slot:
                enhanced_slots.append(enhanced_slot)

        active_names = tuple(
            sorted(
                (slot["position"], slot["state"], slot.get("name") or "")
                for slot in slots
            )
        )
        result = {
            "available": True,
            "label": _normalise(label),
            "slots": slots,
            "signature": active_names,
        }
        if capture is not None:
            capture["label"] = result["label"]
            capture["enhanced_config"] = {
                "available": True,
                "label": result["label"],
                "slots": enhanced_slots,
                "signature": tuple(sorted(
                    (item["position"], item["state"], item.get("name") or "")
                    for item in enhanced_slots
                )),
            }
        # logger.info(
        #     f"AgentInfoCollector: 命盘扫描 label={result['label']!r}, "
        #     f"slots={[(s['position'], s['state'], s.get('name')) for s in slots]}"
        # )
        return result

    def _collect_discs(self, main: dict, comparison: Optional[dict] = None) -> list[dict]:
        image = self._require_page("main")
        self.click(self._roi_config["clicks"]["disc_entry"], image, settle_ms=0)
        self._require_page("disc")
        operator = self._operator_by_id(main.get("operator_id"))
        first = self._scan_disc_config(operator, comparison=comparison)
        configs = [dict(first, index=1)]

        first_signature = first.get("signature", ())
        first_label = first.get("label", "")
        switched = self._toggle_disc(first_label)
        # logger.info(
        #     f"AgentInfoCollector: 命盘切换尝试 from={first_label!r}, switched={switched}"
        # )
        second = self._scan_disc_config(operator, comparison=comparison) if switched else {}
        second_signature = second.get("signature", ())
        second_label = second.get("label", "")
        changed = bool(
            second_signature
            and (
                second_signature != first_signature
                or (first_label and second_label and first_label != second_label)
            )
        )
        if changed:
            configs.append(dict(second, index=2))
        else:
            configs.append(
                {
                    "index": 2,
                    "available": False,
                    "reason": "切换后命盘内容未发生变化",
                    "slots": [],
                }
            )

        if comparison is not None:
            first_enhanced = comparison["captures"][0].pop("enhanced_config")
            enhanced_configs = [dict(first_enhanced, index=1)]
            if switched:
                second_enhanced = comparison["captures"][1].pop("enhanced_config")
                enhanced_changed = bool(
                    second_enhanced["signature"]
                    and (second_enhanced["signature"] != first_enhanced["signature"]
                         or (first_label and second_label and first_label != second_label))
                )
                enhanced_configs.append(
                    dict(second_enhanced, index=2) if enhanced_changed
                    else {"index": 2, "available": False, "reason": "切换后命盘内容未发生变化", "slots": []}
                )
            else:
                enhanced_configs.append({"index": 2, "available": False, "reason": "未切换命盘", "slots": []})
            comparison["enhanced_configs"] = enhanced_configs

        # The collector is read-only: always restore the configuration present on entry,
        # including the one-configuration/no-change case.
        restored_label = first_label
        restored_ok = not switched
        if switched:
            if not self._toggle_disc(second_label):
                logger.warning("AgentInfoCollector: 命盘原配置恢复切换未执行")
            restored_image = self._require_page("disc")
            restored_label = _normalise(
                self.ocr_text(restored_image, (180, 142, 101, 42))
            )
            restored_ok = restored_label == first_label
            if not restored_ok:
                logger.warning("AgentInfoCollector: 命盘原配置恢复结果未确认")

        image = self._require_page("disc")
        self.click(self._roi_config["clicks"]["disc_back"], image, settle_ms=0)
        self._require_page("main")
        if comparison is not None:
            comparison["navigation"] = {
                "switched": switched,
                "restored": restored_ok,
                "restored_label": restored_label,
            }
        logger.info(
            f"AgentInfoCollector: 命盘采集完成 first={first_label!r}, "
            f"second={second_label!r}, changed={changed}, "
            f"restored={restored_ok}, restored_label={restored_label!r}"
        )
        return configs

    def collect_current(self) -> Optional[dict]:
        image = self.screenshot()
        if image is None:
            return None
        return self.collect_current_from_main(self._read_main(image))

    def collect_current_from_main(self, main: dict) -> Optional[dict]:
        if not main.get("name"):
            return None
        record = dict(main)
        record["oddities"] = self._collect_details(main)
        cached_discs = record.pop("_prefetched_discs", None)
        record["disc_configs"] = cached_discs if cached_discs is not None else self._collect_discs(main)
        name_match_operator_id = record.pop("_original_name_match_operator_id", record.get("operator_id"))
        operator = self._confirm_operator_from_discs(
            main,
            record["disc_configs"],
            log_failure=not bool(name_match_operator_id),
        )
        if operator and operator.get("id") != name_match_operator_id:
            if name_match_operator_id:
                logger.warning(
                    "AgentInfoCollector: 命盘专属名称推翻主界面名称匹配 "
                    f"name={main.get('name_raw')!r}, "
                    f"from={name_match_operator_id!r}, to={operator.get('id')!r}"
                )
            record["operator_id"] = operator.get("id")
            record["name"] = str(operator.get("name"))
            record["operator_lookup"] = True
            record["_operator_match"] = "disc"
            self._resolve_locked_disc_names(record["disc_configs"], operator)
        record["huaji"] = self._collect_huaji(record)
        record["collection_debug"] = {
            "name_raw": main.get("name_raw"),
            "name_cleaned": main.get("name_cleaned"),
            "name_ocr_fallback": main.get("_name_ocr_fallback", []),
            "operator_lookup": record.get("operator_lookup", False),
            "operator_match": record.get("_operator_match"),
            "name_match_operator_id": (
                name_match_operator_id
                if name_match_operator_id != record.get("operator_id")
                else None
            ),
            "operator_candidates": main.get("_operator_candidates", []),
            "disc_labels": [config.get("label") for config in record["disc_configs"]],
        }
        record.pop("_operator_match", None)
        record.pop("_operator_candidates", None)
        record.pop("_name_exact", None)
        record.pop("_name_ocr_fallback", None)
        logger.info(
            f"AgentInfoCollector: 当前密探采集完成 name={record.get('name')!r}, "
            f"operator_id={record.get('operator_id')!r}"
        )
        return record

    def _output_path(self) -> Path:
        output = Path(str(self.params.get("output") or "AgentInfoReport.json"))
        return output if output.is_absolute() else REPO_ROOT / output

    def _record_key(self, record: dict) -> str:
        return str(
            record.get("operator_id")
            or _clean_operator_name(
                record.get("name_cleaned") or record.get("name") or ""
            )
        )

    def _load_records(self) -> list[dict]:
        output = self._output_path()
        try:
            with output.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError):
            logger.warning("AgentInfoCollector: 无法读取已有 v3 报告: %s", output)
            return []

        if (
            not isinstance(document, dict)
            or document.get("format") != "myshare-operator-exchange"
            or document.get("version") != 3
        ):
            logger.warning("AgentInfoCollector: 已有报告不是 v3 交换文档: %s", output)
            return []
        records = document.get("records")
        if not isinstance(records, list):
            logger.warning(
                "AgentInfoCollector: 已有 v3 报告缺少 records 数组: %s", output
            )
            return []
        scrub_v3_equipment(document)
        valid = [record for record in records if self._exchange_record_key(record)]
        logger.info(
            f"AgentInfoCollector: 已加载 v3 断点报告 records={len(valid)}, output={output}"
        )
        return valid

    @staticmethod
    def _exchange_record_key(record: Any) -> str:
        if (
            not isinstance(record, dict)
            or record.get("record_type") != "operator_snapshot"
        ):
            return ""
        entries = record.get("entries")
        unmatched = record.get("unmatched")
        if (
            isinstance(entries, list)
            and len(entries) == 1
            and isinstance(entries[0], dict)
        ):
            operator_id = str(entries[0].get("operator_id") or "")
            return f"id:{operator_id}" if operator_id else ""
        if (
            isinstance(unmatched, list)
            and len(unmatched) == 1
            and isinstance(unmatched[0], dict)
        ):
            raw_name = _operator_name_key(unmatched[0].get("raw_name"))
            return f"name:{raw_name}" if raw_name else ""
        return ""

    @staticmethod
    def _exchange_record_match_evidence(record: dict) -> tuple[str, str]:
        entries = record.get("entries")
        if (
            not isinstance(entries, list)
            or len(entries) != 1
            or not isinstance(entries[0], dict)
        ):
            return "", ""
        entry = entries[0]
        debug = entry.get("diagnostics", {}).get("collection_debug", {})
        if not isinstance(debug, dict):
            return "", ""
        return (
            str(debug.get("name_match_operator_id") or ""),
            _normalise(debug.get("name_raw")),
        )

    def _upsert_exchange_record(self, records: list[dict], record: dict) -> bool:
        key = self._exchange_record_key(record)
        replaced_operator_id, raw_name = self._exchange_record_match_evidence(record)
        for index, existing in enumerate(records):
            if self._exchange_record_key(existing) == key:
                records[index] = record
                return True
            existing_entries = (
                existing.get("entries") if isinstance(existing, dict) else None
            )
            existing_entry = (
                existing_entries[0]
                if isinstance(existing_entries, list)
                and len(existing_entries) == 1
                and isinstance(existing_entries[0], dict)
                else {}
            )
            existing_debug = existing_entry.get("diagnostics", {}).get(
                "collection_debug", {}
            )
            if (
                replaced_operator_id
                and existing_entry.get("operator_id") == replaced_operator_id
                and raw_name
                and isinstance(existing_debug, dict)
                and _normalise(existing_debug.get("name_raw")) == raw_name
            ):
                records[index] = record
                return True
        records.append(record)
        return False

    def run(self) -> bool:
        try:
            self.context.run_task("进入界面-密探")
            self._require_page("main")
        except InterruptedError:
            raise
        except Exception:
            logger.exception("AgentInfoCollector: 进入密探界面失败")
            return False

        resume = self.params.get("resume", True)
        resume_enabled = not (
            resume is False or str(resume).strip().lower() in {"0", "false", "no"}
        )
        exchange_records = self._load_records() if resume_enabled else []
        origin_key: Optional[str] = None
        for _ in range(self.max_operators):
            logger.info(f"AgentInfoCollector: 开始读取第 {_ + 1} 位密探")
            image = self.screenshot()
            if image is None:
                break
            main = self._read_main(image)
            if not main.get("name"):
                logger.error("AgentInfoCollector: 无法读取当前密探名称，停止遍历")
                break
            key = self._record_key(main)
            if origin_key is None:
                origin_key = key
            if key == origin_key and _ > 0:
                logger.info(f"AgentInfoCollector: 已绕行一圈，断点采集完成 key={key!r}")
                break

            prepared = self._prepare_growth_filter(main)
            if self._should_collect(prepared):
                record = self.collect_current_from_main(prepared)
                if not record:
                    logger.error("AgentInfoCollector: 无法采集当前密探，停止遍历")
                    return False
                # Disc evidence may correct the identity read on the main page.
                if self._should_collect(record):
                    replaced = self._publish_checkpoint(exchange_records, record)
                    logger.info(
                        f"AgentInfoCollector: {'已更新' if replaced else '已新增'} "
                        f"name={record.get('name')!r}, "
                        f"operator_id={record.get('operator_id')!r}, count={len(exchange_records)}"
                    )

            if self.params.get("scan_single", False):
                break

            previous_name = str(main.get("name_raw") or main.get("name") or "")
            self.click(self._roi_config["clicks"]["next_operator"], image, settle_ms=0)
            if (
                self._wait_for_text_change(
                    self._roi_config["main_name"], previous_name, "下一位密探"
                )
                is None
            ):
                break

        if not exchange_records:
            return bool(self.params.get("active_only") and origin_key is not None)
        logger.info(
            f"AgentInfoCollector: 采集结束，v3 报告共 {len(exchange_records)} 位密探"
        )
        return True

    def _prepare_growth_filter(self, main: dict) -> dict:
        if not self.params.get("active_only") or main.get("_name_exact"):
            return main
        logger.info("AgentInfoCollector: 首屏身份不确定，先读取命盘确认养成筛选对象")
        prepared = dict(main)
        # Do not use a partial name match to resolve locked discs: that would
        # turn an unconfirmed guess into identity evidence.
        probe = dict(main, operator_id=None)
        configs = self._collect_discs(probe)
        operator = self._confirm_operator_from_discs(main, configs)
        prepared["_original_name_match_operator_id"] = main.get("operator_id")
        prepared["_prefetched_discs"] = configs
        prepared["operator_id"] = operator.get("id") if operator else None
        prepared["operator_lookup"] = bool(operator)
        if operator:
            prepared["name"] = str(operator.get("name"))
            prepared["_operator_match"] = "disc"
            self._resolve_locked_disc_names(configs, operator)
        return prepared

    def _should_collect(self, record: dict) -> bool:
        if not self.params.get("active_only"):
            return True
        operator_id = record.get("operator_id")
        states = self.params["growth_states"]
        allowed = bool(operator_id) and states.get(operator_id, "active") == "active"
        if not allowed:
            logger.info(f"AgentInfoCollector: 跳过非养成中或身份未确认的密探 name={record.get('name')!r}")
        return allowed

    def _publish_checkpoint(self, records: list[dict], current_record: dict) -> bool:
        self.publish_sequence += 1
        checkpoint_scan_id = f"{self.scan_id}-{self.publish_sequence:04d}"
        # OpenAPI receives the frozen single-operator document.  The local v3
        # file accumulates those same records so it remains directly importable
        # after an interrupted scan.
        upload_document = build_v3_document(
            [current_record],
            checkpoint_scan_id,
            self.game,
            effective_at=self.scan_started_at,
        )
        schema_path_value = self.params.get("schema_path")
        schema_path = None
        if schema_path_value:
            candidate = Path(str(schema_path_value))
            schema_path = (
                candidate if candidate.is_absolute() else REPO_ROOT / candidate
            )
        else:
            schema_path = discover_v3_schema()
        validate_v3_document(upload_document, schema_path)
        current_exchange_record = upload_document["records"][0]
        replaced = self._upsert_exchange_record(records, current_exchange_record)
        local_document = dict(upload_document)
        local_document["records"] = records
        validate_v3_document(local_document, schema_path)
        output = write_v3_document(local_document, self._output_path())
        logger.info(
            f"AgentInfoCollector: 已更新本地 v3 报告 record_id={current_exchange_record['record_id']!r}, "
            f"records={len(records)}, output={output}"
        )
        self._upload_v3_if_enabled(upload_document)
        return replaced

    def _upload_v3_if_enabled(self, document: dict[str, Any]) -> None:
        if not self.params.get("upload"):
            return
        try:
            node_data = self.context.get_node_data("在线上传认证")
            attach = node_data.get("attach", {}) if isinstance(node_data, dict) else {}
            token = str(attach.get("token", "")).strip()
            base_url = str(attach.get("base_url", "")).strip().rstrip("/")
            if not token or not base_url:
                raise ValueError("在线上传认证缺少 token 或 base_url")
            preview = preview_v3_document(document, base_url, token)
            # logger.info(f"AgentInfoCollector: {summarize_preview(preview)}")
            if self.params.get("commit", True):
                result = commit_v3_document(document, base_url, token)
                response_text = json.dumps(
                    result,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).replace(token, "<redacted>")
                # logger.info(
                #     "AgentInfoCollector: v3 自动上报 commit 完成 "
                #     f"record_id={document['records'][0]['record_id']!r}, "
                #     f"response={response_text[:4000]}"
                # )
        except Exception as exc:
            logger.warning(f"AgentInfoCollector: v3 自动上报失败，文档已保存: {exc}")


@AgentServer.custom_action("AgentInfoCollector")
class AgentInfoCollector(CustomAction):
    """Traverse the agent carousel and collect read-only agent information."""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        params = _parse_params(getattr(argv, "custom_action_param", None))
        for node_name, attach_keys in (
            ("密探采集游戏版本配置", ("game",)),
            ("密探采集对比模式配置", ("name_compare",)),
            ("密探采集命盘对比模式配置", ("disc_compare",)),
            ("密探采集上报配置", ("upload", "commit")),
            ("密探采集本地文件配置", ("output",)),
            ("密探采集单人配置", ("scan_single",)),
            ("密探采集养成筛选配置", ("active_only",)),
        ):
            try:
                node_data = context.get_node_data(node_name)
            except Exception:
                node_data = None
            attach = node_data.get("attach", {}) if isinstance(node_data, dict) else {}
            if isinstance(attach, dict):
                for key in attach_keys:
                    if key in attach:
                        params[key] = attach[key]
        try:
            if params.get("disc_compare"):
                success = _AgentInfoReader(context, params).run_disc_comparison()
                return CustomAction.RunResult(success=success)
            if params.get("name_compare"):
                success = _AgentInfoReader(context, params).run_name_comparison()
                return CustomAction.RunResult(success=success)
            if params.get("active_only") and not params.get("upload"):
                raise ValueError("仅扫描养成中密探需要启用同步至 YuanHub 并填写连接码")
            if params.get("upload"):
                settings = read_upload_settings(context)
                account = get_bound_account(settings)
                account_part = bound_account_report_filename(account)
                params["output"] = f"YuanHubMyBox-{account_part}.json"
                if params.get("active_only"):
                    try:
                        params["growth_states"] = read_growth_states(settings.base_url, settings.token, account.id)
                    except (ValueError, RuntimeError) as exc:
                        logger.error(f"AgentInfoCollector: {exc}")
                        return CustomAction.RunResult(success=False)
                logger.info(
                    "AgentInfoCollector: 已按 Token 绑定子账号选择本地报告 "
                    f"account_id={account.id!r}, output={params['output']!r}"
                )
            success = _AgentInfoReader(context, params).run()
        except InterruptedError:
            logger.info("AgentInfoCollector: 任务已停止")
            success = False
        except Exception:
            logger.exception("AgentInfoCollector: 采集流程异常")
            success = False
        return CustomAction.RunResult(success=success)
