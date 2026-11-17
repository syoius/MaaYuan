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
from custom.action.operator_scan_diagnostics import ScanFailureDiagnostics
from custom.action.operator_scan_upload import OperatorScanUploadQueue, UploadFailure
from custom.action.toolbox_diagnostics import call as diagnostic_call, start_diagnostics
from custom.action.inventory_reporting import (
    TOKEN_REQUIRED_HINT,
    UploadSettings,
    bound_account_report_filename,
    get_bound_account,
    read_upload_settings,
)
from custom.action.operator_growth_exchange import (
    DEFAULT_GAME,
    ExchangeApiError,
    _canonical_disc_description,
    _canonical_disc_name,
    _canonical_disc_name_missing_one,
    _disc_description_aliases,
    _disc_loadouts,
    _disc_name_aliases,
    _normalise_disc_name,
    _operator_rarity,
    build_v3_document,
    discover_v3_schema,
    read_growth_states,
    scrub_v3_equipment,
    set_operator_catalog,
    stable_scan_id,
    upload_response_verified,
    validate_v3_document,
    write_v3_document,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OCR_NODE = "密探信息采集-OCR"
DISC_PROMPT_NODE = "密探信息采集-命盘单套提示"
NETWORK_WEAK_PROMPT_NODE = "密探信息采集-弱网提示"
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 1280

# Page-specific markers confirm transitions; pipeline waits let animations settle.
PAGE_NODES = {
    "main": "密探信息采集-主界面就绪",
    "detail": "密探信息采集-详情界面就绪",
    "huaji": "密探信息采集-化极界面就绪",
    "disc": "密探信息采集-命盘界面就绪",
}
PAGE_PRE_DELAY_MS = 500
RETRY_BUDGET = 10
RETRYABLE_REASONS = frozenset({
    "page_timeout", "form_switch_failed", "form_tip_close_failed",
    "form_identity_unconfirmed", "disc_plan_unknown", "disc_switch_unconfirmed",
    "disc_restore_failed", "disc_tip_close_failed", "identity_changed_on_reread",
    "current_operator_failed", "name_fuzzy_disc_unconfirmed", "screenshot_failed",
    "ocr_failed", "page_turn_unconfirmed", "page_arrow_failed", "name_and_disc_unresolved",
})
SP_FORM_NAME_CHANGE_THRESHOLD = 5.0  # Mean grayscale difference; ignore small animation noise.
OPERATOR_FORM_PAIRS = {
    "char_013_chendeng": "char_084_chendengsp",
    "char_084_chendengsp": "char_013_chendeng",
    "char_023_shizimiao": "char_085_shizimiaosp",
    "char_085_shizimiaosp": "char_023_shizimiao",
}

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
DISC_LOCK_KEYWORDS = ("可解锁", "未解锁", "修为", "信赖", "好感", "达到")

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
            "previous_operator": (33, 638),
        },
        "main_name": MAIN_NAME_ROI,
        "main_stats": MAIN_STAT_ROIS,
        "oddity_rows": ODDITY_ROWS,
        "huaji_stars": HUAJI_STAR_CENTERS,
        "huaji_blue_stars": ((278, 862), (332, 862), (385, 862), (438, 862)),
        "huaji_blue_advance_stars": ((128, 862), (182, 862), (235, 862), (289, 862)),
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
            "previous_operator": (33, 638),
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
        "huaji_blue_stars": ((278, 862), (332, 862), (385, 862), (438, 862)),
        "huaji_blue_advance_stars": ((128, 862), (182, 862), (235, 862), (289, 862)),
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
            logger.bind(console=False).warning("AgentInfoCollector: custom_action_param 不是有效 JSON")
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


def _diagnostic_name_samples(main: dict) -> list[Optional[str]]:
    raw_name = main.get("name_raw")
    names = [raw_name or "", *(item.get("name") or "" for item in main.get("_operator_candidates", []))]
    name_chars = set(re.findall(r"[\u3400-\u9fff]", "".join(names)))
    samples = [raw_name]
    for reading in main.get("_name_ocr_fallback", []):
        reading = re.sub(r"\s+", "", reading)
        if len(reading) >= 2 and name_chars.intersection(reading):
            samples.append(reading)
    return samples


def _disc_ocr_name_matches(raw_name: Any, catalog_name: Any) -> bool:
    """Accept a single letter, or 0/00/0O after a name ending in a Han character."""
    raw = _normalise_disc_name(_normalise(raw_name))
    catalog = _normalise_disc_name(_normalise(catalog_name))
    if not raw or not catalog:
        return False
    suffix = r"[A-Za-z]"
    if unicodedata.name(catalog[-1], "").startswith(("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")):
        suffix = r"(?:[A-Za-z]|0|00|0O)"
    return raw == catalog or bool(re.fullmatch(re.escape(catalog) + suffix, raw))


def _disc_description_key(value: Any) -> str:
    """Fold OCR punctuation differences before comparing unlock descriptions."""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", _normalise(value).replace("生命值", "生命"))


def _has_awakened_badge(text: Any) -> bool:
    return "觉醒" in _normalise(text)


def _is_pending_awaken_layout(text: Any) -> bool:
    return "待觉醒" in _normalise(text)


def _is_max_huaji_layout(text: Any) -> bool:
    return any(marker in _normalise(text) for marker in ("最高等级", "已觉醒"))


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


def _reading_order_text(items: list[dict[str, Any]]) -> str:
    if any(item["box"] is None or item["box"][3] <= 0 for item in items):
        return "".join(item["text"] for item in items)
    rows: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda item: item["box"][1] + item["box"][3] / 2):
        box = item["box"]
        if rows:
            anchor = rows[-1][0]["box"]
            if abs(box[1] + box[3] / 2 - anchor[1] - anchor[3] / 2) <= min(box[3], anchor[3]) / 2:
                rows[-1].append(item)
                continue
        rows.append([item])
    return "".join(
        item["text"] for row in rows
        for item in sorted(row, key=lambda item: item["box"][0])
    )


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


def _ratio(text: str) -> tuple[Optional[int | float], Optional[int | float]]:
    match = re.search(
        r"(?<![\d.,])([\d,]+(?:\.\d)?)\s*%?\s*/\s*([\d,]+(?:\.\d)?)(?![\d.,])",
        text or "",
    )
    if not match:
        return None, None
    try:
        return (
            round(float(match.group(1).replace(",", "")), 1) if "." in match.group(1) else int(match.group(1).replace(",", "")),
            round(float(match.group(2).replace(",", "")), 1) if "." in match.group(2) else int(match.group(2).replace(",", "")),
        )
    except ValueError:
        return None, None


def _remove_disc_state(text: str) -> str:
    value = _normalise(text)
    for token in ("生效中", "可解锁", "锁定", "解锁"):
        value = value.replace(token, "")
    return value.strip(" :：|-")


def _unlock_description(text: str, operator: Optional[dict] = None) -> str:
    value = _normalise(text)
    value = value.replace("消耗材料可以解锁", "")
    requirement = r"(?:达到修为\+\d+|信赖值达到\d+)"
    prefix = r"解锁命盘[一二三四五六七八九十百零〇\dA-Za-z]+需要"
    names = {_normalise((operator or {}).get(key)) for key in ("name", "alt_name")}
    for name in sorted(filter(None, names), key=len, reverse=True):
        escaped = re.escape(name)
        value = re.sub(prefix + escaped + requirement + r"(?:解锁)?", "", value)
        value = re.sub(escaped + requirement + "解锁", "", value)
    generic_name = r"[\u3400-\u9fff](?:·?[\u3400-\u9fff]){0,5}"
    value = re.sub(prefix + generic_name + requirement + r"(?:解锁)?", "", value)
    # Unknown names need a boundary so the fallback cannot eat adjacent description text.
    value = re.sub(r"(?<![\u3400-\u9fff·])" + generic_name + requirement + "解锁", "", value)
    match = re.search(r"解锁.{0,2}获得[:：]?(.*)", value)
    if match:
        value = match.group(1)
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


class CollectorError(RuntimeError):
    def __init__(self, message, reason):
        super().__init__(message)
        self.reason = reason


class CollectorConfigError(ValueError):
    def __init__(self, message, reason):
        super().__init__(message)
        self.reason = reason


def _sync_failure_reason(error: Exception) -> str:
    cause = error.__cause__ or error
    message = str(error)
    status = getattr(cause, "code", None)
    if status == 401 or "HTTP 401" in message:
        return "连接码无效或已被删除；" + TOKEN_REQUIRED_HINT
    if status == 403 or "HTTP 403" in message:
        return "连接码权限不足或绑定账号不匹配"
    if status == 429 or "HTTP 429" in message:
        return "请求过于频繁，请稍后再试"
    if isinstance(cause, TimeoutError) or isinstance(getattr(cause, "reason", None), TimeoutError):
        return "网络超时，请检查网络连接"
    if getattr(error, "code", None) == "upload_network_failed" or "网络连接失败" in message:
        return "网络连接失败，请检查网络连接"
    if TOKEN_REQUIRED_HINT in message:
        return TOKEN_REQUIRED_HINT
    if getattr(error, "reason", None) == "upload_auth_missing" or "token" in message.lower():
        return "连接配置不可用，请检查服务地址；" + TOKEN_REQUIRED_HINT
    return {
        "upload_http_failed": "服务暂时不可用，请稍后再试",
        "upload_rejected": "服务未接受这份数据，请核对采集信息",
    }.get(getattr(error, "code", None), "服务响应异常，请稍后再试")


def _collection_failure_reason(error: Exception, stage: str = "") -> str:
    reason = getattr(error, "reason", None)
    if reason == "page_timeout":
        page = next((label for key, label in (
            ("main", "密探"), ("detail", "详情"), ("huaji", "化极"), ("disc", "命盘"),
        ) if str(error).startswith(f"等待 {key} 页面超时")), "所需")
        return f"{page}界面没有打开，可能是游戏卡顿或被弹窗挡住"
    return {
        "network_weak_prompt": "游戏提示网络信号弱，请检查网络后重试",
        "resource_language_mismatch": "游戏语言与所选资源不一致，请调整资源语言后重试",
        "enter_operator_page_failed": "无法进入密探界面，请确认游戏状态后重试",
        "disc_plan_unknown": "未能确认当前命盘方案，请回到密探界面后重试",
        "disc_tip_close_failed": "命盘提示未能关闭，可能有弹窗遮挡",
        "disc_switch_unconfirmed": "未能确认命盘切换，请检查游戏状态后重试",
        "disc_restore_failed": "未能恢复原命盘方案，请检查游戏中的命盘选择",
        "identity_changed_on_reread": "补读时密探发生变化，请回到目标密探后重试",
        "page_arrow_failed": "未能切换到下一位密探，请检查游戏状态",
        "list_boundary_unverified": "未能确认密探名单边界，请检查游戏状态",
        "page_turn_unconfirmed": "未能确认已切换密探，可能是游戏卡顿",
        "return_first_limit": "未能返回名单首位，请调整采集起点后重试",
        "current_operator_failed": "无法读取当前密探信息，请检查游戏状态后重试",
        "form_identity_unconfirmed": "切换形态后未能确认密探身份，请检查当前形态",
        "form_switch_failed": "未能切换或恢复密探形态，请检查游戏状态",
        "form_tip_close_failed": "未解锁形态的提示未能关闭，请检查弹窗",
        "name_and_disc_unresolved": "姓名和命盘均无法确认密探身份，请检查画面是否清晰",
        "name_fuzzy_disc_unconfirmed": "未能通过姓名和命盘确认密探身份，请重新采集",
        "screenshot_failed": "无法获取游戏画面，请检查设备连接",
        "ocr_failed": "无法识别游戏文字，请检查画面是否清晰",
        "catalog_load_failed": "密探目录不可用，请检查网络后重试",
        "active_only_requires_sync": "仅扫描养成中密探需要启用同步；" + TOKEN_REQUIRED_HINT,
        "unknown_scan_mode": "扫描模式无效，请重新选择扫描模式",
    }.get(reason, "密探目录读取失败，请检查网络后重试" if stage == "catalog"
          else "读取密探信息时出现异常，请稍后重试")


class _AgentInfoReader:
    scan_mode = "enhanced"
    max_retries = 2
    retry_budget = RETRY_BUDGET
    retries_used = 0
    _failure_diagnostics = None
    _toolbox_diagnostics = None
    _ui_script = None
    _upload_queue: Optional[OperatorScanUploadQueue] = None

    def __init__(self, context: Context, params: dict, *, upload_settings: Optional[UploadSettings] = None):
        self.context = context
        self.params = params
        self._upload_settings = upload_settings
        self._toolbox_diagnostics = params.get("_toolbox_diagnostics")
        self._ui_script = None
        self.scan_mode = params.get("scan_mode", "enhanced")
        if self.scan_mode not in ("normal", "enhanced"):
            raise CollectorConfigError(f"未知密探扫描模式: {self.scan_mode!r}", "unknown_scan_mode")
        logger.bind(console=False).info(f"AgentInfoCollector: 扫描模式={self.scan_mode}")
        resource = str(params.get("resource", "") or "").lower()
        self.language = (
            "zh-tw" if "zh_tw" in resource or "zh-tw" in resource else "zh-cn"
        )
        self.game = str(
            params.get("game") or ("如鸢" if self.language == "zh-tw" else DEFAULT_GAME)
        )
        self.roi = ROI_CONFIGS.get(self.game, ROI_CONFIGS[DEFAULT_GAME])
        self._failure_diagnostics = (
            ScanFailureDiagnostics(self.scan_mode, self.game)
            if params.get("failure_diagnostics") is True else None
        )
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
        self._page_image: Optional[np.ndarray] = None
        self.max_operators = max(1, min(300, int(params.get("max_operators", 200))))
        try:
            self.max_retries = max(0, min(5, int(params.get("max_retries", 2))))
        except (TypeError, ValueError, OverflowError):
            self.max_retries = 2
        self.retry_budget = RETRY_BUDGET
        self.retries_used = 0
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
        self._drain_upload_results()
        if self._should_stop():
            logger.bind(console=False).info("AgentInfoCollector: 收到停止请求，终止采集")
            raise InterruptedError("AgentInfoCollector stopped")

    def _sleep_checked(self, milliseconds: int) -> None:
        deadline = time.monotonic() + max(0, milliseconds) / 1000.0
        while time.monotonic() < deadline:
            self._ensure_running()
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    def _load_operators(self) -> dict[str, dict]:
        self._ensure_running()
        try:
            data = refresh_operator_catalog(REPO_ROOT / "agent" / "operators.json")
        except InterruptedError:
            raise
        except Exception as exc:
            diagnostic_call(self._toolbox_diagnostics, "issue", "catalog_unavailable", "catalog", exception=exc, reason="catalog_load_failed")
            raise
        diagnostic_call(self._toolbox_diagnostics, "catalog_info", data, getattr(data, "diagnostic", {}))
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
            image = self.context.tasker.controller.post_screencap().wait().get()
            if image is not None:
                diagnostic_call(self._toolbox_diagnostics, "observe_frame", image)
            if self._failure_diagnostics is not None and image is not None:
                self._failure_diagnostics.last_frame = image
            if image is None:
                diagnostic_call(self._toolbox_diagnostics, "issue", "screenshot_or_ocr_failed", "screenshot", reason="screenshot_failed")
            return image
        except InterruptedError:
            raise
        except Exception as exc:
            diagnostic_call(self._toolbox_diagnostics, "issue", "screenshot_or_ocr_failed", "screenshot", exception=exc, reason="screenshot_failed")
            logger.bind(console=False).exception("AgentInfoCollector: 截图失败")
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
        self._page_image = None
        self.context.tasker.controller.post_click(x, y).wait()
        delay = self.wait_ms if settle_ms is None else max(0, settle_ms)
        deadline = time.monotonic() + delay / 1000.0
        while time.monotonic() < deadline:
            self._ensure_running()
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    def _diagnose_ui_script(self) -> str:
        if self._ui_script is None:
            self._ui_script = "unknown"
            image = self.screenshot()
            if image is not None:
                diagnostic_call(self._toolbox_diagnostics, "enable_trace_text", False)
                try:
                    text = self.ocr_text(image, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
                finally:
                    diagnostic_call(self._toolbox_diagnostics, "enable_trace_text", True)
                simplified = traditional = 0
                for char in re.findall(r"[\u3400-\u9fff]", text):
                    traditional += convert(char, "zh-hans") != char
                    simplified += convert(char, "zh-hant") != char
                total = simplified + traditional
                if total:
                    if min(simplified, traditional) / total >= 0.2:
                        self._ui_script = "mixed"
                    else:
                        self._ui_script = "simplified" if simplified > traditional else "traditional"
            diagnostic_call(self._toolbox_diagnostics, "set_ui_script", self._ui_script)
            if self._ui_script == "simplified" and self.language == "zh-tw":
                logger.warning("检测到游戏界面为简体中文，但当前资源为繁中，请在 MaaYuan 资源设置中选择『简中（游戏界面）』后重试")
            elif self._ui_script == "traditional" and self.language == "zh-cn":
                logger.warning("检测到游戏界面为繁体中文，但当前资源为简中，请在 MaaYuan 资源设置中选择『繁中（游戏界面）』后重试")
        mismatch = ((self._ui_script == "simplified" and self.language == "zh-tw")
                    or (self._ui_script == "traditional" and self.language == "zh-cn"))
        return "resource_language_mismatch" if mismatch else "page_timeout"

    def _wait_for_page(self, page: str) -> Optional[np.ndarray]:
        self._ensure_running()
        self._page_failure_reason = "page_timeout"
        diagnostic_call(self._toolbox_diagnostics, "set_stage", page)
        if self._failure_diagnostics is not None:
            self._failure_diagnostics.stage = page
        if self.current_page != page:
            self._page_image = None
            node = PAGE_NODES[page]
            if self._run_task(node) is not True:
                self._page_failure_reason = self._diagnose_ui_script()
                diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", page, reason=self._page_failure_reason)
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
                logger.bind(console=False).error(
                    f"AgentInfoCollector: 等待页面超时 page={page!r}, "
                    f"node={node!r}{diagnostics}"
                )
                return None
            self.current_page = page
            # logger.info(
            #     f"AgentInfoCollector: 页面已就绪 page={page!r}, node={node!r}"
            # )
        if getattr(self, "_page_image", None) is None:
            self._page_image = self.screenshot()
        return self._page_image

    def _require_page(self, page: str) -> np.ndarray:
        image = self._wait_for_page(page)
        if image is None:
            raise CollectorError(f"等待 {page} 页面超时", getattr(self, "_page_failure_reason", "page_timeout"))
        return image

    def _continue_network_wait(self, image: Optional[np.ndarray], attempts: int) -> tuple[Optional[np.ndarray], int]:
        while image is not None:
            self._ensure_running()
            prompt = self.context.run_recognition(NETWORK_WEAK_PROMPT_NODE, image)
            self._ensure_running()
            if not prompt or not prompt.hit:
                break
            if attempts >= 3:
                raise CollectorError("游戏提示网络信号弱，请检查网络后重试", "network_weak_prompt")
            attempts += 1
            logger.warning(f"游戏提示网络信号弱，点击继续等待（{attempts}/3）")
            if self._run_task(NETWORK_WEAK_PROMPT_NODE) is not True:
                raise CollectorError("游戏提示网络信号弱，请检查网络后重试", "network_weak_prompt")
            self._sleep_checked(1000)
            image = self.screenshot()
        return image, attempts

    def _click_until_page(self, click_key: str, image: np.ndarray, target: str, source: str,
                          network_attempts: int = 0) -> np.ndarray:
        diagnostic_call(self._toolbox_diagnostics, "set_stage", target)
        if self._failure_diagnostics is not None:
            self._failure_diagnostics.stage = target
        point = self._roi_config["clicks"][click_key]
        started = time.monotonic()
        target_ready = False
        for attempt in range(3):
            self._ensure_running()
            previous_network_attempts = network_attempts
            image, network_attempts = self._continue_network_wait(image, network_attempts)
            if image is None:
                raise CollectorError("无法获取游戏画面", "screenshot_failed")
            if network_attempts > previous_network_attempts:
                ready = self.context.run_recognition(PAGE_NODES[target], image)
                self._ensure_running()
                if ready and ready.hit:
                    target_ready = True
                    break
            self.click(point, image, settle_ms=0)
            deadline = time.monotonic() + 3.0
            while True:
                self._sleep_checked(self.poll_interval_ms)
                self._ensure_running()
                frame = self.screenshot()
                previous_network_attempts = network_attempts
                frame, network_attempts = self._continue_network_wait(frame, network_attempts)
                if network_attempts > previous_network_attempts:
                    deadline = time.monotonic() + 3.0
                if frame is not None:
                    ready = self.context.run_recognition(PAGE_NODES[target], frame)
                    self._ensure_running()
                    if ready and ready.hit:
                        target_ready = True
                        break
                if time.monotonic() >= deadline:
                    break
            if target_ready or attempt == 2:
                break
            retry_allowed = True
            for _ in range(2):
                self._sleep_checked(600)
                self._ensure_running()
                frame = self.screenshot()
                frame, network_attempts = self._continue_network_wait(frame, network_attempts)
                if frame is None:
                    retry_allowed = False
                    continue
                ready = self.context.run_recognition(PAGE_NODES[target], frame)
                self._ensure_running()
                if ready and ready.hit:
                    target_ready = True
                    break
                ready = self.context.run_recognition(PAGE_NODES[source], frame)
                self._ensure_running()
                if not ready or not ready.hit:
                    retry_allowed = False
            if target_ready or not retry_allowed:
                break
            logger.bind(console=False).warning(
                f"AgentInfoCollector: 页面点击重试 click_key={click_key!r}, "
                f"source={source!r}, target={target!r}, retry={attempt + 1}, "
                f"elapsed_ms={int((time.monotonic() - started) * 1000)}"
            )
        if target_ready:
            self.current_page = None
            self._page_image = None
        try:
            result = self._require_page(target)
        except CollectorError:
            # The prompt may have appeared during the pipeline's final page wait.
            _, network_attempts = self._continue_network_wait(self.screenshot(), network_attempts)
            if not network_attempts:
                raise
            try:
                result = self._require_page(target)
            except CollectorError as exc:
                raise CollectorError("游戏提示网络信号弱，请检查网络后重试", "network_weak_prompt") from exc
        logger.bind(console=False).info(
            f"AgentInfoCollector: 页面点击成功 click_key={click_key!r}, "
            f"target={target!r}, clicks={attempt + 1}, "
            f"elapsed_ms={int((time.monotonic() - started) * 1000)}"
        )
        return result

    def _wait_for_text_change(
        self,
        roi: tuple[int, int, int, int],
        previous: str,
        description: str,
        previous_readings: Iterable[str] = (),
    ) -> Optional[str]:
        previous_keys = {_operator_name_key(text) for text in (previous, *previous_readings)} - {""}
        deadline = time.monotonic() + self.transition_timeout_ms / 1000.0
        last_text = previous
        candidate = ""
        self._page_image = None
        while True:
            self._ensure_running()
            image = self.screenshot()
            if image is not None:
                last_text = _operator_name_key(self.ocr_text(image, roi))
                if last_text not in self.operators:
                    readings = self._horizontal_name_readings(image)
                    fallback_names = {
                        _operator_name_key(self.operators[key]["name"])
                        for text in readings
                        if (key := _operator_name_key(text)) in self.operators
                    }
                    if not fallback_names:
                        readings.extend(self._horizontal_name_readings(image, full_charset=True))
                        fallback_names = {
                            _operator_name_key(self.operators[key]["name"])
                            for text in readings
                            if (key := _operator_name_key(text)) in self.operators
                        }
                    if len(fallback_names) == 1:
                        last_text = next(iter(fallback_names))
                    elif not fallback_names:
                        # A stable reading confirms movement, not identity; _read_main still requires disc evidence.
                        last_text = max((_operator_name_key(text) for text in readings if _operator_name_key(text)), key=len, default=last_text)
                    else:
                        last_text = ""
                if last_text and last_text not in previous_keys:
                    if last_text == candidate:
                        self._page_image = image
                        return last_text
                    candidate = last_text
                else:
                    candidate = ""
            else:
                candidate = ""
            if time.monotonic() >= deadline:
                logger.bind(console=False).error(
                    f"AgentInfoCollector: 等待{description}更新超时 "
                    f"previous={previous!r}, last_text={last_text!r}"
                )
                return None
            self._sleep_checked(self.poll_interval_ms)

    def _run_task(self, name: str) -> Optional[bool]:
        self._ensure_running()
        self._page_image = None
        try:
            result = self.context.run_task(name)
        except InterruptedError:
            raise
        except Exception:
            return None
        self._ensure_running()
        status = getattr(result, "status", None)
        return bool(status and getattr(status, "succeeded", False))

    def _close_prompt_stably(self, node: str, reason: str, message: str) -> None:
        self._ensure_running()
        first_hit = time.monotonic()
        deadline = first_hit + self.transition_timeout_ms / 1000.0
        hits = 1  # The caller has already recognized the first frame.
        misses = 0
        while True:
            self._sleep_checked(self.poll_interval_ms)
            self._ensure_running()
            image = self.screenshot()
            prompt = self.context.run_recognition(node, image) if image is not None else None
            self._ensure_running()
            if prompt is not None:
                hits = hits + 1 if prompt.hit else 0
                misses = 0 if prompt.hit else misses + 1
                if misses >= 2:
                    return
                if prompt.hit and hits >= 2 and time.monotonic() >= first_hit + 0.6:
                    break
            if time.monotonic() >= deadline:
                diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason=reason)
                raise CollectorError(message, reason)

        for _ in range(3):
            self._ensure_running()
            self._run_task(node)
            self._ensure_running()
            deadline = time.monotonic() + 1.5
            misses = 0
            while True:
                self._sleep_checked(self.poll_interval_ms)
                self._ensure_running()
                image = self.screenshot()
                prompt = self.context.run_recognition(node, image) if image is not None else None
                self._ensure_running()
                if prompt is not None:
                    misses = 0 if prompt.hit else misses + 1
                    if misses >= 2:
                        return
                if time.monotonic() >= deadline:
                    break
        diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason=reason)
        raise CollectorError(message, reason)

    def _toggle_disc(
        self, previous_label: str = "", previous_fingerprint: Optional[tuple[str, ...]] = None
    ) -> Optional[bool]:
        """Wait for a confirmed label/content change or the single-config dialog."""
        previous_label = _normalise(previous_label)
        if not previous_label:
            raise CollectorError(f"无法确认当前命盘方案: {previous_label!r}", "disc_plan_unknown")
        image = self._require_page("disc")
        self.click(self._roi_config["clicks"]["disc_switch"], image, settle_ms=0)
        clicked_at = time.monotonic()
        deadline = clicked_at + self.transition_timeout_ms / 1000.0
        candidate = ""
        candidate_fingerprint = None
        last_fingerprint_check = clicked_at
        saw_different_label = False
        prompt_node = DISC_PROMPT_NODE
        closing_prompt = False
        while True:
            self._ensure_running()
            image = self.screenshot()
            if image is not None:
                prompt = self.context.run_recognition(prompt_node, image)
                if prompt is None:
                    candidate = ""
                    candidate_fingerprint = None
                elif prompt.hit:
                    candidate = ""
                    candidate_fingerprint = None
                    if not closing_prompt:
                        self._close_prompt_stably(prompt_node, "disc_tip_close_failed", "无法关闭命盘单套提示")
                        closing_prompt = True
                        if time.monotonic() >= deadline:
                            deadline = time.monotonic() + self.transition_timeout_ms / 1000.0
                elif closing_prompt:
                    ready = self.context.run_recognition(PAGE_NODES["disc"], image)
                    if ready and ready.hit:
                        self._page_image = image
                        return False
                else:
                    label = _normalise(self.ocr_text(image, (180, 142, 101, 42)))
                    if label and label != previous_label:
                        saw_different_label = True
                        candidate_fingerprint = None
                        if label == candidate:
                            self._page_image = image
                            return True
                        candidate = label
                    else:
                        candidate = ""
                        if label == previous_label and previous_fingerprint is not None:
                            now = time.monotonic()
                            if now - clicked_at >= 1.0 and now - last_fingerprint_check >= 1.0:
                                last_fingerprint_check = now
                                fingerprint = self._disc_cells_fingerprint(image)
                                if any(fingerprint) and fingerprint != previous_fingerprint:
                                    if fingerprint == candidate_fingerprint:
                                        self._page_image = image
                                        return True
                                    candidate_fingerprint = fingerprint
                                else:
                                    candidate_fingerprint = None
                        else:
                            candidate_fingerprint = None
            else:
                candidate = ""
                candidate_fingerprint = None
            if time.monotonic() >= deadline:
                if closing_prompt:
                    raise CollectorError("命盘单套提示关闭未确认", "disc_tip_close_failed")
                if previous_fingerprint is not None and not saw_different_label:
                    return None
                raise CollectorError(f"命盘方案切换未确认: {previous_label!r}", "disc_switch_unconfirmed")
            self._sleep_checked(self.poll_interval_ms)

    def ocr(self, image: np.ndarray, roi: tuple[int, int, int, int], *, only_rec: bool = False) -> Any:
        self._ensure_running()
        height, width = image.shape[:2]
        scaled = _scale_roi(roi, width, height)
        clipped = _clip_roi(scaled, width, height)
        if only_rec:
            x, y, w, h = clipped
            image = np.ascontiguousarray(image[y : y + h, x : x + w])
            clipped = [0, 0, w, h]
        override = {
            OCR_NODE: {
                "recognition": {
                    "type": "OCR",
                    "param": {
                        "roi": clipped,
                        "expected": "",
                        "only_rec": only_rec,
                    },
                }
            }
        }
        try:
            result = self.context.run_recognition(OCR_NODE, image, override)
            self._ensure_running()
            if result is None:
                diagnostic_call(self._toolbox_diagnostics, "issue", "screenshot_or_ocr_failed", "ocr", reason="ocr_failed")
            return result
        except InterruptedError:
            raise
        except Exception as exc:
            diagnostic_call(self._toolbox_diagnostics, "issue", "screenshot_or_ocr_failed", "ocr", exception=exc, reason="ocr_failed")
            logger.bind(console=False).exception("AgentInfoCollector: OCR 失败，roi=%s", roi)
            return None

    def ocr_text(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> str:
        return _joined_text(self.ocr(image, roi))

    def _read_stat(self, image: np.ndarray, key: str, roi: tuple[int, int, int, int]) -> Optional[int]:
        if key == "level" and self.scan_mode == "enhanced":
            height, width = image.shape[:2]
            x, y, w, h = _clip_roi(_scale_roi(roi, width, height), width, height)
            crop = image[y:y + h, x:x + w]
            # Dark decorative lines turn low level digits into letters or other digits.
            mask = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY)[1]
            points = cv2.findNonZero(mask)
            if points is None:
                return None
            left, top, width, height = cv2.boundingRect(points)
            crop = 255 - mask[top:top + height, left:left + width]
            crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            image = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            roi = (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        detected = _normalise(self.ocr_text(image, roi))
        # The cultivation plus sign can be recognized as 十; reject all other noise.
        pattern = r"[0-9]+"
        if key == "cultivation":
            pattern = r"[+十]?[0-9]+" if self.scan_mode == "enhanced" else r"\+?[0-9]+"
        if re.fullmatch(pattern, detected):
            value = int(detected.lstrip("+十"))
            if value > 0 or key == "cultivation":
                return value
        logger.bind(console=False).warning(f"AgentInfoCollector: 数值待核对 field={key}, detected={detected!r}")
        return None

    def _horizontal_name_readings(self, image: np.ndarray, *, full_charset: bool = False) -> list[str]:
        """Retry a vertical name as two/three upright glyphs in a horizontal line."""
        if self.scan_mode == "normal":
            return []
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
        bounds_options = ((88, 149, 215), (28, 88, 149, 215))
        layouts = [(bounds, glyph_left) for bounds in bounds_options]
        if full_charset:
            # Red costume pixels can resemble the badge; check both name columns.
            layouts.extend((bounds, 35 - glyph_left) for bounds in bounds_options)
        for bounds, left in layouts:
            self._ensure_running()
            parts = [
                cv2.resize(crop[top:bottom, left:left + 71], (71, 66))
                for top, bottom in zip(bounds, bounds[1:])
            ]
            horizontal = np.concatenate(parts, axis=1)
            variants = [horizontal]
            if full_charset:
                gray = cv2.cvtColor(horizontal, cv2.COLOR_BGR2GRAY)
                binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
                variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
            override = {
                node: {
                    "recognition": {
                        "type": "OCR",
                        "param": {
                            "roi": [0, 0, horizontal.shape[1], horizontal.shape[0]],
                            "expected": "",
                            "only_rec": True,
                            "replace": [],
                            "model": "operator_names" if full_charset else "",
                        },
                    }
                }
            }
            try:
                for variant in variants:
                    result = self.context.run_recognition(node, variant, override)
                    self._ensure_running()
                    readings.append(_joined_text(result))
            except InterruptedError:
                raise
            except Exception:
                logger.bind(console=False).exception("AgentInfoCollector: 横排姓名 OCR 失败")
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
                logger.bind(console=False).info("密探姓名 OCR 对比：已绕行一圈")
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
            logger.bind(console=False).info(
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
                logger.bind(console=False).warning("密探姓名 OCR 对比：切换密探后姓名区域未变化，停止扫描")
                break

        logger.bind(console=False).info(f"密探姓名 OCR 对比报告：{report_path}")
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
                logger.bind(console=False).info("命盘 OCR 对比：已绕行一圈")
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
            logger.bind(console=False).info(
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
                logger.bind(console=False).warning("命盘 OCR 对比：切换密探后姓名区域未变化，停止扫描")
                break

        logger.bind(console=False).info(f"命盘 OCR 对比报告：{report_path}")
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
        fallback_readings: list[str],
    ) -> None:
        if candidates:
            logger.bind(console=False).warning(
                f"AgentInfoCollector: 密探名称 OCR 仅能模糊匹配 {raw_name!r}，"
                "等待命盘专属名称二次确认 "
                f"horizontal={fallback_readings!r}, "
                f"candidates={[(item['name'], round(item['score'], 3)) for item in candidates]}"
            )
        else:
            logger.bind(console=False).warning(
                "AgentInfoCollector: 密探名称无法匹配 operators.json: "
                f"{raw_name!r}, horizontal={fallback_readings!r}"
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
            if not candidates:
                fallback_readings.extend(self._horizontal_name_readings(image, full_charset=True))
                candidates = {
                    _operator_name_key(text)
                    for text in fallback_readings
                    if _operator_name_key(text) in self.operators
                }
            if len(candidates) == 1:
                operator = self.operators[next(iter(candidates))]
            elif len(candidates) > 1:
                logger.bind(console=False).warning(
                    "AgentInfoCollector: 横排姓名候选冲突，等待命盘确认 "
                    f"name={raw_name!r}, readings={fallback_readings!r}"
                )
        candidate_by_id = {}
        if not operator:
            for text in (raw_name, *fallback_readings):
                for candidate in self._fuzzy_operator_candidates(text):
                    existing = candidate_by_id.get(candidate["operator_id"])
                    if existing is None or candidate["score"] > existing["score"]:
                        candidate_by_id[candidate["operator_id"]] = candidate
        fuzzy_candidates = sorted(candidate_by_id.values(), key=lambda item: (-item["score"], str(item["name"])))
        diagnostic_call(self._toolbox_diagnostics, "observe_main", image, self._roi_config, operator.get("id") if operator else None)
        if not operator:
            self._log_unconfirmed_name(raw_name, fuzzy_candidates, fallback_readings)
        name = str(operator.get("name")) if operator else cleaned_name
        stats: dict[str, Optional[int]] = {}
        for key, roi in self._roi_config["main_stats"].items():
            stats[key] = self._read_stat(image, key, roi)
        if self._failure_diagnostics is not None:
            self._failure_diagnostics.observe_main(image, stats, bool(operator))
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
            if index == 2:
                leading_zero = re.search(r"(?<![\d.,])(0\d)\s*%?\s*/", value_raw)
                if leading_zero:
                    current = round(int(leading_zero.group(1)) / 10, 1)
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
        image = self._click_until_page("detail_entry", image, "detail", "main")
        oddities = self._read_oddities(image)
        image = self._click_until_page("detail_close", image, "main", "detail")
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
        image = self._click_until_page("huaji_entry", image, "huaji", "main")

        status_detail = self.ocr(image, self._roi_config["huaji_status"])
        status_text = _joined_text(status_detail)
        action_text = _normalise(self.ocr_text(image, self._roi_config["huaji_action"]))
        pending_text = _normalise(
            self.ocr_text(image, self._roi_config["huaji_pending"])
        )
        max_text = _normalise(self.ocr_text(image, self._roi_config["huaji_max"]))
        awakened = _has_awakened_badge(main.get("name_raw")) or "已觉醒" in action_text
        if _is_max_huaji_layout(max_text) or _is_max_huaji_layout(status_text):
            image = self._click_until_page("huaji_back", image, "main", "huaji")
            if _operator_rarity(main.get("operator_id")) == 3 and "最高等级" in _normalise(max_text + status_text):
                return {"layout": "max", "stars": 4, "nodes": [], "awakened": False}
            return {
                "layout": "awakened",
                "stars": 5,
                "nodes": [],
                "awakened": True,
            }
        if _is_pending_awaken_layout(pending_text):
            image = self._click_until_page("huaji_back", image, "main", "huaji")
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
            image = self._click_until_page("huaji_back", image, "main", "huaji")
            return {
                "layout": "sp",
                "stars": stars,
                "nodes": [],
                "awakened": awakened,
            }

        # Blue operators have four star slots, offset half a step from the
        # five-slot gold layout in both the regular and the advance groups.
        is_blue = _operator_rarity(main.get("operator_id")) == 3 and not operator_id.endswith("sp")
        advance_state = _regular_huaji_advance_state(
            image,
            self._roi_config["huaji_blue_advance_stars" if is_blue else "huaji_sp_stars"],
            self._roi_config["huaji_nodes"],
        )
        if advance_state is not None:
            image = self._click_until_page("huaji_back", image, "main", "huaji")
            return {
                "layout": "regular",
                **advance_state,
                "awakened": awakened,
            }

        star_group = "huaji_blue_stars" if is_blue else "huaji_stars"
        star_ratios = [_gold_ratio(image, center) for center in self._roi_config[star_group]]
        stars = sum(ratio > 0.035 for ratio in star_ratios)
        logger.bind(console=False).debug(
            f"AgentInfoCollector: 化极计星 group={star_group}, "
            f"gold_ratios=[{', '.join(f'{ratio:.3f}' for ratio in star_ratios)}], stars={stars}"
        )
        nodes = _huaji_nodes(image, self._roi_config["huaji_nodes"])

        image = self._click_until_page("huaji_back", image, "main", "huaji")
        return {
            "layout": "regular",
            "stars": stars,
            "nodes": nodes,
            "awakened": awakened,
        }

    def _lookup_disc(self, operator: Optional[dict], description: str) -> Optional[str]:
        if not operator:
            return None
        needle_key = _disc_description_key(description)
        if not needle_key:
            return None
        candidates = [
            (disc["ot_name"], _disc_description_key(candidate))
            for disc in operator.get("discs", [])
            if disc.get("ot_name")
            for candidate in _disc_description_aliases(disc)
        ]
        for exact in (True, False):
            matches = {
                name
                for name, candidate_key in candidates
                if candidate_key and (
                    candidate_key == needle_key if exact
                    else candidate_key in needle_key or needle_key in candidate_key
                )
            }
            if len(matches) > 1:
                logger.bind(console=False).warning(
                    "AgentInfoCollector: 命盘描述匹配不唯一，保留待核对 "
                    f"operator_id={operator.get('id')!r}, description={description!r}, "
                    f"matches={sorted(matches)!r}"
                )
                return None
            if matches:
                return next(iter(matches))
        segments = [
            _disc_description_key(segment)
            for segment in re.split(r"[，,。．.;；:：、!！?？()（）\[\]{}<>《》\s]+", _normalise(description))
        ]
        segments = [segment for segment in segments if len(segment) >= 4]
        if sum(len(segment) for segment in segments) < 8:
            return None
        matches = {
            name
            for name, candidate_key in candidates
            if all(segment in candidate_key for segment in segments)
        }
        if len(matches) > 1:
            logger.bind(console=False).warning(
                "AgentInfoCollector: 命盘描述匹配不唯一，保留待核对 "
                f"operator_id={operator.get('id')!r}, description={description!r}, "
                f"matches={sorted(matches)!r}"
            )
        if len(matches) == 1:
            return next(iter(matches))
        return None

    def _unique_operator_for_disc_name(self, name: Any) -> Optional[dict]:
        """Return the sole operator owning an exact disc name."""
        needle = self._known_disc_name(name)
        if not needle:
            return None
        owners = {
            str(operator.get("id")): operator
            for operator in self.operators.values()
            if operator.get("id")
            and any(
                _disc_ocr_name_matches(needle, disc.get("ot_name"))
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
                if any(_disc_ocr_name_matches(name, alias) for alias in _disc_name_aliases(disc)):
                    matches.append(str(disc.get("ot_name")))
        unique = {_normalise_disc_name(match) for match in matches}
        return matches[0] if len(unique) == 1 else None

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
                source = self._known_disc_name(slot.get("name"))
                operator = self._unique_operator_for_disc_name(source)
                operator_id = str(operator.get("id")) if operator else ""
                if operator_id:
                    evidence.setdefault(operator_id, set()).add(str(source))

        if len(evidence) != 1:
            if not log_failure:
                return None
            logger.bind(console=False).warning(
                "AgentInfoCollector: 命盘未能唯一确认模糊名称 "
                f"name={main.get('name_raw')!r}, evidence={evidence!r}"
            )
            return None
        operator_id, sources = next(iter(evidence.items()))
        if operator_id not in candidate_ids and len(sources) < 2:
            if not log_failure:
                return None
            logger.bind(console=False).warning(
                "AgentInfoCollector: 名称无匹配，仅有一条专属命盘证据，"
                "暂不确认身份 "
                f"name={main.get('name_raw')!r}, evidence={evidence!r}"
            )
            return None

        operator = self._operator_by_id(operator_id)
        if operator:
            diagnostic_call(self._toolbox_diagnostics, "resolve_operator", operator_id)
            logger.bind(console=False).info(
                "AgentInfoCollector: 命盘专属名称确认密探 "
                f"{main.get('name_raw')!r} -> {operator.get('name')!r}, "
                f"evidence={sorted(sources)!r}"
            )
        return operator

    def _clear_duplicate_locked_disc_names(self, slots: list[dict], operator: Optional[dict]) -> None:
        names = [_normalise_disc_name(slot.get("name")) for slot in slots]
        for slot in slots:
            name = slot.get("name")
            if ((slot.get("state") == "locked" or slot.get("locked") is True)
                    and name and names.count(_normalise_disc_name(name)) > 1):
                logger.bind(console=False).warning(
                    "AgentInfoCollector: 清空锁定命盘重复名字 "
                    f"operator_id={(operator or {}).get('id')!r}, "
                    f"position={slot.get('position')!r}, name={name!r}"
                )
                slot["name"] = None
                slot.pop("name_source", None)
                slot.pop("unlock_description", None)

    def _resolve_locked_by_elimination(self, slots: list[dict], operator: dict) -> None:
        unresolved = [
            slot for slot in slots
            if (slot.get("state") == "locked" or slot.get("locked") is True)
            and not slot.get("name")
        ]
        if len(unresolved) != 1:
            return
        disc_names = {
            _normalise_disc_name(disc["ot_name"]): disc["ot_name"]
            for disc in operator.get("discs", []) if disc.get("ot_name")
        }
        known_names = {
            _normalise_disc_name(self._known_disc_name(slot["name"]))
            for slot in slots if slot.get("name")
        } & disc_names.keys()
        remaining = disc_names.keys() - known_names
        if len(remaining) == 1:
            unresolved[0]["name"] = disc_names[next(iter(remaining))]
            unresolved[0]["name_source"] = "elimination"

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
            self._clear_duplicate_locked_disc_names(config.get("slots", []), operator)
            self._resolve_locked_by_elimination(config.get("slots", []), operator)
            config["signature"] = tuple(
                sorted(
                    (slot["position"], slot["state"], _normalise_disc_name(slot.get("name")))
                    for slot in config.get("slots", [])
                )
            )

    def _disc_slot_from_text(
        self, position: str, text: str
    ) -> Optional[dict[str, Any]]:
        normalised = _normalise(text)
        is_locked = any(keyword in normalised for keyword in DISC_LOCK_KEYWORDS)
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

    def _disc_cells_fingerprint(self, image: np.ndarray) -> tuple[str, ...]:
        return tuple(
            _normalise(self.ocr_text(image, roi))
            for _, roi in self._roi_config["disc_cells"]
        )

    def _read_disc_label_stable(self, *, fallback_to_last: bool = False) -> str:
        previous_label = label = ""
        for read in range(3):
            self._page_image = None
            label = _normalise(self.ocr_text(self._require_page("disc"), (180, 142, 101, 42)))
            if label and label == previous_label:
                return label
            previous_label = label
            if read < 2:
                self._sleep_checked(300)
        return label if fallback_to_last else ""

    def _scan_disc_config(self, operator: Optional[dict], comparison: Optional[dict] = None) -> dict:
        image = self._require_page("disc")
        if image is None:
            return {"available": False, "slots": []}
        label = self._read_disc_label_stable(fallback_to_last=True)
        cells = self._roi_config["disc_cells"]
        left = min(roi[0] for _, roi in cells)
        top = min(roi[1] for _, roi in cells)
        right = max(roi[0] + roi[2] for _, roi in cells)
        bottom = max(roi[1] + roi[3] for _, roi in cells)
        diagnostic_call(self._toolbox_diagnostics, "crop", image,
                        (left, top, right - left, bottom - top), "disc_cells")
        slots = []
        cells_fingerprint = []
        enhanced_slots = []
        previous_locked_position = previous_locked_ordered = None
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
            cells_fingerprint.append(_normalise(cell_text))
            diagnostic_call(self._toolbox_diagnostics, "observe_disc", cell_text)
            slot = self._disc_slot_from_text(position, cell_text)
            name_raw = None
            if slot and slot["state"] == "active" and not slot.get("locked"):
                name_raw = self.ocr_text(image, self._disc_name_roi(roi))
                if name_raw:
                    slot["name"] = _remove_disc_state(name_raw)
            elif slot is None and cell_text.strip():
                name_raw = self.ocr_text(image, self._disc_name_roi(roi))
                slot = self._disc_slot_from_text(position, name_raw)
            if slot and slot["state"] == "active" and not slot.get("locked"):
                cell_name = _remove_disc_state(cell_text)
                if cell_name and _normalise_disc_name(cell_name) != _normalise_disc_name(slot.get("name")):
                    slot["cell_name"] = cell_name
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
                if enhanced_slot and enhanced_slot["state"] == "active" and not enhanced_slot.get("locked"):
                    cell_name = _remove_disc_state(enhanced_text)
                    if cell_name and _normalise_disc_name(cell_name) != _normalise_disc_name(enhanced_slot.get("name")):
                        enhanced_slot["cell_name"] = cell_name
                capture["cells"].append({
                    "position": position, "roi": list(roi), "pixel_roi": [x, y, w, h],
                    "image": f"cells/{original_path.name}",
                    "enhanced_image": f"cells/{enhanced_path.name}",
                    "original": {"raw": cell_text, "name_raw": name_raw, "slot": slot},
                    "enhanced": {"raw": enhanced_text, "name_raw": enhanced_name_raw, "slot": enhanced_slot},
                    "same_text": _normalise(cell_text) == _normalise(enhanced_text),
                })
            if (slot and slot.get("locked")) or (enhanced_slot and enhanced_slot.get("locked")):
                pre_click_ordered = _reading_order_text(
                    _result_items(self.ocr(image, (100, 900, 600, 220)))
                )
                deadline = time.monotonic() + 1.2
                self.click(
                    (roi[0] + roi[2] // 2, roi[1] + roi[3] // 2),
                    image,
                    settle_ms=220,
                )
                detail_raw = detail_text = detail_ordered = ""
                pending = None
                scanned_names = {
                    _normalise_disc_name(candidate["name"]) for candidate in (*slots, *enhanced_slots)
                    if candidate.get("position") != position and candidate.get("name")
                }
                rejected_reasons = set()
                while True:
                    self._ensure_running()
                    detail_image = self.screenshot()
                    detail_items = _result_items(
                        self.ocr(detail_image, (100, 900, 600, 220))
                        if detail_image is not None
                        else None
                    )
                    last_raw = "".join(item["text"] for item in detail_items)
                    last_ordered = _reading_order_text(detail_items)
                    last_text = _unlock_description(last_ordered, operator)
                    detail_name = self._lookup_disc(operator, last_text)
                    stale_panel = (position != previous_locked_position
                                   and last_ordered == previous_locked_ordered)
                    pre_click_stale = bool(pre_click_ordered) and last_ordered == pre_click_ordered
                    duplicate_name = _normalise_disc_name(detail_name) in scanned_names
                    if stale_panel:
                        rejected_reasons.add("旧面板")
                    if duplicate_name:
                        rejected_reasons.add("名字重复")
                    if stale_panel or duplicate_name:
                        detail_name = None
                    if pre_click_stale and detail_name:
                        if pending is None:
                            pending = (detail_name, last_raw, last_text, last_ordered)
                        detail_name = None
                        rejected_reasons.add("面板未刷新")
                    if not pre_click_stale and (detail_name or len(last_text) >= len(detail_text)):
                        detail_raw, detail_text, detail_ordered = last_raw, last_text, last_ordered
                    if detail_name:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if pending is not None:
                            detail_name, detail_raw, detail_text, detail_ordered = pending
                        logger.bind(console=False).warning(
                            "AgentInfoCollector: 锁定命盘描述轮询超时 "
                            f"operator_id={(operator or {}).get('id')!r}, position={position!r}, "
                            f"reason={('、'.join(sorted(rejected_reasons)) or '未反查到命盘名')!r}, "
                            f"raw={last_raw[:200]!r}, ordered={last_ordered[:200]!r}, "
                            f"description={last_text[:200]!r}"
                        )
                        break
                    self._sleep_checked(min(self.poll_interval_ms, max(1, int(remaining * 1000))))
                previous_locked_position, previous_locked_ordered = position, detail_ordered
                for candidate in (slot, enhanced_slot):
                    if candidate and candidate.get("locked"):
                        candidate["unlock_description"] = detail_text
                        candidate["name"] = detail_name
                if capture is not None:
                    capture["cells"][-1]["detail_raw"] = detail_raw
                    capture["cells"][-1]["detail_text"] = detail_text
            if slot:
                slots.append(slot)
            if enhanced_slot:
                enhanced_slots.append(enhanced_slot)

        for config_slots in (slots, enhanced_slots):
            self._clear_duplicate_locked_disc_names(config_slots, operator)
            if operator:
                self._resolve_locked_by_elimination(config_slots, operator)
        active_names = tuple(
            sorted(
                (slot["position"], slot["state"], _normalise_disc_name(slot.get("name")))
                for slot in slots
            )
        )
        result = {
            "available": True,
            "label": _normalise(label),
            "slots": slots,
            "signature": active_names,
            "cells_fingerprint": tuple(cells_fingerprint),
        }
        if capture is not None:
            capture["label"] = result["label"]
            capture["enhanced_config"] = {
                "available": True,
                "label": result["label"],
                "slots": enhanced_slots,
                "signature": tuple(sorted(
                    (item["position"], item["state"], _normalise_disc_name(item.get("name")))
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
        image = self._click_until_page("disc_entry", image, "disc", "main")
        operator = self._operator_by_id(main.get("operator_id"))
        first = self._scan_disc_config(operator, comparison=comparison)
        configs = [dict(first, index=1)]

        first_signature = first.get("signature", ())
        first_label = first.get("label", "")
        switched = self._toggle_disc(first_label, first.get("cells_fingerprint"))
        restored_label = first_label
        restored_ok = not switched
        second = {}
        try:
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
            elif switched is None:
                configs.append({"index": 2, "available": False, "reason": "切换命盘后名称和内容均未变化，无法确认第二套命盘", "slots": []})
            elif switched is False:
                configs.append({"index": 2, "available": False, "single_config": True, "reason": "单套命盘", "slots": []})
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
                elif switched is False or switched is None:
                    enhanced_configs.append(dict(configs[1]))
                else:
                    enhanced_configs.append({"index": 2, "available": False, "reason": "未切换命盘", "slots": []})
                comparison["enhanced_configs"] = enhanced_configs

        finally:
            # Restore even if reading the second configuration fails; cancellation forbids further clicks.
            if switched and not self._should_stop():
                restored_label = ""
                try:
                    current_label = _normalise(self.ocr_text(self._require_page("disc"), (180, 142, 101, 42)))
                    same_label = bool(second) and _normalise(first_label) == _normalise(second.get("label", ""))
                    if same_label:
                        if not self._toggle_disc(current_label, second.get("cells_fingerprint")):
                            raise CollectorError("命盘原配置恢复切换未执行", "disc_restore_failed")
                        restored_fingerprint = self._disc_cells_fingerprint(self._require_page("disc"))
                        restored_ok = bool(first.get("cells_fingerprint")) and restored_fingerprint == first["cells_fingerprint"]
                        if not restored_ok:
                            raise CollectorError("命盘原配置恢复内容未确认", "disc_restore_failed")
                    elif current_label != first_label and not self._toggle_disc(current_label):
                        raise CollectorError("命盘原配置恢复切换未执行", "disc_restore_failed")
                    restored_label = self._read_disc_label_stable()
                    if not same_label:
                        restored_ok = restored_label == first_label
                    if not restored_ok or not restored_label:
                        raise CollectorError(f"命盘原配置恢复结果未确认: expected={first_label!r}, actual={restored_label!r}", "disc_restore_failed")
                except CollectorError as exc:
                    if exc.reason == "disc_restore_failed":
                        try:
                            frame = self.screenshot()
                            restored_label = _normalise(self.ocr_text(frame, (180, 142, 101, 42))) if frame is not None else ""
                            current_cells = self._disc_cells_fingerprint(frame) if frame is not None else ()
                        except InterruptedError:
                            raise
                        except Exception:
                            current_cells = ()
                        logger.bind(console=False).warning(
                            f"AgentInfoCollector: 命盘恢复校验失败 first_label={first_label!r}, "
                            f"second_label={second.get('label', '')!r}, restored_label={restored_label!r}, "
                            f"original_cells={str(first.get('cells_fingerprint', ()))[:200]!r}, "
                            f"current_cells={str(current_cells)[:200]!r}, "
                            f"cells_match={bool(current_cells) and current_cells == first.get('cells_fingerprint')}"
                        )
                    raise

        image = self._require_page("disc")
        image = self._click_until_page("disc_back", image, "main", "disc")
        if comparison is not None:
            comparison["navigation"] = {
                "switched": switched,
                "restored": restored_ok,
                "restored_label": restored_label,
            }
        logger.bind(console=False).info(
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
        if main.get("_name_exact") is False or not main.get("name"):
            main = self._prepare_growth_filter(main)
            if not main.get("name"):
                return None
        record = dict(main)
        record["_observed_at"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
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
                logger.bind(console=False).warning(
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
        if self.scan_mode == "enhanced" and record.get("operator_id") and any(value is None for value in record.get("stats", {}).values()):
            # Moving effects can corrupt one frame; retry after returning from the detail pages.
            self._page_image = None
            retry = self._read_main(self._require_page("main"))
            if retry.get("operator_id") and retry["operator_id"] != record["operator_id"]:
                raise CollectorError("数值补读时密探身份不符，停止采集", "identity_changed_on_reread")
            if retry.get("operator_id") == record["operator_id"]:
                record["stats"] = {
                    key: retry["stats"].get(key) if value is None else value
                    for key, value in record["stats"].items()
                }
                logger.bind(console=False).info(f"AgentInfoCollector: 新截图数值补读 name={record.get('name')!r}, stats={record['stats']!r}")
            else:
                logger.bind(console=False).warning("AgentInfoCollector: 数值补读身份未确认，保留缺失项")
        record["collection_debug"] = {
            "scan_mode": self.scan_mode,
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
        logger.bind(console=False).info(
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
            logger.bind(console=False).warning("AgentInfoCollector: 无法读取已有 v3 报告: %s", output)
            return []

        if (
            not isinstance(document, dict)
            or document.get("format") != "myshare-operator-exchange"
            or document.get("version") != 3
        ):
            logger.bind(console=False).warning("AgentInfoCollector: 已有报告不是 v3 交换文档: %s", output)
            return []
        records = document.get("records")
        if not isinstance(records, list):
            logger.bind(console=False).warning(
                "AgentInfoCollector: 已有 v3 报告缺少 records 数组: %s", output
            )
            return []
        scrub_v3_equipment(document)
        valid = [record for record in records if self._exchange_record_key(record)]
        logger.bind(console=False).info(
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

    def _operator_arrow_visible(self, image: np.ndarray, direction: str) -> bool:
        frame = cv2.flip(image, 1) if direction == "previous" else image
        height, width = frame.shape[:2]
        node = "密探信息采集-翻页箭头"
        detail = self.context.run_recognition(node, frame, {
            node: {"recognition": {"param": {"roi": _scale_roi((660, 614, 50, 65), width, height)}}}
        })
        self._ensure_running()
        if detail is None:
            raise CollectorError("密探翻页箭头识别未能执行", "page_arrow_failed")
        return bool(detail.hit)

    def _move_operator(self, direction: str, main: dict) -> bool:
        diagnostic_call(self._toolbox_diagnostics, "set_stage", "navigation")
        image = self._require_page("main")
        if not self._operator_arrow_visible(image, direction):
            self._sleep_checked(PAGE_PRE_DELAY_MS)
            image = self.screenshot()
            if image is None:
                raise CollectorError("无法复核密探列表边界", "list_boundary_unverified")
            ready = self.context.run_recognition(PAGE_NODES["main"], image)
            self._ensure_running()
            if not ready or not ready.hit:
                raise CollectorError("密探列表边界复核时不在主界面", "list_boundary_unverified")
            self._page_image = image
            if not self._operator_arrow_visible(image, direction):
                logger.bind(console=False).info(f"AgentInfoCollector: 已确认名单{'首位' if direction == 'previous' else '末位'}")
                return False
        self.click(self._roi_config["clicks"][f"{direction}_operator"], image, settle_ms=0)
        if self._wait_for_text_change(
            self._roi_config["main_name"], str(main.get("name_raw") or main.get("name") or ""),
            "上一位密探" if direction == "previous" else "下一位密探",
            previous_readings=(str(main.get("name") or ""), *main.get("_name_ocr_fallback", [])),
        ) is None:
            raise CollectorError("密探翻页未确认，停止采集", "page_turn_unconfirmed")
        return True

    def _seek_first_operator(self) -> None:
        for _ in range(300):
            main = self._read_main(self._require_page("main"))
            if not self._move_operator("previous", main):
                return
        raise CollectorError("回到首位密探超过安全上限", "return_first_limit")

    def _recover_to_main(self) -> bool:
        network_attempts = 0
        for attempt in range(4):
            self._ensure_running()
            self.current_page = None
            self._page_image = None
            image = self.screenshot()
            image, network_attempts = self._continue_network_wait(image, network_attempts)
            if image is not None:
                ready = self.context.run_recognition(PAGE_NODES["main"], image)
                self._ensure_running()
                if ready and ready.hit:
                    self.current_page = "main"
                    self._page_image = image
                    return True
                for node in (DISC_PROMPT_NODE, "密探信息采集-未解锁形态提示"):
                    prompt = self.context.run_recognition(node, image)
                    self._ensure_running()
                    if prompt and prompt.hit:
                        self._run_task(node)
                        break
                else:
                    for page, click_key in (("detail", "detail_close"), ("huaji", "huaji_back"), ("disc", "disc_back")):
                        ready = self.context.run_recognition(PAGE_NODES[page], image)
                        self._ensure_running()
                        if ready and ready.hit:
                            frame = self.screenshot()
                            frame, network_attempts = self._continue_network_wait(frame, network_attempts)
                            if frame is not None:
                                ready = self.context.run_recognition(PAGE_NODES[page], frame)
                                self._ensure_running()
                                if ready and ready.hit:
                                    try:
                                        self._page_image = self._click_until_page(click_key, frame, "main", page,
                                                                                network_attempts=network_attempts)
                                        self.current_page = "main"
                                        return True
                                    except CollectorError as exc:
                                        if exc.reason == "network_weak_prompt":
                                            raise
                            break
            if attempt < 3:
                self._sleep_checked(PAGE_PRE_DELAY_MS)
        if network_attempts:
            raise CollectorError("游戏提示网络信号弱，请检查网络后重试", "network_weak_prompt")
        return False

    def _retry_from_main(self, error: CollectorError, attempt: int, key: Optional[str]) -> Optional[dict]:
        if (error.reason not in RETRYABLE_REASONS
                or attempt >= self.max_retries or self.retries_used >= self.retry_budget
                or self._should_stop()):
            return None
        self.retries_used += 1
        logger.bind(console=False).warning(f"AgentInfoCollector: 第 {attempt + 1} 次重试，原因={error.reason}，密探 key={key!r}")
        diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", exception=error, reason=error.reason)
        try:
            if not self._recover_to_main():
                return None
            main = self._read_main(self._require_page("main"))
            if main.get("_name_exact") is False or not main.get("name"):
                main = self._prepare_growth_filter(main)
            return main
        except CollectorError as exc:
            if exc.reason == "network_weak_prompt":
                raise
            return None

    def run(self) -> bool:
        self._console_progress = {}
        self._rarity_skipped = set()
        try:
            self.context.run_task("进入界面-密探")
            self._require_page("main")
            if not self.params.get("scan_single", False):
                if self.params.get("start_from_current", False):
                    logger.bind(console=False).info("从当前密探开始采集，不返回首位")
                else:
                    self._seek_first_operator()
        except InterruptedError:
            raise
        except Exception as exc:
            diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "enter", exception=exc, reason=getattr(exc, "reason", "enter_operator_page_failed"))
            logger.bind(console=False).exception("AgentInfoCollector: 进入密探界面失败")
            logger.error(f"采集中断：{_collection_failure_reason(exc)}")
            return False

        resume = self.params.get("resume", True)
        resume_enabled = not (
            resume is False or str(resume).strip().lower() in {"0", "false", "no"}
        )
        exchange_records = self._load_records() if resume_enabled else []
        origin_key: Optional[str] = None
        for _ in range(self.max_operators):
            logger.bind(console=False).info(f"AgentInfoCollector: 开始读取第 {_ + 1} 位密探")
            attempt = 0
            key = None
            completed_circle = False
            while True:
                try:
                    if attempt == 0:
                        main = self._read_main(self._require_page("main"))
                        key = self._record_key(main)
                        if main.get("_name_exact") is False or not main.get("name"):
                            main = self._prepare_growth_filter(main)
                    if not main.get("name"):
                        diagnostic_call(self._toolbox_diagnostics, "issue", "identity_unresolved", "identity", reason="name_and_disc_unresolved",
                                        name_samples=_diagnostic_name_samples(main),
                                        candidate_operator_ids=[item.get("operator_id") for item in main.get("_operator_candidates", [])])
                        logger.bind(console=False).error("AgentInfoCollector: 姓名 OCR 与命盘均无法确认当前密探，停止遍历")
                        raise CollectorError("姓名 OCR 与命盘均无法确认当前密探", "name_and_disc_unresolved")
                    if attempt == 0:
                        key = self._record_key(main)
                    if origin_key is None:
                        origin_key = key
                    if attempt == 0 and key == origin_key and _ > 0:
                        logger.bind(console=False).info(f"AgentInfoCollector: 已绕行一圈，断点采集完成 key={key!r}")
                        completed_circle = True
                        break

                    prepared = self._prepare_growth_filter(main)
                    current = self._collect_and_publish(prepared, exchange_records)
                    if not self.params.get("scan_single", False):
                        self._collect_other_form(current, exchange_records)
                    if attempt > 0:
                        diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason="retry_recovered",
                                        operator_id=current.get("operator_id"))
                    break
                except CollectorError as exc:
                    recovered = self._retry_from_main(exc, attempt, key)
                    if recovered is not None:
                        recovered_key = self._record_key(recovered)
                        if recovered_key in {key, OPERATOR_FORM_PAIRS.get(key)}:
                            attempt += 1
                            main = recovered
                            continue
                        logger.bind(console=False).error(f"AgentInfoCollector: 恢复后密探身份改变，停止重试 key={key!r}, actual={recovered_key!r}")
                    if exc.reason == "name_and_disc_unresolved":
                        logger.error(f"采集中断：{_collection_failure_reason(exc)}")
                        return False
                    raise

            if completed_circle or self.params.get("scan_single", False):
                break

            try:
                collection_attempt = attempt
                current_key = self._record_key(current)
                navigation_main = {**main, "name": current.get("name") or main.get("name")}
                while True:
                    try:
                        moved = self._move_operator("next", navigation_main)
                        break
                    except CollectorError as exc:
                        if exc.reason not in {"page_turn_unconfirmed", "page_arrow_failed", "page_timeout"}:
                            raise
                        recovered = self._retry_from_main(exc, attempt, current_key)
                        if recovered is None:
                            raise
                        attempt += 1
                        recovered_key = self._record_key(recovered)
                        if not recovered_key:
                            raise
                        if recovered_key not in {current_key, OPERATOR_FORM_PAIRS.get(current_key)}:
                            moved = True
                            break
                        navigation_main = recovered
                if attempt > collection_attempt:
                    diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason="retry_recovered",
                                    operator_id=current.get("operator_id"))
                if not moved:
                    break
            except InterruptedError:
                raise
            except RuntimeError as exc:
                diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", exception=exc)
                logger.bind(console=False).exception("AgentInfoCollector: 无法继续遍历密探")
                logger.error(f"采集中断：{_collection_failure_reason(exc)}")
                return False

        if not exchange_records:
            filtered = bool(
                (self.params.get("active_only") or self.params.get("min_rarity", 3) > 3)
                and origin_key is not None
            )
            if not filtered:
                logger.error("采集中断：没有采集到可保存的密探信息")
            return filtered
        logger.bind(console=False).info(
            f"AgentInfoCollector: 采集结束，v3 报告共 {len(exchange_records)} 位密探"
        )
        return True

    def _collect_and_publish(self, main: dict, records: list[dict]) -> dict:
        if not self._should_collect(main):
            return main
        record = self.collect_current_from_main(main)
        if not record:
            raise CollectorError("无法采集当前密探，停止遍历", "current_operator_failed")
        # Disc evidence may correct the identity read on the main page.
        if self._should_collect(record):
            replaced = self._publish_checkpoint(records, record)
            logger.bind(console=False).info(
                f"AgentInfoCollector: 本地{'已更新' if replaced else '已新增'} "
                f"name={record.get('name')!r}, "
                f"operator_id={record.get('operator_id')!r}, count={len(records)}"
            )
        return record

    def _wait_for_operator_form(self, expected_id: str) -> dict:
        self.current_page = None
        self._sleep_checked(PAGE_PRE_DELAY_MS)
        deadline = time.monotonic() + self.transition_timeout_ms / 1000.0
        while True:
            self._page_image = None
            main = self._read_main(self._require_page("main"))
            if main.get("operator_id") == expected_id or main.get("_name_exact") is False:
                main = self._prepare_growth_filter(main)
                if main.get("operator_id") == expected_id:
                    return main
            if time.monotonic() >= deadline:
                diagnostic_call(self._toolbox_diagnostics, "issue", "identity_unresolved", "identity", reason="form_identity_unconfirmed",
                                name_samples=_diagnostic_name_samples(main),
                                candidate_operator_ids=[item.get("operator_id") for item in main.get("_operator_candidates", [])])
                raise CollectorError(f"切换形态后身份未确认 expected={expected_id!r}, actual={main.get('operator_id')!r}", "form_identity_unconfirmed")
            self._sleep_checked(self.poll_interval_ms)

    def _click_form_switch(self) -> bool:
        for attempt in range(3):
            self.current_page = None
            self._page_image = None
            self._require_page("main")
            if self._run_task("密探信息采集-切换形态") is True:
                return True
            if attempt < 2:
                self._sleep_checked(self.poll_interval_ms)
        logger.bind(console=False).warning("AgentInfoCollector: 切换形态点击失败，已尝试 3 次")
        return False

    def _collect_other_form(self, main: dict, records: list[dict]) -> None:
        original_id = str(main.get("operator_id") or "")
        other_id = OPERATOR_FORM_PAIRS.get(original_id)
        if not other_id or self.params.get("scan_single"):
            return
        if other_id.endswith("sp"):
            baseline = self._name_signature(self._require_page("main"))
        if not self._click_form_switch():
            if original_id.endswith("sp"):
                diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason="form_switch_failed")
                raise CollectorError(f"未能切换到普通形态 operator_id={original_id!r}", "form_switch_failed")
            diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason="form_switch_failed")
            return
        if other_id.endswith("sp"):
            self.current_page = None
            self._page_image = None
            clicked_at = time.monotonic()
            deadline = clicked_at + self.transition_timeout_ms / 1000.0
            attempts = 1
            confirming_identity = False
            prompt_node = "密探信息采集-未解锁形态提示"
            other = None
            while True:
                self._ensure_running()
                image = self.screenshot()
                try:
                    prompt = self.context.run_recognition(prompt_node, image) if image is not None else None
                except InterruptedError:
                    raise
                except Exception:
                    logger.bind(console=False).warning("AgentInfoCollector: 未解锁提示识别失败，继续通过形态身份确认")
                    prompt = None
                self._ensure_running()
                if prompt and prompt.hit:
                    self._close_prompt_stably(prompt_node, "form_tip_close_failed", "无法关闭未解锁形态提示")
                    self._wait_for_operator_form(original_id)
                    logger.bind(console=False).info(f"AgentInfoCollector: SP 形态未解锁，继续扫描 operator_id={other_id!r}")
                    return
                if image is not None:
                    ready = self.context.run_recognition(PAGE_NODES["main"], image)
                    self._ensure_running()
                    if ready and ready.hit:
                        self.current_page = "main"
                        self._page_image = image
                        if not confirming_identity:
                            difference = np.mean(cv2.absdiff(self._name_signature(image), baseline))
                            if difference < SP_FORM_NAME_CHANGE_THRESHOLD:
                                if time.monotonic() >= clicked_at + 1.5:
                                    if attempts >= 3:
                                        logger.bind(console=False).warning(f"AgentInfoCollector: SP 形态切换无响应，跳过 operator_id={other_id!r}")
                                        diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason="form_switch_failed")
                                        return
                                    if self._run_task("密探信息采集-切换形态") is not True:
                                        logger.bind(console=False).warning(f"AgentInfoCollector: SP 形态切换无响应，跳过 operator_id={other_id!r}")
                                        diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason="form_switch_failed")
                                        return
                                    attempts += 1
                                    clicked_at = time.monotonic()
                                    deadline = clicked_at + self.transition_timeout_ms / 1000.0
                                    self.current_page = None
                                    self._page_image = None
                                self._sleep_checked(self.poll_interval_ms)
                                continue
                            # Identity confirmation gets its own budget after the visual transition.
                            confirming_identity = True
                            deadline = time.monotonic() + self.transition_timeout_ms / 1000.0
                        other = self._read_main(image)
                        if other.get("_name_exact") is False or not other.get("operator_id") or not other.get("name"):
                            other = self._prepare_growth_filter(other)
                        if other.get("operator_id") == other_id:
                            break
                        if other.get("operator_id") == original_id:
                            logger.bind(console=False).warning(f"AgentInfoCollector: SP 形态切换无响应，跳过 operator_id={other_id!r}")
                            diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason="form_switch_failed")
                            return
                if time.monotonic() >= deadline:
                    if other is None:
                        diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "main", reason="page_timeout")
                        raise CollectorError("等待 main 页面超时", "page_timeout")
                    diagnostic_call(self._toolbox_diagnostics, "issue", "identity_unresolved", "identity", reason="form_identity_unconfirmed",
                                    name_samples=_diagnostic_name_samples(other),
                                    candidate_operator_ids=[item.get("operator_id") for item in other.get("_operator_candidates", [])])
                    raise CollectorError(f"切换形态后身份未确认 expected={other_id!r}, actual={other.get('operator_id')!r}", "form_identity_unconfirmed")
                self._sleep_checked(self.poll_interval_ms)
        try:
            if not other_id.endswith("sp"):
                other = self._wait_for_operator_form(other_id)
            self._collect_and_publish(other, records)
        finally:
            # Respect cancellation; do not issue game actions after a stop.
            if not self._should_stop():
                if not self._click_form_switch():
                    diagnostic_call(self._toolbox_diagnostics, "issue", "navigation_failed", "navigation", reason="form_switch_failed")
                    raise CollectorError(f"无法恢复原密探形态 operator_id={original_id!r}", "form_switch_failed")
                self._wait_for_operator_form(original_id)

    def _prepare_growth_filter(self, main: dict) -> dict:
        if "_prefetched_discs" in main or (
            main.get("name")
            and main.get("_name_exact") is not False
        ):
            return main
        logger.bind(console=False).info("AgentInfoCollector: 首屏身份不确定，先读取命盘确认当前密探")
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
        else:
            diagnostic_call(self._toolbox_diagnostics, "issue", "identity_unresolved", "identity", reason="name_fuzzy_disc_unconfirmed",
                            name_samples=_diagnostic_name_samples(main),
                            candidate_operator_ids=[item.get("operator_id") for item in main.get("_operator_candidates", [])])
        return prepared

    def _should_collect(self, record: dict) -> bool:
        operator_id = record.get("operator_id")
        if self.params.get("active_only"):
            states = self.params["growth_states"]
            if not operator_id or states.get(operator_id, "active") != "active":
                logger.bind(console=False).info(f"AgentInfoCollector: 跳过非养成中或身份未确认的密探 name={record.get('name')!r}")
                return False
        min_rarity = self.params.get("min_rarity", 3)
        if min_rarity > 3:
            rarity = _operator_rarity(operator_id)
            if rarity is not None and rarity < min_rarity:
                logger.bind(console=False).info(
                    f"AgentInfoCollector: 跳过低于稀有度门槛的密探 operator_id={operator_id!r}, "
                    f"rarity={rarity}, min_rarity={min_rarity}"
                )
                skipped = getattr(self, "_rarity_skipped", set())
                if operator_id not in skipped:
                    rarity_name = "蓝卡" if rarity == 3 else "紫卡"
                    logger.info(f"跳过 {record.get('name') or operator_id}（{rarity_name}）")
                    skipped.add(operator_id)
                self._rarity_skipped = skipped
                return False
        return True

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
            effective_at=current_record.get("_observed_at") or self.scan_started_at,
        )
        diagnostic_call(self._toolbox_diagnostics, "checkpoint", upload_document)
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
        logger.bind(console=False).info(
            f"AgentInfoCollector: 已更新本地 v3 报告 record_id={current_exchange_record['record_id']!r}, "
            f"records={len(records)}, output={output}"
        )
        upload_pending = self._queue_upload_if_enabled(upload_document)
        self._log_checkpoint(current_record, current_exchange_record, upload_pending=upload_pending)
        return replaced

    @staticmethod
    def _console_name(entry: dict) -> str:
        name = entry.get("name")
        if isinstance(name, str):
            name = re.sub(r"[·・]?SP$", "·SP", name, flags=re.IGNORECASE)
        return name if isinstance(name, str) and re.fullmatch(r"[\u3400-\u9fff·・（） ]+(?:·SP)?", name) else "未确认姓名的密探"

    def _log_checkpoint(self, record: dict, exchange_record: dict, *, upload_pending: bool = False) -> None:
        entries = exchange_record.get("entries", [])
        entry = entries[0] if entries else {}
        statuses = entry.get("section_status", {})
        review_labels = [label for key, label in (
            ("basic", "基础信息"), ("huaji", "化极"), ("oddities", "奇闻"),
            ("combat_stats", "属性"), ("disc_loadouts", "命盘"),
        ) if key in statuses and statuses[key] != "ready"]
        if not entries:
            review_labels.append("姓名")
        details = []
        if "level" in entry:
            details.append(f"等级 {entry['level']}")
        cultivation = record.get("stats", {}).get("cultivation")
        if type(cultivation) is int:
            details.append(f"修为 +{cultivation}")
        huaji = record.get("huaji", {})
        if statuses.get("huaji") == "ready":
            if huaji.get("awakened"):
                details.append("已觉醒")
            elif type(huaji.get("stars")) is int:
                details.append(f"化极 {huaji['stars']} 星")
        loadouts = {loadout["id"]: {disc["ot_name"] for disc in loadout["discs"]}
                    for loadout in entry.get("disc_loadouts", [])}
        for index, config in enumerate((record.get("disc_configs") or [])[:2], 1):
            if not config.get("available"):
                continue
            confirmed = loadouts.get(f"disc_{index}", set())
            names = []
            seen = set()
            for slot in config.get("slots", []):
                if slot.get("state") not in {"active", "locked"}:
                    continue
                raw_names = [str(slot.get(key) or "").strip() for key in ("name", "cell_name")]
                canonical = None
                for matcher in (_canonical_disc_name, _canonical_disc_name_missing_one):
                    for raw_name in filter(None, raw_names):
                        canonical = matcher(record.get("operator_id"), raw_name)
                        if canonical is not None:
                            break
                    if canonical is not None:
                        break
                if canonical is None and not any(raw_names):
                    canonical = _canonical_disc_description(record.get("operator_id"), slot.get("unlock_description") or "")
                names.append(canonical if canonical in confirmed and canonical not in seen else "？")
                seen.add(canonical)
            label = config.get("label") or ("命盘一", "命盘二")[index - 1]
            details.append(f"{label}：{'、'.join(names) or '？'}")
            if (not names or "？" in names) and "命盘" not in review_labels:
                review_labels.append("命盘")
        if self.params.get("upload"):
            details.append("等待同步" if upload_pending else "未同步")
        name = self._console_name(entry)
        message = f"已录入 {name}：{' · '.join(details) or '本地记录已保存'}"
        if review_labels:
            message += f"（{'、'.join(review_labels)}需核对）"
        progress = getattr(self, "_console_progress", {})
        progress[self._record_key(record)] = (False, bool(review_labels))
        self._console_progress = progress
        logger.info(message)

    def _queue_upload_if_enabled(self, document: dict[str, Any]) -> bool:
        if not self.params.get("upload"):
            return False
        entries = [entry for record in document.get("records", []) for entry in record.get("entries", [])]
        if not entries:
            logger.bind(console=False).info("本次无已识别密探可上报，已跳过 YuanHub 上报")
            return False
        entry = entries[0]
        operator_id = entry.get("operator_id", "")
        try:
            if self._upload_queue is None:
                if self._upload_settings is None:
                    raise CollectorConfigError("密探上报认证未初始化；" + TOKEN_REQUIRED_HINT, "upload_auth_missing")
                self._pending_uploads = {}
                self._latest_upload_ids = {}
                self._upload_queue = OperatorScanUploadQueue(
                    self._upload_settings, commit=bool(self.params.get("commit", True))
                )
            record_id = document["records"][0]["record_id"]
            key = self._record_key(entry)
            self._pending_uploads[record_id] = {
                "key": key, "name": self._console_name(entry), "operator_id": operator_id,
                "client_review_count": sum(any(value != "ready" for value in item.get("section_status", {}).values()) for item in entries),
            }
            self._latest_upload_ids[key] = record_id
            self._upload_queue.enqueue(document)
            return True
        except Exception as exc:
            self._report_upload_failure(exc, "preview", operator_id)
            return False

    def _report_upload_failure(self, exc: Exception, stage: str, operator_id: str) -> None:
        code = exc.code if isinstance(exc, ExchangeApiError) else "collector_exception" if isinstance(exc, CollectorConfigError) else "upload_response_unverified"
        reason = {
            "upload_network_failed": "upload_network_error",
            "upload_http_failed": "upload_http_error",
            "upload_rejected": "upload_rejected_by_server",
            "upload_response_unverified": "upload_response_invalid",
        }.get(code, getattr(exc, "reason", None))
        diagnostic_call(self._toolbox_diagnostics, "issue", code, stage, operator_id=operator_id,
                        exception=exc, reason=reason, errors=exc.blocking_errors if isinstance(exc, ExchangeApiError) else ())
        logger.bind(console=False).warning(f"AgentInfoCollector: v3 自动上报失败，文档已保存: {exc}")
        logger.warning(f"同步到 YuanHub 失败：{_sync_failure_reason(exc)}；本地记录已保存，可稍后重新同步")

    def _drain_upload_results(self) -> None:
        if self._upload_queue is None:
            return
        for result in self._upload_queue.take_results():
            pending = self._pending_uploads[result.record_id]
            synced = server_review = False
            if isinstance(result, UploadFailure):
                self._report_upload_failure(result.error, result.stage, pending["operator_id"])
            else:
                diagnostic_call(self._toolbox_diagnostics, "upload_response", result.payload, result.stage,
                                operator_id=pending["operator_id"], client_review_count=pending["client_review_count"])
                if result.stage == "preview":
                    if self.params.get("commit", True):
                        continue
                    diagnostic_call(self._toolbox_diagnostics, "issue", "upload_response_unverified", "commit",
                                    operator_id=pending["operator_id"], reason="upload_response_invalid")
                else:
                    synced = upload_response_verified(result.payload)
                    data = result.payload.get("data", result.payload) if isinstance(result.payload, dict) else {}
                    if isinstance(data, dict):
                        server_review = any(type(data.get(key)) is int and data[key] > 0 for key in ("partial", "review"))
                    if not synced:
                        logger.warning("同步到 YuanHub 的结果未能确认；本地记录已保存，可稍后重新同步")
            key = pending["key"]
            if self._latest_upload_ids.get(key) == result.record_id:
                progress = getattr(self, "_console_progress", {})
                local_review = progress.get(key, (False, bool(pending["client_review_count"])))[1]
                progress[key] = (synced, local_review or server_review)
                self._console_progress = progress
                if synced:
                    suffix = "（同步数据需核对）" if server_review else ""
                    logger.info(f"已同步 {pending['name']}{suffix}")
            del self._pending_uploads[result.record_id]

    def finish_uploads(self) -> None:
        if self._upload_queue is None:
            return

        def poll_stop():
            self._drain_upload_results()
            return self._should_stop()

        try:
            self._upload_queue.finish(poll_stop)
            self._drain_upload_results()
        finally:
            self._upload_queue = None
            self._pending_uploads.clear()


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
            ("密探采集扫描模式配置", ("scan_mode",)),
            ("密探采集诊断配置", ("failure_diagnostics", "toolbox_diagnostics")),
            ("密探采集对比模式配置", ("name_compare",)),
            ("密探采集命盘对比模式配置", ("disc_compare",)),
            ("密探采集上报配置", ("upload", "commit")),
            ("密探采集本地文件配置", ("output",)),
            ("密探采集单人配置", ("scan_single",)),
            ("密探采集起点配置", ("start_from_current",)),
            ("密探采集重试配置", ("max_retries",)),
            ("密探采集养成筛选配置", ("active_only",)),
            ("密探采集稀有度配置", ("min_rarity",)),
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
        min_rarity = params.get("min_rarity", 3)
        if type(min_rarity) is not int or min_rarity not in (3, 4, 5):
            logger.bind(console=False).warning(f"AgentInfoCollector: 非法 min_rarity={min_rarity!r}，按 3 处理")
            min_rarity = 3
        params["min_rarity"] = min_rarity
        reader = None
        settings = None
        diagnostics = start_diagnostics(context, params)
        if diagnostics is not None:
            params["_toolbox_diagnostics"] = diagnostics
        stage = "initialize"
        stopped = False
        success = False
        try:
            if params.get("disc_compare"):
                success = _AgentInfoReader(context, params).run_disc_comparison()
                return CustomAction.RunResult(success=success)
            if params.get("name_compare"):
                success = _AgentInfoReader(context, params).run_name_comparison()
                return CustomAction.RunResult(success=success)
            scan_range = "当前密探" if params.get("scan_single") else "从当前密探开始" if params.get("start_from_current") else "全部密探"
            if params.get("active_only"):
                scan_range += "，仅养成中密探"
            if min_rarity > 3:
                scan_range += "，仅金卡" if min_rarity == 5 else "，跳过蓝卡"
            scan_mode = "普通模式" if params.get("scan_mode", "enhanced") == "normal" else "增强模式"
            logger.info(f"开始采集密探信息（{scan_range} · {scan_mode}）")
            if params.get("active_only") and not params.get("upload"):
                raise CollectorConfigError("仅扫描养成中密探需要启用同步至 YuanHub；" + TOKEN_REQUIRED_HINT, "active_only_requires_sync")
            if params.get("upload"):
                stage = "account"
                settings = read_upload_settings(context)
                account = get_bound_account(settings)
                diagnostic_call(diagnostics, "protect", account.name, account.id)
                account_part = bound_account_report_filename(account)
                params["output"] = f"YuanHubMyBox-{account_part}.json"
                if params.get("active_only"):
                    try:
                        params["growth_states"] = read_growth_states(settings.base_url, settings.token, account.id)
                    except (ValueError, RuntimeError) as exc:
                        diagnostic_call(diagnostics, "issue", "annotations_failed", "annotations", exception=exc, reason="annotations_failed")
                        logger.bind(console=False).error(f"AgentInfoCollector: {exc}")
                        logger.error(f"采集中断：无法读取养成状态，{_sync_failure_reason(exc)}")
                        return CustomAction.RunResult(success=False)
                logger.bind(console=False).info(
                    "AgentInfoCollector: 已按 Token 绑定子账号选择本地报告 "
                    f"account_id={account.id!r}, output={params['output']!r}"
                )
            stage = "catalog"
            reader = _AgentInfoReader(context, params, upload_settings=settings)
            stage = "initialize"
            try:
                success = reader.run()
            finally:
                reader.finish_uploads()
            if success:
                progress = getattr(reader, "_console_progress", {})
                synced = sum(value[0] for value in progress.values())
                review = sum(value[1] for value in progress.values())
                skipped = len(getattr(reader, "_rarity_skipped", set()))
                summary = f"采集完成：共 {len(progress)} 位，已同步 {synced} 位，需核对 {review} 位"
                if min_rarity > 3:
                    summary += f"，跳过 {skipped} 位"
                logger.info(summary)
            if not success and reader._failure_diagnostics is not None and not reader._should_stop():
                with logger.contextualize(console=False):
                    reader._failure_diagnostics.save_failure("task_failed")
        except InterruptedError:
            stopped = True
            logger.bind(console=False).info("AgentInfoCollector: 任务已停止")
            logger.info("密探信息采集已停止")
            success = False
        except Exception as exc:
            diagnostic_call(diagnostics, "issue", "account_lookup_failed" if stage == "account" else "collector_exception",
                            None if reader is not None else stage, exception=exc,
                            reason="account_lookup_failed" if stage == "account" else getattr(exc, "reason", None))
            logger.bind(console=False).exception("AgentInfoCollector: 采集流程异常")
            reason = _sync_failure_reason(exc) if stage == "account" else _collection_failure_reason(exc, stage)
            logger.error(f"采集中断：{reason}")
            if reader is not None and reader._failure_diagnostics is not None and not reader._should_stop():
                with logger.contextualize(console=False):
                    reader._failure_diagnostics.save_failure("exception")
            success = False
        finally:
            try:
                tasker = getattr(context, "tasker", None)
                stopped = stopped or getattr(context, "stop", False) is True or getattr(tasker, "stopping", False) is True or getattr(tasker, "running", None) is False
            except Exception:
                pass
            if reader is not None:
                try:
                    stopped = stopped or reader._should_stop() is True
                except Exception:
                    pass
            diagnostic_call(diagnostics, "finish", success, stopped=stopped)
        return CustomAction.RunResult(success=success)
