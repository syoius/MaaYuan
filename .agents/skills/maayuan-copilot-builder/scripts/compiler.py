# -*- coding: utf-8 -*-
"""
MaaYuan Copilot Compiler
Translates high-level strategy IR into deterministic MaaYuan v3 Copilot pipelines.

CORE SEMANTICS:
- Game Round: Actual in-game turn verified by MaaFramework PureNum OCR (ROI [641, 50, 43, 27]).
- Action Segment: Batches of action opportunities within the SAME game round.
  E.g. certain boss/character mechanics ("牢狱", "再动", "追击") grant full-team actions
  without changing the turn counter.
- Execution continuity: All segments within a round are compiled into a single continuous chain
  (回合N行动1 -> 回合N行动2 -> ... -> 回合N行动M).
  Only after the last action of the last segment completes does the flow proceed to the next game round.
- Local Order: Action orders are scoped to their respective segment. Multiple segments may each
  restart numbering from 1.
"""

from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("MaaYuanCompiler")


class ValidationError(Exception):
    """Raised when an IR fails semantic validation."""
    pass


class NoteTier(str, Enum):
    TIER_A = "Tier A: Native Executable"
    TIER_B = "Tier B: Branch/Retry Guard"
    TIER_C = "Tier C: Human Guidance"
    TIER_D = "Tier D: Record Only (Never An Action)"


ACTION_TOKEN_MAP = {
    # ↑ is the guide's up arrow and means cast the skill. The character 个 is a
    # misread of that arrow and must not compile.
    "↑": "ultimate",
    "W": "ultimate",
    "w": "ultimate",
    "大": "ultimate",
    "↓": "defense",
    "S": "defense",
    "s": "defense",
    "防": "defense",
    "下": "defense",
    "A": "normal",
    "a": "normal",
    "普": "normal",
    "O": "normal",
    "o": "normal",
    "M": "normal",
    "m": "normal",
    "圈": "switch_form",  # 点该卡牌下方的形态切换按钮，不会触发普攻（用户 2026-10-06 确认）
    "Q": "target_left",
    "q": "target_left",
    "左": "target_left",
    "E": "target_right",
    "e": "target_right",
    "右": "target_right",
    "C": "check_orange",
    "c": "check_orange"
}

# One local order may expand into several physical actions: "3圈后↑".
# "后" only joins actions that share the preceding number. A new digit starts a new order,
# so "4A9A" and "2↑8↓" stay two orders.
_CHAIN_CONNECTORS = frozenset("后")
_ORDER_RE = re.compile(r"(\d+)")


def _action_chain_from_chars(chars: List[str]) -> List[Dict[str, Any]]:
    chain = []
    for index, ch in enumerate(chars):
        kind = ACTION_TOKEN_MAP.get(ch)
        if not kind:
            raise ValidationError(f"unknown action character '{ch}'")
        chain.append({"kind": kind, "subaction_index": index, "token": ch})
    return chain


def parse_cell_tokens(slot: int, cell_text: str, round_num: int) -> List[Dict[str, Any]]:
    """
    Parses a strategy table cell into action objects.

    A digit starts a local order. The glyphs after it, joined by 后, belong to that
    same order. Example: 3圈后↑ is order 3, SwitchForm then Ultimate (↑ means cast skill).
    A later digit starts another order, so 4A9A and 2↑8↓ stay multi-order cells.
    """
    if not isinstance(cell_text, str):
        raise ValidationError(f"Round {round_num} slot {slot}: cell content must be string, got {type(cell_text)}")

    cleaned = re.sub(r"\s+", "", cell_text)
    if cleaned in ("", "-", "/", "——", "无", "空"):
        return []

    actions: List[Dict[str, Any]] = []
    i = 0
    length = len(cleaned)
    while i < length:
        match = _ORDER_RE.match(cleaned, i)
        if not match:
            raise ValidationError(
                f"Round {round_num} slot {slot}: no valid action token found in cell '{cell_text}'"
            )
        order = int(match.group(1))
        if order <= 0:
            raise ValidationError(f"Round {round_num} slot {slot}: action order must be positive integer, got {order}")
        i = match.end()
        chars: List[str] = []
        while i < length and not cleaned[i].isdigit():
            ch = cleaned[i]
            if ch in _CHAIN_CONNECTORS:
                if not chars:
                    raise ValidationError(
                        f"Round {round_num} slot {slot}: '后' in '{cell_text}' must follow an action"
                    )
                i += 1
                if i >= length or cleaned[i].isdigit() or cleaned[i] in _CHAIN_CONNECTORS:
                    raise ValidationError(
                        f"Round {round_num} slot {slot}: '后' in '{cell_text}' must be followed by an action"
                    )
                continue
            if ch not in ACTION_TOKEN_MAP:
                raise ValidationError(
                    f"Round {round_num} slot {slot}: cell '{cell_text}' contains unrecognized characters: '{cleaned[i:]}'"
                )
            chars.append(ch)
            i += 1
        if not chars:
            raise ValidationError(f"Round {round_num} slot {slot}: no valid action token found in cell '{cell_text}'")
        chain = _action_chain_from_chars(chars)
        actions.append({
            "order": order,
            "slot": slot,
            "kind": chain[0]["kind"],
            "raw_token": f"{order}{''.join(chars)}",
            "raw_cell": cleaned,
            "actions": chain,
        })
    return actions


def _compiled_action_spec(kind: str, tpl: Dict[str, Any], special_ops: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Gesture stored for a later stall replay. Callers must not rebuild coordinates."""
    if kind == "switch_form":
        #从档位配置读取，不在此处硬编码模板/ROI/阈值。
        sf = ((special_ops or {}).get("switch_form") or {})
        #该函数拿不到具体站位（回放时才知道），故给全部站位的模板作为候选。
        templates: List[str] = []
        for slot_cfg in sf.values():
            if isinstance(slot_cfg, dict):
                for t in slot_cfg.get("recognition", {}).get("param", {}).get("template", []):
                    if t not in templates:
                        templates.append(t)
        if not templates:
            templates = ["copilot/lb_l2h.png", "copilot/lb_h2l.png"]
        roi = [15, 1072, 690, 95]
        for slot_cfg in sf.values():
            if isinstance(slot_cfg, dict):
                roi = slot_cfg.get("recognition", {}).get("param", {}).get("roi", roi)
                break
        return {
            "action": "SwitchForm",
            "template": templates,
            "roi": roi,
        }
    spec: Dict[str, Any] = {"action": tpl.get("action")}
    for key in ("target", "begin", "end", "duration"):
        if key in tpl:
            spec[key] = tpl[key]
    return spec


def expand_action_chains(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten one local order into consecutive physical actions, preserving source order."""
    expanded: List[Dict[str, Any]] = []
    for act in actions:
        chain = act.get("actions") or [{"kind": act.get("kind"), "subaction_index": 0}]
        for index, sub in enumerate(chain):
            kind = sub.get("kind") if isinstance(sub, dict) else sub
            sub_index = sub.get("subaction_index", index) if isinstance(sub, dict) else index
            item = {k: v for k, v in act.items() if k != "actions"}
            item["kind"] = kind
            item["source_order"] = act.get("order")
            item["subaction_index"] = sub_index
            item["raw_cell"] = act.get("raw_cell") or act.get("raw_token")
            expanded.append(item)
    return expanded


# Notes that look like orders but describe something the pipeline cannot do.
# They are echoed into the job details for the human, never compiled into a node.
# Keep this list narrow: 练度/等级/冷却 requirements are ordinary Tier C advice and
# must not be swept in here.
NOTE_RECORD_ONLY_KEYWORDS = (
    # 敌方 AI 倾向，作业无法选择敌方目标
    "默念", "别打", "不要打", "尽量不打", "避开", "仇恨",
    # 手动重来的操作，不属于本局流水
    "回溯", "重来",
    # 开打前的配装准备，与 1~5 号位站位无关
    "带星石", "星石", "配装", "装备准备",
)


def classify_note(note_text: str) -> Dict[str, Any]:
    """
    Classifies a strategy note into Tier A, Tier B, or Tier C.
    Character names in notes are preserved as human guidance without character lookup.
    """
    text = note_text.strip()
    # Tier A: Native executable targets or checks
    if any(k in text for k in ["集火左", "打左", "左侧目标", "切换左", "1Q", "Q"]):
        return {
            "tier": NoteTier.TIER_A,
            "action_type": "target_left",
            "supported": True,
            "raw": text
        }
    if any(k in text for k in ["集火右", "打右", "右侧目标", "切换右", "1E", "E"]):
        return {
            "tier": NoteTier.TIER_A,
            "action_type": "target_right",
            "supported": True,
            "raw": text
        }
    if any(k in text for k in ["检测橙星", "橙星检测", "刷橙星"]):
        return {
            "tier": NoteTier.TIER_A,
            "action_type": "check_orange",
            "supported": True,
            "raw": text
        }

    # Tier B: Timeline branch / retry guard
    if any(k in text for k in ["暴击", "重开", "掉出", "龙气", "龙脉", "被控", "眩晕", "退场", "阵亡", "鹦鹉"]):
        is_native = any(k in text for k in ["阵亡", "退场", "鹦鹉", "橙星"]) and not any(k in text for k in ["暴击", "龙气", "龙脉", "眩晕"])
        return {
            "tier": NoteTier.TIER_B,
            "supported": is_native,
            "raw": text,
            "guard_type": "native" if is_native else "unsupported_native_guard"
        }

    # Tier D: record-only. These read like instructions but are NOT commands the
    # pipeline can execute, so they must never turn into a node or a warning --
    # only into a line in the job's details for the human reader.
    #
    # Confirmed by the user 2026-10-06 for these three shapes:
    #   - 默念/倾向类「默念别打诸葛亮」: the enemy's AI decides its target; the
    #     copilot cannot pick which enemy to hit.
    #   - 回溯回合: a manual "rewind and retry" step the user performs on failure.
    #   - 带星石/配装: pre-fight loadout, unrelated to the 1~5 slot layout.
    #
    # Deliberately NOT included: 练度/等级/冷却数值. Those stay Tier C, they are
    # ordinary strategy advice (test_notes_three_tier_classification pins this).
    if any(k in text for k in NOTE_RECORD_ONLY_KEYWORDS):
        return {
            "tier": NoteTier.TIER_D,
            "supported": True,
            "generates_action": False,
            "raw": text
        }

    # Tier C: Human guidance / strategy advice
    return {
        "tier": NoteTier.TIER_C,
        "supported": True,
        "raw": text
    }


def _count_action_orders(cells: Dict[str, Any]) -> int:
    """
    How many distinct action orders this segment really has.

    NOT the cell count: one cell may carry several orders ("1↓2↑" is two actions
    in one cell), which is exactly why the 金乌枝 round 9 legitimately reaches
    order 6 with only 5 cells. Count the orders themselves.
    """
    orders: Set[int] = set()
    for raw in cells.values():
        if not isinstance(raw, str):
            continue
        text = re.sub(r"\s+", "", raw)
        if text in ("", "-", "/", "——", "无", "空"):
            continue
        for chunk in _ORDER_RE.findall(text):
            orders.add(int(chunk))
    return len(orders)


def _check_orders_within_actor_count(
    labelled: Dict[int, Tuple[str, str]],
    cells: Dict[str, Any],
    round_num: int,
    seg_label: str,
) -> None:
    """
    Reject a table whose labelled order leaves a hole in the sequence.

    A cell like {"1": "1A", "2": "6B"} looks harmless (no duplicates) but cannot
    be right: the orders present are 1 and 6, so 2~5 belong to nobody. Left alone
    it would compile into a chain that silently skips four slots.
    """
    if not labelled:
        return
    worst = max(labelled)
    actual = _count_action_orders(cells)
    if worst > actual:
        raise ValidationError(
            f"Round {round_num}{seg_label}: action order {worst} exceeds the number of "
            f"action orders in this segment ({actual}); orders={sorted(labelled)} — "
            f"some slots are unaccounted for"
        )


def assign_missing_orders(cells: Dict[str, Any], round_num: int, seg_label: str = "") -> Dict[str, Any]:
    """
    Fill in action order for cells that omit it.

    Strategy tables frequently number only some characters. Two patterns occur:

    1. "只写前几个": orders 1..k are labelled, the rest are plain symbols
       (e.g. {'2':'1↓','3':'2A','1':'↓','4':'↓','5':'↓'}). The unlabelled
       characters fill the remaining order slots **left to right**.
    2. "某人必须最后": order k is explicitly assigned to a late character while
       earlier slots are unlabelled; those still fill left to right into the
       gaps, leaving order k untouched.

    Rules (confirmed by the user 2026-10-06):
    - A labelled cell keeps its number, ALWAYS. Never renumber it.
    - Unlabelled cells take the leftover order numbers in ascending slot order.
    - A cell left empty ("", null, "-", "要退场" already stripped by the author)
      consumes no order number at all.

    Returns a new cells dict where every non-empty cell starts with its order.
    """
    labelled: Dict[int, Tuple[str, str]] = {}
    unlabelled: List[Tuple[str, str]] = []

    for slot_key in sorted(cells.keys(), key=lambda s: str(s)):
        raw = cells[slot_key]
        if not isinstance(raw, str):
            continue
        text = re.sub(r"\s+", "", raw)
        if text in ("", "-", "/", "——", "无", "空"):
            continue  # empty cell consumes no order
        m = _ORDER_RE.match(text)
        if m:
            order = int(m.group(1))
            if order in labelled:
                # 措辞保留 "Duplicate action order"，与 validate_ir 里的同名校验一致，
                # 两处触发同一类问题，只是本处在补序阶段就能定位到具体站位。
                raise ValidationError(
                    f"Round {round_num}{seg_label}: Duplicate action order {order} is used by both "
                    f"slot {labelled[order][0]} and slot {slot_key}"
                )
            labelled[order] = (slot_key, text)
        else:
            unlabelled.append((slot_key, text))

    if not unlabelled:
        _check_orders_within_actor_count(labelled, cells, round_num, seg_label)
        return dict(cells)

    total = len(labelled) + len(unlabelled)
    _check_orders_within_actor_count(labelled, cells, round_num, seg_label)

    used = set(labelled)
    available = [i for i in range(1, total + 1) if i not in used]
    if len(available) < len(unlabelled):
        raise ValidationError(
            f"Round {round_num}{seg_label}: not enough free order slots for "
            f"{len(unlabelled)} unlabelled cell(s); labelled={sorted(used)}"
        )

    result: Dict[str, Any] = dict(cells)
    for (slot_key, text), order in zip(unlabelled, available):
        logger.info(
            "Round %s%s: slot %s has no order label -> auto-assigned order %d (left to right)",
            round_num, seg_label, slot_key, order,
        )
        result[slot_key] = f"{order}{text}"
    return result


def normalize_round_segments(r_info: Dict[str, Any], round_num: int) -> List[Dict[str, Any]]:
    """
    Normalizes a round definition into a list of segments.
    Supports both explicit multi-segment rounds:
        {"round": 12, "segments": [{"label": "12回合", "cells": ...}, {"label": "牢狱", "cells": ...}]}
    and backwards-compatible single-segment rounds:
        {"round": 1, "cells": {...}}
    """
    if "segments" in r_info and isinstance(r_info["segments"], list):
        if len(r_info["segments"]) == 0:
            raise ValidationError(f"Round {round_num}: 'segments' list cannot be empty")
        segments = r_info["segments"]
    elif "cells" in r_info or "actions" in r_info:
        segments = [{
            "label": f"第{round_num}回合",
            "cells": r_info.get("cells"),
            "actions": r_info.get("actions"),
            "notes": r_info.get("notes", [])
        }]
    else:
        raise ValidationError(f"Round {round_num}: must contain 'segments', 'cells', or 'actions'")

    # Auto-fill missing action orders here so every downstream stage (validation,
    # action flattening, doc generation) sees fully-numbered cells and needs no
    # special-casing of its own.
    for seg in segments:
        if isinstance(seg.get("cells"), dict):
            label = seg.get("label") or ""
            seg["cells"] = assign_missing_orders(seg["cells"], round_num, label)

    return segments


def validate_ir(ir: Dict[str, Any]) -> None:
    """
    Validates IR semantics strictly.
    Checks:
    - title and stage_name
    - non-empty rounds
    - slot in 1..5
    - explicit order >= 1
    - no duplicate orders WITHIN THE SAME SEGMENT
    - no empty rounds or empty segments
    """
    if not isinstance(ir, dict):
        raise ValidationError("IR must be a dictionary")

    title = ir.get("title")
    if not title or not isinstance(title, str):
        raise ValidationError("IR 'title' must be a non-empty string")

    stage_name = ir.get("stage_name")
    if not stage_name or not isinstance(stage_name, str):
        raise ValidationError("IR 'stage_name' must be a non-empty string")

    rounds = ir.get("rounds")
    if not isinstance(rounds, list) or len(rounds) == 0:
        raise ValidationError("IR 'rounds' must be a non-empty list")

    for r_idx, r in enumerate(rounds, start=1):
        round_num = r.get("round")
        if not isinstance(round_num, int) or round_num <= 0:
            raise ValidationError(f"Round at index {r_idx} has invalid 'round' number: {round_num}")

        segments = normalize_round_segments(r, round_num)
        total_actions_in_round = 0

        for s_idx, seg in enumerate(segments, start=1):
            seg_label = seg.get("label", f"段{s_idx}")
            actions = []
            if "cells" in seg and isinstance(seg["cells"], dict):
                for slot_key, cell_val in seg["cells"].items():
                    try:
                        slot = int(slot_key)
                    except ValueError:
                        raise ValidationError(f"Round {round_num} segment '{seg_label}': invalid slot key '{slot_key}'")
                    if slot < 1 or slot > 5:
                        raise ValidationError(f"Round {round_num} segment '{seg_label}': slot {slot} out of range [1, 5]")
                    parsed = parse_cell_tokens(slot, cell_val, round_num)
                    actions.extend(parsed)
            elif "actions" in seg and isinstance(seg["actions"], list):
                actions = seg["actions"]
            else:
                raise ValidationError(f"Round {round_num} segment '{seg_label}': must contain 'cells' or 'actions'")

            if len(actions) == 0:
                raise ValidationError(f"Round {round_num} segment '{seg_label}': segment has 0 valid actions")

            # Validate orders WITHIN THIS SEGMENT (duplicate orders across different segments are valid)
            seen_orders_in_seg: Set[int] = set()
            for act in actions:
                order = act.get("order")
                if order is None or not isinstance(order, int) or order <= 0:
                    raise ValidationError(f"Round {round_num} segment '{seg_label}': action missing valid 'order': {act}")
                slot = act.get("slot")
                if slot is None or not isinstance(slot, int) or slot < 1 or slot > 5:
                    raise ValidationError(f"Round {round_num} segment '{seg_label}': action missing valid 'slot' [1-5]: {act}")
                kind = act.get("kind")
                if not kind or kind not in ("normal", "ultimate", "defense", "target_left", "target_right", "check_orange", "switch_form"):
                    raise ValidationError(f"Round {round_num} segment '{seg_label}': invalid action 'kind': {kind}")
                for sub in act.get("actions") or []:
                    sub_kind = sub.get("kind") if isinstance(sub, dict) else sub
                    if sub_kind not in ("normal", "ultimate", "defense", "target_left", "target_right", "check_orange", "switch_form"):
                        raise ValidationError(f"Round {round_num} segment '{seg_label}': invalid chain kind: {sub_kind}")

                if order in seen_orders_in_seg:
                    raise ValidationError(f"Round {round_num} segment '{seg_label}': Duplicate action order {order} detected on slot {slot}")
                seen_orders_in_seg.add(order)

            total_actions_in_round += len(actions)

        if total_actions_in_round == 0:
            raise ValidationError(f"Round {round_num}: round has 0 total valid actions")


def _load_profile(profile_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a runtime profile from references/runtime_profiles/<id>.json.

    That JSON is the single source of truth for coordinates, ROIs, thresholds and
    special operations. Keep it authoritative rather than duplicating values in
    DEFAULT_RUNTIME_PROFILE -- drift between the two caused the compiler to reject
    valid jobs ("档位配置缺少 special_operations.switch_form") when only the JSON
    had been updated.

    Falls back to the built-in dict when the file is unavailable so the compiler
    never hard-fails on a missing profile.
    """
    profiles_dir = Path(__file__).resolve().parent.parent / "references" / "runtime_profiles"
    if profile_id is None:
        # 默认取目录里唯一的 profile；若有多份，取带 default 标记或名字排序第一份
        candidates = sorted(profiles_dir.glob("*.json")) if profiles_dir.is_dir() else []
        if not candidates:
            return dict(DEFAULT_RUNTIME_PROFILE)
        preferred = [p for p in candidates if "default" in p.stem.lower()]
        target = preferred[0] if preferred else candidates[0]
    else:
        target = profiles_dir / f"{profile_id}.json"

    if not target.is_file():
        return dict(DEFAULT_RUNTIME_PROFILE)

    try:
        with open(target, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("档位配置 %s 读取失败(%s)，回退内置默认档位", target, exc)
        return dict(DEFAULT_RUNTIME_PROFILE)

    # JSON 缺字段时用内置值兜底，保证结构完整
    merged = dict(DEFAULT_RUNTIME_PROFILE)
    for key, value in loaded.items():
        if key in ("slot_coordinates", "special_operations", "recognition_rois") and isinstance(value, dict):
            base = dict(merged.get(key, {}))
            base.update(value)
            merged[key] = base
        else:
            merged[key] = value
    return merged


# Built-in fallback runtime profile (720x1280 portrait baseline).
# The JSON file under references/runtime_profiles/ is the SINGLE SOURCE OF TRUTH —
# this dict is only a last-resort fallback so the compiler still works if the
# profile file is missing. Loading the file (see _load_profile) keeps the two from
# drifting apart, which previously caused "档位配置缺少 special_operations.switch_form".
DEFAULT_RUNTIME_PROFILE = {
    "display": {"target_width": 720, "target_height": 1280, "dpi": 320},
    "interaction_baseline": {"y_card_baseline": 1060},
    "slot_coordinates": {
        "1": {
            "normal": {"action": "Click", "target": [56, 1060, 5, 5], "post_delay": 6000, "focus": "行动:1号位普攻"},
            "ultimate": {"action": "Swipe", "begin": [77, 1060, 10, 1], "end": [77, 670, 10, 1], "duration": 800, "post_delay": 10000, "focus": "行动:1号位大招"},
            "defense": {"action": "Swipe", "begin": [73, 1060, 1, 1], "end": [73, 1258, 1, 1], "duration": 800, "post_delay": 6000, "focus": "行动:1号位防御"}
        },
        "2": {
            "normal": {"action": "Click", "target": [180, 1060, 5, 5], "post_delay": 6000, "focus": "行动:2号位普攻"},
            "ultimate": {"action": "Swipe", "begin": [220, 1060, 1, 1], "end": [225, 668, 1, 1], "duration": 800, "post_delay": 10000, "focus": "行动:2号位大招"},
            "defense": {"action": "Swipe", "begin": [221, 1060, 1, 1], "end": [221, 1251, 1, 1], "duration": 800, "post_delay": 6000, "focus": "行动:2号位防御"}
        },
        "3": {
            "normal": {"action": "Click", "target": [357, 1060, 5, 5], "post_delay": 6000, "focus": "行动:3号位普攻"},
            "ultimate": {"action": "Swipe", "begin": [357, 1060, 1, 1], "end": [357, 714, 1, 1], "duration": 800, "post_delay": 10000, "focus": "行动:3号位大招"},
            "defense": {"action": "Swipe", "begin": [357, 1060, 1, 1], "end": [357, 1237, 1, 1], "duration": 800, "post_delay": 6000, "focus": "行动:3号位防御"}
        },
        "4": {
            "normal": {"action": "Click", "target": [496, 1060, 5, 5], "post_delay": 6000, "focus": "行动:4号位普攻"},
            "ultimate": {"action": "Swipe", "begin": [496, 1060, 1, 1], "end": [496, 679, 1, 1], "duration": 800, "post_delay": 10000, "focus": "行动:4号位大招"},
            "defense": {"action": "Swipe", "begin": [496, 1060, 1, 1], "end": [496, 1258, 1, 1], "duration": 800, "post_delay": 6000, "focus": "行动:4号位防御"}
        },
        "5": {
            "normal": {"action": "Click", "target": [646, 1060, 5, 5], "post_delay": 6000, "focus": "行动:5号位普攻"},
            "ultimate": {"action": "Swipe", "begin": [646, 1060, 1, 1], "end": [642, 700, 1, 1], "duration": 800, "post_delay": 10000, "focus": "行动:5号位大招"},
            "defense": {"action": "Swipe", "begin": [646, 1060, 1, 1], "end": [646, 1258, 1, 1], "duration": 800, "post_delay": 6000, "focus": "行动:5号位防御"}
        }
    },
    "special_operations": {
        "target_left": {"text_doc": "左侧目标", "action": "Click", "target": [154, 648, 1, 1], "post_delay": 2000, "duration": 800, "focus": "行动:切换左侧目标"},
        "target_right": {"text_doc": "右侧目标", "action": "Click", "target": [603, 413, 18, 21], "post_delay": 2000, "duration": 800, "focus": "行动:切换右侧目标"}
    }
}


def compile_ir_to_copilot(
    ir: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
    verification_mode: str = "off"
) -> Dict[str, Any]:
    """
    Compiles verified IR into MaaYuan v3 Copilot format.
    Key guarantees:
    - Segments within a game round are sorted internally by local order, then flattened sequentially.
    - Continuous execution chain: 回合N行动1 -> ... -> 回合N行动M.
    - Segment transition: The last action of segment K links directly to the first action of segment K+1.
      No PureNum turn check is inserted between segments.
    - Turn transition: Only the final action of the round links to victory-check and the next game round.
    - Multi-action cells (e.g. 4A9A) expanded and chronologically ordered within their segment.
    - Operators field is optional metadata; execution strictly relies on slot coordinates.
    """
    validate_ir(ir)

    prof = profile if profile is not None else _load_profile()
    slot_coords = prof.get("slot_coordinates", DEFAULT_RUNTIME_PROFILE["slot_coordinates"])
    special_ops = prof.get("special_operations", DEFAULT_RUNTIME_PROFILE["special_operations"])

    title = ir["title"]
    stage_name = ir["stage_name"]
    raw_operators = ir.get("operators")
    opers = [{"name": op} for op in raw_operators] if raw_operators else []
    custom_delays = ir.get("delays", {"attack": 6000, "ultimate": 10000, "defense": 6000})

    actions: Dict[str, Any] = {}
    unsupported_guards: List[str] = []
    strategy_notes: List[str] = []
    record_only_notes: List[str] = []
    observer_metadata: Dict[str, Any] = {}

    rounds_data = ir["rounds"]
    total_rounds = len(rounds_data)

    for r_idx, r_info in enumerate(rounds_data, start=1):
        round_num = r_info["round"]
        round_key = f"检测回合{round_num}"

        segments = normalize_round_segments(r_info, round_num)

        # 1. Flatten all segments in top-to-bottom order
        all_round_actions: List[Dict[str, Any]] = []
        for s_idx, seg in enumerate(segments, start=1):
            seg_label = seg.get("label", f"段{s_idx}")
            seg_actions: List[Dict[str, Any]] = []
            if "cells" in seg and isinstance(seg["cells"], dict):
                for s_key, c_val in seg["cells"].items():
                    slot = int(s_key)
                    seg_actions.extend(parse_cell_tokens(slot, c_val, round_num))
            elif "actions" in seg and isinstance(seg["actions"], list):
                seg_actions = list(seg["actions"])

            # Sort actions strictly by local order within this segment
            seg_actions_sorted = sorted(seg_actions, key=lambda a: a["order"])
            for act in seg_actions_sorted:
                act["_segment_label"] = seg_label
            all_round_actions.extend(expand_action_chains(seg_actions_sorted))

        # 2. Gather notes from round and all segments (with de-duplication)
        raw_notes = []
        if "notes" in r_info and isinstance(r_info["notes"], list):
            raw_notes.extend(r_info["notes"])
        if "segments" in r_info and isinstance(r_info["segments"], list):
            for seg in r_info["segments"]:
                if "notes" in seg and isinstance(seg["notes"], list):
                    raw_notes.extend(seg["notes"])
        round_notes = []
        seen_note_texts = set()
        for n in raw_notes:
            if n not in seen_note_texts:
                seen_note_texts.add(n)
                round_notes.append(n)

        tier_a_target: Optional[str] = None
        has_orange_check = False

        for n_str in round_notes:
            classified = classify_note(n_str)
            tier = classified["tier"]
            if tier == NoteTier.TIER_A:
                act_type = classified.get("action_type")
                if act_type in ("target_left", "target_right"):
                    tier_a_target = act_type
                elif act_type == "check_orange":
                    has_orange_check = True
            elif tier == NoteTier.TIER_B:
                if not classified.get("supported", False):
                    warn_msg = f"第{round_num}回合: {n_str} (当前 MaaYuan 原生图像识别不支持该条件分支，需人工留意)"
                    unsupported_guards.append(warn_msg)
                    logger.warning(f"[MaaYuanCompiler] {warn_msg}")
            elif tier == NoteTier.TIER_D:
                # 只记录，不生成动作，也不当作「不支持的条件分支」报警。
                record_only_notes.append(f"第{round_num}回合: {n_str}")
                logger.info(f"[MaaYuanCompiler] 第{round_num}回合旁注仅记录，不生成动作: {n_str}")
            else:
                strategy_notes.append(f"第{round_num}回合: {n_str}")

        # 3. Entry node: target switch or action 1
        first_node_key = f"回合{round_num}行动1"
        if tier_a_target:
            target_node_key = f"回合{round_num}切换目标"
            target_spec = special_ops[tier_a_target].copy()
            target_spec["next"] = ["史子眇sp", first_node_key]
            actions[target_node_key] = target_spec
            round_entry_node = target_node_key
        else:
            round_entry_node = first_node_key

        # 4. Turn recognition node (PureNum) - EXACTLY ONE per game round!
        round_next = ["史子眇sp", round_entry_node]
        if has_orange_check or r_info.get("restart_if_no_orange"):
            round_next.insert(0, f"第{round_num}回合橙星检测")

        actions[round_key] = {
            "recognition": "Custom",
            "custom_recognition": "PureNum",
            "custom_recognition_param": {
                "roi": [641, 50, 43, 27],
                "expected": str(round_num)
            },
            "text_doc": f"回合{round_num}",
            "focus": f"当前：第{round_num}回合",
            "next": round_next,
            "on_error": ["回合停滞恢复"],
            "timeout": 20000,
            "post_delay": 4000
        }

        # 5. Build seamless continuous action chain across all segments
        total_actions = len(all_round_actions)
        for a_idx, act in enumerate(all_round_actions, start=1):
            act_node_key = f"回合{round_num}行动{a_idx}"
            slot = act["slot"]
            kind = act["kind"]
            order = act["order"]
            seg_label = act.get("_segment_label", "")
            # 只有 switch_form 会用到兜底坐标；其余动作走固定站位坐标，不需要。
            fallback_target = None

            if kind == "switch_form":
                # 切形态 = 点该卡牌下方的形态切换按钮（不会触发普攻）。
                #
                # 为什么不能沿用金乌枝作业里那套整排 ROI + 0.8 阈值（2026-10-06 实测教训）：
                # 模板 lb_l2h/lb_h2l 是在吕布站1 号位时截的，背景带进了 1 号位的卡面。
                # 吕布改站 4 号位后，模板与实际画面相似度只有 0.66~0.74，稳定够不到 0.8，
                # 重试 12 次后整个任务被判定失败。引擎定位其实是准的（命中框就在按钮上），
                # 纯粹是外观差异。
                #
                # 现在三重保障：
                #   1. 模板数组首位放按站位实测截的新模板（lb_form_s<N>.png），命中率高；
                #   2. 保留旧模板兜底，两种形态状态都能认；
                #   3. ROI 只框本卡牌下方，阈值降到 0.65。
                # 见references/runtime_profiles/<profile>.json 的 special_operations.switch_form。
                sf = special_ops.get("switch_form", {}).get(str(slot))
                if sf is None:
                    raise ValidationError(
                        f"档位配置缺少 special_operations.switch_form.{slot}，无法编译切形态动作"
                    )
                tpl = {
                    "recognition": dict(sf["recognition"]),
                    "action": {"type": "Click", "param": {}},
                    "post_delay": sf.get("post_delay", 1500),
                }
                # 识别失败的兜底：直接点按钮中心。放在 next 的第一位之前由
                # switch_form_fallback 节点承载，见下方 next 组装。
                fallback_target = sf.get("fallback_target")
                orig_post_delay = sf.get("post_delay", 1500)
                act_type_name = "SwitchForm"
                doc_suffix = "切形态"
            else:
                tpl = slot_coords[str(slot)][kind].copy()
                if kind == "normal":
                    orig_post_delay = custom_delays.get("attack", 6000)
                    act_type_name = "Attack"
                    doc_suffix = "普"
                elif kind == "ultimate":
                    orig_post_delay = custom_delays.get("ultimate", 10000)
                    act_type_name = "Ultimate"
                    doc_suffix = "大"
                else:
                    orig_post_delay = custom_delays.get("defense", 6000)
                    act_type_name = "Defense"
                    doc_suffix = "下"

            # Record observer metadata mapping
            observer_metadata[act_node_key] = {
                "round": round_num,
                "segment": seg_label,
                "action_index": a_idx,
                "source_order": act.get("source_order", order),
                "subaction_index": act.get("subaction_index", 0),
                "slot": slot,
                "action_type": act_type_name,
                "raw_cell": act.get("raw_cell") or act.get("raw_token"),
                "action_spec": _compiled_action_spec(kind, tpl, special_ops),
            }

            tpl["text_doc"] = f"{slot}{doc_suffix}"
            tpl["focus"] = f"第{round_num}回合[{seg_label}] 第{order}动(总第{a_idx}动): {slot}号位{doc_suffix}"

            eff_mode = ir.get("verification_mode", verification_mode)
            if eff_mode == "shadow":
                action_def = {"action": tpl.get("action", "Click")}
                if "target" in tpl:
                    action_def["target"] = tpl["target"]
                if "begin" in tpl:
                    action_def["begin"] = tpl["begin"]
                if "end" in tpl:
                    action_def["end"] = tpl["end"]
                if "duration" in tpl:
                    action_def["duration"] = tpl["duration"]

                shadow_param = {
                    "slot": slot,
                    "action_type": act_type_name,
                    "action_def": action_def,
                    "original_post_delay": orig_post_delay,
                    "round": round_num,
                    "segment": seg_label,
                    "action_index": a_idx,
                    "node_name": act_node_key
                }

                tpl["recognition"] = "DirectHit"
                tpl["action"] = "Custom"
                tpl["custom_action"] = "ShadowObserver"
                tpl["custom_action_param"] = json.dumps(shadow_param, ensure_ascii=False)
                tpl["post_delay"] = 0
                tpl.pop("target", None)
                tpl.pop("begin", None)
                tpl.pop("end", None)
                tpl.pop("duration", None)
            else:
                tpl["post_delay"] = orig_post_delay

            # Next pointers
            next_list = ["史子眇sp"]
            if a_idx < total_actions:
                # Continuous transition: next action in same round (even across segments!)
                next_list.append(f"回合{round_num}行动{a_idx + 1}")
            else:
                # FINAL ACTION of the game round: proceed to victory check or next round
                if r_idx < total_rounds:
                    next_round_num = rounds_data[r_idx]["round"]
                    next_list.append("抄作业战斗胜利-check")
                    next_list.append(f"检测回合{next_round_num}")
                else:
                    # Final round of whole strategy
                    next_list.append("抄作业战斗胜利")

            # 切形态节点额外挂一个坐标兜底分支：模板识别超时后，直接按按钮中心点一下。
            # 没有这层兜底时，一次识别失败会让整个任务在20 秒后被判失败（2026-10-06 实测）。
            if kind == "switch_form" and fallback_target:
                fb_node = f"{act_node_key}兜底"
                actions[fb_node] = {
                    "recognition": "DirectHit",
                    "action": "Click",
                    "target": fallback_target,
                    "post_delay": orig_post_delay,
                    "focus": f"{tpl.get('focus', '')}（模板未命中，按坐标兜底）",
                    "next": list(next_list),
                }
                next_list = [fb_node] + next_list

            tpl["next"] = next_list
            actions[act_node_key] = tpl

    # 6. Common navigation and lifecycle nodes
    actions["抄作业点左上角重开"] = {
        "recognition": "TemplateMatch",
        "template": "back.png",
        "green_mask": True,
        "threshold": 0.5,
        "roi": [6, 8, 123, 112],
        "action": "Click",
        "pre_delay": 500,
        "post_delay": 2000,
        "next": ["[JumpBack]抄作业确定左上角重开", "抄作业找到关卡-OCR"],
        "focus": "正在尝试点左上角重开",
        "timeout": 20000
    }
    actions["作业信息"] = {
        "focus": f"[color:#1890FF] 该作业由 MaaYuan AI Skill 编译生成：{title} [/color]"
    }
    actions["抄作业胜利后继续"] = {
        "focus": "战斗胜利，尝试继续",
        "next": ["抄作业准备开始战斗"]
    }
    actions["抄作业自定义延时"] = {
        "attack_delay": str(custom_delays.get("attack", 6000)),
        "ult_delay": str(custom_delays.get("ultimate", 10000)),
        "defense_delay": str(custom_delays.get("defense", 6000))
    }
    actions["抄作业找到关卡-OCR"] = {
        "recognition": "OCR",
        "expected": title,
        "roi": [0, 249, 720, 1030],
        "action": "Click",
        "pre_delay": 2000,
        "next": ["抄作业战斗开始"],
        "timeout": 20000
    }

    # 7. Assemble documentation details
    doc_lines = []
    base_details = ir.get("details", "").strip()
    if base_details:
        doc_lines.append(base_details)

    if unsupported_guards:
        doc_lines.append("")
        doc_lines.append("⚠️【条件分支与重试提示 (需人工关注)】")
        for g in unsupported_guards:
            doc_lines.append(f"- {g}")

    if strategy_notes:
        doc_lines.append("")
        doc_lines.append("💡【策略建议与旁注】")
        for sn in strategy_notes:
            doc_lines.append(f"- {sn}")

    if record_only_notes:
        doc_lines.append("")
        doc_lines.append("📝【仅记录，不影响本作业执行】")
        for ro in record_only_notes:
            doc_lines.append(f"- {ro}")

    full_details = "\n".join(doc_lines).strip()

    # 8. Final v3 Copilot structure
    copilot_doc = {
        "version": 3,
        "stage_name": stage_name,
        "difficulty": 0,
        "minimum_required": "v4.0.0",
        "level_meta": {
            "stage_id": stage_name,
            "level_id": f"custom/{stage_name}",
            "name": title,
            "cat_one": "自定义",
            "cat_two": title,
            "cat_three": "无",
            "width": 0,
            "height": 0
        },
        "doc": {
            "title": title,
            "details": full_details
        },
        "opers": opers,
        "groups": [],
        "actions": actions,
        "_observer_metadata": observer_metadata
    }

    return copilot_doc


def _resolve_sidecar_path(output_path: str) -> Path:
    """
    Decide where the observer metadata sidecar goes.

    MaaFramework scans EVERY *.json under resource/base/pipeline as a pipeline
    definition. The sidecar is a bare node map without recognition/action, so if
    it lands there the whole resource bundle fails to load and the client shows
    "资源加载失败". pipeline_metadata/ exists exactly to keep such files out of
    that scan.

    The job file itself may legitimately live in copilot-cache/ (which the client
    reads separately), so keying the decision off "pipeline" in the path is not
    enough: the sidecar must always go to a pipeline_metadata directory, next to
    whichever resource root the job was exported under.
    """
    out_p = Path(output_path)
    parts = list(out_p.parts)
    # Find the resource root segment ("resource") so the sidecar can be placed
    # under <resource>/base/pipeline_metadata/copilot/.
    try:
        res_idx = parts.index("resource")
    except ValueError:
        sidecar_dir = Path(*parts[:-1])
    else:
        # metadata is always bucketed under pipeline_metadata/copilot/ so that no
        # part of the original path can leak it back into base/pipeline/.
        sidecar_dir = Path(*parts[: res_idx + 1], "base", "pipeline_metadata", "copilot")
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    return sidecar_dir / f"{out_p.stem}.observer_metadata.json"


def _sync_to_base_pipeline(copilot_doc: Dict[str, Any], output_path: str) -> Optional[Path]:
    """
    Mirror the job into resource/base/pipeline/copilot/ as a bare node map.

    Why this is mandatory (learned the hard way on 2026-10-06):

    MaaFramework scans EVERY *.json under resource/base/pipeline as a pipeline
    definition. A job file has scalar top-level keys (version, difficulty, opers,
    level_meta...) which blow up parsing with "value is not object [key=difficulty]"
    and take the whole resource bundle down with them.

    So the job body must reach the engine as a *bare node map* (just `actions`),
    while the full job document stays in copilot-cache/ for the UI to list.

    Two hard constraints, both of which produce "资源加载失败":

    1. Strip every node that already exists in copilot_config.json. MaaFramework
       rejects duplicate keys ("key already exists"), so shadowing a base node
       is never allowed -- not even to disable it with {"enabled": false}.
    2. Remove any OTHER job file already sitting in that directory. The client
       writes the active job there itself, and two job files collide on ~19
       shared node names (检测回合N, 回合N行动M, 作业信息, 抄作业*...).

    Returns the path written, or None when the job was not exported under a
    resource root (nothing to mirror onto).

    NOTE: these constraints are version-specific. If the client is upgraded and
    jobs start failing to load (or silently stop running), do NOT guess -- re-derive
    the layout with references/resource_load_troubleshooting.md, which documents
    how to read the client's own output files and dll strings to establish where
    it expects job nodes to live.
    """
    out_p = Path(output_path)
    parts = list(out_p.parts)
    try:
        res_idx = parts.index("resource")
    except ValueError:
        return None

    base_copilot_dir = Path(*parts[: res_idx + 1], "base", "pipeline", "copilot")
    config_path = base_copilot_dir / "copilot_config.json"

    actions = dict(copilot_doc.get("actions") or {})
    if not actions:
        return None

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            base_nodes = set(json.load(f).keys())
        # Drop shadowing nodes rather than silently producing a broken bundle.
        for key in [k for k in actions if k in base_nodes]:
            del actions[key]

    base_copilot_dir.mkdir(parents=True, exist_ok=True)
    # Evict other jobs -- two flat job files cannot coexist (duplicate node keys).
    for stale in base_copilot_dir.glob("*.json"):
        if stale.name in ("copilot_config.json", out_p.name):
            continue
        try:
            stale.unlink()
        except OSError:
            pass

    flat_path = base_copilot_dir / out_p.name
    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(actions, f, indent=2, ensure_ascii=False)
    return flat_path


def export_copilot_json(copilot_doc: Dict[str, Any], output_path: str, *, sync_pipeline: bool = False) -> None:
    """Export the complete job only; explicit sync is for isolated staging.

    A cache export must not evict the user's active job. The GUI extracts the
    embedded observer metadata and pipeline when the user activates this job.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(copilot_doc, f, indent=2, ensure_ascii=False)

    meta = copilot_doc.get("_observer_metadata")
    if sync_pipeline and meta:
        sidecar_path = _resolve_sidecar_path(output_path)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    if sync_pipeline:
        _sync_to_base_pipeline(copilot_doc, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile strategy IR to MaaYuan v3 Copilot pipeline")
    parser.add_argument("ir_file", help="Path to strategy IR JSON file")
    parser.add_argument("-o", "--output", help="Path to output Copilot JSON file", default=None)
    parser.add_argument(
        "--verification-mode",
        choices=["off", "shadow"],
        default="off",
        help="Combat action verification mode: 'off' (native Click/Swipe) or 'shadow' (ShadowObserver CustomAction)"
    )
    args = parser.parse_args()

    with open(args.ir_file, "r", encoding="utf-8") as f:
        ir_data = json.load(f)

    copilot = compile_ir_to_copilot(ir_data, verification_mode=args.verification_mode)

    out_file = args.output or f"compiled_{Path(args.ir_file).stem}.json"
    export_copilot_json(copilot, out_file)
    print(f"Successfully compiled {args.ir_file} to {out_file} (verification_mode={args.verification_mode})")
