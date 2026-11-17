"""Draw per-cell recognition evidence on a diagnostic copy of a backpack ROI."""
from functools import lru_cache
import math
import os
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=1)
def _font():
    paths = (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/simhei.ttf",
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in paths:
        try:
            return ImageFont.truetype(str(path), 14), True
        except OSError:
            continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14), False
    except OSError:
        return ImageFont.load_default(), False


REASONS = {
    "match-score-below-threshold": ("模板分数不足", "low template score"),
    "coarse-score-below-threshold": ("粗匹配分数不足", "low coarse score"),
    "match-margin-below-threshold": ("候选分差不足", "ambiguous candidates"),
    "match-score-below-threshold-after-refine": ("模板分数不足", "low template score"),
    "refine-score-below-threshold": ("元素分数不足", "low element score"),
    "refine-margin-below-threshold": ("元素分差不足", "ambiguous element"),
    "refine-candidates-insufficient": ("元素候选不足", "missing element candidates"),
    "count-not-recognized": ("数量未确认", "count unresolved"),
    "count-too-close-to-roi-bottom": ("数量靠近底边", "count at bottom edge"),
}


def render_bag_page(crop, roi, page, index, other_index, entity_type, resolved=None):
    font, chinese = _font()
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    names = {}
    for catalogue in (index, other_index):
        if catalogue is not None:
            for raw_id, operator_id, name in zip(catalogue.agent_ids, catalogue.operator_ids, catalogue.operator_names):
                names[str(raw_id)] = names[str(operator_id)] = str(name)

    def label(raw_id):
        # Only catalogue names/IDs, never OCR text, exception messages or account data.
        raw_id = str(raw_id or "unknown")
        value = names.get(raw_id, raw_id) if chinese else raw_id
        return re.sub(r"[^\u3400-\u9fffA-Za-z0-9_·【】（）() .+-]", "", value)[:32]

    def score_text(value):
        return f"{float(value):.3f}" if value is not None and math.isfinite(float(value)) else "?"

    def fit(text, available):
        if draw.textbbox((0, 0), text, font=font)[2] <= available:
            return text
        while text and draw.textbbox((0, 0), text + "...", font=font)[2] > available:
            text = text[:-1]
        return text + "..."

    entries = {(entry["row"], entry["column"]): entry for entry in page.results + page.rejected}
    counts = {"ok": 0, "pending": 0, "other": 0}
    palette = {"ok": (30, 148, 84), "pending": (182, 105, 0), "other": (92, 100, 111)}
    column_width = width / max(1, page.column_count)
    for (row, column), center in sorted(page.centers.items()):
        entry = entries.get((row, column), {})
        chosen = (resolved or {}).get((row, column))
        identity = page.row_entities[row].get(column, "")
        if chosen is not None:
            status = "ok"
            state = ("详情确认" if chosen.get("_count_source") == "detail" else "识别成功") if chinese else "OK"
            entity_id = chosen.get("operator_id") if chosen["entity_type"] == "agent" else chosen.get("item_id")
            lines = [f"R{row + 1}C{column + 1} {state}", f"{label(entity_id)} x{chosen['count']}"]
        elif identity and not identity.startswith(entity_type + ":"):
            status = "other"
            lines = [f"R{row + 1}C{column + 1} " + ("非本分区" if chinese else "OTHER"), label(identity.split(":", 1)[1])]
        else:
            status = "pending"
            lines = [f"R{row + 1}C{column + 1} " + ("待确认" if chinese else "UNRESOLVED")]
        counts[status] += 1
        if status != "other":
            candidates = []
            if entry.get("refine_candidates"):
                candidates = [(candidate["item_id"], candidate["score"], "元素" if chinese else "element")
                              for candidate in sorted(entry["refine_candidates"], key=lambda item: item["score"], reverse=True)[:2]]
            else:
                candidates = [(entry.get("best_agent_id") or entry.get("item_id"), entry.get("match_score"), "模板" if chinese else "match"),
                              (entry.get("match_runner_up_agent_id"), entry.get("match_runner_up_score"), "模板" if chinese else "match")]
            for raw_id, score, kind in candidates:
                if raw_id:
                    lines.append(f"{label(raw_id)} {kind}{score_text(score)}")
            if status == "pending":
                reason = REASONS.get(entry.get("reason"), ("详情或数量未确认", "detail/count unresolved"))
                lines.append(reason[0 if chinese else 1])
        lines = lines[:4]
        color = palette[status]
        box = entry.get("cell_box") or (center[0] - 53, center[1] - 53, 106, 106)
        x, y, w, h = [int(round(value)) for value in box]
        left, top = max(0, x - roi[0]), max(0, y - roi[1])
        right, bottom = min(width - 1, x + w - roi[0]), min(height - 1, y + h - roi[1])
        if right <= left or bottom <= top:
            continue
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
        panel_width = max(1, int(column_width) - 6)
        panel_x = max(0, min(width - panel_width, int(center[0] - roi[0] - panel_width / 2)))
        panel_y = max(24, min(height - len(lines) * 17 - 3, bottom + 3))
        draw.rectangle((panel_x, panel_y, panel_x + panel_width, panel_y + len(lines) * 17 + 2), fill=(252, 252, 250))
        for n, line in enumerate(lines):
            draw.text((panel_x + 3, panel_y + n * 17), fit(line, panel_width - 6), font=font,
                      fill=color if n == 0 else (38, 38, 38))
    title = (f"本页 {len(page.centers)} 格 | 已确认 {counts['ok']} | 待确认 {counts['pending']} | 非本分区 {counts['other']}"
             if chinese else f"Cells {len(page.centers)} | OK {counts['ok']} | Unresolved {counts['pending']} | Other {counts['other']}")
    draw.rectangle((0, 0, width, 23), fill=(252, 252, 250))
    draw.text((5, 2), fit(title, width - 10), font=font, fill=(25, 25, 25))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), counts["pending"] > 0
