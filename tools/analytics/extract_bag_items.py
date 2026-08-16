#!/usr/bin/env python3
r"""Extract ordinary bag-item templates until the first heart-paper item.

The OCR/controller plumbing and 720x1280 normalization are shared with
``extract_bag_portraits.py``.  Labels are processed in screenshot, row and
column order.  The item that first contains ``心纸``/``心紙`` and everything
after it are not exported.

Example (PowerShell):

    python tools/analytics/extract_bag_items.py `
        tools/analytics/bag-items-sample-*.png `
        -o tools/analytics/bag-items-extracted `
        --debug
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from extract_bag_portraits import (
    DEFAULT_MODEL,
    GRID_COLUMNS,
    TARGET_SIZE,
    MaaOcr,
    OcrToken,
    expand_inputs,
    merge_ocr_tokens,
    nearest_column,
    normalize_screenshot,
    normalize_text,
    read_image,
    safe_slug,
    write_png,
)


CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
DEFAULT_OUTPUT_SUFFIX = "-bag"


@dataclass
class ItemCandidate:
    source: str
    raw_text: str
    name: str
    ocr_score: float
    row: int
    column: int
    label_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    image: np.ndarray
    status: str = "candidate"
    output: str = ""

    def report_dict(self) -> dict:
        return {
            "source": self.source,
            "raw_text": self.raw_text,
            "name": self.name,
            "ocr_score": self.ocr_score,
            "row": self.row + 1,
            "column": self.column + 1,
            "label_box": list(self.label_box),
            "crop_box": list(self.crop_box),
            "status": self.status,
            "output": self.output,
        }


def is_item_label(token: OcrToken, args: argparse.Namespace) -> bool:
    """Reject count OCR and vertically merged decoration before row grouping."""

    text = normalize_text(token.text)
    return bool(
        args.label_min_y <= token.y <= args.label_max_y
        and args.label_min_height <= token.h <= args.label_max_height
        and CJK_PATTERN.search(text)
        and nearest_column(token.cx, args.column_tolerance) is not None
    )


def group_complete_rows(
    tokens: Sequence[OcrToken], args: argparse.Namespace
) -> list[list[OcrToken]]:
    """Return top-to-bottom rows containing one OCR label in every column."""

    lines: list[list[OcrToken]] = []
    for token in sorted(
        (item for item in tokens if is_item_label(item, args)),
        key=lambda item: (item.cy, item.x),
    ):
        line = next(
            (
                current
                for current in lines
                if abs(token.cy - sum(item.cy for item in current) / len(current))
                <= args.row_tolerance
            ),
            None,
        )
        if line is None:
            lines.append([token])
        else:
            line.append(token)

    complete: list[list[OcrToken]] = []
    for line in lines:
        by_column: dict[int, OcrToken] = {}
        for token in line:
            column_x = nearest_column(token.cx, args.column_tolerance)
            if column_x is None:
                continue
            column = GRID_COLUMNS.index(column_x)
            previous = by_column.get(column)
            if previous is None or token.score > previous.score:
                by_column[column] = token
        if len(by_column) == len(GRID_COLUMNS):
            complete.append([by_column[column] for column in range(len(GRID_COLUMNS))])

    complete.sort(key=lambda row: sum(token.cy for token in row) / len(row))
    return complete[: args.max_full_rows]


def crop_item(
    image: np.ndarray,
    source: Path,
    token: OcrToken,
    row: int,
    column: int,
    args: argparse.Namespace,
) -> ItemCandidate:
    name = normalize_text(token.text)
    center_x = GRID_COLUMNS[column]
    crop_x = int(round(center_x - args.crop_width / 2))
    crop_y = int(round(token.y - args.label_to_crop_top))
    crop_box = (crop_x, crop_y, args.crop_width, args.crop_height)
    height, width = image.shape[:2]
    if (
        crop_x < 0
        or crop_y < 0
        or crop_x + args.crop_width > width
        or crop_y + args.crop_height > height
    ):
        raise ValueError(f"裁切框超出截图: {crop_box}")

    cropped = image[
        crop_y : crop_y + args.crop_height,
        crop_x : crop_x + args.crop_width,
    ].copy()
    if cropped.shape[:2] != (args.crop_height, args.crop_width):
        raise ValueError(f"裁切尺寸异常: {cropped.shape}")
    return ItemCandidate(
        source=str(source),
        raw_text=token.text,
        name=name,
        ocr_score=token.score,
        row=row,
        column=column,
        label_box=(token.x, token.y, token.w, token.h),
        crop_box=crop_box,
        image=cropped,
    )


def unique_output_path(
    output: Path,
    name: str,
    suffix: str,
    used: set[Path],
    overwrite: bool,
) -> Path:
    base = safe_slug(name)
    candidate = output / f"{base}{suffix}.png"
    if candidate not in used and (overwrite or not candidate.exists()):
        used.add(candidate)
        return candidate
    number = 2
    while True:
        candidate = output / f"{base}-{number}{suffix}.png"
        if candidate not in used and not candidate.exists():
            used.add(candidate)
            return candidate
        number += 1


def draw_debug(
    image: np.ndarray,
    rows: Sequence[Sequence[OcrToken]],
    accepted: Sequence[ItemCandidate],
    marker: OcrToken | None,
) -> np.ndarray:
    canvas = image.copy()
    for row in rows:
        for token in row:
            color = (0, 0, 255) if token is marker else (0, 180, 0)
            cv2.rectangle(
                canvas,
                (token.x, token.y),
                (token.x + token.w, token.y + token.h),
                color,
                2,
            )
    for item in accepted:
        x, y, w, h = item.crop_box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 80, 0), 2)
    return canvas


def write_report(
    output: Path,
    records: Sequence[ItemCandidate],
    inputs: Sequence[Path],
    marker_record: dict | None,
    args: argparse.Namespace,
) -> None:
    rows = [item.report_dict() for item in records]
    manifest = {
        "inputs": [str(path) for path in inputs],
        "stop_marker": "心纸",
        "marker_found": marker_record is not None,
        "marker": marker_record,
        "target_size": list(TARGET_SIZE),
        "template_size": [args.crop_width, args.crop_height],
        "max_full_rows": args.max_full_rows,
        "items": rows,
    }
    with (output / "bag-items-extraction-report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    fields = [
        "status",
        "name",
        "ocr_score",
        "source",
        "row",
        "column",
        "raw_text",
        "label_box",
        "crop_box",
        "output",
    ]
    with (output / "bag-items-extraction-report.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按背包顺序提取普通道具 70x58 模板，遇到第一个心纸后停止。"
    )
    parser.add_argument("inputs", nargs="+", help="截图、目录或 glob")
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描输入目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同名输出")
    parser.add_argument("--debug", action="store_true", help="输出带定位框的调试图")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Maa OCR 模型目录")
    parser.add_argument("--ocr-threshold", type=float, default=0.25)
    parser.add_argument("--ocr-roi-top", type=int, default=300)
    parser.add_argument("--ocr-roi-bottom", type=int, default=1060)
    parser.add_argument("--label-min-y", type=int, default=380)
    parser.add_argument("--label-max-y", type=int, default=1050)
    parser.add_argument("--label-min-height", type=int, default=15)
    parser.add_argument("--label-max-height", type=int, default=36)
    parser.add_argument("--row-tolerance", type=float, default=16.0)
    parser.add_argument("--column-tolerance", type=int, default=58)
    parser.add_argument(
        "--max-full-rows",
        type=int,
        default=4,
        help="每张图只取顶部若干个四列完整行；默认 4，可避开底边截断行",
    )
    parser.add_argument("--crop-width", type=int, default=70)
    parser.add_argument("--crop-height", type=int, default=58)
    parser.add_argument("--label-to-crop-top", type=int, default=96)
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help="文件名后缀，默认 -bag；传空字符串可得到 <OCR名称>.png",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    screenshots = expand_inputs(args.inputs, args.recursive)
    screenshots.sort(
        key=lambda path: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))
        ]
    )
    if not screenshots:
        print("没有找到可处理的截图。", file=sys.stderr)
        return 2
    if args.max_full_rows < 1:
        raise ValueError("--max-full-rows 必须至少为 1")
    if args.ocr_roi_bottom <= args.ocr_roi_top:
        raise ValueError("--ocr-roi-bottom 必须大于 --ocr-roi-top")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    debug_dir = output / "debug"
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    ocr = MaaOcr(args.model.resolve(), args.ocr_threshold)
    accepted: list[ItemCandidate] = []
    used_outputs: set[Path] = set()
    marker_record: dict | None = None
    stop = False

    roi = (
        0,
        args.ocr_roi_top,
        TARGET_SIZE[0],
        args.ocr_roi_bottom - args.ocr_roi_top,
    )
    for image_number, path in enumerate(screenshots, start=1):
        print(f"[{image_number}/{len(screenshots)}] OCR: {path}")
        image = normalize_screenshot(read_image(path))
        tokens = merge_ocr_tokens(ocr.recognize(image, roi))
        rows = group_complete_rows(tokens, args)
        image_items: list[ItemCandidate] = []
        marker_token: OcrToken | None = None

        for row_number, row in enumerate(rows):
            for column, token in enumerate(row):
                normalized = normalize_text(token.text)
                if "心纸" in normalized:
                    marker_token = token
                    marker_record = {
                        "source": str(path),
                        "row": row_number + 1,
                        "column": column + 1,
                        "raw_text": token.text,
                        "normalized_text": normalized,
                        "score": token.score,
                        "box": [token.x, token.y, token.w, token.h],
                    }
                    stop = True
                    break

                item = crop_item(image, path, token, row_number, column, args)
                output_path = unique_output_path(
                    output,
                    item.name,
                    args.output_suffix,
                    used_outputs,
                    args.overwrite,
                )
                write_png(output_path, item.image)
                item.output = str(output_path)
                item.status = "written"
                accepted.append(item)
                image_items.append(item)
            if stop:
                break

        print(
            f"  完整名称行 {len(rows)}，新增模板 {len(image_items)}"
            + ("，已遇到心纸" if stop else "")
        )
        if args.debug:
            write_png(
                debug_dir / f"{path.stem}-debug.png",
                draw_debug(image, rows, image_items, marker_token),
            )
        if stop:
            break

    write_report(output, accepted, screenshots, marker_record, args)
    print(
        f"完成：写入 {len(accepted)} 个模板；"
        f"心纸停止标记{'已找到' if marker_record else '未找到'}。"
    )
    print(f"报告：{output / 'bag-items-extraction-report.json'}")
    return 0


def main() -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args())
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
