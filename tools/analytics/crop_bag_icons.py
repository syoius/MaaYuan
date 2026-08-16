#!/usr/bin/env python3
"""Convert standalone circular heart-paper icons into MaaY -bag templates.

The inventory UI renders the source icon at approximately 108x108 on a
720x1280 screenshot.  Existing screenshot-derived templates correspond to
the rectangle x=18, y=20, w=70, h=58 in that normalized icon.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
NORMALIZED_SIZE = 108
CROP_BOX = (18, 20, 70, 58)


def expand_inputs(values: Sequence[str], recursive: bool) -> list[Path]:
    found: set[Path] = set()
    for raw in values:
        matches = [Path(item) for item in glob.glob(raw, recursive=recursive)]
        if not matches:
            matches = [Path(raw)]
        for path in matches:
            if path.is_dir():
                iterator = path.rglob("*") if recursive else path.glob("*")
                found.update(
                    item.resolve()
                    for item in iterator
                    if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
                )
            elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                found.add(path.resolve())
    return sorted(found)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("无法解码图片")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    encoded.tofile(path)


def convert_icon(image: np.ndarray) -> tuple[np.ndarray, str]:
    height, width = image.shape[:2]
    if width != height:
        raise ValueError(f"源图必须为正方形，实际为 {width}x{height}")
    if width < CROP_BOX[2] or height < CROP_BOX[3]:
        raise ValueError(f"源图过小：{width}x{height}")

    if width > NORMALIZED_SIZE:
        interpolation = cv2.INTER_AREA
        interpolation_name = "INTER_AREA"
    elif width < NORMALIZED_SIZE:
        interpolation = cv2.INTER_CUBIC
        interpolation_name = "INTER_CUBIC"
    else:
        interpolation = cv2.INTER_NEAREST
        interpolation_name = "none"

    if width == NORMALIZED_SIZE:
        normalized = image
    else:
        normalized = cv2.resize(
            image,
            (NORMALIZED_SIZE, NORMALIZED_SIZE),
            interpolation=interpolation,
        )

    x, y, crop_width, crop_height = CROP_BOX
    cropped = normalized[y : y + crop_height, x : x + crop_width].copy()
    if cropped.shape[:2] != (crop_height, crop_width):
        raise RuntimeError(f"裁剪尺寸异常：{cropped.shape}")

    alpha_min = int(cropped[:, :, 3].min())
    if alpha_min < 250:
        raise ValueError(f"目标裁剪区域含明显透明像素，最小 alpha={alpha_min}")
    cropped[:, :, 3] = 255
    return cropped, interpolation_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将独立圆形心纸图标转换为统一的 70x58 -bag 模板。"
    )
    parser.add_argument("inputs", nargs="+", help="图片、目录或 glob")
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描输入目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有同名模板")
    return parser


def run(args: argparse.Namespace) -> int:
    inputs = expand_inputs(args.inputs, args.recursive)
    if not inputs:
        print("没有找到可处理图片。", file=sys.stderr)
        return 2

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    converted_count = 0

    for path in inputs:
        record = {"source": str(path), "status": "", "output": ""}
        try:
            image = read_image(path)
            source_height, source_width = image.shape[:2]
            cropped, interpolation = convert_icon(image)
            output_path = output / f"{path.stem}-bag.png"
            record.update(
                {
                    "source_size": [source_width, source_height],
                    "normalized_size": [NORMALIZED_SIZE, NORMALIZED_SIZE],
                    "crop_box": list(CROP_BOX),
                    "interpolation": interpolation,
                    "output": str(output_path),
                }
            )
            if output_path.exists() and not args.overwrite:
                record["status"] = "exists-not-overwritten"
            else:
                write_png(output_path, cropped)
                record["status"] = "converted"
                converted_count += 1
        except Exception as exc:
            record.update({"status": "skipped", "reason": str(exc)})
        records.append(record)

    report = {
        "normalized_size": [NORMALIZED_SIZE, NORMALIZED_SIZE],
        "crop_box": list(CROP_BOX),
        "records": records,
    }
    report_path = output / "crop-bag-icons-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    skipped = sum(record["status"] == "skipped" for record in records)
    protected = sum(
        record["status"] == "exists-not-overwritten" for record in records
    )
    print(
        f"完成：发现 {len(inputs)} 张，转换 {converted_count} 张，"
        f"跳过 {skipped} 张，保留已有 {protected} 张。"
    )
    print(f"报告：{report_path}")
    return 0 if converted_count or protected else 1


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
