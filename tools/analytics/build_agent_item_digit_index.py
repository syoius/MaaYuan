#!/usr/bin/env python3
"""Build the tiny 0-9 glyph index used by AgentItemRecognition."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "agent" / "agent-item-digit-index.npz"
FEATURE_SIZE = (20, 28)  # width, height
GLYPH_CONTENT_SIZE = (14, 22)  # width, height
COUNT_BOX = (-35, 19, 95, 44)  # relative to the matched portrait center
BINARY_THRESHOLD = 170
BADGE_THRESHOLD = 190
BADGE_CLOSE_KERNEL = (9, 5)  # width, height


@dataclass(frozen=True)
class TrainingCell:
    source: str
    center: tuple[int, int]
    value: str
    binary_threshold: int = BINARY_THRESHOLD
    badge_threshold: int = BADGE_THRESHOLD


def _grid_cells(
    source: str,
    row_tops: list[list[int]],
    values: list[list[int]],
) -> list[TrainingCell]:
    columns = (118, 286, 454, 622)
    cells: list[TrainingCell] = []
    for row, (tops, row_values) in enumerate(zip(row_tops, values)):
        if len(tops) != 4 or len(row_values) != 4:
            raise ValueError(f"训练网格必须为四列: {source}, row={row}")
        for column, center_x in enumerate(columns):
            cells.append(
                TrainingCell(
                    source=source,
                    center=(center_x, int(tops[column]) + 34),
                    value=str(row_values[column]),
                )
            )
    return cells


def _bag_3_1_cells(
    source: str,
    row_centers: list[int],
    values: list[list[int | None]],
) -> list[TrainingCell]:
    columns = (118, 279, 440, 601)
    cells: list[TrainingCell] = []
    for row, (center_y, row_values) in enumerate(zip(row_centers, values)):
        if len(row_values) != 4:
            raise ValueError(f"训练网格必须为四列: {source}, row={row}")
        for center_x, value in zip(columns, row_values):
            if value is None:
                continue
            cells.append(
                TrainingCell(
                    source=source,
                    center=(center_x, center_y),
                    value=str(value),
                    binary_threshold=165,
                    badge_threshold=180,
                )
            )
    return cells


# Fully visible item rows and the current cultivation-tab screenshots provide
# the 1.0 inventory glyphs.  The final cells add the 0.9 Shouchun rendering.
TRAINING_CELLS = [
    *_grid_cells(
        "tools/analytics/debug/bag-3-2/4.png",
        [
            [304, 304, 304, 303],
            [493, 493, 493, 493],
            [681, 681, 681, 681],
            [870, 870, 870, 869],
        ],
        [
            [63, 4, 17, 3],
            [57, 17, 2, 13],
            [10, 10, 10, 7],
            [25, 4, 1, 44],
        ],
    ),
    *_grid_cells(
        "tools/analytics/debug/bag-3-2/5.png",
        [
            [306, 306, 305, 305],
            [494, 495, 495, 495],
            [683, 683, 683, 684],
            [874, 872, 872, 872],
        ],
        [
            [20, 69, 1, 1],
            [20, 4, 8, 5],
            [5, 12, 29, 20],
            [11, 7, 10, 6],
        ],
    ),
    TrainingCell(
        "tools/analytics/debug/bag-3-2/2.png", (118, 530), "184"
    ),
    TrainingCell(
        "tools/analytics/debug/bag-3-2/2.png", (622, 718), "28"
    ),
    # Fully visible ordinary-item rows.  Besides improving the shared glyph
    # shapes, these samples cover three- to five-digit inventory quantities.
    *_grid_cells(
        "tools/analytics/bag-items-sample-1.png",
        [
            [355, 355, 356, 355],
            [543, 543, 543, 543],
            [731, 732, 732, 732],
            [920, 920, 919, 922],
        ],
        [
            [15, 46, 110, 27],
            [30, 147, 216, 30],
            [2, 34, 1, 56],
            [150, 233, 413, 48],
        ],
    ),
    *_grid_cells(
        "tools/analytics/bag-items-sample-2.png",
        [
            [323, 323, 322, 322],
            [512, 512, 513, 512],
            [700, 700, 700, 700],
            [889, 890, 890, 889],
        ],
        [
            [1043, 3556, 111, 63],
            [66, 7520, 43, 12365],
            [556, 8820, 356, 337],
            [131, 366, 21, 526],
        ],
    ),
    *_grid_cells(
        "tools/analytics/bag-items-sample-3.png",
        [
            [318, 317, 317, 317],
            [506, 506, 507, 508],
            [695, 695, 695, 695],
            [883, 884, 884, 884],
        ],
        [
            [117, 474, 14, 45],
            [184, 588, 833, 10],
            [1, 1, 239, 19],
            [186, 174, 6, 182],
        ],
    ),
    *_grid_cells(
        "tools/analytics/bag-items-sample-4.png",
        [[328, 328, 328, 328]],
        [[173, 102, 46, 21]],
    )[:3],
    # The cultivation-tab inventory uses a slightly softer digit edge than the
    # earlier item samples.  Threshold 165 preserves leading 1/6/9 strokes.
    *_bag_3_1_cells(
        "tools/analytics/debug/bag-3-1/1.png",
        [389, 577, 765, 954],
        [
            [2, None, 1072, 6561],
            [91, 32, 99, 36],
            [33, 129, 3, 8],
            [2, 223, 152, 161],
        ],
    ),
    *_bag_3_1_cells(
        "tools/analytics/debug/bag-3-1/2.png",
        [329, 518, 707, 895, 1085],
        [
            [2, 223, 152, 161],
            [520, 2201, 211, 94],
            [5, 2755, 65, 2200],
            [340, 1530, 226, 1097],
            [None, 562, 681, 1607],
        ],
    ),
    *_bag_3_1_cells(
        "tools/analytics/debug/bag-3-1/3.png",
        [449, 637, 826, 1015],
        [
            [161, 221, 28, 301],
            [42, 160, 431, 11],
            [4, 7, 198, 43],
            [296, 142, 37, 142],
        ],
    ),
    *_bag_3_1_cells(
        "tools/analytics/debug/bag-3-1/4.png",
        [388, 576],
        [
            [296, 142, 37, 142],
            [79, 11, 175, None],
        ],
    ),
    # This softer leading 7 is otherwise almost indistinguishable from 1 at
    # threshold 165; keep the full 786 badge as a production-shaped sample.
    TrainingCell(
        "tools/analytics/digit-samples/jiezhuping-786.png",
        (35, -19),
        "786",
        binary_threshold=165,
        badge_threshold=200,
    ),
    TrainingCell("tools/analytics/shouchun-sample-1.png", (98, 871), "1"),
    TrainingCell("tools/analytics/shouchun-sample-1.png", (228, 872), "1"),
    TrainingCell("tools/analytics/shouchun-sample-1.png", (359, 871), "1"),
]


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取训练截图: {path}")
    return image


def normalize_glyph(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    target_width, target_height = GLYPH_CONTENT_SIZE
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        mask,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    feature_width, feature_height = FEATURE_SIZE
    canvas = np.zeros((feature_height, feature_width), dtype=np.uint8)
    x = (feature_width - resized_width) // 2
    y = (feature_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized

    vector = canvas.astype(np.float32).reshape(-1) / 255.0
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        raise ValueError("数字字形没有有效前景")
    return np.ascontiguousarray(vector / norm), canvas


def extract_glyphs(
    image: np.ndarray,
    center: tuple[float, float],
    count_box: tuple[int, int, int, int] = COUNT_BOX,
    binary_threshold: int = BINARY_THRESHOLD,
    badge_threshold: int = BADGE_THRESHOLD,
) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    offset_x, offset_y, width, height = count_box
    x = int(round(center[0] + offset_x))
    y = int(round(center[1] + offset_y))
    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        return [], (x, y, width, height)

    gray = cv2.cvtColor(image[y : y + height, x : x + width], cv2.COLOR_BGR2GRAY)
    dark_badge = np.where(gray < badge_threshold, 255, 0).astype(np.uint8)
    kernel_width, kernel_height = BADGE_CLOSE_KERNEL
    badge = cv2.morphologyEx(
        dark_badge,
        cv2.MORPH_CLOSE,
        np.ones((kernel_height, kernel_width), dtype=np.uint8),
    )
    binary = np.where((gray >= binary_threshold) & (badge > 0), 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    # Anti-aliasing occasionally splits one digit (notably 5) into vertically
    # disconnected components.  Keep small fragments, merge overlapping
    # x-ranges, then select the right-aligned run of baseline-aligned glyphs.
    groups: list[dict] = []
    for component in range(1, count):
        glyph_x, glyph_y, glyph_width, glyph_height, area = (
            int(value) for value in stats[component]
        )
        touches_edge = (
            glyph_x == 0
            or glyph_y == 0
            or glyph_x + glyph_width == width
            or glyph_y + glyph_height == height
        )
        if touches_edge:
            continue
        if not (
            glyph_y >= 11
            and 1 <= glyph_width <= 18
            and 5 <= glyph_height <= 30
            and 8 <= area <= 250
        ):
            continue
        target = next(
            (
                group
                for group in groups
                if min(glyph_x + glyph_width, group["x"] + group["width"])
                - max(glyph_x, group["x"])
                >= 1
            ),
            None,
        )
        if target is None:
            groups.append(
                {
                    "x": glyph_x,
                    "y": glyph_y,
                    "width": glyph_width,
                    "height": glyph_height,
                    "area": area,
                    "components": [component],
                }
            )
            continue
        x1 = min(target["x"], glyph_x)
        y1 = min(target["y"], glyph_y)
        x2 = max(target["x"] + target["width"], glyph_x + glyph_width)
        y2 = max(target["y"] + target["height"], glyph_y + glyph_height)
        target.update(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            area=target["area"] + area,
        )
        target["components"].append(component)

    groups = [
        group
        for group in groups
        if 1 <= group["width"] <= 18
        and 12 <= group["height"] <= 20
        and 8 <= group["area"] <= 250
    ]
    groups.sort(key=lambda group: group["x"])
    if not groups:
        return [], (x, y, width, height)

    run = [groups[-1]]
    for previous in reversed(groups[:-1]):
        following = run[-1]
        gap = following["x"] - (previous["x"] + previous["width"])
        previous_bottom = previous["y"] + previous["height"]
        following_bottom = following["y"] + following["height"]
        if gap > 5 or abs(previous_bottom - following_bottom) > 5:
            break
        run.append(previous)
    run.reverse()

    glyphs: list[np.ndarray] = []
    for group in run:
        glyph_x, glyph_y = group["x"], group["y"]
        glyph_width, glyph_height = group["width"], group["height"]
        region = labels[
            glyph_y : glyph_y + glyph_height,
            glyph_x : glyph_x + glyph_width,
        ]
        glyphs.append(
            np.where(np.isin(region, group["components"]), 255, 0).astype(np.uint8)
        )
    return glyphs, (x, y, width, height)


def build_index(output: Path) -> dict:
    image_cache: dict[str, np.ndarray] = {}
    labels: list[str] = []
    features: list[np.ndarray] = []
    glyphs: list[np.ndarray] = []
    sources: list[str] = []
    source_values: list[str] = []

    for cell in TRAINING_CELLS:
        image = image_cache.get(cell.source)
        if image is None:
            image = read_bgr(REPO_ROOT / cell.source)
            image_cache[cell.source] = image
        extracted, _ = extract_glyphs(
            image,
            cell.center,
            binary_threshold=cell.binary_threshold,
            badge_threshold=cell.badge_threshold,
        )
        if len(extracted) != len(cell.value):
            raise RuntimeError(
                f"数字分割数量不一致: {cell.source}, center={cell.center}, "
                f"value={cell.value}, glyphs={len(extracted)}"
            )
        for label, glyph in zip(cell.value, extracted):
            feature, normalized = normalize_glyph(glyph)
            labels.append(label)
            features.append(feature)
            glyphs.append(normalized)
            sources.append(cell.source)
            source_values.append(cell.value)

    labels_array = np.asarray(labels)
    features_array = np.ascontiguousarray(np.stack(features), dtype=np.float32)
    glyphs_array = np.ascontiguousarray(np.stack(glyphs), dtype=np.uint8)

    predictions: list[str] = []
    scores: list[float] = []
    for index, feature in enumerate(features_array):
        candidates = np.flatnonzero(np.arange(len(labels_array)) != index)
        similarities = features_array[candidates] @ feature
        best = int(candidates[int(np.argmax(similarities))])
        predictions.append(str(labels_array[best]))
        scores.append(float(similarities.max()))
    incorrect = [
        f"{index}:{expected}->{actual}"
        for index, (expected, actual) in enumerate(zip(labels, predictions))
        if expected != actual
    ]
    correct = len(labels) - len(incorrect)
    if correct / len(labels) < 0.95:
        raise RuntimeError(f"数字模板留一验证失败: {incorrect[:10]}")

    per_digit = {
        digit: int(np.count_nonzero(labels_array == digit)) for digit in "0123456789"
    }
    if any(count < 2 for count in per_digit.values()):
        raise RuntimeError(f"每个数字至少需要两个训练字形: {per_digit}")

    source_hash = hashlib.sha256()
    for source in sorted(image_cache):
        source_hash.update(source.encode("utf-8"))
        source_hash.update((REPO_ROOT / source).read_bytes())
    metadata = {
        "schema": "maay.agent-item-digit-index",
        "version": 1,
        "sample_count": len(labels),
        "per_digit": per_digit,
        "feature_size": list(FEATURE_SIZE),
        "glyph_content_size": list(GLYPH_CONTENT_SIZE),
        "count_box": list(COUNT_BOX),
        "binary_threshold": BINARY_THRESHOLD,
        "badge_threshold": BADGE_THRESHOLD,
        "badge_close_kernel": list(BADGE_CLOSE_KERNEL),
        "validation": {
            "leave_one_out_correct": correct,
            "leave_one_out_total": len(labels),
            "min_score": min(scores),
            "errors": incorrect,
        },
        "source_sha256": source_hash.hexdigest(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(1, dtype=np.int32),
        labels=labels_array,
        features=features_array,
        glyphs=glyphs_array,
        feature_size=np.asarray(FEATURE_SIZE, dtype=np.int32),
        glyph_content_size=np.asarray(GLYPH_CONTENT_SIZE, dtype=np.int32),
        count_box=np.asarray(COUNT_BOX, dtype=np.int32),
        binary_threshold=np.asarray(BINARY_THRESHOLD, dtype=np.int32),
        badge_threshold=np.asarray(BADGE_THRESHOLD, dtype=np.int32),
        badge_close_kernel=np.asarray(BADGE_CLOSE_KERNEL, dtype=np.int32),
        source_files=np.asarray(sources),
        source_values=np.asarray(source_values),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成角色物品数量的 0-9 字形索引。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        metadata = build_index(args.output.resolve())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(f"已生成：{args.output.resolve()}")
    print(f"字形数：{metadata['sample_count']}")
    print(f"每个数字：{metadata['per_digit']}")
    print(
        "留一验证："
        f"{metadata['validation']['leave_one_out_correct']}/"
        f"{metadata['validation']['leave_one_out_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
