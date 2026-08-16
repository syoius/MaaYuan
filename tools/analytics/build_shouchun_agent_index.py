#!/usr/bin/env python3
"""Build the compact Shouchun heart-paper recognition index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATES = Path(__file__).resolve().parent / "bag-templates"
DEFAULT_OUTPUT = REPO_ROOT / "agent" / "shouchun-agent-index.npz"
DEFAULT_OPERATORS = REPO_ROOT / "agent" / "operators.json"
SCALES = (0.89, 0.90, 0.91)
FEATURE_SIZE = (48, 44)  # width, height
BASE_TEMPLATE_SIZE = (70, 58)  # width, height


def normalize_scales(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    scales = tuple(float(value) for value in values)
    if not scales:
        raise ValueError("至少需要一个模板缩放比例")
    keys: set[str] = set()
    for scale in scales:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"无效缩放比例: {scale}")
        key = template_array_key(scale)
        if key in keys:
            raise ValueError(f"缩放比例生成了重复模板键: {scale} -> {key}")
        keys.add(key)
    return scales


def template_array_key(scale: float) -> str:
    scale_percent = int(round(float(scale) * 100))
    # NPZ stores scales as float32, so allow its small round-trip error.
    if not math.isclose(float(scale) * 100, scale_percent, abs_tol=1e-4):
        raise ValueError(f"缩放比例最多支持两位小数: {scale}")
    return f"templates_{scale_percent:03d}"


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3]
        if int(alpha.min()) < 250:
            raise ValueError(f"模板包含透明像素: {path}")
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise ValueError(f"不支持的通道数: {path} {image.shape}")
    return np.ascontiguousarray(image)


def template_id(path: Path) -> str:
    suffix = "-bag.png"
    if not path.name.endswith(suffix):
        raise ValueError(f"模板文件名必须以 {suffix} 结尾: {path.name}")
    value = path.name[: -len(suffix)]
    if not value:
        raise ValueError(f"模板 ID 为空: {path.name}")
    return value


def load_operator_catalog(path: Path) -> dict[str, tuple[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        operators = json.load(file).get("OPERATORS", [])

    entries: list[tuple[dict, str]] = []
    counts: dict[str, int] = {}
    for operator in operators:
        aliases = re.findall(
            r"\b[a-z][a-z0-9_-]*\b", str(operator.get("alias", "")).lower()
        )
        if not aliases:
            continue
        base_slug = aliases[0]
        entries.append((operator, base_slug))
        counts[base_slug] = counts.get(base_slug, 0) + 1

    catalog: dict[str, tuple[str, str]] = {}
    for operator, base_slug in entries:
        slug = base_slug
        if counts[base_slug] > 1:
            parts = str(operator.get("id", "")).split("_")
            number = next((part for part in parts if part.isdigit()), "unknown")
            slug = f"{base_slug}-{number}"
        if slug in catalog:
            raise ValueError(f"operators.json 中模板 ID 仍有冲突: {slug}")
        catalog[slug] = (str(operator.get("id", "")), str(operator.get("name", "")))
    return catalog


def resize_templates(images: list[np.ndarray], scale: float) -> np.ndarray:
    resized = [
        cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        for image in images
    ]
    shapes = {image.shape for image in resized}
    if len(shapes) != 1:
        raise ValueError(f"缩放后的模板尺寸不一致: scale={scale}, shapes={shapes}")
    return np.stack(resized).astype(np.uint8, copy=False)


def build_gray_features(
    templates: np.ndarray,
    feature_size: tuple[int, int] = FEATURE_SIZE,
) -> np.ndarray:
    feature_width, feature_height = feature_size
    rows: list[np.ndarray] = []
    for image in templates:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        x = (width - feature_width) // 2
        y = (height - feature_height) // 2
        if x < 0 or y < 0:
            raise ValueError(
                f"模板小于特征区域: template={width}x{height}, "
                f"feature={feature_width}x{feature_height}"
            )
        vector = gray[y : y + feature_height, x : x + feature_width]
        vector = vector.astype(np.float32).reshape(-1)
        vector -= float(vector.mean())
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            raise ValueError("模板特征没有有效对比度")
        rows.append(vector / norm)
    return np.ascontiguousarray(np.stack(rows), dtype=np.float32)


def combined_source_digest(files: list[Path]) -> tuple[str, np.ndarray]:
    combined = hashlib.sha256()
    file_digests: list[str] = []
    for path in files:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        file_digests.append(digest)
        combined.update(path.name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(digest))
    return combined.hexdigest(), np.asarray(file_digests)


def validate_top1(
    ids: np.ndarray,
    features: np.ndarray,
    template_variants: dict[float, np.ndarray],
    feature_size: tuple[int, int],
) -> dict[str, dict[str, float | int]]:
    expected = np.arange(len(ids))
    result: dict[str, dict[str, float | int]] = {}
    for scale, templates in template_variants.items():
        queries = build_gray_features(templates, feature_size)
        scores = queries @ features.T
        order = np.argsort(scores, axis=1)
        predicted = order[:, -1]
        own = scores[expected, expected]
        runner_up = scores[expected, order[:, -2]]
        correct = int(np.count_nonzero(predicted == expected))
        if correct != len(ids):
            misses = [
                f"{ids[index]}->{ids[predicted[index]]}"
                for index in np.flatnonzero(predicted != expected)[:10]
            ]
            raise RuntimeError(
                f"scale={scale:.2f} Top-1 自检失败: "
                f"{correct}/{len(ids)}, {misses}"
            )
        result[f"{scale:.2f}"] = {
            "correct": correct,
            "total": len(ids),
            "min_own_score": float(own.min()),
            "mean_own_score": float(own.mean()),
            "min_margin": float((own - runner_up).min()),
            "mean_margin": float((own - runner_up).mean()),
        }
    return result


def build_index(
    templates_dir: Path,
    output: Path,
    operators_path: Path,
    scales: tuple[float, ...] | list[float] = SCALES,
    feature_scale: float | None = None,
) -> dict:
    scales = normalize_scales(scales)
    if feature_scale is None:
        feature_scale = scales[len(scales) // 2]
    feature_scale = float(feature_scale)
    feature_index = next(
        (
            index
            for index, scale in enumerate(scales)
            if math.isclose(scale, feature_scale, abs_tol=1e-6)
        ),
        None,
    )
    if feature_index is None:
        raise ValueError("feature-scale 必须是 scales 中的一个值")
    feature_scale = scales[feature_index]

    files = sorted(templates_dir.glob("*-bag.png"))
    if not files:
        raise ValueError(f"没有找到 *-bag.png: {templates_dir}")

    ids_list = [template_id(path) for path in files]
    if len(ids_list) != len(set(ids_list)):
        raise ValueError("模板 ID 不唯一")

    images = [read_bgr(path) for path in files]
    expected_shape = (BASE_TEMPLATE_SIZE[1], BASE_TEMPLATE_SIZE[0], 3)
    invalid_shapes = [
        (path.name, image.shape)
        for path, image in zip(files, images)
        if image.shape != expected_shape
    ]
    if invalid_shapes:
        raise ValueError(f"模板不是统一的 70x58 BGR: {invalid_shapes[:10]}")

    catalog = load_operator_catalog(operators_path)
    unmapped = sorted(set(ids_list) - set(catalog))
    if unmapped:
        raise ValueError(f"以下模板无法映射 operators.json: {unmapped}")

    agent_ids = np.asarray(ids_list)
    operator_ids = np.asarray([catalog[item][0] for item in ids_list])
    operator_names = np.asarray([catalog[item][1] for item in ids_list])
    source_files = np.asarray([path.name for path in files])
    source_digest, source_sha256 = combined_source_digest(files)

    variants: dict[float, np.ndarray] = {
        scale: resize_templates(images, scale) for scale in scales
    }
    features = build_gray_features(variants[feature_scale], FEATURE_SIZE)
    validation = validate_top1(agent_ids, features, variants, FEATURE_SIZE)
    template_keys = [template_array_key(scale) for scale in scales]

    metadata = {
        "schema": "maay.shouchun-agent-index",
        "version": 2,
        "agent_count": len(agent_ids),
        "color_order": "BGR",
        "base_template_size": list(BASE_TEMPLATE_SIZE),
        "scales": list(scales),
        "template_keys": template_keys,
        "feature": {
            "kind": "centered-grayscale-zero-mean-l2",
            "scale": feature_scale,
            "size": list(FEATURE_SIZE),
            "dimensions": int(features.shape[1]),
        },
        "source_digest_sha256": source_digest,
        "validation": validation,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": np.asarray(2, dtype=np.int32),
        "agent_ids": agent_ids,
        "operator_ids": operator_ids,
        "operator_names": operator_names,
        "source_files": source_files,
        "source_sha256": source_sha256,
        "scales": np.asarray(scales, dtype=np.float32),
        "base_template_size": np.asarray(BASE_TEMPLATE_SIZE, dtype=np.int32),
        "feature_scale": np.asarray(feature_scale, dtype=np.float32),
        "feature_size": np.asarray(FEATURE_SIZE, dtype=np.int32),
        "features_gray": features,
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    payload.update(
        {key: variants[scale] for key, scale in zip(template_keys, scales)}
    )
    np.savez_compressed(output, **payload)
    return metadata


def verify_saved_index(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as index:
        required = {
            "agent_ids",
            "operator_ids",
            "operator_names",
            "features_gray",
            "scales",
            "feature_size",
            "metadata_json",
        }
        missing = required - set(index.files)
        if missing:
            raise RuntimeError(f"索引回读缺少字段: {sorted(missing)}")
        metadata = json.loads(str(index["metadata_json"]))
        scales = normalize_scales(index["scales"].tolist())
        template_keys = [template_array_key(scale) for scale in scales]
        missing_templates = set(template_keys) - set(index.files)
        if missing_templates:
            raise RuntimeError(
                f"索引回读缺少模板数组: {sorted(missing_templates)}"
            )
        count = len(index["agent_ids"])
        if count != metadata["agent_count"]:
            raise RuntimeError("索引回读角色数量不一致")
        feature_size = tuple(int(value) for value in index["feature_size"])
        if index["features_gray"].shape != (
            count,
            feature_size[0] * feature_size[1],
        ):
            raise RuntimeError("索引回读特征矩阵尺寸异常")
        for key in template_keys:
            templates = index[key]
            if len(templates) != count or templates.ndim != 4 or templates.shape[-1] != 3:
                raise RuntimeError(f"索引回读模板数组尺寸异常: {key}")
        norms = np.linalg.norm(index["features_gray"], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise RuntimeError("索引回读特征未正确 L2 归一化")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成寿春心纸角色识别索引 NPZ。")
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--operators", type=Path, default=DEFAULT_OPERATORS)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(SCALES),
        help="写入索引的模板比例，例如 --scales 1.0；默认 0.89 0.90 0.91。",
    )
    parser.add_argument(
        "--feature-scale",
        type=float,
        default=None,
        help="生成粗筛特征所用比例；默认取 scales 的中间项。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        metadata = build_index(
            args.templates.resolve(),
            args.output.resolve(),
            args.operators.resolve(),
            args.scales,
            args.feature_scale,
        )
        verify_saved_index(args.output.resolve())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(f"已生成：{args.output.resolve()}")
    print(f"角色数：{metadata['agent_count']}")
    print(f"源模板摘要：{metadata['source_digest_sha256']}")
    for scale, validation in metadata["validation"].items():
        print(
            f"scale={scale}: Top-1 {validation['correct']}/{validation['total']}, "
            f"最小自身分数 {validation['min_own_score']:.4f}, "
            f"最小间隔 {validation['min_margin']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
