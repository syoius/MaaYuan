#!/usr/bin/env python3
"""Build the mixed agent/item index used by dispatch reward screens."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from build_bag_item_index import load_catalog
from build_shouchun_agent_index import (
    BASE_TEMPLATE_SIZE,
    FEATURE_SIZE,
    build_gray_features,
    load_operator_catalog,
    normalize_scales,
    read_bgr,
    resize_templates,
    template_array_key,
    template_id,
    validate_top1,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "dispatch-reward-index.json"


def _repo_path(value: object, field: str) -> Path:
    path = (REPO_ROOT / str(value)).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{field} 必须位于项目目录内: {value}") from exc
    return path


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if (
        manifest.get("schema") != "maay.dispatch-reward-index-source"
        or int(manifest.get("version", 0)) != 1
    ):
        raise ValueError("派遣索引清单 schema/version 不受支持")

    item_ids = [str(value) for value in manifest.get("item_ids", [])]
    if not item_ids or len(item_ids) != len(set(item_ids)):
        raise ValueError("派遣索引清单 item_ids 必须是非空且不重复的数组")
    manifest["item_ids"] = item_ids
    manifest["scales"] = normalize_scales(manifest.get("scales", []))
    feature_scale = float(manifest.get("feature_scale", 0))
    if not any(
        math.isclose(scale, feature_scale, abs_tol=1e-6)
        for scale in manifest["scales"]
    ):
        raise ValueError("feature_scale 必须是 scales 中的一个值")
    manifest["feature_scale"] = feature_scale
    return manifest


def build_index(manifest_path: Path) -> tuple[Path, dict]:
    manifest = load_manifest(manifest_path)
    templates_dir = _repo_path(manifest["agent_templates"], "agent_templates")
    operators_path = _repo_path(manifest["operators"], "operators")
    items_path = _repo_path(manifest["items"], "items")
    output = _repo_path(manifest["output"], "output")

    agent_files = sorted(templates_dir.glob("*-bag.png"))
    if not agent_files:
        raise ValueError(f"没有找到密探模板: {templates_dir}")
    agent_catalog = load_operator_catalog(operators_path)
    agent_template_ids = [template_id(path) for path in agent_files]
    unknown_agents = sorted(set(agent_template_ids) - set(agent_catalog))
    if unknown_agents:
        raise ValueError(f"密探模板无法映射 operators.json: {unknown_agents}")

    items, _ = load_catalog(items_path)
    item_catalog = {item["id"]: item for item in items}
    unknown_items = sorted(set(manifest["item_ids"]) - set(item_catalog))
    if unknown_items:
        raise ValueError(f"派遣索引清单包含未知道具: {unknown_items}")
    selected_items = [item_catalog[item_id] for item_id in manifest["item_ids"]]
    item_files = [item["template"] for item in selected_items]

    ids = np.asarray(agent_template_ids + manifest["item_ids"])
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("密探模板 ID 与道具 ID 存在冲突")
    entity_types = np.asarray(
        ["agent"] * len(agent_files) + ["item"] * len(item_files)
    )
    operator_ids = np.asarray(
        [agent_catalog[item_id][0] for item_id in agent_template_ids]
        + manifest["item_ids"]
    )
    names = np.asarray(
        [agent_catalog[item_id][1] for item_id in agent_template_ids]
        + [item["name"] for item in selected_items]
    )
    source_paths = agent_files + item_files
    source_files = np.asarray(
        [path.relative_to(REPO_ROOT).as_posix() for path in source_paths]
    )
    images = [read_bgr(path) for path in source_paths]
    expected_shape = (BASE_TEMPLATE_SIZE[1], BASE_TEMPLATE_SIZE[0], 3)
    invalid_shapes = [
        (path.name, image.shape)
        for path, image in zip(source_paths, images)
        if image.shape != expected_shape
    ]
    if invalid_shapes:
        raise ValueError(f"模板不是统一的 70x58 BGR: {invalid_shapes[:10]}")

    scales = manifest["scales"]
    variants = {scale: resize_templates(images, scale) for scale in scales}
    feature_scale = next(
        scale
        for scale in scales
        if math.isclose(scale, manifest["feature_scale"], abs_tol=1e-6)
    )
    features = build_gray_features(variants[feature_scale], FEATURE_SIZE)
    validation = validate_top1(ids, features, variants, FEATURE_SIZE)
    template_keys = [template_array_key(scale) for scale in scales]

    metadata = {
        "schema": "maay.dispatch-reward-index",
        "version": 1,
        "entry_count": len(ids),
        "agent_count": len(agent_files),
        "item_count": len(item_files),
        "item_ids": manifest["item_ids"],
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
        "validation": validation,
    }
    payload = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "entity_type": np.asarray("mixed"),
        "entity_types": entity_types,
        "agent_ids": ids,
        "operator_ids": operator_ids,
        "operator_names": names,
        "source_files": source_files,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output, metadata


def verify_index(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "entity_type",
            "entity_types",
            "agent_ids",
            "operator_ids",
            "operator_names",
            "features_gray",
            "scales",
            "feature_size",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"索引回读缺少字段: {sorted(missing)}")
        count = len(data["agent_ids"])
        if not (
            len(data["entity_types"])
            == len(data["operator_ids"])
            == len(data["operator_names"])
            == data["features_gray"].shape[0]
            == count
        ):
            raise RuntimeError("索引回读条目数量不一致")
        if set(data["entity_types"].astype(str)) != {"agent", "item"}:
            raise RuntimeError("索引回读对象类型异常")
        feature_size = tuple(int(value) for value in data["feature_size"])
        if data["features_gray"].shape[1] != feature_size[0] * feature_size[1]:
            raise RuntimeError("索引回读特征尺寸异常")
        for scale in data["scales"]:
            key = template_array_key(float(scale))
            if key not in data.files or len(data[key]) != count:
                raise RuntimeError(f"索引回读模板数组异常: {key}")
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("entry_count") != count:
            raise RuntimeError("索引 metadata 条目数量不一致")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="生成派遣奖励密探/道具混合索引。")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        output, metadata = build_index(args.manifest.resolve())
        verify_index(output)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(f"已生成：{output}")
    print(
        f"条目数：{metadata['entry_count']} "
        f"（密探 {metadata['agent_count']}，道具 {metadata['item_count']}）"
    )
    for scale, validation in metadata["validation"].items():
        print(
            f"scale={scale}: Top-1 {validation['correct']}/{validation['total']}，"
            f"最小间隔 {validation['min_margin']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
