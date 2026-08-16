#!/usr/bin/env python3
"""Build the recursive 1.0-scale bag-item recognition index."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from build_shouchun_agent_index import (
    BASE_TEMPLATE_SIZE,
    FEATURE_SIZE,
    build_gray_features,
    combined_source_digest,
    read_bgr,
    resize_templates,
    template_array_key,
    validate_top1,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ITEMS = REPO_ROOT / "agent" / "items.json"
DEFAULT_OUTPUT = REPO_ROOT / "agent" / "bag-item-index.npz"
ITEM_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
SCALE = 1.0


def _resolve_repo_path(value: str, field: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{field} 必须位于项目目录内: {value}") from exc
    return path


def load_catalog(path: Path) -> tuple[list[dict], list[dict]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("schema") != "maay.items" or int(payload.get("version", 0)) != 1:
        raise ValueError("items.json schema/version 不受支持")

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("items.json 的 items 必须是非空数组")

    ids: set[str] = set()
    template_paths: set[Path] = set()
    normalized: list[dict] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{position}] 必须是对象")
        item_id = str(item.get("id", ""))
        name = str(item.get("name", "")).strip()
        category = str(item.get("category", "")).strip()
        template_value = str(item.get("template", ""))
        if not ITEM_ID_PATTERN.fullmatch(item_id):
            raise ValueError(f"无效道具 ID: {item_id!r}")
        if item_id in ids:
            raise ValueError(f"重复道具 ID: {item_id}")
        if not name or not category:
            raise ValueError(f"道具缺少 name/category: {item_id}")
        template = _resolve_repo_path(template_value, f"items[{position}].template")
        if not template.is_file():
            raise FileNotFoundError(f"模板不存在: {template}")
        if template.name != f"{item_id}-bag.png":
            raise ValueError(
                f"模板文件名必须与 ID 一致: {item_id} -> {template.name}"
            )
        if template in template_paths:
            raise ValueError(f"模板路径重复: {template}")
        ids.add(item_id)
        template_paths.add(template)
        normalized.append(
            {
                "id": item_id,
                "name": name,
                "category": category,
                "template": template,
                "template_relative": template.relative_to(REPO_ROOT).as_posix(),
            }
        )

    roots = payload.get("template_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("items.json 的 template_roots 必须是非空数组")
    discovered: set[Path] = set()
    for position, value in enumerate(roots):
        root = _resolve_repo_path(str(value), f"template_roots[{position}]")
        if not root.is_dir():
            raise FileNotFoundError(f"模板根目录不存在: {root}")
        discovered.update(path.resolve() for path in root.rglob("*-bag.png"))
    missing_from_catalog = sorted(discovered - template_paths)
    missing_from_roots = sorted(template_paths - discovered)
    if missing_from_catalog or missing_from_roots:
        raise ValueError(
            "递归模板与 items.json 不一致: "
            f"未登记={[str(path) for path in missing_from_catalog]}, "
            f"不在根目录={[str(path) for path in missing_from_roots]}"
        )

    refine_groups = payload.get("refine_groups", [])
    if not isinstance(refine_groups, list):
        raise ValueError("refine_groups 必须是数组")
    group_ids: set[str] = set()
    occupied_member_ids: set[str] = set()
    normalized_groups: list[dict] = []
    for position, group in enumerate(refine_groups):
        if not isinstance(group, dict):
            raise ValueError(f"refine_groups[{position}] 必须是对象")
        group_id = str(group.get("id", ""))
        member_ids = [str(value) for value in group.get("item_ids", [])]
        box = [int(value) for value in group.get("box", [])]
        threshold = float(group.get("threshold", 0.9))
        min_margin = float(group.get("min_margin", 0.08))
        mode = str(group.get("mode", "color_ncc"))
        if not ITEM_ID_PATTERN.fullmatch(group_id) or group_id in group_ids:
            raise ValueError(f"无效或重复复核组 ID: {group_id!r}")
        if len(member_ids) < 2 or len(member_ids) != len(set(member_ids)):
            raise ValueError(f"复核组成员必须至少两个且不重复: {group_id}")
        unknown = set(member_ids) - ids
        if unknown:
            raise ValueError(f"复核组包含未知道具: {group_id}: {sorted(unknown)}")
        overlap = occupied_member_ids.intersection(member_ids)
        if overlap:
            raise ValueError(f"道具不能同时属于多个复核组: {sorted(overlap)}")
        if mode != "color_ncc":
            raise ValueError(f"不支持的复核模式: {mode}")
        if len(box) != 4:
            raise ValueError(f"复核框必须为 [x,y,w,h]: {group_id}")
        x, y, width, height = box
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > BASE_TEMPLATE_SIZE[0]
            or y + height > BASE_TEMPLATE_SIZE[1]
        ):
            raise ValueError(f"复核框超出 70x58 模板: {group_id}: {box}")
        if not (0 <= threshold <= 1 and 0 <= min_margin <= 1):
            raise ValueError(f"复核阈值必须位于 [0,1]: {group_id}")
        group_ids.add(group_id)
        occupied_member_ids.update(member_ids)
        normalized_groups.append(
            {
                "id": group_id,
                "item_ids": member_ids,
                "box": box,
                "mode": mode,
                "threshold": threshold,
                "min_margin": min_margin,
                "expand_candidates": bool(group.get("expand_candidates", True)),
            }
        )

    normalized.sort(key=lambda item: item["id"])
    return normalized, normalized_groups


def build_index(items_path: Path, output: Path) -> dict:
    items, refine_groups = load_catalog(items_path)
    paths = [item["template"] for item in items]
    images = [read_bgr(path) for path in paths]
    expected_shape = (BASE_TEMPLATE_SIZE[1], BASE_TEMPLATE_SIZE[0], 3)
    invalid = [
        (path.name, image.shape)
        for path, image in zip(paths, images)
        if image.shape != expected_shape
    ]
    if invalid:
        raise ValueError(f"模板不是统一的 70x58 BGR: {invalid[:10]}")

    templates = resize_templates(images, SCALE)
    features = build_gray_features(templates, FEATURE_SIZE)
    item_ids = np.asarray([item["id"] for item in items])
    item_names = np.asarray([item["name"] for item in items])
    categories = np.asarray([item["category"] for item in items])
    source_files = np.asarray([item["template_relative"] for item in items])
    source_digest, source_sha256 = combined_source_digest(paths)
    validation = validate_top1(
        item_ids,
        features,
        {SCALE: templates},
        FEATURE_SIZE,
    )
    metadata = {
        "schema": "maay.item-index",
        "version": 1,
        "entity_type": "item",
        "item_count": len(items),
        "base_template_size": list(BASE_TEMPLATE_SIZE),
        "scales": [SCALE],
        "feature": {
            "kind": "centered-grayscale-zero-mean-l2",
            "scale": SCALE,
            "size": list(FEATURE_SIZE),
        },
        "refine_groups": refine_groups,
        "source_digest_sha256": source_digest,
        "validation": validation,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    # The legacy agent/operator arrays keep existing releases loadable.  New
    # runtimes prefer item_ids/item_names and expose generic result fields.
    np.savez_compressed(
        output,
        schema_version=np.asarray(1, dtype=np.int32),
        entity_type=np.asarray("item"),
        item_ids=item_ids,
        item_names=item_names,
        item_categories=categories,
        agent_ids=item_ids,
        operator_ids=item_ids,
        operator_names=item_names,
        source_files=source_files,
        source_sha256=source_sha256,
        scales=np.asarray([SCALE], dtype=np.float32),
        base_template_size=np.asarray(BASE_TEMPLATE_SIZE, dtype=np.int32),
        feature_scale=np.asarray(SCALE, dtype=np.float32),
        feature_size=np.asarray(FEATURE_SIZE, dtype=np.int32),
        features_gray=features,
        templates_100=templates,
        refine_groups_json=np.asarray(json.dumps(refine_groups, ensure_ascii=False)),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def verify_index(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "item_ids",
            "item_names",
            "item_categories",
            "features_gray",
            "templates_100",
            "refine_groups_json",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"索引回读缺少字段: {sorted(missing)}")
        count = len(data["item_ids"])
        if not (
            len(data["item_names"])
            == len(data["item_categories"])
            == data["features_gray"].shape[0]
            == data["templates_100"].shape[0]
            == count
        ):
            raise RuntimeError("索引回读道具数量不一致")
        if data["templates_100"].shape[1:] != (58, 70, 3):
            raise RuntimeError("索引回读模板尺寸异常")
        norms = np.linalg.norm(data["features_gray"], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise RuntimeError("索引回读粗筛特征未 L2 归一化")
        metadata = json.loads(str(data["metadata_json"]))
        json.loads(str(data["refine_groups_json"]))
        if metadata.get("item_count") != count:
            raise RuntimeError("索引 metadata 道具数量不一致")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 items.json 递归生成背包道具索引。")
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        metadata = build_index(args.items.resolve(), args.output.resolve())
        verify_index(args.output.resolve())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    validation = metadata["validation"]["1.00"]
    print(f"已生成：{args.output.resolve()}")
    print(f"道具数：{metadata['item_count']}")
    print(f"递归源模板摘要：{metadata['source_digest_sha256']}")
    print(
        f"粗筛 Top-1 自检：{validation['correct']}/{validation['total']}，"
        f"最小间隔 {validation['min_margin']:.4f}"
    )
    print(f"条件复核组：{len(metadata['refine_groups'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
