#!/usr/bin/env python3
"""Import new heart-paper icons, rebuild both indexes, and archive the originals."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

import build_dispatch_reward_index as dispatch
import build_shouchun_agent_index as bag
from crop_bag_icons import IMAGE_SUFFIXES, convert_icon, read_image, write_png


ANALYTICS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ANALYTICS_DIR / "xinzhi-update"
DEFAULT_ARCHIVE = ANALYTICS_DIR / "xinzhi-archive"
DEFAULT_BAG_OUTPUT = bag.REPO_ROOT / "agent" / "bag-agent-index.npz"


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_operator(path: Path, operators: list[dict]) -> tuple[str, str]:
    name = path.stem.removesuffix("-bag").strip()
    matches = [operator for operator in operators if operator.get("id") == name]
    if not matches:
        matches = [
            operator
            for operator in operators
            if name and name in {
                str(operator.get(key, "")).strip()
                for key in ("name", "name_en", "alt_name")
            }
        ]
    if len(matches) != 1:
        raise ValueError(
            f"{path.name}: 无法唯一匹配 operators.json（匹配 {len(matches)} 项），"
            "请使用准确的角色名或完整角色 ID 命名"
        )
    operator_id = str(matches[0]["id"])
    if not re.fullmatch(r"[A-Za-z0-9_]+", operator_id):
        raise ValueError(f"角色 ID 不能用作模板文件名: {operator_id!r}")
    return operator_id, str(matches[0].get("name", operator_id))


def replace_from_copy(source: Path, destination: Path) -> None:
    """Publish atomically with permissions inherited from the destination folder."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Windows moves retain the source ACL. A TemporaryDirectory has a private
    # ACL, so moving its files directly would make indexes unreadable to users
    # other than the account running the update (e.g. a sandbox account).
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False,
    ) as temporary:
        sibling = Path(temporary.name)
    try:
        # copy2 preserves timestamps/mode, but does not copy Windows ACLs.
        shutil.copy2(source, sibling)
        sibling.replace(destination)
    finally:
        sibling.unlink(missing_ok=True)


def publish_files(files: list[tuple[Path, Path]], backup_dir: Path) -> None:
    """Back up first; roll back all completed replacements if publishing fails."""
    backup_dir.mkdir()
    backups: dict[Path, Path] = {}
    for index, (_, destination) in enumerate(files):
        if destination.exists():
            backup = backup_dir / f"{index:03d}-{destination.name}"
            shutil.copy2(destination, backup)
            backups[destination] = backup
    replaced: list[Path] = []
    try:
        for staged, destination in files:
            replace_from_copy(staged, destination)
            replaced.append(destination)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for destination in reversed(replaced):
            try:
                if destination in backups:
                    replace_from_copy(backups[destination], destination)
                else:
                    destination.unlink()
            except OSError as rollback_exc:
                rollback_errors.append(f"{destination}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                f"写入失败且部分文件无法回滚: {rollback_errors}；原文件备份: {backup_dir}"
            ) from exc
        raise


def update_indexes(
    input_dir: Path = DEFAULT_INPUT,
    archive_dir: Path = DEFAULT_ARCHIVE,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    manifest_path: Path = dispatch.DEFAULT_MANIFEST,
    bag_output: Path = DEFAULT_BAG_OUTPUT,
) -> dict:
    input_dir, archive_dir = input_dir.resolve(), archive_dir.resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(input_dir.iterdir())
    if not inputs:
        print(f"没有新增图片：{input_dir}")
        return {"status": "empty", "records": []}
    invalid = [
        path.name for path in inputs
        if path.is_symlink() or not path.is_file()
        or path.suffix.lower() not in IMAGE_SUFFIXES
    ]
    if invalid:
        raise ValueError(f"输入目录只能直接放置图片，存在不支持的文件或子目录: {invalid}")

    manifest_path, bag_output = manifest_path.resolve(), bag_output.resolve()
    manifest = dispatch.load_manifest(manifest_path)
    templates_dir = dispatch._repo_path(manifest["agent_templates"], "agent_templates")
    operators_path = dispatch._repo_path(manifest["operators"], "operators")
    dispatch_output = dispatch._repo_path(manifest["output"], "output")
    # Keep cleanup strictly separate from templates, outputs, and the archive.
    for protected in (archive_dir, templates_dir, bag_output.parent, dispatch_output.parent):
        if protected.is_relative_to(input_dir) or input_dir.is_relative_to(protected):
            raise ValueError(f"输入目录与归档/输出目录不能重叠: {input_dir}, {protected}")
    if bag_output == dispatch_output:
        raise ValueError("背包索引与派遣索引不能使用同一个输出路径")
    with operators_path.open(encoding="utf-8") as file:
        operators = json.load(file)["OPERATORS"]
    bag.load_operator_catalog(operators_path)  # Check duplicate catalog IDs early.

    # One update at a time, including runs with a different input directory.
    lock = ANALYTICS_DIR / ".xinzhi-update.lock"
    try:
        lock_file = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(
            f"另一个更新正在运行；若上次异常退出，确认进程已停止后删除 {lock}"
        ) from exc
    try:
        with lock_file, tempfile.TemporaryDirectory(
            prefix=".xinzhi-stage-", dir=ANALYTICS_DIR
        ) as temporary:
            stage = Path(temporary)
            staged_templates = stage / "templates"
            shutil.copytree(templates_dir, staged_templates)
            staged_archive = stage / "archive"
            originals = staged_archive / "originals"
            originals.mkdir(parents=True)
            records: list[dict] = []
            seen: set[str] = set()
            changed: list[str] = []
            for source in inputs:
                operator_id, name = resolve_operator(source, operators)
                if operator_id in seen:
                    raise ValueError(f"本批次有多张图片对应同一角色: {operator_id}")
                seen.add(operator_id)
                snapshot = originals / source.name
                shutil.copy2(source, snapshot)
                image = read_image(snapshot)
                if image.shape[:2] == (58, 70):
                    template = bag.read_bgr(snapshot)
                    conversion = "already-70x58"
                else:
                    template, conversion = convert_icon(image)
                    template = template[:, :, :3]
                filename = f"{operator_id}-bag.png"
                target = staged_templates / filename
                same = target.exists() and np.array_equal(bag.read_bgr(target), template)
                if target.exists() and not same and not overwrite:
                    raise ValueError(f"{filename} 已存在且内容不同；确认替换时请加 --overwrite")
                if not same:
                    write_png(target, template)
                    changed.append(filename)
                records.append({
                    "source": source.name,
                    "sha256": file_digest(snapshot),
                    "operator_id": operator_id,
                    "operator_name": name,
                    "template": filename,
                    "conversion": conversion,
                    "unchanged": same,
                })
                print(f"{source.name} -> {filename}")

            staged_bag = stage / "bag-agent-index.npz"
            staged_dispatch = stage / "dispatch-reward-index.npz"
            bag.build_index(staged_templates, staged_bag, operators_path, scales=(1.0,))
            bag_metadata = bag.verify_saved_index(staged_bag)
            dispatch.build_index(manifest_path, templates_dir=staged_templates, output=staged_dispatch)
            dispatch_metadata = dispatch.verify_index(staged_dispatch)
            report = {
                "status": "validated",
                "created_at": datetime.now().astimezone().isoformat(),
                "input_dir": str(input_dir),
                "bag_output": str(bag_output),
                "dispatch_output": str(dispatch_output),
                "records": records,
                "bag_index": bag_metadata,
                "dispatch_index": dispatch_metadata,
            }
            print(
                f"自检通过：背包 {bag_metadata['agent_count']} 位密探；"
                f"派遣 {dispatch_metadata['entry_count']} 个条目"
            )
            if dry_run:
                print("预演完成，未更新索引、归档或移除输入图片。")
                return report

            for source, record in zip(inputs, records):
                if source.is_symlink() or file_digest(source) != record["sha256"]:
                    raise RuntimeError(f"处理期间输入图片发生变化，已取消更新: {source}")
            batch = archive_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
            batch.parent.mkdir(parents=True, exist_ok=True)
            write_report(staged_archive / "report.json", report)
            shutil.copytree(staged_archive, batch)
            for record in records:
                if file_digest(batch / "originals" / record["source"]) != record["sha256"]:
                    raise RuntimeError(f"归档校验失败，输入图片保留: {batch}")
            print(f"原图归档已校验：{batch}")
            files = [(staged_templates / name, templates_dir / name) for name in changed]
            files.extend([(staged_bag, bag_output), (staged_dispatch, dispatch_output)])
            try:
                publish_files(files, batch / "previous")
            except BaseException:
                report["status"] = "publish-failed"
                write_report(batch / "report.json", report)
                raise
            report["status"] = "updated"
            write_report(batch / "report.json", report)
            # Delete only the exact files processed, after both indexes and archives are safe.
            for source, record in zip(inputs, records):
                if source.is_symlink() or file_digest(source) != record["sha256"]:
                    raise RuntimeError(
                        f"索引已更新，但输入图片发生变化，已保留待处理文件: {source}；"
                        f"归档: {batch}"
                    )
            for source in inputs:
                source.unlink()
            report["status"] = "complete"
            write_report(batch / "report.json", report)
            print(f"已更新：{bag_output}\n已更新：{dispatch_output}\n归档：{batch}")
            print(f"已从输入目录移除并归档 {len(inputs)} 张图片。")
            return report
    finally:
        lock.unlink()


def main() -> int:
    # Some Windows consoles use GBK, which cannot encode names such as 士䵋.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="从新增心纸图片更新背包、派遣索引，成功后归档并清理输入。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="待处理图片目录")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="按批次保存原图与报告的目录")
    parser.add_argument("--overwrite", action="store_true", help="允许替换内容不同的已有模板")
    parser.add_argument("--dry-run", action="store_true", help="完整构建和自检，但不更新、归档或清理")
    args = parser.parse_args()
    try:
        update_indexes(args.input, args.archive, overwrite=args.overwrite, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("已取消；未清理的图片可重新运行处理。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
