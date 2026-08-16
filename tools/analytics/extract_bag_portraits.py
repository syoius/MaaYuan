#!/usr/bin/env python3
r"""Batch-extract heart-paper portraits from full-screen bag screenshots.

The script uses MaaFramework's bundled PP-OCR model.  A label must be
successfully recognized as containing ``心纸`` (traditional ``心紙`` is
normalized when zhconv is available) before its portrait can be exported.

Examples (PowerShell):

    python tools/analytics/extract_bag_portraits.py `
        tools/analytics/bag-template-*.png `
        -o tools/analytics/bag-extracted

    python tools/analytics/extract_bag_portraits.py D:\screenshots `
        -o assets/resource/base/image/agent/shouchun --recursive
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    import zhconv
except ImportError:  # pragma: no cover - MaaY's normal environment includes it.
    zhconv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "assets" / "MaaCommonAssets" / "OCR" / "ppocr_v6" / "small"
DEFAULT_OPERATORS = REPO_ROOT / "agent" / "operators.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TARGET_SIZE = (720, 1280)  # width, height
GRID_COLUMNS = (118, 286, 454, 622)
NAME_PATTERN = re.compile(r"^[\u3400-\u9fff·]{1,12}$")
BRACKET_NAME_PATTERN = re.compile(r"[【\[（(]([^】\]）)]+)[】\]）)]")
NAME_CORRECTIONS = {
    # Confirmed PP-OCRv6 small mistakes in the bag's decorative typeface.
    "鄂公珠": "酆公珠",
    "称衡": "祢衡",
    "张部": "张郃",
}


@dataclass
class OcrToken:
    text: str
    score: float
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class Candidate:
    source: str
    raw_text: str
    normalized_text: str
    name: str
    slug: str
    mapped_operator: bool
    ocr_score: float
    label_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    sharpness: float
    image: np.ndarray
    status: str = "candidate"
    output: str = ""

    def report_dict(self) -> dict:
        return {
            "source": self.source,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "name": self.name,
            "slug": self.slug,
            "mapped_operator": self.mapped_operator,
            "ocr_score": self.ocr_score,
            "label_box": list(self.label_box),
            "crop_box": list(self.crop_box),
            "sharpness": self.sharpness,
            "status": self.status,
            "output": self.output,
        }


class _OfflineController:
    """Create the smallest controller Maa Tasker accepts for offline OCR."""

    def __new__(cls):
        # Importing Maa lazily keeps --help useful even in a plain Python env.
        from maa.controller import CustomController

        class Controller(CustomController):
            def connect(self) -> bool:
                return True

            def connected(self) -> bool:
                return True

            def request_uuid(self) -> str:
                return "offline-bag-extractor"

            def start_app(self, intent: str) -> bool:
                return False

            def stop_app(self, intent: str) -> bool:
                return False

            def screencap(self) -> np.ndarray:
                return np.zeros((1, 1, 3), dtype=np.uint8)

            def click(self, x: int, y: int) -> bool:
                return False

            def swipe(
                self, x1: int, y1: int, x2: int, y2: int, duration: int
            ) -> bool:
                return False

            def touch_down(
                self, contact: int, x: int, y: int, pressure: int
            ) -> bool:
                return False

            def touch_move(
                self, contact: int, x: int, y: int, pressure: int
            ) -> bool:
                return False

            def touch_up(self, contact: int) -> bool:
                return False

            def click_key(self, keycode: int) -> bool:
                return False

            def input_text(self, text: str) -> bool:
                return False

            def key_down(self, keycode: int) -> bool:
                return False

            def key_up(self, keycode: int) -> bool:
                return False

            def scroll(self, dx: int, dy: int) -> bool:
                return False

        return Controller()


class MaaOcr:
    def __init__(self, model_path: Path, threshold: float) -> None:
        try:
            from maa.pipeline import JOCR
            from maa.resource import Resource
            from maa.tasker import Tasker
            from maa.toolkit import Toolkit
        except ImportError as exc:
            raise RuntimeError(
                "未安装 maafw；请使用 MaaYuan 自带 Python，或先安装 requirements.txt"
            ) from exc

        if not model_path.is_dir():
            raise FileNotFoundError(f"OCR 模型目录不存在: {model_path}")

        Toolkit.init_option(str(REPO_ROOT))
        self._resource = Resource()
        self._resource.use_cpu()
        model_job = self._resource.post_ocr_model(model_path).wait()
        if not model_job.succeeded:
            raise RuntimeError(f"加载 Maa OCR 模型失败: {model_path}")

        self._controller = _OfflineController()
        connection = self._controller.post_connection().wait()
        if not connection.succeeded:
            raise RuntimeError("初始化离线 Maa Controller 失败")

        self._tasker = Tasker()
        if not self._tasker.bind(self._resource, self._controller):
            raise RuntimeError("Maa Tasker 绑定 OCR 资源失败")
        if not self._tasker.inited:
            raise RuntimeError("Maa Tasker 未完成初始化")

        self._param_type = JOCR
        self._threshold = threshold

    def recognize(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> list[OcrToken]:
        param = self._param_type(
            roi=roi,
            threshold=self._threshold,
            order_by="Vertical",
            only_rec=False,
        )
        job = self._tasker.post_recognition("OCR", param, image).wait()
        if not job.succeeded:
            return []
        task_detail = job.get()
        if not task_detail:
            return []

        # MaaFramework's standalone post_recognition returns a TaskDetail.  Keep
        # the direct RecognitionDetail fallback for compatibility with older
        # Python bindings.
        detail = task_detail
        if hasattr(task_detail, "nodes"):
            detail = next(
                (
                    node.recognition
                    for node in task_detail.nodes
                    if getattr(node, "recognition", None) is not None
                ),
                None,
            )
        if detail is None:
            return []

        results = detail.filtered_results or detail.all_results or []
        tokens: list[OcrToken] = []
        for item in results:
            box = item.box
            if hasattr(box, "x"):
                x, y, w, h = box.x, box.y, box.w, box.h
            else:
                x, y, w, h = box
            tokens.append(
                OcrToken(
                    text=str(item.text),
                    score=float(item.score),
                    x=int(x),
                    y=int(y),
                    w=int(w),
                    h=int(h),
                )
            )
        return tokens


def normalize_text(text: str) -> str:
    value = zhconv.convert(text, "zh-cn") if zhconv else text.replace("紙", "纸")
    value = value.replace(" ", "").replace("\u3000", "")
    return value.strip()


def extract_name(text: str) -> str | None:
    """Conservatively extract the item name after a recognized 心纸 marker."""

    normalized = normalize_text(text)
    marker = normalized.find("心纸")
    if marker < 0:
        return None

    tail = normalized[marker + 2 :]
    bracket_match = BRACKET_NAME_PATTERN.search(tail)
    if bracket_match:
        name = bracket_match.group(1)
    else:
        name = tail.strip("：:·-—_【】[]（）()")

    # OCR sometimes retains punctuation around an otherwise good name.
    name = name.replace(" ", "").strip("：:·-—_【】[]（）()")
    return name if NAME_PATTERN.fullmatch(name) else None


def merge_ocr_tokens(tokens: Sequence[OcrToken]) -> list[OcrToken]:
    """Merge nearby OCR fragments without joining labels from adjacent slots."""

    remaining = sorted(tokens, key=lambda t: (t.cy, t.x))
    lines: list[list[OcrToken]] = []
    for token in remaining:
        target: list[OcrToken] | None = None
        for line in lines:
            line_cy = sum(item.cy for item in line) / len(line)
            if abs(token.cy - line_cy) <= max(12.0, token.h * 0.55):
                target = line
                break
        if target is None:
            lines.append([token])
        else:
            target.append(token)

    merged: list[OcrToken] = []
    for line in lines:
        ordered = sorted(line, key=lambda t: t.x)
        groups: list[list[OcrToken]] = []
        for token in ordered:
            if not groups:
                groups.append([token])
                continue
            previous = groups[-1][-1]
            gap = token.x - (previous.x + previous.w)
            if gap <= 34:
                groups[-1].append(token)
            else:
                groups.append([token])

        for group in groups:
            x1 = min(t.x for t in group)
            y1 = min(t.y for t in group)
            x2 = max(t.x + t.w for t in group)
            y2 = max(t.y + t.h for t in group)
            merged.append(
                OcrToken(
                    text="".join(t.text for t in group),
                    score=min(t.score for t in group),
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                )
            )
    return sorted(merged, key=lambda t: (t.y, t.x))


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {path}")
    encoded.tofile(path)


def normalize_screenshot(image: np.ndarray) -> np.ndarray:
    target_w, target_h = TARGET_SIZE
    h, w = image.shape[:2]
    source_ratio = w / h
    target_ratio = target_w / target_h
    if abs(source_ratio - target_ratio) > 0.015:
        raise ValueError(
            f"截图宽高比为 {w}x{h}，不是预期的竖屏 9:16；请先去除模拟器边框"
        )
    if (w, h) == TARGET_SIZE:
        return image
    interpolation = cv2.INTER_AREA if w > target_w else cv2.INTER_CUBIC
    return cv2.resize(image, TARGET_SIZE, interpolation=interpolation)


def load_operator_slugs(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    entries: list[tuple[dict, str]] = []
    slug_counts: dict[str, int] = {}
    for operator in payload.get("OPERATORS", []):
        alias = str(operator.get("alias", ""))
        ascii_aliases = re.findall(r"\b[a-z][a-z0-9_-]*\b", alias.lower())
        if not ascii_aliases:
            continue
        slug = ascii_aliases[0]
        entries.append((operator, slug))
        slug_counts[slug] = slug_counts.get(slug, 0) + 1

    mapping: dict[str, str] = {}
    for operator, base_slug in entries:
        slug = base_slug
        if slug_counts[base_slug] > 1:
            operator_id = str(operator.get("id", ""))
            number = re.search(r"(?:^|_)(\d+)(?:_|$)", operator_id)
            suffix = number.group(1) if number else safe_slug(operator_id)
            slug = f"{base_slug}-{suffix}"
        for key in ("name", "name_en", "alt_name"):
            value = operator.get(key)
            if value:
                mapping[normalize_text(str(value))] = slug
    return mapping


def find_missing_operators(path: Path, extracted_names: set[str]) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    slug_map = load_operator_slugs(path)
    missing: list[dict] = []
    for operator in payload.get("OPERATORS", []):
        name = normalize_text(str(operator.get("name", "")))
        if not name or name in extracted_names:
            continue
        missing.append(
            {
                "id": str(operator.get("id", "")),
                "name": name,
                "slug": slug_map.get(name, ""),
            }
        )
    return missing


def safe_slug(name: str) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", name).strip(" ._")
    return value or "unnamed"


def nearest_column(x: float, tolerance: int) -> int | None:
    column = min(GRID_COLUMNS, key=lambda value: abs(value - x))
    return column if abs(column - x) <= tolerance else None


def crop_candidate(
    image: np.ndarray,
    source: Path,
    token: OcrToken,
    operator_slugs: dict[str, str],
    args: argparse.Namespace,
) -> tuple[Candidate | None, str]:
    normalized = normalize_text(token.text)
    if "心纸" not in normalized:
        return None, "not-heart-paper"

    name = extract_name(normalized)
    if not name:
        return None, "invalid-or-missing-name"
    name = NAME_CORRECTIONS.get(name, name)

    cx = nearest_column(token.cx, args.column_tolerance)
    if cx is None:
        return None, "not-on-grid"

    crop_x = int(cx - args.crop_width // 2)
    crop_y = int(token.y - args.label_to_crop_top)
    crop_w = int(args.crop_width)
    crop_h = int(args.crop_height)
    height, width = image.shape[:2]

    if (
        crop_y < args.safe_top
        or crop_y + crop_h > args.safe_bottom
        or crop_x < 0
        or crop_x + crop_w > width
    ):
        return None, "occluded-or-partial"

    portrait = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
    gray = cv2.cvtColor(portrait, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if float(gray.std()) < args.min_stddev:
        return None, "blank-or-low-detail"

    slug = operator_slugs.get(name, safe_slug(name))
    return (
        Candidate(
            source=str(source),
            raw_text=token.text,
            normalized_text=normalized,
            name=name,
            slug=slug,
            mapped_operator=name in operator_slugs,
            ocr_score=token.score,
            label_box=(token.x, token.y, token.w, token.h),
            crop_box=(crop_x, crop_y, crop_w, crop_h),
            sharpness=sharpness,
            image=portrait,
        ),
        "candidate",
    )


def candidate_quality(candidate: Candidate) -> float:
    # OCR confidence is most important; sharpness only breaks close ties.
    return candidate.ocr_score * 1000.0 + math.log1p(candidate.sharpness)


def image_ncc(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return -1.0
    a = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    b = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else -1.0


def select_candidates(candidates: list[Candidate], conflict_ncc: float) -> list[Candidate]:
    by_name: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_name.setdefault(candidate.name, []).append(candidate)

    selected: list[Candidate] = []
    for group in by_name.values():
        group.sort(key=candidate_quality, reverse=True)
        winner = group[0]
        winner.status = "selected"
        selected.append(winner)
        for duplicate in group[1:]:
            similarity = image_ncc(winner.image, duplicate.image)
            duplicate.status = (
                "duplicate" if similarity >= conflict_ncc else "duplicate-conflict"
            )
    return sorted(selected, key=lambda c: c.slug)


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


def draw_debug(
    image: np.ndarray,
    tokens: Sequence[OcrToken],
    accepted: Sequence[Candidate],
) -> np.ndarray:
    canvas = image.copy()
    accepted_boxes = {candidate.label_box for candidate in accepted}
    for token in tokens:
        box = (token.x, token.y, token.w, token.h)
        color = (0, 180, 0) if box in accepted_boxes else (0, 140, 255)
        cv2.rectangle(canvas, (token.x, token.y), (token.x + token.w, token.y + token.h), color, 2)
    for candidate in accepted:
        x, y, w, h = candidate.crop_box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 80, 0), 2)
        cv2.putText(
            canvas,
            candidate.slug,
            (x, max(18, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 80, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_reports(
    output: Path,
    candidates: Sequence[Candidate],
    rejected: Sequence[dict],
    missing_operators: Sequence[dict],
) -> None:
    records = [candidate.report_dict() for candidate in candidates]
    manifest = {
        "target_marker": "心纸",
        "target_size": list(TARGET_SIZE),
        "candidates": records,
        "rejected": list(rejected),
        "missing_operators": list(missing_operators),
    }
    with (output / "bag-extraction-report.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    fields = [
        "status",
        "name",
        "slug",
        "mapped_operator",
        "ocr_score",
        "raw_text",
        "source",
        "output",
        "sharpness",
        "label_box",
        "crop_box",
    ]
    with (output / "bag-extraction-report.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    with (output / "missing-operators.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["id", "name", "slug"])
        writer.writeheader()
        writer.writerows(missing_operators)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从全屏背包截图中 OCR 定位并提取名称包含“心纸”的角色纸片。"
    )
    parser.add_argument("inputs", nargs="+", help="截图、目录或 glob（如 bag-template-*.png）")
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描输入目录")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有的同名 -bag.png")
    parser.add_argument("--debug", action="store_true", help="保存带 OCR/裁切框的调试截图")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Maa OCR 模型目录")
    parser.add_argument("--operators", type=Path, default=DEFAULT_OPERATORS, help="operators.json 路径")
    parser.add_argument("--ocr-threshold", type=float, default=0.25)
    parser.add_argument("--ocr-roi-top", type=int, default=250)
    parser.add_argument("--ocr-roi-bottom", type=int, default=1190)
    parser.add_argument("--safe-top", type=int, default=300, help="低于该 y 的裁图视为被顶部遮挡")
    parser.add_argument("--safe-bottom", type=int, default=1125, help="超过该 y 的裁图视为被底部遮挡")
    parser.add_argument("--crop-width", type=int, default=70)
    parser.add_argument("--crop-height", type=int, default=58)
    parser.add_argument("--label-to-crop-top", type=int, default=96)
    parser.add_argument("--column-tolerance", type=int, default=58)
    parser.add_argument("--min-stddev", type=float, default=12.0)
    parser.add_argument("--duplicate-ncc", type=float, default=0.94)
    return parser


def run(args: argparse.Namespace) -> int:
    screenshots = expand_inputs(args.inputs, args.recursive)
    if not screenshots:
        print("没有找到可处理的截图。", file=sys.stderr)
        return 2

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output / "debug"
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    operator_slugs = load_operator_slugs(args.operators.resolve())
    ocr = MaaOcr(args.model.resolve(), args.ocr_threshold)
    all_candidates: list[Candidate] = []
    rejected: list[dict] = []

    roi_height = args.ocr_roi_bottom - args.ocr_roi_top
    if roi_height <= 0:
        raise ValueError("--ocr-roi-bottom 必须大于 --ocr-roi-top")
    ocr_roi = (0, args.ocr_roi_top, TARGET_SIZE[0], roi_height)

    for index, path in enumerate(screenshots, start=1):
        print(f"[{index}/{len(screenshots)}] OCR: {path}")
        try:
            image = normalize_screenshot(read_image(path))
        except Exception as exc:
            rejected.append({"source": str(path), "reason": "invalid-image", "detail": str(exc)})
            print(f"  跳过: {exc}")
            continue

        tokens = merge_ocr_tokens(ocr.recognize(image, ocr_roi))
        image_candidates: list[Candidate] = []
        for token in tokens:
            if "心纸" not in normalize_text(token.text):
                continue
            candidate, reason = crop_candidate(
                image, path, token, operator_slugs, args
            )
            if candidate:
                all_candidates.append(candidate)
                image_candidates.append(candidate)
            else:
                rejected.append(
                    {
                        "source": str(path),
                        "reason": reason,
                        "text": token.text,
                        "score": token.score,
                        "box": [token.x, token.y, token.w, token.h],
                    }
                )

        print(f"  OCR 条目 {len(tokens)}，有效心纸候选 {len(image_candidates)}")
        if args.debug:
            write_png(debug_dir / f"{path.stem}-debug.png", draw_debug(image, tokens, image_candidates))

    selected = select_candidates(all_candidates, args.duplicate_ncc)
    for candidate in selected:
        output_path = args.output / f"{candidate.slug}-bag.png"
        if output_path.exists() and not args.overwrite:
            candidate.status = "exists-not-overwritten"
            candidate.output = str(output_path)
            continue
        write_png(output_path, candidate.image)
        candidate.output = str(output_path)

    missing_operators = find_missing_operators(
        args.operators.resolve(), {candidate.name for candidate in selected}
    )
    write_reports(args.output, all_candidates, rejected, missing_operators)
    conflict_count = sum(c.status == "duplicate-conflict" for c in all_candidates)
    unmapped_count = sum(not c.mapped_operator for c in selected)
    print(
        f"完成：输入 {len(screenshots)} 张，识别候选 {len(all_candidates)} 个，"
        f"去重后 {len(selected)} 个，冲突 {conflict_count} 个，"
        f"未映射 operators.json {unmapped_count} 个，"
        f"operators.json 中尚缺 {len(missing_operators)} 个。"
    )
    print(f"报告：{args.output / 'bag-extraction-report.json'}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
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
