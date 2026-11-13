"""Supplement missing bag heart-paper templates without changing the bundled index."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from urllib import request
from urllib.parse import urlsplit

import cv2
import numpy as np

from custom.reco.agent_item import (
    AgentIndex,
    REPO_ROOT,
    _load_index,
    _normalized_gray_feature,
    _template_array_key,
)
from utils import logger
from utils.remote_file_sync import create_https_context

_HOST = "https://api-hub.maayuan.com"
_CACHE_DIR = REPO_ROOT / "agent" / "xinzhi-cache"
_BACKGROUND = REPO_ROOT / "agent" / "resources" / "xinzhi-q5.png"
_MAX_BYTES = 200 * 1024
_LOCK = threading.Lock()


def load_xinzhi_catalog(path: Path) -> dict:
    # Catalog sync preserves extra fields and writes the remote file, not operators.json.
    remote = path.with_name("operators.remote.json")
    if remote.is_file():
        try:
            data = json.loads(remote.read_text(encoding="utf-8"))
            operators = data["OPERATORS"]
            ids = [operator["id"] for operator in operators]
            if (
                not isinstance(operators, list)
                or not operators
                or any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(ids) != len(set(ids))
                or any(not isinstance(operator["name"], str) for operator in operators)
            ):
                raise ValueError("远程目录缓存格式无效")
            return data
        except (OSError, ValueError, KeyError, TypeError):
            logger.bind(console=False).warning("心纸目录缓存不可用，使用内置名单")
    return json.loads(path.read_text(encoding="utf-8"))


def _trusted_url(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "api-hub.maayuan.com"
        and parsed.username is None
        and parsed.password is None
    )


class _TrustedRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _trusted_url(newurl):
            raise ValueError("心纸下载重定向到不受信任的地址")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _atomic_file(path: Path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _decode_png(payload: bytes) -> np.ndarray:
    if (
        len(payload) > _MAX_BYTES
        or payload[:8] != b"\x89PNG\r\n\x1a\n"
        or payload[12:16] != b"IHDR"
        or int.from_bytes(payload[16:20], "big") != 140
        or int.from_bytes(payload[20:24], "big") != 140
    ):
        raise ValueError("心纸必须为不超过 200 KiB 的 140×140 PNG")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (140, 140, 4) or image.dtype != np.uint8:
        raise ValueError("心纸必须为 140×140 四通道 PNG")
    return image


def _icon(operator: dict) -> np.ndarray:
    operator_id = operator["id"]
    sha = operator["xinzhi_sha256"]
    url = operator["xinzhi_url"]
    if not isinstance(operator_id, str) or not re.fullmatch(r"char_[A-Za-z0-9_]+", operator_id):
        raise ValueError("密探 ID 无效")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("心纸 SHA-256 无效")
    if not isinstance(url, str):
        raise ValueError("心纸地址无效")
    parsed = urlsplit(url)
    if (
        parsed.scheme or parsed.netloc or parsed.fragment
        or parsed.path != f"/xinzhi/{operator_id}.png"
        or "\\" in url or any(ord(char) < 33 for char in url)
    ):
        raise ValueError("心纸地址必须为 YuanHub 相对路径")
    path = _CACHE_DIR / f"{operator_id}-{sha[:16]}.png"
    if path.is_file():
        try:
            with path.open("rb") as stream:
                payload = stream.read(_MAX_BYTES + 1)
            if hashlib.sha256(payload).hexdigest() != sha:
                raise ValueError("缓存 SHA-256 不一致")
            return _decode_png(payload)
        except (OSError, ValueError, cv2.error):
            logger.bind(console=False).warning(f"心纸图片缓存失效：{operator_id}")
    opener = request.build_opener(
        request.HTTPSHandler(context=create_https_context()), _TrustedRedirect()
    )
    req = request.Request(_HOST + url, headers={"User-Agent": "MaaY-Xinzhi/1.0"})
    with opener.open(req, timeout=8) as response:
        if response.status != 200 or not _trusted_url(response.geturl()):
            raise ValueError("心纸下载响应无效")
        payload = response.read(_MAX_BYTES + 1)
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError("心纸 SHA-256 不一致")
    image = _decode_png(payload)
    try:
        with _atomic_file(path) as stream:
            stream.write(payload)
    except OSError as exc:
        logger.bind(console=False).warning(f"心纸图片缓存写入失败：{operator_id}：{exc}")
    return image


def _compose(background: np.ndarray, icon: np.ndarray) -> np.ndarray:
    alpha = icon[:, :, 3:4].astype(np.float32) / 255.0
    bgr = np.rint(icon[:, :, :3] * alpha + background[:, :, :3] * (1.0 - alpha))
    resized = cv2.resize(bgr.astype(np.uint8), (109, 109), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized[20:78, 18:88])


def _feature(template: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    x = (template.shape[1] - width) // 2
    y = (template.shape[0] - height) // 2
    if x < 0 or y < 0:
        raise ValueError("模板小于特征区域")
    return _normalized_gray_feature(template[y:y + height, x:x + width], size)


def _merge(builtin: AgentIndex, extra: AgentIndex) -> AgentIndex:
    if builtin.feature_size != extra.feature_size or len(builtin.variants) != len(extra.variants):
        raise ValueError("补充索引参数不兼容")
    for (scale, templates), (other_scale, other_templates) in zip(builtin.variants, extra.variants):
        if scale != other_scale or templates.shape[1:] != other_templates.shape[1:]:
            raise ValueError("补充索引模板尺寸不兼容")
    if set(builtin.operator_ids).intersection(extra.operator_ids):
        raise ValueError("补充索引包含内置密探")
    return replace(
        builtin,
        entity_types=np.concatenate((builtin.entity_types, extra.entity_types)),
        agent_ids=np.concatenate((builtin.agent_ids, extra.agent_ids)),
        operator_ids=np.concatenate((builtin.operator_ids, extra.operator_ids)),
        operator_names=np.concatenate((builtin.operator_names, extra.operator_names)),
        features=np.concatenate((builtin.features, extra.features)),
        variants=tuple(
            (scale, np.concatenate((templates, other_templates)))
            for (scale, templates), (_, other_templates) in zip(builtin.variants, extra.variants)
        ),
    )


def _validate(template: np.ndarray, position: int, index: AgentIndex, builtin_count: int):
    queries = [_feature(template, index.feature_size)]
    queries.extend(_feature(templates[position], index.feature_size) for _, templates in index.variants)
    for query in queries:
        scores = query @ index.features.T
        if not np.isfinite(scores).all() or int(np.argmax(scores)) != position:
            raise ValueError("灰度粗筛 Top-1 自检失败")
    best_scores = np.full(len(index.agent_ids), -1.0, dtype=np.float32)
    for _, templates in index.variants:
        for candidate, image in enumerate(templates):
            if image.shape[0] > template.shape[0] or image.shape[1] > template.shape[1]:
                raise ValueError("模板超出 70×58 自检图")
            score = float(cv2.minMaxLoc(cv2.matchTemplate(template, image, cv2.TM_CCOEFF_NORMED))[1])
            if not np.isfinite(score):
                raise ValueError("完整模板分数无效")
            best_scores[candidate] = max(best_scores[candidate], score)
    if int(np.argmax(best_scores)) != position:
        raise ValueError("完整模板 Top-1 自检失败")
    if np.any(best_scores[:builtin_count] > 0.80):
        raise ValueError("与内置模板过于相似（超过 0.80）")


def _supplement(builtin: AgentIndex, operators: list[dict]) -> AgentIndex:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    background_bytes = _BACKGROUND.read_bytes()
    expected = {
        "version": 1,
        "builtin_sha256": _digest(builtin.path),
        "background_sha256": hashlib.sha256(background_bytes).hexdigest(),
        "operators": [
            {key: operator[key] for key in ("id", "name", "xinzhi_url", "xinzhi_sha256")}
            for operator in operators
        ],
    }
    index_path = _CACHE_DIR / "supplement.npz"
    manifest_path = _CACHE_DIR / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source"] == expected and manifest["index_sha256"] == _digest(index_path):
            extra = _load_index(index_path)
            if list(extra.operator_ids) != [operator["id"] for operator in operators]:
                raise ValueError("补充索引清单不一致")
            merged = _merge(builtin, extra)
            logger.info(f"已在线补全 {len(extra.agent_ids)} 名新密探心纸识别")
            return merged
    except Exception as exc:
        logger.bind(console=False).debug(f"心纸补充索引需重建：{exc}")

    background = _decode_png(background_bytes)
    with np.load(builtin.path, allow_pickle=False) as data:
        feature_scale = float(data["feature_scale"].item())
        base_size = tuple(int(value) for value in data["base_template_size"])
    if base_size != (70, 58):
        raise ValueError("内置索引基础模板尺寸不兼容")
    feature_variant = next(
        position for position, (scale, _) in enumerate(builtin.variants)
        if abs(scale - feature_scale) <= 1e-6
    )
    entries, images, variants, features = [], [], [], []
    for operator in operators:
        try:
            image = _compose(background, _icon(operator))
            resized = [
                cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                for scale, _ in builtin.variants
            ]
            if any(
                template.shape != templates.shape[1:]
                for template, (_, templates) in zip(resized, builtin.variants)
            ):
                raise ValueError("补充模板尺寸与内置索引不一致")
            feature = _feature(resized[feature_variant], builtin.feature_size)
            entries.append(operator)
            images.append(image)
            variants.append(resized)
            features.append(feature)
        except Exception as exc:
            _warn(operator, exc)
    if not entries:
        return builtin

    extra = AgentIndex(
        path=index_path, entity_type="agent",
        entity_types=np.asarray(["agent"] * len(entries)),
        agent_ids=np.asarray([operator["id"] for operator in entries]),
        operator_ids=np.asarray([operator["id"] for operator in entries]),
        operator_names=np.asarray([operator["name"] for operator in entries]),
        features=np.ascontiguousarray(np.stack(features), dtype=np.float32),
        variants=tuple(
            (scale, np.stack([entry[position] for entry in variants]))
            for position, (scale, _) in enumerate(builtin.variants)
        ),
        feature_size=builtin.feature_size, refine_groups=(),
    )
    merged = _merge(builtin, extra)
    accepted = []
    for position, (operator, image) in enumerate(zip(entries, images)):
        try:
            _validate(image, len(builtin.agent_ids) + position, merged, len(builtin.agent_ids))
            accepted.append(position)
        except Exception as exc:
            _warn(operator, exc)
    if not accepted:
        return builtin
    extra = replace(
        extra,
        entity_types=extra.entity_types[accepted], agent_ids=extra.agent_ids[accepted],
        operator_ids=extra.operator_ids[accepted], operator_names=extra.operator_names[accepted],
        features=extra.features[accepted],
        variants=tuple((scale, templates[accepted]) for scale, templates in extra.variants),
    )
    if len(accepted) != len(entries):
        del merged
        merged = _merge(builtin, extra)
    payload = {
        "entity_type": np.asarray("agent"), "agent_ids": extra.agent_ids,
        "operator_ids": extra.operator_ids, "operator_names": extra.operator_names,
        "features_gray": extra.features, "feature_size": np.asarray(extra.feature_size),
        "feature_scale": np.asarray(feature_scale, dtype=np.float32),
        "base_template_size": np.asarray(base_size),
        "scales": np.asarray([scale for scale, _ in extra.variants], dtype=np.float32),
    }
    payload.update({_template_array_key(scale): templates for scale, templates in extra.variants})
    try:
        with _atomic_file(index_path) as stream:
            np.savez_compressed(stream, **payload)
        manifest = {"source": expected, "index_sha256": _digest(index_path)}
        with _atomic_file(manifest_path) as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
    except Exception as exc:
        logger.warning("心纸补充索引缓存写入失败，本次仍使用已补全模板")
        logger.bind(console=False).warning(f"心纸缓存写入失败：{exc}")
    logger.info(f"已在线补全 {len(accepted)} 名新密探心纸识别")
    return merged


def _warn(operator: dict, error: Exception):
    logger.warning(f"心纸在线补全失败：{operator.get('name') or operator['id']}，已跳过")
    logger.bind(console=False).warning(f"心纸在线补全失败：{operator['id']}：{error}")


def supplement_xinzhi_index(builtin: AgentIndex, catalog_path: Path, target_ids: tuple[str, ...]) -> AgentIndex:
    try:
        catalog = load_xinzhi_catalog(catalog_path)
        existing = {
            str(operator_id) for entity_type, operator_id in zip(builtin.entity_types, builtin.operator_ids)
            if str(entity_type) == "agent"
        }
        missing = set(target_ids) - existing
        operators = [
            operator for operator in catalog["OPERATORS"]
            if operator["id"] in missing
            and operator.get("xinzhi_url") and operator.get("xinzhi_sha256")
        ]
        if not operators:
            return builtin
        with _LOCK:
            return _supplement(builtin, operators)
    except Exception as exc:
        logger.warning("心纸在线补全失败，继续使用内置识别模板")
        logger.bind(console=False).warning(f"心纸在线补全失败：{exc}")
        return builtin
