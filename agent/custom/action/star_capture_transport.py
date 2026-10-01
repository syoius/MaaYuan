"""Private B4 main-star transport: manifest construction and multipart upload only."""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, TypeAlias
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote

from utils import logger

from custom.action.inventory_reporting import (
    AUTO_UPLOAD_MODE,
    UploadResult,
    UploadSettings,
    _retry_wait,
    read_upload_settings,
)


STAR_CAPTURE_PATH = "/open-api/star/captures"
INIT_UPLOAD_TIMEOUT_SECONDS = 10.0
IMAGE_UPLOAD_TIMEOUT_SECONDS = 20.0
FINALIZE_UPLOAD_TIMEOUT_SECONDS = 10.0
_IMAGE_NAME = re.compile(r"^capture-\d{2}\.png$")
_SAFE_CAPTURE_ID = re.compile(r"^[A-Za-z0-9:_-]{1,160}$")
_SECTION_NAMES = ("main", "support", "experience")
_WireImage: TypeAlias = tuple[str, Path]


def build_main_capture_manifest(
    run_dir: Path, session: dict[str, Any], game_version: str
) -> tuple[dict[str, Any], list[Path]]:
    """Build the smoke manifest without changing B capture diagnostics or images."""
    if session.get("success") is not True or session.get("stop_reason") != "bottom_no_move":
        raise ValueError("只有 bottom_no_move 成功主星采集可以上传")
    if game_version not in {"如鸢", "代号鸢"}:
        raise ValueError("game_version 必须是 如鸢 或 代号鸢")
    image_names = session.get("retained_images")
    if not isinstance(image_names, list) or not image_names:
        raise ValueError("连续采集缺少 retained_images")
    if len(set(image_names)) != len(image_names):
        raise ValueError("连续采集 retained_images 重复")

    resolved_dir = run_dir.resolve()
    paths: list[Path] = []
    for name in image_names:
        if not isinstance(name, str) or not _IMAGE_NAME.fullmatch(name):
            raise ValueError("连续采集图片文件名无效")
        path = (resolved_dir / name).resolve()
        if path.parent != resolved_dir or not path.is_file():
            raise ValueError("连续采集图片不存在")
        paths.append(path)

    # run_dir is immutable for one successful capture, so retries retain this ID
    # without adding mutable transport fields to session.json.
    capture_id = "star-" + uuid.uuid5(uuid.NAMESPACE_URL, resolved_dir.as_uri()).hex
    source_id_by_name = {
        path.name: f"{capture_id}:main:{index:03d}"
        for index, path in enumerate(paths)
    }
    relations: list[dict[str, str]] = []
    raw_relations = session.get("adjacent_relations")
    if not isinstance(raw_relations, list):
        raise ValueError("连续采集 adjacent_relations 无效")
    for relation in raw_relations:
        if not isinstance(relation, dict) or relation.get("relation") != "overlap":
            raise ValueError("连续采集 overlap 关系无效")
        previous = source_id_by_name.get(str(relation.get("previous_image", "")))
        current = source_id_by_name.get(str(relation.get("current_image", "")))
        if not previous or not current:
            raise ValueError("连续采集 overlap 未指向保留图片")
        relations.append(
            {
                "previous_source_image_id": previous,
                "current_source_image_id": current,
                "relation": "overlap",
            }
        )
    return (
        {
            "schema_version": 1,
            "capture_id": capture_id,
            "game_version": game_version,
            "section": "main",
            "stop_reason": "bottom_no_move",
            "images": [
                {
                    "source_image_id": source_id_by_name[path.name],
                    "source_order": index + 1,
                    "file_name": path.name,
                }
                for index, path in enumerate(paths)
            ],
            "adjacent_relations": relations,
        },
        paths,
    )


def build_full_capture_manifest(run_dir: Path, batch: dict[str, Any]) -> tuple[dict[str, Any], list[_WireImage]]:
    """Translate the canonical local CaptureBatchV1 at the transport boundary."""
    if batch.get("schemaVersion") != 1 or batch.get("source") != "maayuan":
        raise ValueError("capture-batch.json 不是 CaptureBatchV1")
    capture_id = str(batch.get("captureId", "")).strip()
    game_version = str(batch.get("gameVersion", "")).strip()
    sections = batch.get("sections")
    if not _SAFE_CAPTURE_ID.fullmatch(capture_id) or game_version not in {"如鸢", "代号鸢"}:
        raise ValueError("capture-batch.json 顶层字段无效")
    if not isinstance(sections, dict) or set(sections) != set(_SECTION_NAMES):
        raise ValueError("capture-batch.json 必须包含完整的三段 sections")

    resolved_run_dir = run_dir.resolve()
    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    seen_wire_names: set[str] = set()
    wire_images: list[_WireImage] = []
    wire_sections: dict[str, dict[str, Any]] = {}
    for section_name in _SECTION_NAMES:
        section = sections[section_name]
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} section 无效")
        images = section.get("images")
        relations = section.get("adjacentRelations")
        complete = section.get("complete")
        stop_reason = section.get("stopReason")
        if not isinstance(images, list) or not isinstance(relations, list):
            raise ValueError(f"{section_name} section 图片或关联无效")
        expected_stop = "single_capture" if section_name == "experience" else "bottom_no_move"
        if complete is not True or stop_reason != expected_stop:
            raise ValueError(f"{section_name} section 未完成或停止原因无效")
        if section_name == "experience" and (len(images) != 1 or relations):
            raise ValueError("experience 必须恰有一张图片且无关联")
        if section_name != "experience" and not images:
            raise ValueError(f"{section_name} 必须至少有一张图片")

        source_orders: dict[str, int] = {}
        converted_images: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            if not isinstance(image, dict):
                raise ValueError(f"{section_name} 图片无效")
            source_id = str(image.get("sourceImageId", "")).strip()
            source_order = image.get("sourceOrder")
            local_name = str(image.get("fileName", "")).strip()
            expected_local_name = f"{section_name}/capture-{index:02d}.png"
            if (
                not _SAFE_CAPTURE_ID.fullmatch(source_id)
                or not isinstance(source_order, int)
                or source_order < 1
                or local_name != expected_local_name
                or source_id in seen_ids
                or source_order in seen_orders
            ):
                raise ValueError(f"{section_name} 图片字段无效")
            local_path = (resolved_run_dir / local_name).resolve()
            if local_path.parent != (resolved_run_dir / section_name) or not local_path.is_file():
                raise ValueError(f"{section_name} 图片不存在或越界")
            wire_name = f"{section_name}-{index:03d}.png"
            if wire_name in seen_wire_names:
                raise ValueError("传输图片名重复")
            seen_ids.add(source_id)
            seen_orders.add(source_order)
            seen_wire_names.add(wire_name)
            source_orders[source_id] = source_order
            wire_images.append((wire_name, local_path))
            converted_images.append({"source_image_id": source_id, "source_order": source_order, "file_name": wire_name})
        converted_relations: list[dict[str, str]] = []
        relation_keys: set[tuple[str, str]] = set()
        for relation in relations:
            if not isinstance(relation, dict):
                raise ValueError(f"{section_name} overlap 关系无效")
            previous = str(relation.get("previousSourceImageId", "")).strip()
            current = str(relation.get("currentSourceImageId", "")).strip()
            key = (previous, current)
            if (
                relation.get("relation") != "overlap"
                or previous not in source_orders
                or current not in source_orders
                or source_orders[current] != source_orders[previous] + 1
                or key in relation_keys
            ):
                raise ValueError(f"{section_name} overlap 关系无效")
            relation_keys.add(key)
            converted_relations.append({"previous_source_image_id": previous, "current_source_image_id": current, "relation": "overlap"})
        wire_sections[section_name] = {"images": converted_images, "adjacent_relations": converted_relations, "complete": True, "stop_reason": expected_stop}
    if sorted(seen_orders) != list(range(1, len(seen_orders) + 1)):
        raise ValueError("CaptureBatchV1 source_order 必须全局连续")
    return ({"schema_version": 1, "capture_id": capture_id, "source": "maayuan", "game_version": game_version, "sections": wire_sections}, wire_images)


def _multipart_body(manifest: dict[str, Any], paths: list[Path] | list[_WireImage]) -> tuple[bytes, str]:
    boundary = "----MaaYuanStarCapture" + uuid.uuid4().hex
    chunks: list[bytes] = []

    def add(headers: list[str], payload: bytes) -> None:
        chunks.append(("--" + boundary + "\r\n").encode("ascii"))
        chunks.extend((header + "\r\n").encode("utf-8") for header in headers)
        chunks.append(b"\r\n")
        chunks.append(payload)
        chunks.append(b"\r\n")

    add(
        [
            'Content-Disposition: form-data; name="manifest"',
            "Content-Type: application/json; charset=utf-8",
        ],
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )
    for item in paths:
        wire_name, path = item if isinstance(item, tuple) else (item.name, item)
        add(
            [
                f'Content-Disposition: form-data; name="files"; filename="{wire_name}"',
                "Content-Type: image/png",
            ],
            path.read_bytes(),
        )
    chunks.append(("--" + boundary + "--\r\n").encode("ascii"))
    return b"".join(chunks), boundary


def upload_full_capture_batch(
    context: Any,
    run_dir: Path,
    *,
    init_timeout_seconds: float = INIT_UPLOAD_TIMEOUT_SECONDS,
    image_timeout_seconds: float = IMAGE_UPLOAD_TIMEOUT_SECONDS,
    finalize_timeout_seconds: float = FINALIZE_UPLOAD_TIMEOUT_SECONDS,
    max_attempts: int = 3,
) -> UploadResult | None:
    """Upload a complete local three-section batch only in existing auto-upload mode."""
    settings = read_upload_settings(context)
    if settings.mode != AUTO_UPLOAD_MODE:
        return None
    try:
        batch = json.loads((run_dir / "capture-batch.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取 capture-batch.json") from exc
    manifest, paths = build_full_capture_manifest(run_dir, batch)
    if not settings.token or min(init_timeout_seconds, image_timeout_seconds, finalize_timeout_seconds) <= 0 or max_attempts < 1:
        raise ValueError("星石截图上传参数无效")
    started = time.monotonic()
    capture_id = manifest["capture_id"]
    sizes = [path.stat().st_size for _, path in paths]
    logger.info(
        f"star_capture_upload capture_id={capture_id} image_count={len(paths)} "
        f"total_image_bytes={sum(sizes)} largest_image_bytes={max(sizes, default=0)} max_attempts={max_attempts}"
    )

    def finish(result: UploadResult) -> UploadResult:
        logger.info(f"star_capture_upload capture_id={capture_id} result={'success' if result.success else 'failure'} "
                    f"http_status={result.status_code} total_elapsed_seconds={time.monotonic() - started:.6f}")
        return result

    body = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    result, _ = _upload_request(settings, STAR_CAPTURE_PATH + "/init", body, "application/json; charset=utf-8",
                                capture_id, "init", init_timeout_seconds, max_attempts)
    if not result.success:
        return finish(result)
    path_by_name = dict(paths)
    images = sorted((image for section in manifest["sections"].values() for image in section["images"]),
                    key=lambda image: image["source_order"])
    capture_path = STAR_CAPTURE_PATH + "/" + quote(capture_id, safe="")
    for image in images:
        image_bytes = path_by_name[image["file_name"]].read_bytes()
        body, boundary = _image_multipart_body(image["file_name"], image_bytes)
        result, _ = _upload_request(
            settings, capture_path + "/images/" + quote(image["source_image_id"], safe=""), body,
            f"multipart/form-data; boundary={boundary}", capture_id, "image", image_timeout_seconds, max_attempts,
            source_image_id=image["source_image_id"],
            metrics=f"source_order={image['source_order']} image_bytes={len(image_bytes)} multipart_body_bytes={len(body)}",
        )
        if not result.success:
            return finish(result)
    result, missing_ids = _upload_request(settings, capture_path + "/finalize", b"", "application/json",
                                         capture_id, "finalize", finalize_timeout_seconds, max_attempts,
                                         metrics=f"image_count={len(images)}")
    if missing_ids:
        logger.info(f"star_capture_upload stage=finalize capture_id={capture_id} missing_count={len(missing_ids)} "
                    f"missing_source_image_ids={','.join(missing_ids)}")
    return finish(result)


def _image_multipart_body(file_name: str, image_bytes: bytes) -> tuple[bytes, str]:
    boundary = "----MaaYuanStarImage" + uuid.uuid4().hex
    header = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
              'Content-Type: image/png\r\n\r\n').encode("utf-8")
    return header + image_bytes + f"\r\n--{boundary}--\r\n".encode("ascii"), boundary


def _upload_request(
    settings: UploadSettings, path: str, body: bytes, content_type: str, capture_id: str, stage: str,
    timeout_seconds: float, max_attempts: int, *, source_image_id: str | None = None, metrics: str = "",
) -> tuple[UploadResult, list[str]]:
    request = urllib_request.Request(settings.base_url + path, data=body, method="POST", headers={
        "Accept": "application/json", "Authorization": f"Bearer {settings.token}",
        "Content-Type": content_type, "User-Agent": "MaaYuan-StarCapture/1",
    })
    for attempt in range(max_attempts):
        started = time.monotonic()
        status = None
        response_body = b""
        exception = None
        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                status = int(response.status)
                response_body = response.read()
        except urllib_error.HTTPError as exc:
            status, exception = int(exc.code), exc
            try:
                with exc:
                    response_body = exc.read()
            except (urllib_error.URLError, TimeoutError, OSError) as read_error:
                exception = read_error
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            exception = exc
        reason = getattr(exception, "reason", None)
        logger.info(f"star_capture_upload stage={stage} capture_id={capture_id} "
                    f"source_image_id={source_image_id or 'none'} {metrics} attempt={attempt + 1}/{max_attempts} "
                    f"timeout_seconds={timeout_seconds} http_attempt_elapsed_seconds={time.monotonic() - started:.6f} "
                    f"http_status={status} exception_class={type(exception).__name__ if exception else 'none'} "
                    f"reason_class={type(reason).__name__ if isinstance(reason, BaseException) else 'none'}")
        missing_ids: list[str] = []
        try:
            payload = json.loads(response_body.decode("utf-8"))
            data = payload.get("data") or {}
            missing_ids = [item for item in data.get("missing_source_image_ids", [])
                           if isinstance(item, str) and _SAFE_CAPTURE_ID.fullmatch(item)]
            valid = (isinstance(data, dict) and data.get("capture_id") == capture_id
                     and payload.get("status_code") == 200
                     and (source_image_id is None or data.get("source_image_id") == source_image_id))
        except (ValueError, AttributeError, TypeError):
            valid = False
        if status is not None and 200 <= status < 300:
            if valid and not missing_ids:
                return UploadResult(True, status, f"{stage}: HTTP {status}"), []
            return UploadResult(False, status, f"{stage}: invalid upload response"), missing_ids
        if (status is None or status in {408, 429} or status >= 500) and attempt + 1 < max_attempts:
            _retry_wait(attempt)
            continue
        message = f"{stage}: HTTP {status}" if status is not None else f"{stage}: network failure"
        if missing_ids:
            message += "; missing sourceImageIds=" + ",".join(missing_ids)
        return UploadResult(False, status, message), missing_ids
    raise RuntimeError("星石截图上传重试循环异常结束")


def upload_main_capture_manifest(
    manifest: dict[str, Any],
    paths: list[Path] | list[_WireImage],
    settings: UploadSettings,
    timeout_seconds: float = 20.0,
    max_attempts: int = 3,
) -> UploadResult:
    if settings.mode != AUTO_UPLOAD_MODE or not settings.token:
        raise ValueError("只有配置 Token 的自动上报模式可以上传星石截图")
    if timeout_seconds <= 0 or max_attempts < 1:
        raise ValueError("星石截图上传参数无效")
    round_started = time.monotonic()
    capture_id = manifest.get("capture_id", "")
    if not isinstance(capture_id, str) or not _SAFE_CAPTURE_ID.fullmatch(capture_id):
        capture_id = "invalid"
    build_started = time.monotonic()
    try:
        image_sizes = [(item[1] if isinstance(item, tuple) else item).stat().st_size for item in paths]
        body, boundary = _multipart_body(manifest, paths)
    except Exception as exc:
        logger.info(
            f"star_capture_upload capture_id={capture_id} result=build_failure "
            f"exception_class={type(exc).__name__} "
            f"multipart_build_elapsed_seconds={time.monotonic() - build_started:.6f}"
        )
        raise
    logger.info(
        f"star_capture_upload capture_id={capture_id} image_count={len(image_sizes)} "
        f"total_image_bytes={sum(image_sizes)} largest_image_bytes={max(image_sizes, default=0)} "
        f"multipart_body_bytes={len(body)} timeout_seconds={timeout_seconds} "
        f"max_attempts={max_attempts} "
        f"multipart_build_elapsed_seconds={time.monotonic() - build_started:.6f}"
    )

    def log_attempt(attempt: int, started: float, status: int | None, exception: BaseException | None = None) -> None:
        reason = getattr(exception, "reason", None)
        failure_class = type(reason).__name__ if isinstance(reason, BaseException) else "none"
        logger.info(
            f"star_capture_upload capture_id={capture_id} attempt={attempt + 1}/{max_attempts} "
            f"http_attempt_elapsed_seconds={time.monotonic() - started:.6f} "
            f"http_status={status} exception_class={type(exception).__name__ if exception else 'none'} "
            f"reason_class={failure_class}"
        )

    def finish(result: UploadResult, attempt: int) -> UploadResult:
        logger.info(
            f"star_capture_upload capture_id={capture_id} "
            f"result={'success' if result.success else 'failure'} attempts={attempt + 1}/{max_attempts} "
            f"http_status={result.status_code} total_elapsed_seconds={time.monotonic() - round_started:.6f}"
        )
        return result

    request = urllib_request.Request(
        settings.base_url + STAR_CAPTURE_PATH,
        data=body,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {settings.token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "MaaYuan-StarCapture/1",
        },
        method="POST",
    )
    for attempt in range(max_attempts):
        attempt_started = time.monotonic()
        status_code = None
        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                status_code = int(getattr(response, "status", response.getcode()))
                response_body = response.read().decode("utf-8", errors="replace")
            log_attempt(attempt, attempt_started, status_code)
            if 200 <= status_code < 300:
                return finish(UploadResult(True, status_code, f"HTTP {status_code}"), attempt)
            if status_code >= 500 and attempt + 1 < max_attempts:
                _retry_wait(attempt)
                continue
            return finish(UploadResult(False, status_code, _response_message(status_code, response_body)), attempt)
        except urllib_error.HTTPError as exc:
            log_attempt(attempt, attempt_started, int(exc.code), exc)
            response_body = exc.read().decode("utf-8", errors="replace")
            if exc.code >= 500 and attempt + 1 < max_attempts:
                _retry_wait(attempt)
                continue
            return finish(UploadResult(False, int(exc.code), _response_message(exc.code, response_body)), attempt)
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            log_attempt(attempt, attempt_started, status_code, exc)
            if attempt + 1 < max_attempts:
                _retry_wait(attempt)
                continue
            return finish(UploadResult(False, None, f"网络连接失败：{getattr(exc, 'reason', exc)}"), attempt)
    raise RuntimeError("星石截图上传重试循环异常结束")


def _response_message(status_code: int, response_body: str) -> str:
    try:
        payload = json.loads(response_body)
        message = payload.get("message") or (payload.get("error") or {}).get("message")
        if isinstance(message, str) and message.strip():
            return f"HTTP {status_code}：{message.strip()}"
    except (json.JSONDecodeError, AttributeError):
        pass
    return f"HTTP {status_code}"
