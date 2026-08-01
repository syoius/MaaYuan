import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from .logger import logger


FileValidator = Callable[[Path], bool]


@dataclass(frozen=True)
class SyncResult:
    path: Path
    temporary: bool = False
    updated: bool = False

    def cleanup(self):
        if not self.temporary:
            return
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            logger.warning(f"清理临时更新文件失败: {self.path}")


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_meta(path: Path, display_name: str) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning(f"读取 {display_name} 同步元数据失败，将重建元数据")
        return {}


def _write_meta(path: Path, meta: Dict, display_name: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except Exception:
        logger.warning(f"写入 {display_name} 同步元数据失败")


def _validate(path: Path, validator: FileValidator, source: str) -> bool:
    if not path.exists():
        return False
    try:
        if validator(path):
            return True
    except Exception as error:
        logger.warning(f"{source}校验失败: {error}")
        return False
    logger.warning(f"{source}结构无效")
    return False


def _fetch_remote_file(
    remote_url: str,
    meta: Dict,
    use_conditional: bool,
    timeout_sec: int,
    max_size_bytes: int,
    user_agent: str,
    display_name: str,
) -> Tuple[str, Optional[bytes], Dict]:
    headers = {"User-Agent": user_agent}
    if use_conditional:
        etag = str(meta.get("etag", "") or "").strip()
        last_modified = str(meta.get("last_modified", "") or "").strip()
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

    request = urllib_request.Request(remote_url, headers=headers, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=timeout_sec) as response:
            status = int(getattr(response, "status", response.getcode()))
            if status != 200:
                logger.warning(f"拉取远程 {display_name} 失败，HTTP {status}")
                return "error", None, {}

            payload = response.read(max_size_bytes + 1)
            if len(payload) > max_size_bytes:
                logger.error(f"远程 {display_name} 超过大小限制，忽略本次更新")
                return "error", None, {}

            updates: Dict[str, str] = {
                "remote_hash": hashlib.sha256(payload).hexdigest()
            }
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
            if etag:
                updates["etag"] = str(etag)
            if last_modified:
                updates["last_modified"] = str(last_modified)
            return "updated", payload, updates
    except urllib_error.HTTPError as error:
        if error.code == 304:
            updates: Dict[str, str] = {}
            etag = error.headers.get("ETag") if error.headers else None
            last_modified = (
                error.headers.get("Last-Modified") if error.headers else None
            )
            if etag:
                updates["etag"] = str(etag)
            if last_modified:
                updates["last_modified"] = str(last_modified)
            return "not_modified", None, updates
        logger.warning(f"拉取远程 {display_name} 失败，HTTP {error.code}")
    except urllib_error.URLError as error:
        logger.warning(f"拉取远程 {display_name} 失败: {error}")
    except Exception:
        logger.exception(f"拉取远程 {display_name} 时发生异常")
    return "error", None, {}


def sync_remote_file(
    local_path: Path,
    remote_url: str,
    validator: FileValidator,
    *,
    display_name: Optional[str] = None,
    meta_path: Optional[Path] = None,
    check_interval_sec: int = 15 * 24 * 60 * 60,
    retry_interval_sec: int = 12 * 60 * 60,
    timeout_sec: int = 8,
    max_size_bytes: int = 20 * 1024 * 1024,
    user_agent: str = "MaaY-RemoteFileSync/1.0",
    force_remote_check: bool = False,
) -> SyncResult:
    """检查并原子更新本地文件，失败时回退到可用的本地副本。"""
    local_path = Path(local_path)
    display_name = display_name or local_path.name
    meta_path = meta_path or local_path.with_suffix(".sync.meta")

    local_available = _validate(
        local_path, validator, f"本地 {display_name}"
    )
    meta = _read_meta(meta_path, display_name)
    now = int(time.time())
    last_check_ts = _to_int(meta.get("last_check_ts"), 0)
    interval = check_interval_sec if local_available else retry_interval_sec

    if not force_remote_check and now - last_check_ts < max(1, interval):
        return SyncResult(local_path)

    if not local_available:
        logger.warning(f"本地 {display_name} 不可用，尝试从远程下载")

    status, payload, meta_updates = _fetch_remote_file(
        remote_url=remote_url,
        meta=meta,
        use_conditional=local_available,
        timeout_sec=timeout_sec,
        max_size_bytes=max_size_bytes,
        user_agent=user_agent,
        display_name=display_name,
    )

    meta["last_check_ts"] = now
    if status == "not_modified":
        meta["last_success_ts"] = now
        meta.update(meta_updates)
        _write_meta(meta_path, meta, display_name)
        return SyncResult(local_path)

    if status == "updated" and payload is not None:
        temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as file:
                file.write(payload)

            if not _validate(temp_path, validator, f"远程 {display_name}"):
                temp_path.unlink(missing_ok=True)
                _write_meta(meta_path, meta, display_name)
                return SyncResult(local_path)

            remote_hash = str(meta_updates.get("remote_hash", "") or "")
            local_hash = _file_hash(local_path) if local_available else ""
            meta.update(meta_updates)
            meta["last_success_ts"] = now

            if local_available and remote_hash == local_hash:
                temp_path.unlink(missing_ok=True)
                _write_meta(meta_path, meta, display_name)
                return SyncResult(local_path)

            try:
                temp_path.replace(local_path)
                logger.info(f"{display_name} 已更新为远程版本")
                _write_meta(meta_path, meta, display_name)
                return SyncResult(local_path, updated=True)
            except Exception:
                logger.warning(
                    f"写入本地 {display_name} 失败，当前运行将直接使用远程数据"
                )
                _write_meta(meta_path, meta, display_name)
                return SyncResult(temp_path, temporary=True, updated=True)
        except Exception:
            logger.exception(f"写入 {display_name} 更新文件失败")
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    _write_meta(meta_path, meta, display_name)
    if not local_available:
        logger.error(f"远程 {display_name} 拉取失败，且本地无可用数据")
    else:
        logger.warning(f"远程 {display_name} 检查失败，继续使用本地数据")
    return SyncResult(local_path)
