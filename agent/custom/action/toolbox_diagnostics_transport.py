"""Bounded, authenticated diagnostic delivery; never follow redirects."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib import error, request
from urllib.parse import quote, urlsplit
from uuid import uuid4

MAX_MANIFEST = 64 * 1024
MAX_IMAGE = 512 * 1024
MAX_REQUEST = 2 * 1024 * 1024
MAX_RESPONSE = 16 * 1024
TRUSTED_BASE = "https://api-hub.maayuan.com"


def trusted_base(value):
    parsed = urlsplit(value)
    return (parsed.scheme == "https" and parsed.hostname == "api-hub.maayuan.com"
            and parsed.port in (None, 443) and not parsed.username and not parsed.password
            and parsed.path in ("", "/") and not parsed.query and not parsed.fragment)


def trusted_upload_url(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "https" and parsed.port in (None, 443)
                and bool(parsed.hostname) and parsed.hostname.endswith(".myqcloud.com")
                and parsed.username is None and parsed.password is None)
    except ValueError:
        return False


def multipart(manifest, files):
    if len(manifest) > MAX_MANIFEST or len(files) > 3:
        raise ValueError("diagnostics_size")
    boundary = "MaaYuan" + uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="manifest"; filename="manifest.json"\r\nContent-Type: application/json\r\n\r\n'.encode(), manifest, b"\r\n"]
    for index, data in enumerate(files):
        if not data.startswith(b"\xff\xd8") or len(data) > MAX_IMAGE:
            raise ValueError("diagnostics_image")
        parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="crop-{index}.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode(), data, b"\r\n"])
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    if len(body) > MAX_REQUEST:
        raise ValueError("diagnostics_size")
    return body, f"multipart/form-data; boundary={boundary}"


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_once(base_url, token, body, content_type, timeout, path="/open-api/maayuan/diagnostics"):
    from utils.remote_file_sync import create_https_context

    if not trusted_base(base_url):
        return {"status": 400}
    opener = request.build_opener(_NoRedirect(), request.HTTPSHandler(context=create_https_context()))
    req = request.Request(base_url.rstrip("/") + path, data=body,
                          headers={"Authorization": "Bearer " + token, "Content-Type": content_type}, method="POST")
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE + 1)
            payload = json.loads(raw) if len(raw) <= MAX_RESPONSE else {}
            return {"status": response.status, "payload": payload}
    except error.HTTPError as exc:
        # Do not read, retain or log a server error body.
        retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
        exc.close()
        return {"status": exc.code, "retry_after": retry_after[:128]}
    except (ValueError, UnicodeError):
        return {"status": 200, "payload": {}}
    except (error.URLError, OSError, TimeoutError):
        return {"status": 0}


def _put_once(url, headers, body, timeout):
    from utils.remote_file_sync import create_https_context

    if not trusted_upload_url(url):
        return {"status": 400}
    try:
        opener = request.build_opener(_NoRedirect(), request.HTTPSHandler(context=create_https_context()))
        opener.addheaders = []
        req = request.Request(url, data=body, headers=headers, method="PUT")
        with opener.open(req, timeout=timeout) as response:
            return {"status": response.status}
    except error.HTTPError as exc:
        status = exc.code
        exc.close()
        return {"status": status}
    except (error.URLError, OSError, ValueError):
        return {"status": 0}


def _isolated_once(base_url, token, body, content_type, timeout, *, path="/open-api/maayuan/diagnostics", upload=None):
    # A killable process bounds DNS, TLS and slow-drip reads too; socket timeouts cannot.
    payload = json.dumps({"base_url": base_url, "token": token, "body": base64.b64encode(body).decode(),
                          "content_type": content_type, "timeout": timeout, "path": path, "upload": upload}).encode()
    try:
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve())], input=payload,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout, check=False,
                                **({"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}))
        return json.loads(result.stdout) if result.returncode == 0 else {"status": 0}
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return {"status": 0}


def retry_delay(value):
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else 0.5
    except (ValueError, TypeError):
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0.5


def _upload_one(upload, data):
    if not isinstance(upload, dict):
        return False
    headers = upload.get("headers")
    if (upload.get("method") != "PUT" or not trusted_upload_url(upload.get("url"))
            or not isinstance(headers, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   or "\r" in key + value or "\n" in key + value for key, value in headers.items())):
        return False
    result = _isolated_once("", "", data, "", 15.0,
                            upload={"url": upload["url"], "headers": headers})
    return 200 <= result.get("status", 0) < 300


def _upload_images(payload, files, base_url, token, event_id, log=None):
    from utils import logger

    uploads = payload.get("uploads", [])
    uploads = uploads if isinstance(uploads, list) else []
    succeeded = 0
    for index, data in enumerate(files):
        matches = [item for item in uploads if isinstance(item, dict)
                   and type(item.get("index")) is int and item["index"] == index]
        if len(matches) != 1:
            continue
        if _upload_one(matches[0], data):
            succeeded += 1
    if files:
        logger.bind(console=False).info(f"诊断图片直传：成功 {succeeded} 张 / 失败 {len(files) - succeeded} 张")
    if log is not None:
        _upload_one(payload.get("log_upload"), log)
    if files or log is not None:
        _isolated_once(base_url, token, b"{}", "application/json", 5.0,
                       path="/open-api/maayuan/diagnostics/" + quote(event_id, safe="") + "/complete")


def deliver(manifest, files, base_url, token, *, direct=False, log=None):
    """Return (accepted|permanent|pending, retry epoch). No credential is persisted."""
    deadline = time.monotonic() + 8.0
    if not token or not trusted_base(base_url):
        return "permanent", 0
    document = json.loads(manifest)
    if document.get("feature") in {"inventory_bag_scan", "star_backpack_scan"}:
        from custom.action.toolbox_diagnostics import encode_manifest

        try:
            encode_manifest(document, (token,))
        except (ValueError, TypeError, KeyError):
            return "permanent", 0
        if document.get("feature") == "star_backpack_scan":
            log = None
    event_id = document["event_id"]
    if direct:
        from custom.action.toolbox_diagnostics_log import valid_log

        if log is not None and not valid_log(log):
            log = None
        if (len(manifest) > MAX_MANIFEST
                or len(files) > (12 if document.get("feature") == "inventory_bag_scan" else
                                 4 if document.get("feature") == "star_backpack_scan" else 3)
                or len(document.get("images", [])) != len(files)
                or any(not data.startswith(b"\xff\xd8") or not 1 <= len(data) <= MAX_IMAGE for data in files)):
            return "permanent", 0
        intent = {"manifest": document, "uploads": [
            {"size": len(data), "content_md5": base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode()}
            for data in files]}
        if log is not None:
            intent["log"] = {"size": len(log), "content_md5": base64.b64encode(hashlib.md5(log, usedforsecurity=False).digest()).decode()}
        body = json.dumps(intent, ensure_ascii=False, separators=(",", ":")).encode()
        if len(body) > MAX_MANIFEST:
            return "permanent", 0
        content_type = "application/json"
    else:
        body, content_type = multipart(manifest, files)
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        result = _isolated_once(base_url, token, body, content_type, min(5.0, remaining),
                                **({"path": "/open-api/maayuan/diagnostics/intents"} if direct else {}))
        status = result.get("status", 0)
        if direct and status == 400 and log is not None and document.get("feature") != "inventory_bag_scan":
            log = None
            intent.pop("log")
            body = json.dumps(intent, ensure_ascii=False, separators=(",", ":")).encode()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            result = _isolated_once(base_url, token, body, content_type, min(5.0, remaining),
                                    path="/open-api/maayuan/diagnostics/intents")
            status = result.get("status", 0)
        if direct and status in (404, 405, 409):
            if document.get("feature") in {"inventory_bag_scan", "star_backpack_scan"}:
                return "permanent", 0
            return deliver(manifest, files, base_url, token)
        if status == 200:
            payload = result.get("payload")
            if (isinstance(payload, dict) and payload.get("event_id") == event_id
                    and payload.get("status") in {"accepted", "duplicate", "accepted_without_images"}):
                if direct and payload["status"] == "accepted":
                    _upload_images(payload, files, base_url, token, event_id, log)
                return "accepted", 0
            return "permanent", 0
        if status not in (0, 408, 429) and not 500 <= status <= 599:
            return "permanent", 0
        delay = retry_delay(result.get("retry_after", "0.5"))
        if attempt == 1 or delay >= deadline - time.monotonic():
            return "pending", time.time() + delay
        time.sleep(delay)
    return "pending", 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        data = json.loads(sys.stdin.buffer.read(3 * MAX_REQUEST))
        body = base64.b64decode(data["body"])
        if data.get("upload") is not None:
            result = _put_once(data["upload"]["url"], data["upload"]["headers"], body, data["timeout"])
        else:
            result = _http_once(data["base_url"], data["token"], body,
                                data["content_type"], data["timeout"], data["path"])
        sys.stdout.write(json.dumps(result))
    except Exception:
        sys.stdout.write('{"status":0}')
