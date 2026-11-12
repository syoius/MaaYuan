"""Collect and redact this diagnostic session's local file log before queuing."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import gzip
import io
import re
import zipfile
import zlib

MAX_LOG = 256 * 1024
MAX_LOG_TEXT = 200 * 1024
MAX_EXPANDED_LOG = 1024 * 1024
_TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) \|")
_VARIABLE = re.compile(r"^\s*[\u2500-\u257f].*(?:\n|$)", re.MULTILINE)
_HOME = re.compile(r"(?i)([a-z]:[\\/]+Users[\\/]+|/(?:Users|home)/)[^\\/\r\n\"'<>]+")
_CREDENTIAL = re.compile(
    r"(?i)((?:\b[a-z_]*token\b|连接码|\bauthorization\b|\bbearer\b)[\"']*\s*[:=]?\s*)"
    r"(?:bearer\s+)?(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;\"'<>}\]]+)"
)
_OPAQUE = re.compile(
    r"(?<![A-Za-z0-9_+/=-])(?=[A-Za-z0-9_+/=-]*[0-9])"
    r"(?=[A-Za-z0-9_+/=-]*[A-Za-z])[A-Za-z0-9_+/=-]{24,}(?![A-Za-z0-9_+/=-])"
)
_EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def redact_log(text, secrets):
    text = _VARIABLE.sub("", text)
    # Replace longest first so overlapping secrets cannot leave a suffix behind.
    for secret in sorted({value for value in secrets if isinstance(value, str) and value}, key=len, reverse=True):
        text = text.replace(secret, "***")
    text = _HOME.sub(r"\1***", text)
    text = _CREDENTIAL.sub(r"\1***", text)
    text = _OPAQUE.sub("***", text)
    return _EMAIL.sub("***", text)


def valid_log(data):
    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_LOG:
        return False
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            raw = stream.read(MAX_EXPANDED_LOG + 1)
        return len(raw) <= MAX_EXPANDED_LOG and bool(raw.decode("utf-8"))
    except (OSError, EOFError, UnicodeError, ValueError, zlib.error):
        return False


def collect_log(directory, started_at, secrets):
    def unavailable(reason, exception=None):
        try:
            from utils import logger

            logger.bind(console=False).warning(
                f"ToolboxDiagnostics: 日志收集失败 directory={redact_log(directory.name, secrets)!r}, "
                f"reason={reason}"
            )
        except Exception:
            # A logging failure must not prevent the diagnostic itself.
            pass
        try:
            def error_text(exc):
                return redact_log(f"{type(exc).__name__}: {exc}", secrets)[:300]

            details = [
                "（日志收集失败）",
                f"失败原因={reason}",
                f"异常类型={type(exception).__name__ if exception is not None else '无'}",
                f"异常消息={redact_log(str(exception), secrets)[:300]!r}" if exception is not None else "异常消息=无",
                f"会话开始时间={started_at.isoformat()}",
            ]
            try:
                details.append(f"日志目录是否存在={directory.exists()}")
            except Exception as exc:
                details.append(f"日志目录是否存在=未知 ({error_text(exc)!r})")
            try:
                details.append(f"当前工作目录名={Path.cwd().name!r}")
            except Exception as exc:
                details.append(f"当前工作目录名=未知 ({error_text(exc)!r})")

            files = []
            try:
                for path in directory.glob("*.log*"):
                    try:
                        stat = path.stat()
                        files.append((path, stat.st_mtime, f"{stat.st_size} bytes"))
                    except Exception as exc:
                        files.append((path, float("-inf"), f"未知 ({error_text(exc)!r})"))
            except Exception as exc:
                details.append(f"日志文件枚举异常={error_text(exc)!r}")
            files.sort(key=lambda item: (item[1], item[0].name), reverse=True)
            details.append("日期相关日志文件（最多10个）:")
            details.extend(f"  {path.name!r}: {size}" for path, _, size in files[:10])
            if not files:
                details.append("  无")

            first_timestamp = last_timestamp = "不可读取"
            if files:
                latest = files[0][0]
                details.append(f"最新日志文件名={latest.name!r}")

                def timestamps(stream):
                    first = last = next(stream, "")
                    for line in stream:
                        last = line
                    return tuple(
                        match[1] if (match := _TIMESTAMP.match(line)) else "未识别"
                        for line in (first, last)
                    )

                try:
                    if latest.suffix == ".zip":
                        with zipfile.ZipFile(latest) as archive:
                            names = sorted(name for name in archive.namelist() if name.endswith(".log"))
                            if names:
                                with archive.open(names[-1]) as stream:
                                    first_timestamp, last_timestamp = timestamps(
                                        io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
                                    )
                    else:
                        with latest.open(encoding="utf-8", errors="replace") as stream:
                            first_timestamp, last_timestamp = timestamps(stream)
                except Exception as exc:
                    details.append(f"日志时间戳读取异常={error_text(exc)!r}")
            details.extend((f"最新日志首行时间戳={first_timestamp}", f"最新日志末行时间戳={last_timestamp}"))
            raw = redact_log("\n".join(details), secrets).encode("utf-8")
            data = gzip.compress(raw, mtime=0)
            return data if valid_log(data) else None
        except Exception:
            return None

    try:
        from utils import logger

        if not directory.is_dir():
            return unavailable("日志目录不存在或不是目录")
        complete = getattr(logger, "complete", None)
        if callable(complete):
            complete()
        lines = []
        include = False

        def read(stream):
            nonlocal include
            for line in stream:
                match = _TIMESTAMP.match(line)
                if match:
                    include = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S.%f") >= started_at
                if include:
                    lines.append(line)

        today = datetime.now().date().isoformat()
        # Loguru may zip the previous day's log during midnight rotation.
        for path in sorted(directory.glob("*.log*")):
            if not started_at.date().isoformat() <= path.name[:10] <= today:
                continue
            include = False
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for name in sorted(archive.namelist()):
                        with archive.open(name) as stream:
                            read(io.TextIOWrapper(stream, encoding="utf-8", errors="replace"))
            elif path.suffix == ".log":
                with path.open(encoding="utf-8", errors="replace") as stream:
                    read(stream)
        text = redact_log("".join(lines), secrets)
        raw = "".join(
            line[:1000] + "…（本行已截断）\n" if len(line.rstrip("\r\n")) > 1000 else line
            for line in text.splitlines(keepends=True)
        ).encode("utf-8")
        if len(raw) > MAX_LOG_TEXT:
            raw = "（前面的日志已截断）\n".encode("utf-8") + raw[-MAX_LOG_TEXT:].decode("utf-8", errors="ignore").encode("utf-8")
        if not raw:
            return unavailable("没有匹配的日志条目")
        data = gzip.compress(raw, mtime=0)
        if len(data) > MAX_LOG:
            return unavailable("压缩后超过大小上限")
        if not valid_log(data):
            return unavailable("压缩日志无效或解压后超过大小上限")
        return data
    except Exception as exc:
        # File logging is optional and must never prevent the diagnostic itself.
        return unavailable(f"读取或处理异常({type(exc).__name__})", exc)
