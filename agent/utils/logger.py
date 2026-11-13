import importlib
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime

try:
    from loguru import logger as _logger
except ImportError:
    _logger = None


def setup_logger(log_dir="debug/custom", console_level="INFO"):
    """设置 loguru logger

    Args:
        log_dir: 日志文件目录
        console_level: 控制台输出等级 (DEBUG, INFO, WARNING, ERROR)
    """
    os.makedirs(log_dir, exist_ok=True)
    _logger.remove()

    # 定义日志级别的简短格式
    def format_level(record):
        if record["extra"].get("console") is False:
            return False
        level_map = {
            "INFO": "info",
            "ERROR": "err",
            "WARNING": "warn",
            "DEBUG": "debug",
            "CRITICAL": "critical",
            "SUCCESS": "success",
            "TRACE": "trace",
        }
        record["extra"]["level_short"] = level_map.get(
            record["level"].name, record["level"].name.lower()
        )
        return True

    _logger.add(
        sys.stderr,
        format="<level>{extra[level_short]}</level>:<level>{message}</level>",
        colorize=True,
        level=console_level,
        filter=format_level,
    )
    _logger.add(
        f"{log_dir}/{{time:YYYY-MM-DD}}.log",
        rotation="00:00",  # midnight
        retention="2 weeks",
        compression="zip",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,  # 包含完整的异常回溯信息
        diagnose=True,  # 包含变量值信息
    )
    return _logger


def change_console_level(level="DEBUG"):
    """动态修改控制台日志等级"""
    setup_logger(console_level=level)
    _logger.info(f"控制台日志等级已更改为: {level}")


class ShortLevelFormatter(logging.Formatter):
    """自定义 Formatter，使用简短的日志级别名称"""

    level_map = {
        "INFO": "info",
        "ERROR": "err",
        "WARNING": "warn",
        "DEBUG": "debug",
        "CRITICAL": "critical",
    }

    def format(self, record):
        record.level_short = self.level_map.get(
            record.levelname, record.levelname.lower()
        )
        return super().format(record)


class _FallbackLogger:
    def __init__(self):
        self._target = logging
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        handler.setFormatter(ShortLevelFormatter("%(level_short)s:%(message)s"))
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.DEBUG)
        self._handlers = [handler]
        try:
            log_dir = "debug/custom"
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                f"{log_dir}/{datetime.now():%Y-%m-%d}.log", mode="a", encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logging.root.addHandler(file_handler)
            self._handlers.append(file_handler)
        except Exception:
            # File logging must not prevent the first startup.
            pass

    def __getattr__(self, name):
        return getattr(self._target, name)

    def bind(self, **kwargs):
        return self if self._target is logging else self._target.bind(**kwargs)

    def contextualize(self, **kwargs):
        return nullcontext() if self._target is logging else self._target.contextualize(**kwargs)


logger = setup_logger() if _logger is not None else _FallbackLogger()


def upgrade_logger():
    """Switch existing fallback references to Loguru once dependencies are installed."""
    global _logger
    if not isinstance(logger, _FallbackLogger) or logger._target is not logging:
        return
    try:
        importlib.invalidate_caches()
        from loguru import logger as _logger

        target = setup_logger()
    except Exception:
        return
    logger._target = target
    for handler in logger._handlers:
        logging.root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    logger._handlers.clear()


def get_logger_backend():
    return "fallback" if isinstance(logger, _FallbackLogger) and logger._target is logging else "loguru"
