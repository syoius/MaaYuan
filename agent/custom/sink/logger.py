import os
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# 检查是否安装了 loguru
HAS_LOGURU = False
_logger = None
try:
    from loguru import logger as _logger

    HAS_LOGURU = True
except ImportError:
    pass


class StructuredFormatter(logging.Formatter):
    """结构化 JSONL 日志格式器，便于分析"""

    def format(self, record):
        # 获取 sink_type（只取字符串值）
        sink_type = getattr(record, "sink_type", "unknown")
        if isinstance(sink_type, dict):
            sink_type = sink_type.get("sink_type", "unknown")

        # 基础格式
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]

        # MaaFramework 格式：message 是消息类型，details 是详细信息
        message = record.getMessage()
        details = getattr(record, "details", {})

        structured_data = {
            "timestamp": timestamp,
            "sink_type": sink_type,
            "message": message,
            "details": details,
        }

        return json.dumps(structured_data, ensure_ascii=False)


class SinkLoggerAdapter(logging.LoggerAdapter):
    """适配器，为日志添加 sink 相关上下文"""

    def __init__(self, logger, sink_type):
        super().__init__(logger, {"sink_type": sink_type})

    def process(self, msg, kwargs):
        """处理日志记录，正确合并 extra 参数"""
        # 获取传递的 extra 参数
        extra = kwargs.get("extra", {})

        # 合并 self.extra 和传递的 extra
        merged_extra = {}
        merged_extra.update(self.extra)
        merged_extra.update(extra)

        # 更新 kwargs
        kwargs["extra"] = merged_extra

        return msg, kwargs

    def bind(self, **kwargs):
        """模拟 loguru 的 bind 方法，返回带有额外上下文的适配器"""
        new_extra = self.extra.copy()
        new_extra.update(kwargs)
        return SinkLoggerAdapter(self.logger, new_extra)

    def log_event(self, event_type, message, *args, **kwargs):
        """记录特定类型的事件"""
        extra = kwargs.get("extra", {})
        extra.update({"event_type": event_type})
        kwargs["extra"] = extra
        self.info(message, *args, **kwargs)


# 全局变量，用于存储共享的 sink logger
_shared_sink_logger = None
_console_handler_id = None
_file_handler_id = None


def cleanup_global_logger_conflicts():
    """清理全局 logger 中可能导致 KeyError 的冲突 handler"""
    if HAS_LOGURU:
        try:
            from loguru import logger as global_logger

            # 检查是否有冲突的 sink logger handler
            has_conflict = False
            try:
                # 尝试一个简单的日志记录，看是否会触发 KeyError
                test_record = {
                    "name": "test",
                    "level": "INFO",
                    "message": "test",
                    "extra": {},
                    "time": None,
                    "file": None,
                    "function": None,
                    "line": 0,
                    "module": None,
                    "process": None,
                    "thread": None,
                    "elapsed": None,
                    "exception": None,
                }

                # 测试所有 handler
                for handler_id in list(global_logger._core.handlers.keys()):
                    try:
                        handler = global_logger._core.handlers[handler_id]
                        if (
                            hasattr(handler, "_format")
                            and hasattr(handler, "_format")
                            and "sink_type" in str(handler._format)
                        ):
                            has_conflict = True
                            break
                    except:
                        pass
            except:
                pass

            if has_conflict:
                print("检测到冲突的 sink logger handler，正在清理...")
                # 只移除冲突的 handler，而不是全部移除
                handlers_to_remove = []
                for handler_id in list(global_logger._core.handlers.keys()):
                    try:
                        handler = global_logger._core.handlers[handler_id]
                        if hasattr(handler, "_format") and "sink_type" in str(
                            handler._format
                        ):
                            handlers_to_remove.append(handler_id)
                    except:
                        pass

                for handler_id in handlers_to_remove:
                    try:
                        global_logger.remove(handler_id)
                        print(f"移除了冲突的 handler: {handler_id}")
                    except:
                        pass

        except Exception as e:
            # 如果清理失败，尝试完全重置
            try:
                global_logger.remove()
                print("完全重置了全局 logger")
            except:
                pass


def get_shared_sink_logger():
    """获取共享的 sink logger 实例"""
    global _shared_sink_logger, _console_handler_id, _file_handler_id

    if _shared_sink_logger is None and HAS_LOGURU and _logger is not None:
        # 首先清理可能的全局冲突
        cleanup_global_logger_conflicts()

        # 创建共享的 loguru logger 实例
        from loguru import logger as shared_logger

        # 配置控制台输出 - 只显示所有 sink 消息
        _console_handler_id = shared_logger.add(
            sys.stderr,
            format="<level>[SINK]</level> <level>{level: <8}</level> | <cyan>{time:HH:mm:ss}</cyan> | <level>{message}</level>",
            colorize=True,
            level="INFO",
        )

        # 配置文件输出 - 只记录所有 sink 消息
        log_dir = "debug/sink"
        os.makedirs(log_dir, exist_ok=True)
        _file_handler_id = shared_logger.add(
            f"{log_dir}/sink_{{time:YYYY-MM-DD}}.log",
            rotation="00:00",
            retention="30 days",
            compression="zip",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {extra[sink_type]:<10} | {extra[event_type]:<15} | {message}",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

        _shared_sink_logger = shared_logger

    return _shared_sink_logger


def create_sink_logger_instance(sink_type: str = "unknown"):
    """创建 sink logger 实例，使用标准 logging 避免全局冲突

    Args:
        sink_type: sink 类型，默认 "unknown"
    """
    # 为了避免 loguru 的全局状态问题，我们统一使用标准 logging
    logger = logging.getLogger(f"maa_sink_{sink_type}")
    logger.setLevel(logging.DEBUG)

    # 清除现有处理器
    logger.handlers.clear()

    # 移除控制台输出，只保留文件日志

    # 文件处理器
    log_dir_path = Path("debug/sink")
    log_dir_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir_path / f"sink_{datetime.now().strftime('%Y-%m-%d')}.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)

    return SinkLoggerAdapter(logger, sink_type)


def setup_sink_logger(log_dir="debug/sink", console_level="INFO", file_level="DEBUG"):
    """设置全局 sink logger - 仅用于兼容性，建议使用 create_sink_logger"""
    import warnings

    warnings.warn(
        "setup_sink_logger() 影响全局 logger，建议使用 create_sink_logger() 创建独立的 logger 实例",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sink_logger_instance("global")


def create_sink_logger(sink_type: str):
    """为特定类型的 sink 创建 logger 实例

    Args:
        sink_type: sink 类型，如 'resource', 'controller', 'tasker', 'context'
    """
    return create_sink_logger_instance(sink_type)


# 注意：不再在模块级别自动初始化全局 logger
# 如需使用默认 sink logger，请调用 get_default_sink_logger()
def get_default_sink_logger():
    """获取默认的 sink logger"""
    return setup_sink_logger()
