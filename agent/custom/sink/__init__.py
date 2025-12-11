from maa.agent.agent_server import AgentServer

from maa.resource import ResourceEventSink
from maa.controller import ControllerEventSink
from maa.tasker import TaskerEventSink
from maa.context import Context, ContextEventSink

from .logger import create_sink_logger


@AgentServer.resource_sink()
class MyResSink(ResourceEventSink):
    def __init__(self):
        self.logger = create_sink_logger("resource")

    def on_raw_notification(self, resource, msg: str, details: dict):
        self.logger.info(msg, extra={"details": details})


@AgentServer.controller_sink()
class MyCtrlSink(ControllerEventSink):
    def __init__(self):
        self.logger = create_sink_logger("controller")

    def on_raw_notification(self, controller, msg: str, details: dict):
        self.logger.info(msg, extra={"details": details})


@AgentServer.tasker_sink()
class MyTaskerSink(TaskerEventSink):
    def __init__(self):
        self.logger = create_sink_logger("tasker")

    def on_raw_notification(self, tasker, msg: str, details: dict):
        self.logger.info(msg, extra={"details": details})


@AgentServer.context_sink()
class MyCtxSink(ContextEventSink):
    def __init__(self):
        self.logger = create_sink_logger("context")

    def on_raw_notification(self, context: Context, msg: str, details: dict):
        # 增强：获取识别和动作的详细信息
        enhanced_details = details.copy()

        try:
            # 识别成功/失败：获取完整的识别详情
            if "Recognition.Succeeded" in msg or "Recognition.Failed" in msg:
                reco_id = details.get("reco_id")
                if reco_id:
                    reco_detail = context.tasker.get_recognition_detail(reco_id)
                    if reco_detail:
                        enhanced_details["recognition"] = {
                            "algorithm": str(reco_detail.algorithm),
                            "hit": reco_detail.hit,
                            "box": list(reco_detail.box) if reco_detail.box else None,
                            "best_result": self._serialize_recognition_result(
                                reco_detail.best_result
                            ),
                            "filtered_count": len(reco_detail.filtered_results),
                            "all_count": len(reco_detail.all_results),
                        }

            # 动作成功/失败：获取完整的动作详情
            elif "Action.Succeeded" in msg or "Action.Failed" in msg:
                action_id = details.get("action_id")
                if action_id:
                    action_detail = context.tasker.get_action_detail(action_id)
                    if action_detail:
                        enhanced_details["action"] = {
                            "action_type": str(action_detail.action),
                            "box": (
                                list(action_detail.box) if action_detail.box else None
                            ),
                            "success": action_detail.success,
                            "result": self._serialize_action_result(
                                action_detail.action, action_detail.result
                            ),
                        }

        except Exception as e:
            # 如果查询失败，不影响日志记录，但记录警告
            self.logger.warning(f"Failed to get detail for {msg}: {e}")

        self.logger.info(msg, extra={"details": enhanced_details})

    def _serialize_recognition_result(self, result):
        """序列化识别结果"""
        if not result:
            return None

        from maa.define import OCRResult, TemplateMatchResult

        result_dict = {
            "box": list(result.box) if hasattr(result, "box") else None,
            "score": result.score if hasattr(result, "score") else None,
        }

        # OCR 特殊处理
        if isinstance(result, OCRResult):
            result_dict["text"] = result.text

        return result_dict

    def _serialize_action_result(self, action_type, result):
        """序列化动作结果"""
        if not result:
            return None

        from maa.define import (
            ClickActionResult,
            SwipeActionResult,
            InputTextActionResult,
        )

        if isinstance(result, ClickActionResult):
            return {"point": list(result.point)}
        elif isinstance(result, SwipeActionResult):
            return {
                "begin": list(result.begin),
                "end": [list(p) for p in result.end],
                "duration": result.duration,
            }
        elif isinstance(result, InputTextActionResult):
            return {"text": result.text}
        else:
            return {"type": str(type(result).__name__)}
