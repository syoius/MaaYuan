from typing import Union, Optional

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.define import RectType


@AgentServer.custom_recognition("CheckStopping")
class CheckStopping(CustomRecognition):
    """
    检查任务是否即将停止。
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        if context.tasker.stopping:
            return CustomRecognition.AnalyzeResult(
                box=[0, 0, 0, 0],
                detail={"node": "CheckStopping", "stopping": True},
            )
        else:
            return None
