from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from utils import logger


@AgentServer.custom_action("CopilotInfo")
class CopilotInfo(CustomAction):
    """
    读取并打印作业文件中的"作业信息"
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        context.run_task("作业信息")
        return CustomAction.RunResult(success=True)
