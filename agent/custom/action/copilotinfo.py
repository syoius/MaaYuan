import json
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.library import *
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


@AgentServer.custom_action("DownRestart")
class DownRestart(CustomAction):
    """
    ColorMatch 检测指定位置密探是否已阵亡，是则改写 next 为左上角重开

    Args:
        - "node": "当前节点名称"
        - "position": [1,5]
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        COLORMATCH_ROIS = [
            [],
            [21, 811, 124, 378],
            [156, 810, 128, 375],
            [298, 811, 129, 378],
            [439, 809, 127, 376],
            [579, 808, 126, 378],
        ]
        params = json.loads(argv.custom_action_param)
        current_node_name = params["node"]
        # logger.info(f"{current_node_name}")
        position = params["position"]
        cmroi = COLORMATCH_ROIS[position]
        img = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition(
            "downTest", img, {"downTest": {"roi": cmroi}}
        )
        if reco_detail and getattr(reco_detail, "hit", False):
            context.override_next(current_node_name, ["抄作业点左上角重开"])
            logger.info(f"检测到{position}号位阵亡，正在尝试点左上角重开")
            return CustomAction.RunResult(success=True)
        else:
            logger.info(f"检测到{position}号位存活，正常执行后续动作")
            return CustomAction.RunResult(success=True)


@AgentServer.custom_action("RetreatRestart")
class RetreatRestart(CustomAction):
    """
    OCR 检测指定位置密探是否已退场，是则改写 next 为左上角重开

    Args:
        - "node": "当前节点名称"
        - "position": [1,5]
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        RETREAT_ROIS = [
            [],
            [21, 811, 124, 378],
            [156, 810, 128, 375],
            [298, 811, 129, 378],
            [439, 809, 127, 376],
            [579, 808, 126, 378],
        ]
        params = json.loads(argv.custom_action_param)
        current_node_name = params["node"]
        # logger.info(f"{current_node_name}")
        position = params["position"]
        retreatroi = RETREAT_ROIS[position]
        img = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition(
            "RetreatCheck", img, {"RetreatCheck": {"roi": retreatroi}}
        )
        if reco_detail.hit:
            context.override_next(current_node_name, ["抄作业点左上角重开"])
            logger.info(f"检测到{position}号位已退场，正在尝试点左上角重开")
            return CustomAction.RunResult(success=True)
        else:
            logger.info(f"检测到{position}号位未退场，正常执行后续动作")
            return CustomAction.RunResult(success=True)


@AgentServer.custom_action("BirdRestart")
class BirdRestart(CustomAction):
    """
    TemplateMatch 检测指定位置密探是否有鹦鹉图片，如有则正常执行后续动作，无则改写 next 为左上角重开

    Args:
        - "node": "当前节点名称"
        - "position": [1,5]
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        BIRD_ROIS = [
            [],
            [21, 811, 124, 378],
            [156, 810, 128, 375],
            [298, 811, 129, 378],
            [439, 809, 127, 376],
            [579, 808, 126, 378],
        ]
        params = json.loads(argv.custom_action_param)
        current_node_name = params["node"]
        # logger.info(f"{current_node_name}")
        position = params["position"]
        birdroi = BIRD_ROIS[position]
        img = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition(
            "BirdCheck", img, {"BirdCheck": {"roi": birdroi}}
        )
        if reco_detail.hit:
            logger.info(f"检测到{position}号位有鹦鹉，正常执行后续动作")
            return CustomAction.RunResult(success=True)
        else:
            context.override_next(current_node_name, ["抄作业点左上角重开"])
            logger.info(f"检测到{position}号位无鹦鹉，正在尝试点左上角重开")
            return CustomAction.RunResult(success=True)