import re
import os
import json
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.library import *
from utils import logger


@AgentServer.custom_action("EnergyCheck")
class EnergyCheck(CustomAction):
    """
    检测体力值，根据体力值决定下一步操作
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        # 获取截图
        img = context.tasker.controller.post_screencap().wait().get()
        
        # 使用 OCR 识别体力数字，roi: [605,63,38,28]
        override = {
            "EnergyCheck": {
                "recognition": {
                    "type": "OCR",
                    "param": {
                        "roi": [605, 63, 38, 28],
                        "model": "en",
                        "only_rec": True,
                    },
                }
            }
        }
        
        result = context.run_recognition("EnergyCheck", img, override)
        
        # 提取数字
        energy_value = None
        if result and getattr(result, "best_result", None):
            text = result.best_result.text.strip()
            # 使用正则表达式提取数字
            numbers = re.findall(r'\d+', text)
            if numbers:
                try:
                    energy_value = int(numbers[0])
                except ValueError:
                    logger.info(f"无法解析体力值: {text}")
        
        if energy_value is None:
            logger.info("未能识别到体力值，保持原流程")
            return CustomAction.RunResult(success=True)
        
        logger.info(f"识别到体力值: {energy_value}")
        
        # 根据体力值决定下一步
        if energy_value >= 10:
            logger.info("检测到体力大于等于10，继续刷取雪山秘宝")
        elif energy_value < 10:
            # 体力小于10，改写"寒夜厄境-确定结算"的next为stop
            context.override_next("寒夜厄境-获得雪山秘宝", ["寒夜厄境-切换到资源牌"])
            context.override_next("寒夜厄境-确定结算", ["stop"])
            logger.info("检测到体力小于10，尝试开启所有雪山秘宝并结束任务")
        
        return CustomAction.RunResult(success=True)

