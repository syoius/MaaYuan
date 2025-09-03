import json
import os
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger

from custom.reco.monopoly import MonopolyStatsRecord


@AgentServer.custom_action("MonopolyLapRecord")
class MonopolyLapRecord(CustomAction):
    """
    每次开始新一圈时记录这一次的两轮PK信息到文件
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        resource = json.loads(argv.custom_action_param)["resource"]
        file_path = f"resource/data/monopoly_{resource}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["PK1"] = {"stat_name": stat_name, "value": value}

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


@AgentServer.custom_action("MonopolySetShipDestination")
class MonopolySetShipDestination(CustomAction):
    """
    找到六项属性中数值最低的那项，并点击对应的目的地

    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        stats = MonopolyStatsRecord.stats
        STATS_CLICK_ROIS = [
            [110, 798, 19, 15],
            [216, 795, 4, 12],
            [308, 802, 11, 11],
            [404, 801, 15, 16],
            [506, 812, 1, 1],
            [596, 800, 11, 12],
        ]

        min_stat = min(stats)
        min_index = stats.index(min_stat)
        target_roi = STATS_CLICK_ROIS[min_index]

        context.run_task("大富翁-点击操作", {"大富翁-点击操作": {"target": target_roi}})

        STAT_NAMES = ["智慧", "武力", "运气", "领袖", "气质", "口才"]
        logger.info(
            f"当前属性中数值最低的是{STAT_NAMES[min_index]}:{min_stat}，已设定为本次出航目的地"
        )

        return CustomAction.RunResult(success=True)
