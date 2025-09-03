import json
import os
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger

from custom.reco.monopoly import MonopolyStatsRecord, MonopolySinglePkStats


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


@AgentServer.custom_action("MonopolySinglePkStrategy")
class MonopolySinglePkStrategy(CustomAction):
    """
    （在有泻药的时候）比较当前属性是否能够PK胜利，若会失败根据结果是否恶性来决定是否使用泻药
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        STAT_NAMES = ["智慧", "武力", "运气", "领袖", "气质", "口才"]
        # [stat_name, value, description, label, suggestion, pc_stats]
        pk_stats = MonopolySinglePkStats.pkstats

        test_name = pk_stats[0]
        test_name_index = STAT_NAMES.index(test_name)
        test_standard = pk_stats[1]
        pc_stats = pk_stats[5]
        pc_test_value = pc_stats[test_name_index]

        # 若无法通过PK，则override next list
        if pc_test_value < test_standard:
            logger.info(
                f"当前PK需要{test_name}>={test_standard}, 玩家当前属性为{pc_test_value}，无法通过PK"
            )
            # 先检查是否为炸房/降税事件
            if pk_stats[4]:
                logger.info("由于失败后果严重，将使用泻药")
                context.override_next("大富翁-PK方案", ["大富翁-打开泻药使用界面"])
            else:
                logger.info("失败后果不严重，故不使用泻药")
                context.override_next("大富翁-PK方案", ["大富翁-PK结果预测-无泻药"])
        else:
            logger.info(
                f"当前PK需要{test_name}>={test_standard}, 玩家当前属性为{pc_test_value}，可以通过PK，无需使用泻药"
            )
            context.override_next("大富翁-PK方案", ["大富翁-PK结果预测-无泻药"])
        return CustomAction.RunResult(success=True)
