import json
import os
from pathlib import Path
import random

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger

from custom.reco.monopoly import *


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


@AgentServer.custom_action("MonopolyOfficeStrategy")
class MonopolyOfficeStrategy(CustomAction):
    """
    根据label找到对应倾向的选项

    args:
        - label: 贤明 | 混沌
    """

    def __init__(self):
        super().__init__()
        self.data = pd.read_excel("agent/monopoly.xlsx", sheet_name=1)
        # logger.info(f"列名: {list(self.data.columns)}")

    def find_event_options(self, event_name: str) -> List[Dict]:
        """
        根据事件名称查找所有相关的选项
        Args:
            event_name: 事件名称
        Returns:
            包含所有选项的列表
        """
        if self.data is None:
            raise ValueError("数据未加载")

        # 查找匹配的事件
        matching_rows = self.data[self.data["事件名称"] == event_name]

        if matching_rows.empty:
            return []

        options = []
        for _, row in matching_rows.iterrows():
            # 提取选项文本和结果文本
            option_text = row.get("选项文本", "")
            ocr_text = row.get("OCR用", "")
            # result_text = row.get("结果文本", "")
            label = row.get("label", "")

            if pd.notna(option_text) and option_text.strip():
                options.append(
                    {
                        "option_text": option_text.strip(),
                        "ocr_text": ocr_text.strip(),
                        # "result_text": (
                        #     result_text.strip() if pd.notna(result_text) else ""
                        # ),
                        "label": label.strip() if pd.notna(label) else "",
                        "row_index": row.name,
                    }
                )

        return options

    def get_decision(self, event_name: str, decision_type: str = "贤明") -> Dict:
        """
        根据事件名称和决策类型获取决策结果

        Args:
            event_name: 事件名称
            decision_type: 决策类型 ("贤明" 或 "混沌")

        Returns:
            包含选择的选项信息的字典
        """
        # 查找事件的所有选项
        options = self.find_event_options(event_name)

        if not options:
            return {
                "success": False,
                "message": f'未找到事件"{event_name}"的相关选项',
                "event_name": event_name,
                "decision_type": decision_type,
            }

        # 根据决策类型筛选选项
        filtered_options = []

        if decision_type == "贤明":
            # 贤明决策：选择标签为"贤明"的选项，如果没有则选择没有特定标签的选项
            filtered_options = [opt for opt in options if opt["label"] == "贤明"]
            if not filtered_options:
                filtered_options = [
                    opt
                    for opt in options
                    if opt["label"] == "" or opt["label"] == "混沌没有发生变化"
                ]

        elif decision_type == "混沌":
            # 混沌决策：选择标签为"混沌"的选项，如果没有则随机选择
            filtered_options = [opt for opt in options if opt["label"] == "混沌"]
            if not filtered_options:
                filtered_options = options

        else:
            # 其他情况：返回所有选项
            filtered_options = options

        if not filtered_options:
            return {
                "success": False,
                "message": f'事件"{event_name}"没有适合"{decision_type}"类型的选项',
                "event_name": event_name,
                "decision_type": decision_type,
                "all_options": options,
            }

        # 选择一个选项（如果有多个，随机选择）
        chosen_option = random.choice(filtered_options)

        return {
            "success": True,
            "event_name": event_name,
            "decision_type": decision_type,
            "chosen_option": chosen_option["option_text"],
            "ocr_text": chosen_option["ocr_text"],
            # "result": chosen_option["result_text"],
            "label": chosen_option["label"],
            "row_index": chosen_option["row_index"],
            "total_options": len(options),
            "filtered_options": len(filtered_options),
        }

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        event_name = MonopolyOfficeRecord.event_name
        # logger.info(f"已读取公务事件名称：{event_name}")
        label = json.loads(argv.custom_action_param)["label"]
        opposite_label = "混沌" if label == "贤明" else "贤明"
        result = self.get_decision(event_name, label)
        if not result["success"]:
            logger.info(f"该事件不含【{label}】方向的选项，将选择具有相反倾向的选项")
            result = self.get_decision(event_name, opposite_label)
        logger.info(
            f"选择【{result['decision_type']}】方向的选项：{result['chosen_option']}"
        )
        expected = ""
        expected = result["ocr_text"]
        context.run_task(
            "大富翁-点击公务选项", {"大富翁-点击公务选项": {"expected": expected}}
        )
        return CustomAction.RunResult(success=True)


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

        STAT_NAMES = ["智慧", "武力", "幸运", "领袖", "气质", "口才"]
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
        STAT_NAMES = ["智慧", "武力", "幸运", "领袖", "气质", "口才"]
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
