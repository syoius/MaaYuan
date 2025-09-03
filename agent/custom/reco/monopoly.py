import difflib
import json
import string
from typing import Any, Dict, List, Union, Optional
import pandas as pd

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.define import RectType
from utils.logger import logger


@AgentServer.custom_recognition("MonopolyStatsRecord")
class MonopolyStatsRecord(CustomRecognition):
    """
    读取玩家的六项数值
    """

    def analyze(
        self, context: Context, argv: CustomRecognition.AnalyzeArg
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        STATS_ROIS = [
            [118, 156, 57, 28],
            [201, 159, 45, 23],
            [281, 159, 45, 24],
            [124, 207, 43, 21],
            [201, 206, 43, 21],
            [281, 203, 44, 25],
        ]
        stats = []
        for roi in STATS_ROIS:
            reco_detail = context.run_recognition(
                "大富翁-读取个人数值", argv.image, {"大富翁-读取个人数值": {"roi": roi}}
            )
            stat = reco_detail.best_result.text
            # logger.info(f"已在roi:{roi}区域识别到属性数值{stat}")
            stats.append(int(stat))
        MonopolyStatsRecord.stats = stats
        logger.info(
            f"已读取当前属性：智慧{stats[0]}，武力{stats[1]}, 运气{stats[2]}，领袖{stats[3]}，气质{stats[4]}，口才{stats[5]}"
        )
        return CustomRecognition.AnalyzeResult(box=[0, 0, 0, 0], detail=str(stats))


@AgentServer.custom_recognition("MonopolySinglePkStats")
class MonopolySinglePkStats(CustomRecognition):
    """
    在 PK 界面读取属性、数值要求及事件内容
    """

    def __init__(self):
        super().__init__()
        self.description_bank = self.read_excel("agent/monopoly.xlsx")
        self.similarity_threshold = 0.5  # 相似度阈值

    @staticmethod
    def split_name_value(s: str):
        mapping = {
            "智慧": "智慧",
            "武力": "武力",
            "運氣": "运气",
            "运气": "运气",
            "領袖": "领袖",
            "领袖": "领袖",
            "氣質": "气质",
            "气质": "气质",
            "口才": "口才",
        }
        for word in mapping:
            if s.startswith(word):
                return mapping[word], s[len(word) :]
        return None, None

    def clean_text(self, text):
        # 强制转成字符串，避免 int/None 出错
        text = str(text) if text is not None else ""

        # 创建翻译表，将所有标点符号和空格映射为 None
        chinese_punctuation = "，。！？【】（）《》“”‘’；：、——·〈〉……—"
        translator = str.maketrans(
            "", "", string.punctuation + chinese_punctuation + " \t\n\r\u3000"
        )

        # 使用翻译表移除标点符号和空格
        cleaned_text = text.translate(translator)
        return cleaned_text.strip()

    def read_excel(self, file_path):
        # 读取第1个sheet并跳过第一行
        df = pd.read_excel(file_path, sheet_name=0).iloc[1:]

        results = []
        for _, row in df.iterrows():
            description = self.clean_text(row.iloc[0])
            label = row.iloc[1]
            results.append({"d": description, "label": label})
        return results

    def find_label(self, description: str):
        best_match = None
        max_sim = 0

        for item in self.description_bank:
            # 截断到前 25 个字
            item_d = item["d"][:25] if len(item["d"]) > 25 else item["d"]
            sim = difflib.SequenceMatcher(None, description, item_d).ratio()
            if sim > max_sim:
                max_sim = sim
                best_match = item

        if best_match is None:  # 没找到匹配
            logger.info(f"未匹配到恶性事件（炸工坊/降税收）")
            return None, None

        label_map = {
            1: "炸工坊",
            2: "减税收",
        }
        if max_sim > self.similarity_threshold:
            label_text = label_map.get(best_match["label"])
            # 如果找到了，返回结果
            logger.info(
                f"已匹配到【{label_text}】事件: {best_match['d']}, 相似度：{max_sim}"
            )
        return best_match["d"] if max_sim > self.similarity_threshold else None

    def analyze(
        self, context: Context, argv: CustomRecognition.AnalyzeArg
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        reco_detail = context.run_recognition("大富翁-读取PK要求", argv.image)
        stat_name_n_value = reco_detail.best_result.text
        stat_name, value = self.split_name_value(stat_name_n_value)

        description_detail = context.run_recognition(
            "大富翁-读取PK事件内容", argv.image
        )
        description = ""
        if description_detail and description_detail.filterd_results:
            for r in description_detail.filterd_results:
                description = description + r.text
        else:
            logger.info("警告：未能识别到事件内容")
        description = self.clean_text(description)
        # logger.info(f"{description}")

        label = self.find_label(description)
        if label:
            suggestion = True
        else:
            suggestion = False

        pkstats = [stat_name, value, description, label, suggestion]
        # logger.info(f"{pkstats}")
        MonopolySinglePkStats.pkstats = pkstats
        return CustomRecognition.AnalyzeResult(box=[0, 0, 0, 0], detail=str(pkstats))
