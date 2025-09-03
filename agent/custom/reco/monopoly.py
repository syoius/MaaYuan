import json
from typing import Any, Dict, List, Union, Optional

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
