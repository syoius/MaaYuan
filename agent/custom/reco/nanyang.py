import re
from typing import Optional

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition

from utils import logger

_STAMINA_RECO = "南阳-识别当前体力"
_BAISHU_RECO = "南阳-识别当前白薯数"


def _extract_text(detail) -> Optional[str]:
    if not detail:
        return None
    best = getattr(detail, "best_result", None)
    if best is not None:
        text = getattr(best, "text", None)
        if text:
            return str(text)
    for attr in ("filtered_results", "filterd_results", "all_results"):
        results = getattr(detail, attr, None)
        if results:
            for res in results:
                text = getattr(res, "text", None)
                if text:
                    return str(text)
    return None


def _parse_current_value(text: str) -> Optional[int]:
    if not text:
        return None
    cleaned = text.strip()
    match = re.search(r"(\d+)\s*/", cleaned)
    if match:
        return int(match.group(1))
    match = re.search(r"\d+", cleaned)
    if match:
        return int(match.group(0))
    return None


@AgentServer.custom_recognition("NanyangStamina")
class NanyangStamina(CustomRecognition):
    """
    Combine current stamina and sweet potato count to decide if stamina is enough.
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        stamina_detail = context.run_recognition(_STAMINA_RECO, argv.image)
        stamina_text = _extract_text(stamina_detail)
        if not stamina_text:
            logger.info("未识别到当前体力文本")
            return None
        current_stamina = _parse_current_value(stamina_text)
        if current_stamina is None:
            logger.info(f"当前体力识别失败，识别到文本：'{stamina_text}'")
            return None

        baishu_detail = context.run_recognition(_BAISHU_RECO, argv.image)
        baishu_text = _extract_text(baishu_detail)
        if not baishu_text:
            logger.info("未识别到白薯数量文本")
            return None
        current_baishu = _parse_current_value(baishu_text)
        if current_baishu is None:
            logger.info(f"白薯数量识别失败，识别到文本：'{baishu_text}'")
            return None

        stamina_sum = current_stamina + current_baishu * 10
        if stamina_sum >= 120:
            detail = f"{current_stamina}+{current_baishu}*10={stamina_sum}"
            return CustomRecognition.AnalyzeResult(
                box=[0, 0, 0, 0],
                detail=detail,
            )

        logger.info(f"可用体力{stamina_sum} < 120，探索行动取消")
        return None
