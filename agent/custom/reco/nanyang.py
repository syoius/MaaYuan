import json
import re
from typing import Optional, List

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition

from utils import logger

_STAMINA_RECO = "南阳-识别当前体力"
_BAISHU_RECO = "南阳-识别当前白薯数"
_HAT_BULLET_RECO = "南阳-ocr草帽菌数量"
_CORAL_BULLET_RECO = "南阳-ocr公孙珊珊瑚数量"
_GEM_BULLET_RECO = "南阳-ocr酥酪宝石菌数量"


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


def _extract_texts(detail) -> List[str]:
    texts = []
    if not detail:
        return texts
    best = getattr(detail, "best_result", None)
    if best is not None:
        text = getattr(best, "text", None)
        if text:
            texts.append(str(text))
    for attr in ("filtered_results", "filterd_results", "all_results"):
        results = getattr(detail, attr, None)
        if results:
            for res in results:
                text = getattr(res, "text", None)
                if text:
                    texts.append(str(text))
            break
    return texts


def _parse_count_from_detail(detail) -> Optional[int]:
    texts = _extract_texts(detail)
    if not texts:
        return None
    merged = "".join(texts)
    match = re.search(r"\d+", merged)
    if match:
        return int(match.group(0))
    for text in texts:
        match = re.search(r"\d+", text)
        if match:
            return int(match.group(0))
    return None


def _parse_threshold(raw_param: str) -> int:
    if not raw_param:
        return 0
    try:
        params = json.loads(raw_param)
    except json.JSONDecodeError:
        logger.warning(f"NanyangCheckBullets 参数解析失败: {raw_param}")
        return 0
    except Exception:
        logger.exception("NanyangCheckBullets 参数解析异常")
        return 0
    try:
        return int(params.get("threshold", 0))
    except Exception:
        return 0


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

    参数格式:
    {
        "threshold": 120
    }
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        threshold = _parse_threshold(argv.custom_recognition_param)
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
        if stamina_sum >= threshold:
            logger.info(f"可用体力{stamina_sum} >= {threshold}")
            detail = f"{current_stamina}+{current_baishu}*10={stamina_sum}"
            return CustomRecognition.AnalyzeResult(
                box=[0, 0, 0, 0],
                detail=detail,
            )

        logger.info(f"可用体力{stamina_sum} < {threshold}，探索行动取消")
        return None


@AgentServer.custom_recognition("NanyangCheckBullets")
class NanyangCheckBullets(CustomRecognition):
    """
    计算草帽菌/珊瑚菌/宝石菌对应的子弹数量并与阈值比较。

    参数格式:
    {
        "threshold": 60
    }
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        threshold = _parse_threshold(argv.custom_recognition_param)
        hat_detail = context.run_recognition(_HAT_BULLET_RECO, argv.image)
        coral_detail = context.run_recognition(_CORAL_BULLET_RECO, argv.image)
        gem_detail = context.run_recognition(_GEM_BULLET_RECO, argv.image)

        hat_count = _parse_count_from_detail(hat_detail)
        if hat_count is None:
            logger.info("草帽菌数量识别失败")
            return None
        coral_count = _parse_count_from_detail(coral_detail)
        if coral_count is None:
            logger.info("珊瑚菌数量识别失败")
            return None
        gem_count = _parse_count_from_detail(gem_detail)
        if gem_count is None:
            logger.info("宝石菌数量识别失败")
            return None

        bullet_sum = hat_count * 10 + coral_count * 20 + gem_count * 30
        if bullet_sum >= threshold:
            detail = f"{hat_count}*10+{coral_count}*20+{gem_count}*30={bullet_sum}"
            logger.info(f"当前总攻击力 {bullet_sum} >= {threshold}，可以进行宇宙探索")
            return CustomRecognition.AnalyzeResult(
                box=[0, 0, 0, 0],
                detail=detail,
            )

        logger.info(f"当前总攻击力 {bullet_sum} < {threshold}，取消宇宙探索计划")
        return None
