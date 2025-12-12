import json
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from zhconv import convert

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils import logger

# 为了在命盘校验时复用 AutoFormation 解析出的方案数据
try:
    from custom.action.autoformation import AutoFormation
except Exception:
    AutoFormation = None


def _merge_roi(base_roi: Tuple[int, int, int, int], roi: Optional[list]) -> Tuple[int, int, int, int]:
    """优先使用自定义 roi，否则使用节点自带 roi。"""
    if roi and len(roi) == 4:
        return int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    return int(base_roi[0]), int(base_roi[1]), int(base_roi[2]), int(base_roi[3])


def _extract_recognition(detail):
    """兼容 run_recognition 与 run_task 的返回值，统一提取 RecognitionDetail。"""
    if detail is None:
        return None
    if hasattr(detail, "nodes"):
        nodes = getattr(detail, "nodes", None) or []
        if nodes:
            return getattr(nodes[0], "recognition", None)
        return None
    return detail


def _get_results(recognition) -> list:
    """兼容不同字段命名的 OCR 结果列表。"""
    recognition = _extract_recognition(recognition)
    if recognition is None:
        return []
    for attr in ("filterd_results", "filtered_results", "all_results"):
        results = getattr(recognition, attr, None)
        if results:
            return results
    return []


@AgentServer.custom_recognition("OCRWithSimilarity")
class OCRWithSimilarity(CustomRecognition):
    """
    通过 OCR 结果与 expected 的相似度判断命中。

    参数:
    {
        "expected": "密探名",
        "roi": [x, y, w, h],      # 可选，覆盖节点自带的 roi
        "threshold": 0.55,        # 可选，相似度阈值
        "lang": "zh-cn"           # 可选，默认简体，传 zh-tw 则以繁体比对
    }
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        params: Optional[Dict] = {}
        if argv.custom_recognition_param:
            try:
                params = json.loads(argv.custom_recognition_param)
            except json.JSONDecodeError:
                logger.warning(f"OCRWithSimilarity 参数解析失败: {argv.custom_recognition_param}")
            except Exception:
                logger.exception("OCRWithSimilarity 参数解析异常")
        if not isinstance(params, dict):
            params = {}

        lang = params.get("lang", "zh-cn")
        expected: str = str(params.get("expected", "")).strip()
        expected_conv = convert(expected, lang)
        threshold: float = float(params.get("threshold", 0.55))

        roi = _merge_roi(argv.roi, params.get("roi"))
        x, y, w, h = roi
        img = argv.image
        cropped = img[y : y + h, x : x + w] if all(v is not None for v in roi) else img

        override = {
            "自动编队-识别目标密探": {
                "roi": [0, 0, cropped.shape[1], cropped.shape[0]],
            }
        }
        if expected:
            override["自动编队-识别目标密探"]["expected"] = expected

        reco_detail = context.run_recognition("自动编队-识别目标密探", cropped, override)
        reco_detail = _extract_recognition(reco_detail)
        if not reco_detail or not getattr(reco_detail, "hit", False):
            return None

        candidates = _get_results(reco_detail)
        best_box = None
        best_text = ""
        best_score = 0.0
        for res in candidates:
            text_raw = getattr(res, "text", "") or ""
            text_conv = convert(text_raw.strip(), lang)
            score = SequenceMatcher(None, text_conv, expected_conv).ratio() if expected_conv else 1.0
            if score > best_score:
                best_score = score
                best_text = text_conv
                best_box = getattr(res, "box", None)

        if best_score < threshold or not best_box:
            return None

        abs_box = (
            int(best_box[0]) + x,
            int(best_box[1]) + y,
            int(best_box[2]),
            int(best_box[3]),
        )

        return CustomRecognition.AnalyzeResult(box=abs_box, detail=best_text)


@AgentServer.custom_recognition("ActiveDiscText")
class ActiveDiscText(CustomRecognition):
    """
    读取生效命盘区域文字，可用于偏移后的命盘判定。

    参数:
    {
        "roi": [x, y, w, h],
        "expected": "命盘名",     # 可选
        "threshold": 0.5,         # 可选
        "lang": "zh-cn"           # 可选
    }
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        params: Optional[Dict] = {}
        if argv.custom_recognition_param:
            try:
                params = json.loads(argv.custom_recognition_param)
            except json.JSONDecodeError:
                logger.warning(f"ActiveDiscText 参数解析失败: {argv.custom_recognition_param}")
            except Exception:
                logger.exception("ActiveDiscText 参数解析异常")
        if not isinstance(params, dict):
            params = {}

        lang = params.get("lang", "zh-cn")
        expected = convert(str(params.get("expected", "")).strip(), lang)
        threshold: float = float(params.get("threshold", 0.5))

        roi = _merge_roi(argv.roi, params.get("roi"))
        x, y, w, h = roi
        img = argv.image
        cropped = img[y : y + h, x : x + w] if all(v is not None for v in roi) else img

        override = {
            "自动编队-读取生效中命盘": {
                "roi": [0, 0, cropped.shape[1], cropped.shape[0]],
            }
        }
        reco_detail = context.run_recognition("自动编队-读取生效中命盘", cropped, override)
        if not reco_detail or not getattr(reco_detail, "hit", False):
            return None

        candidates = _get_results(reco_detail)
        best_box = None
        best_text = ""
        best_score = 0.0
        for res in candidates:
            text_raw = getattr(res, "text", "") or ""
            text_conv = convert(text_raw.strip(), lang)
            score = SequenceMatcher(None, text_conv, expected).ratio() if expected else 1.0
            if score > best_score:
                best_score = score
                best_text = text_conv
                best_box = getattr(res, "box", None)

        if best_box is None:
            return None
        if expected and best_score < threshold:
            return None

        abs_box = (
            int(best_box[0]) + x,
            int(best_box[1]) + y,
            int(best_box[2]),
            int(best_box[3]),
        )
        return CustomRecognition.AnalyzeResult(box=abs_box, detail=best_text)


@AgentServer.custom_recognition("VerifyActiveDiscs")
class VerifyActiveDiscs(CustomRecognition):
    """
    校验生效命盘列表。

    参数:
    {
        "index": 0,                    # 必填，指示在上次编队方案中的第 index 位密探
        "threshold": 0.55              # 可选，相似度阈值
    }
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        params: Optional[Dict] = {}
        if argv.custom_recognition_param:
            try:
                params = json.loads(argv.custom_recognition_param)
            except json.JSONDecodeError:
                logger.warning(f"VerifyActiveDiscs 参数解析失败: {argv.custom_recognition_param}")
            except Exception:
                logger.exception("VerifyActiveDiscs 参数解析异常")
        if not isinstance(params, dict):
            params = {}

        if AutoFormation is None:
            logger.error("未找到 AutoFormation，无法读取编队信息")
            return None

        if not params or "index" not in params:
            logger.error("VerifyActiveDiscs 缺少必填参数 index")
            return None

        try:
            index = int(params.get("index"))
        except Exception:
            logger.error("index 参数格式错误，必须为整数")
            return None

        plan = AutoFormation.get_last_plan() if hasattr(AutoFormation, "get_last_plan") else {}
        if not plan:
            logger.error("尚未获取到编队方案，无法校验命盘")
            return None

        opers = plan.get("opers", []) or []
        opers_num = plan.get("opers_num", len(opers))
        if index < 0 or index >= opers_num:
            logger.error("index 超出编队人数范围，终止命盘校验")
            return None

        oper = opers[index] if index < len(opers) else None
        if not oper:
            logger.error("未找到对应位置的密探信息，终止命盘校验")
            return None

        required = oper.get("discs_ot_names") or []
        lang = getattr(AutoFormation, "_last_lang", "zh-cn")
        threshold = float(params.get("threshold", 0.55))
        offset = list(getattr(AutoFormation, "EFFECTIVE_DISC_OFFSET", [0, 0, 0, 0]))
        dx, dy, dw, dh = [int(v) for v in offset]

        required_conv = [convert(str(r), lang) for r in required if r]

        if not required_conv:
            logger.error("当前密探未配置命盘要求，无法校验")
            return None

        img = argv.image
        reco = context.run_recognition("自动编队-读取生效中命盘", img)
        if not reco or not getattr(reco, "hit", False):
            logger.error("未识别到生效中命盘")
            return None

        candidates = _get_results(reco)
        texts: List[str] = []
        for res in candidates:
            box = getattr(res, "box", None)
            if not box:
                continue
            roi = (
                int(box[0]) + dx,
                int(box[1]) + dy,
                int(box[2]) + dw,
                int(box[3]) + dh,
            )
            roi = (
                max(0, roi[0]),
                max(0, roi[1]),
                max(1, roi[2]),
                max(1, roi[3]),
            )
            override = {
                "自动编队-读取生效中命盘": {
                    "recognition": {
                        "type": "Custom",
                        "param": {
                            "custom_recognition": "ActiveDiscText",
                            "roi": list(roi),
                            "expected": "",
                            "lang": lang,
                        },
                    }
                }
            }
            detail = context.run_recognition("自动编队-读取生效中命盘", img, override)
            if detail and getattr(detail, "best_result", None):
                text = convert(detail.best_result.text.strip(), lang)
                if text:
                    texts.append(text)

        for need in required_conv:
            if not any(SequenceMatcher(None, need, t).ratio() >= threshold for t in texts):
                logger.error(f"命盘缺失: {need}")
                return None

        first_box = getattr(candidates[0], "box", None)
        abs_box = (
            int(first_box[0]) if first_box else 0,
            int(first_box[1]) if first_box else 0,
            int(first_box[2]) if first_box else 1,
            int(first_box[3]) if first_box else 1,
        )
        return CustomRecognition.AnalyzeResult(box=abs_box, detail="ok")
