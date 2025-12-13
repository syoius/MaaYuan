import json
import unicodedata
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


def _parse_param(raw, name: str) -> Dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"{name} 参数解析失败: {raw}")
        except Exception:
            logger.exception(f"{name} 参数解析异常")
    return {}


def _lang_from_resource_hint(params: Dict, default: str = "zh-cn") -> str:
    res_hint = str(params.get("resource", "") or "").lower()
    if "zh_tw" in res_hint or "zh-tw" in res_hint:
        return "zh-tw"
    return default


def _merge_roi(
    base_roi: Tuple[int, int, int, int], roi: Optional[list]
) -> Tuple[int, int, int, int]:
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


def _strip_punct_and_digits(text: str) -> str:
    """Remove punctuation and digits before similarity comparison."""
    if not text:
        return ""
    return "".join(
        ch
        for ch in text
        if not ch.isdigit() and not unicodedata.category(ch).startswith("P")
    ).strip()


@AgentServer.custom_recognition("OCRWithSimilarity")
class OCRWithSimilarity(CustomRecognition):
    """
    通过 OCR 结果与 expected 的相似度判断命中。

    参数:
    {
        "expected": "密探名",
        "roi": [x, y, w, h],      # 可选，覆盖节点自带的 roi
        "threshold": 0.55,        # 可选，相似度阈值
        "resource": "zh_tw"       # 可选，按资源推断语言（含 zh_tw/zh-tw 视为繁体，否则简体）
    }
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        params = _parse_param(argv.custom_recognition_param, "OCRWithSimilarity")

        lang = _lang_from_resource_hint(params, "zh-cn")
        expected: str = str(params.get("expected", "")).strip()
        expected_conv = convert(expected, lang)
        expected_clean = _strip_punct_and_digits(expected_conv)
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

        reco_detail = context.run_recognition(
            "自动编队-识别目标密探", cropped, override
        )
        reco_detail = _extract_recognition(reco_detail)
        if not reco_detail:
            return None

        candidates = _get_results(reco_detail)
        best_box = None
        best_text = ""
        best_score = 0.0
        for res in candidates:
            text_raw = getattr(res, "text", "") or ""
            text_conv = convert(text_raw.strip(), lang)
            text_clean = _strip_punct_and_digits(text_conv)
            score = (
                SequenceMatcher(None, text_clean, expected_clean).ratio()
                if expected_conv
                else 1.0
            )
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
