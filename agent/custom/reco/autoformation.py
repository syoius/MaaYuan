import json
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils import logger


def _merge_roi_box(base_roi: Tuple[int, int, int, int], roi: Optional[list]) -> Tuple[int, int, int, int]:
    """Return a valid ROI using custom param roi when provided, otherwise fallback to base roi."""
    if roi and len(roi) == 4:
        return int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    return int(base_roi[0]), int(base_roi[1]), int(base_roi[2]), int(base_roi[3])


def _get_results(recognition) -> list:
    """兼容 filterd_results / filtered_results 字段命名差异。"""
    for attr in ("filterd_results", "filtered_results", "all_results"):
        results = getattr(recognition, attr, None)
        if results:
            return results
    return []


@AgentServer.custom_recognition("OCRWithSimilarity")
class OCRWithSimilarity(CustomRecognition):
    """
    通过 OCR 结果与 expected 的相似度来判断命中。

    参数:
    {
        "expected": "密探名",
        "roi": [x, y, w, h],      # 可选，覆盖节点自带的 roi
        "threshold": 0.55         # 可选，相似度阈值
    }
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        params: Dict = {}
        if argv.custom_recognition_param:
            try:
                params = json.loads(argv.custom_recognition_param)
            except json.JSONDecodeError:
                logger.warning(f"OCRWithSimilarity 参数解析失败: {argv.custom_recognition_param}")

        expected: str = str(params.get("expected", "")).strip()
        threshold: float = float(params.get("threshold", 0.55))

        # 使用节点 roi 与自定义 roi 的组合裁切图片
        roi = _merge_roi_box(argv.roi, params.get("roi"))
        x, y, w, h = roi
        img = argv.image
        cropped = img[y : y + h, x : x + w] if all(v is not None for v in roi) else img

        # 复用已有 OCR 节点，强制覆盖 roi，以得到完整的识别结果
        override = {
            "自动编队-读取生效中命盘": {
                "roi": [0, 0, cropped.shape[1], cropped.shape[0]],
            }
        }
        if expected:
            override["自动编队-读取生效中命盘"]["expected"] = expected

        reco_detail = context.run_recognition("自动编队-读取生效中命盘", cropped, override)
        if not reco_detail or not getattr(reco_detail, "hit", False):
            return None

        candidates = _get_results(reco_detail)
        best_box = None
        best_text = ""
        best_score = 0.0
        for res in candidates:
            text = getattr(res, "text", "") or ""
            score = SequenceMatcher(None, text.strip(), expected).ratio() if expected else 1.0
            if score > best_score:
                best_score = score
                best_text = text
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
