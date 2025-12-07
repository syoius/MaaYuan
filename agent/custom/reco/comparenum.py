from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

import json
import cv2
from utils import logger


@AgentServer.custom_recognition("CompareNum")
class CompareNum(CustomRecognition):
    """
    检查OCR结果与给定数字的比较关系

    参数格式:
    {
        "roi": [x,y,w,h],
        "expected": 550,
        "operator": ">="  // 支持 ">", ">=", "<", "<="
    }

    返回结果:
    如果OCR识别的数字满足比较条件，返回识别结果；否则返回None
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        raw_img = argv.image
        params = json.loads(argv.custom_recognition_param)
        expected = params["expected"]
        operator = params["operator"]
        roi = params["roi"]
        if roi and len(roi) == 4:
            x, y, w, h = roi
            roi_img = raw_img[y : y + h, x : x + w]
        else:
            roi_img = raw_img
        # logger.info(
        #     f"已载入图片及参数expected:{expected},roi:{roi},operator:{operator}"
        # )
        # cv2.imwrite("debug_roi.png", roi_img)

        # todo: 直接对roi_img进行识别时疑似因为image size太小导致无法成功识别，后续需要将roi通过pipeline_override传入run_recognition以实现泛化
        digit_detail = context.run_recognition("大富翁-商店货币数", raw_img)
        if not digit_detail or not getattr(digit_detail, "hit", False):
            logger.info("未识别到数字，返回 None")
            return None

        # 尝试提取字符串内容
        digit_text = None
        try:
            digit_text = digit_detail.best_result.text
        except Exception:
            digit_text = str(digit_detail)

        try:
            ocr_number = int(digit_text.strip())
            expected_number = int(expected)

            if operator == ">":
                result = ocr_number > expected_number
            elif operator == ">=":
                result = ocr_number >= expected_number
            elif operator == "<":
                result = ocr_number < expected_number
            elif operator == "<=":
                result = ocr_number <= expected_number
            else:
                logger.error(f"不支持的操作符: {operator}")
                return None

            if result:
                return CustomRecognition.AnalyzeResult(
                    box=(0, 0, roi_img.shape[1], roi_img.shape[0]),
                    detail=f"{ocr_number}",
                )
            else:
                return None

        except ValueError as e:
            logger.error(f"数字转换失败: {e}, OCR结果: '{digit_text}'")
            return None
