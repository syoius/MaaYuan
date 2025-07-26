from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

import json
import cv2
from utils import logger


@AgentServer.custom_recognition("PureNum")
class PureNum(CustomRecognition):
    """
    对截图进行预处理后再进行OCR

    参数格式:
    {
        "roi": [x,y,w,h]
        "expected":"digit"
    }

    返回结果:
    对预处理后的截图进行OCR后得到的数字是否和expected的相同
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        raw_img = argv.image
        expected = json.loads(argv.custom_recognition_param)["expected"]
        # 根据roi裁切
        roi = json.loads(argv.custom_recognition_param)["roi"]
        if roi and len(roi) == 4:
            x, y, w, h = roi
            roi_img = raw_img[y : y + h, x : x + w]
        else:
            roi_img = raw_img
        # logger.info(f"已载入图片及参数expected:{expected},roi:{roi}")
        # cv2.imwrite("debug_roi.png", roi_img)

        # HSV green mask extract
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        lower = (50, 40, 100)
        upper = (90, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
        # cv2.imwrite("debug_bin.png", mask)

        # 对mask做OCR,需要三通道
        img = cv2.merge([mask, mask, mask])
        # cv2.imwrite("debug_img.png", img)
        digit_detail = context.run_recognition("PureNum识别", img)
        # logger.info(f"识别到：{digit_detail}")

        # 尝试提取字符串内容
        digit_text = None
        try:
            digit_text = digit_detail.best_result.text
        except Exception:
            digit_text = str(digit_detail)

        if str(digit_text).startswith(str(expected)):
            return CustomRecognition.AnalyzeResult(
                box=(0, 0, roi_img.shape[1], roi_img.shape[0]),
                detail=digit_text,
            )
        else:
            return None
