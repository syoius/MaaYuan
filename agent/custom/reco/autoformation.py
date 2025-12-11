from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils import logger


@AgentServer.custom_recognition("OCRWithSimilarity")
class OCRWithSimilarity(CustomRecognition):
    """
    检查OCR结果和expected的similarly

    参数格式:
    {
        "expected": "密探name",
    }

    """
