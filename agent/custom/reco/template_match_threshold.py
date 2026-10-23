import json

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from utils import logger


@AgentServer.custom_recognition("TemplateMatchThreshold")
class TemplateMatchThreshold(CustomRecognition):
    """Run TemplateMatch using an integer threshold and explicit matcher params.

    custom_recognition_param example:
    {
        "threshold_x100": 70,
        "template_match_param": {"template": "path/to/template.png"}
    }
    The current node is re-run as TemplateMatch with threshold 0.70.
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        try:
            params = json.loads(argv.custom_recognition_param or "{}")
            threshold_x100 = int(params["threshold_x100"])
            recognition_param = params["template_match_param"]
            if not 0 <= threshold_x100 <= 100:
                raise ValueError("threshold_x100 must be between 0 and 100")
            if not isinstance(recognition_param, dict):
                raise ValueError("template_match_param must be an object")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.error(f"Invalid TemplateMatchThreshold parameters: {exc}")
            return None

        recognition_param = dict(recognition_param)
        recognition_param["threshold"] = threshold_x100 / 100.0
        detail = context.run_recognition(
            argv.node_name,
            argv.image,
            {
                argv.node_name: {
                    "recognition": {
                        "type": "TemplateMatch",
                        "param": recognition_param,
                    }
                }
            },
        )
        score = (
            getattr(getattr(detail, "best_result", None), "score", None)
            if detail
            else None
        )
        logger.debug(
            f"TemplateMatchThreshold node={argv.node_name} "
            f"threshold={threshold_x100 / 100.0} "
            f"hit={bool(detail and getattr(detail, 'hit', False))} score={score}"
        )
        if not detail or not getattr(detail, "hit", False):
            return None
        best = getattr(detail, "best_result", None)
        if best is None:
            return None
        return CustomRecognition.AnalyzeResult(
            box=list(best.box),
            detail={
                "threshold": threshold_x100 / 100.0,
                "score": getattr(best, "score", None),
            },
        )
