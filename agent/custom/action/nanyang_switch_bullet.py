import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger

_HAT_RECO = "南阳-当前为草帽菌"
_CORAL_RECO = "南阳-当前为珊瑚"
_GEM_RECO = "南阳-当前为宝石菌"

_CORAL_ROI = [355, 620, 30, 25]
_HAT_ROI = [156, 619, 32, 28]
_CONFIRM_ROI = [346, 825, 56, 26]

_CLICK_DELAY_SECONDS = 0.5


def _screencap(context: Context):
    try:
        return context.tasker.controller.post_screencap().wait().get()
    except Exception:
        logger.exception("NanyangSwitchBullet: screencap failed")
        return None


def _is_hit(detail) -> bool:
    return bool(detail and getattr(detail, "hit", False))


def _click_roi(context: Context, roi) -> bool:
    if not roi or len(roi) < 4:
        return False
    try:
        x, y, w, h = [int(v) for v in roi]
    except Exception:
        return False
    cx = x + w // 2
    cy = y + h // 2
    context.tasker.controller.post_click(cx, cy).wait()
    return True


@AgentServer.custom_action("NanyangSwitchBullet")
class NanyangSwitchBullet(CustomAction):
    """
    Switch to a lower-tier bullet based on current bullet recognition.
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        img = _screencap(context)
        if img is None:
            logger.warning("NanyangSwitchBullet: no screenshot")
            return CustomAction.RunResult(success=False)

        hat_hit = _is_hit(context.run_recognition(_HAT_RECO, img))
        coral_hit = _is_hit(context.run_recognition(_CORAL_RECO, img))
        gem_hit = _is_hit(context.run_recognition(_GEM_RECO, img))

        current = None
        if gem_hit:
            current = "gem"
        elif coral_hit:
            current = "coral"
        elif hat_hit:
            current = "hat"

        if current is None:
            logger.warning("NanyangSwitchBullet: cannot detect current bullet")
            return CustomAction.RunResult(success=False)

        if current == "hat":
            logger.info("NanyangSwitchBullet: already at lowest bullet")
            context.run_task("stop")
            return CustomAction.RunResult(success=False)

        target_roi = _CORAL_ROI if current == "gem" else _HAT_ROI
        if not _click_roi(context, target_roi):
            logger.warning("NanyangSwitchBullet: failed to click target bullet")
            return CustomAction.RunResult(success=False)

        time.sleep(_CLICK_DELAY_SECONDS)

        if not _click_roi(context, _CONFIRM_ROI):
            logger.warning("NanyangSwitchBullet: failed to click confirm")
            return CustomAction.RunResult(success=False)

        logger.info("NanyangSwitchBullet: bullet switched")
        return CustomAction.RunResult(success=True)
