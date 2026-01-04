import time
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger

_SELL_TEMPLATES = [
    "nanyang/forsell1.png",
    "nanyang/forsell2.png",
    "nanyang/forsell3.png",
    "nanyang/forsell4.png",
]

_FIND_NODE = "南阳-找到要卖的菌"
_PRICE_NODE = "南阳-检测菌的价格"
_SCROLL_BOTTOM_TASK = "南阳-下滑到回收底端"
_PREPARE_UP_TASK = "南阳-准备向上寻找要卖的菌"
_SCROLL_UP_TASK = "南阳-向上寻找要卖的菌"
_SET_MAX_TASK = "南阳-设置最大卖出数量"

_FIND_ROI = (71, 443, 574, 394)
_COMPARE_ROI = (88, 467, 541, 182)

_PIXEL_DIFF_THRESHOLD = 2
_MAX_SCROLL_UP = 20
_SCROLL_SETTLE_SECONDS = 1.0

_EXPECTED_PRICE_BY_TEMPLATE = {
    "nanyang/forsell1.png": "3",
    "nanyang/forsell2.png": "9",
    "nanyang/forsell3.png": "1",
}

_SELL_LABEL_BY_TEMPLATE = {
    "nanyang/forsell1.png": "普通老土草帽菌",
    "nanyang/forsell2.png": "常见实习泡泡菌",
    "nanyang/forsell3.png": "未能长大的草帽菌",
    "nanyang/forsell4.png": "未能长大的泡泡菌",
}
_SET_MAX_REPEAT_BY_TEMPLATE = {
    "nanyang/forsell1.png": 25,
}
_DEFAULT_SET_MAX_REPEAT = 5

_PRICE_X_COEFF = (267, 271)
_PRICE_X_W_COEFF = (8504, 13279)
_PRICE_X_BIAS = (-152689, 13279)
_PRICE_Y_COEFF = (137, 134)
_PRICE_Y_H_COEFF = (641, 268)
_PRICE_Y_BIAS = (-3276, 67)
_PRICE_W_COEFF = (1481, 13279)
_PRICE_W_X_COEFF = (3, 271)
_PRICE_W_BIAS = (639918, 13279)
_PRICE_H_COEFF = (-245, 268)
_PRICE_H_Y_COEFF = (-7, 134)
_PRICE_H_BIAS = (8113, 67)


def _should_stop_ctx(context: Context) -> bool:
    """协作式停止检查，兼容 StopTask/Tasker.stop()."""
    try:
        if bool(getattr(context, "stop", False)):
            return True
        tasker = getattr(context, "tasker", None)
        if tasker is not None:
            if bool(getattr(tasker, "stopping", False)):
                return True
            if not tasker.running:
                return True
    except Exception:
        return False
    return False


def _stop_result() -> CustomAction.RunResult:
    logger.info("检测到任务终止")
    return CustomAction.RunResult(success=False)


def _wait_task(context: Context, task_detail):
    if _should_stop_ctx(context):
        return None
    if not task_detail:
        return None
    status = getattr(task_detail, "status", None)
    if status and getattr(status, "done", False):
        return task_detail
    gen_job = getattr(context.tasker, "_gen_task_job", None)
    if callable(gen_job):
        try:
            return gen_job(task_detail.task_id).wait().get()
        except Exception:
            logger.exception("等待任务完成时异常")
    return task_detail


def _screencap(context: Context):
    if _should_stop_ctx(context):
        return None
    try:
        return context.tasker.controller.post_screencap().wait().get()
    except Exception:
        logger.exception("截图失败")
        return None


def _clamp_roi(roi, img_shape):
    x, y, w, h = roi
    height, width = img_shape[:2]
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def _crop_roi(img, roi):
    if img is None or not roi or len(roi) < 4:
        return None
    x, y, w, h = _clamp_roi(roi, img.shape)
    return img[y : y + h, x : x + w]


def _roi_changed(before, after) -> bool:
    if before is None or after is None:
        return True
    if before.shape != after.shape:
        return True
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16))
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    return bool(np.any(diff > _PIXEL_DIFF_THRESHOLD))


def _calc_price_roi_offset(roi):
    try:
        x, y, w, h = [int(v) for v in roi]
    except Exception:
        return [0, 0, 0, 0]
    target_x = int(
        round(
            x * _PRICE_X_COEFF[0] / _PRICE_X_COEFF[1]
            + w * _PRICE_X_W_COEFF[0] / _PRICE_X_W_COEFF[1]
            + _PRICE_X_BIAS[0] / _PRICE_X_BIAS[1]
        )
    )
    target_y = int(
        round(
            y * _PRICE_Y_COEFF[0] / _PRICE_Y_COEFF[1]
            + h * _PRICE_Y_H_COEFF[0] / _PRICE_Y_H_COEFF[1]
            + _PRICE_Y_BIAS[0] / _PRICE_Y_BIAS[1]
        )
    )
    target_w = int(
        round(
            w * _PRICE_W_COEFF[0] / _PRICE_W_COEFF[1]
            + x * _PRICE_W_X_COEFF[0] / _PRICE_W_X_COEFF[1]
            + _PRICE_W_BIAS[0] / _PRICE_W_BIAS[1]
        )
    )
    target_h = int(
        round(
            h * _PRICE_H_COEFF[0] / _PRICE_H_COEFF[1]
            + y * _PRICE_H_Y_COEFF[0] / _PRICE_H_Y_COEFF[1]
            + _PRICE_H_BIAS[0] / _PRICE_H_BIAS[1]
        )
    )
    return [
        target_x - x,
        target_y - y,
        target_w - w,
        target_h - h,
    ]


def _recognize_sell_target(context: Context, img, template: str):
    override = {
        _FIND_NODE: {
            "recognition": {
                "type": "TemplateMatch",
                "param": {
                    "template": [template],
                    "roi": list(_FIND_ROI),
                    "order_by": "Score",
                },
            }
        }
    }
    return context.run_recognition(_FIND_NODE, img, override)


def _run_price_check(context: Context, img, box, expected: str):
    roi = [int(v) for v in box]
    roi_offset = _calc_price_roi_offset(roi)
    override = {
        _PRICE_NODE: {
            "recognition": {
                "param": {
                    "expected": expected,
                    "roi": roi,
                    "roi_offset": roi_offset,
                }
            }
        }
    }
    return context.run_recognition(_PRICE_NODE, img, override)


def _get_hit_box(detail):
    if not detail or not getattr(detail, "hit", False):
        return None
    best = getattr(detail, "best_result", None)
    if best is not None:
        box = getattr(best, "box", None)
        if box:
            return box
    for attr in ("filtered_results", "filterd_results", "all_results"):
        results = getattr(detail, attr, None)
        if results:
            box = getattr(results[0], "box", None)
            if box:
                return box
    return None


def _get_all_hit_boxes(detail):
    if not detail or not getattr(detail, "hit", False):
        return []
    for attr in ("filtered_results", "filterd_results", "all_results"):
        results = getattr(detail, attr, None)
        if results:
            boxes = []
            for res in results:
                box = getattr(res, "box", None)
                if box:
                    boxes.append(box)
            if boxes:
                return boxes
    best = getattr(detail, "best_result", None)
    if best is not None:
        box = getattr(best, "box", None)
        if box:
            return [box]
    return []


def _click_box(context: Context, box) -> bool:
    if _should_stop_ctx(context):
        return False
    if not box:
        return False
    try:
        x, y, w, h = [int(v) for v in box]
    except Exception:
        return False
    cx = x + w // 2
    cy = y + h // 2
    context.tasker.controller.post_click(cx, cy).wait()
    return True


def _get_set_max_repeat(template: str) -> int:
    repeat = _SET_MAX_REPEAT_BY_TEMPLATE.get(template)
    if repeat is None:
        return _DEFAULT_SET_MAX_REPEAT
    return repeat


@AgentServer.custom_action("NanyangSell")
class NanyangSell(CustomAction):
    """
    依次查找并卖出指定的菌子。
    """

    def _try_find_and_click(self, context: Context, img, template: str) -> bool:
        if _should_stop_ctx(context):
            return False
        if img is None:
            return False
        reco_detail = _recognize_sell_target(context, img, template)
        boxes = _get_all_hit_boxes(reco_detail)
        if not boxes:
            return False
        expected = _EXPECTED_PRICE_BY_TEMPLATE.get(template)
        if not expected:
            # logger.info(f"跳过价格检查: {template}")
            for box in boxes:
                if _should_stop_ctx(context):
                    return False
                if _click_box(context, box):
                    return True
            return False
        for box in boxes:
            if _should_stop_ctx(context):
                return False
            detail = _run_price_check(context, img, box, expected)
            if detail and getattr(detail, "hit", False):
                if _click_box(context, box):
                    return True
        return False

    def _set_max_count(self, context: Context, template: str) -> None:
        repeat = _get_set_max_repeat(template)
        overrides = {_SET_MAX_TASK: {"repeat": repeat}}
        _wait_task(context, context.run_task(_SET_MAX_TASK, overrides))

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        for template in _SELL_TEMPLATES:
            if _should_stop_ctx(context):
                return _stop_result()
            label = _SELL_LABEL_BY_TEMPLATE.get(template, template)
            logger.info(f"开始寻找 {label}")
            _wait_task(context, context.run_task(_SCROLL_BOTTOM_TASK))

            img = _screencap(context)
            found = self._try_find_and_click(context, img, template)

            if not found:
                if _should_stop_ctx(context):
                    return _stop_result()
                _wait_task(context, context.run_task(_PREPARE_UP_TASK))
                prepare_img = _screencap(context)
                found = self._try_find_and_click(context, prepare_img, template)
                if found:
                    self._set_max_count(context, template)
                    continue
                baseline_img = prepare_img
                baseline_roi = _crop_roi(baseline_img, _COMPARE_ROI)
                if baseline_roi is None:
                    logger.warning("无法获取对比区域")
                else:
                    for _ in range(_MAX_SCROLL_UP):
                        if _should_stop_ctx(context):
                            return _stop_result()
                        _wait_task(context, context.run_task(_SCROLL_UP_TASK))
                        if _should_stop_ctx(context):
                            return _stop_result()
                        time.sleep(_SCROLL_SETTLE_SECONDS)
                        img = _screencap(context)
                        current_roi = _crop_roi(img, _COMPARE_ROI)
                        if current_roi is None:
                            break
                        if not _roi_changed(baseline_roi, current_roi):
                            break
                        baseline_roi = current_roi
                        if self._try_find_and_click(context, img, template):
                            found = True
                            break

            if found:
                self._set_max_count(context, template)
            else:
                logger.info(f"未找到目标 {label}")

        return CustomAction.RunResult(success=True)
