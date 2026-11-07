"""Screen-driven Jianmu loop. Every click belongs to a recognized activity page."""

from pathlib import Path
import re
import time

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from utils import logger


TREE = (44, 248, 632, 740)
MAP = (0, 96, 720, 1044)
OCR_NODE = "建木巡检-读字"


class PatrolError(RuntimeError):
    pass


class Stopped(PatrolError):
    pass


class ScanError(PatrolError):
    pass


def red_text(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 95, 155), (13, 255, 255)) | cv2.inRange(
        hsv, (166, 95, 155), (179, 255, 255)
    )


def crop(image, roi):
    x, y, w, h = roi
    return image[y:y + h, x:x + w]


def same_marker(first, second):
    kind, (x, y) = first
    other_kind, (other_x, other_y) = second
    return kind == other_kind and abs(x - other_x) <= 24 and abs(y - other_y) <= 24


def vertical_delta(before, after):
    """Use several strips of the scene that are clear of fixed buttons."""
    deltas = []
    search = after[251:988, 54:669]
    for y in range(251, 939, 54):
        template = before[y:y + 42, 54:669]
        if template.std() < 10:
            continue
        scores = cv2.matchTemplate(search, template, cv2.TM_SQDIFF_NORMED)
        low, _, (_, match_y), _ = cv2.minMaxLoc(scores)
        if low < .025:
            deltas.append(match_y + 251 - y)
    for delta in sorted(deltas, key=lambda d: sum(abs(d - other) <= 5 for other in deltas), reverse=True):
        if sum(abs(delta - other) <= 5 for other in deltas) >= 2:
            return delta
    return None


class Vision:
    def __init__(self):
        root = Path(__file__).resolve().parents[3]
        directory = root / "resource/base/image/jianmu"
        if not directory.is_dir():
            directory = root / "assets/resource/base/image/jianmu"
        self.templates = {
            p.stem: cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            for p in directory.glob("*.png")
        }
        self.templates["plan_missing"] = red_text(self.templates["plan_missing"])

    def hits(self, frame, key, roi, threshold=.97, *, shape=0, mask=False, many=False):
        x, y, _, _ = roi
        image = crop(frame, roi)
        template = self.templates[key]
        if mask:
            image = red_text(image)
        h, w = template.shape[:2]
        if image.shape[0] < h or image.shape[1] < w:
            return []
        if mask:
            scores = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
        else:
            scores = 1 - cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
        if shape:
            scores[cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) < shape] = 0
        found = []
        for _ in range(40 if many else 1):
            _, score, _, (a, b) = cv2.minMaxLoc(scores)
            if score < threshold:
                break
            found.append((x + a + w // 2, y + b + h // 2))
            scores[max(0, b - h):b + h + 1, max(0, a - w):a + w + 1] = 0
        return found

    def hit(self, frame, key, roi, threshold=.97, *, shape=0):
        return next(iter(self.hits(frame, key, roi, threshold, shape=shape)), None)

    def tree(self, frame):
        return (
            self.hit(frame, "tree_title", (40, 145, 123, 50))
            and self.hit(frame, "outside_button", (590, 1118, 98, 106), .97, shape=.85)
        )

    def outside(self, frame):
        return (
            self.hit(frame, "outside_title", (20, 132, 162, 66))
            and self.hit(frame, "return", (591, 1123, 98, 98))
        )

    def markers(self, frame):
        result = []
        for key in ("gold", "purple", "blue", "bang"):
            for x, y in self.hits(frame, "mark_" + key, MAP, .98, shape=.94, many=True):
                if all(abs(x - a) > 25 or abs(y - b) > 25 for _, (a, b) in result):
                    result.append((key, (x, y)))
        return result

    def room_targets(self, frame, key, roi=TREE):
        if key == "upgrade":
            return self.hits(frame, "upgrade_arrow", roi, .95, shape=.85, many=True)
        return self.hits(frame, "plan_missing", roi, .79, shape=.65, mask=True, many=True)


class Patrol:
    def __init__(self, context, vision):
        self.context = context
        self.controller = context.tasker.controller
        self.vision = vision
        self.phase = "启动"

    def check_stop(self):
        if self.context.tasker.stopping:
            raise Stopped()

    def pause(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check_stop()
            time.sleep(min(.25, max(0, end - time.monotonic())))

    def shot(self):
        for _ in range(3):
            self.check_stop()
            job = self.controller.post_screencap().wait()
            frame = job.get() if job.succeeded else None
            if frame is not None and frame.shape[:2] == (1280, 720):
                return frame
            self.pause(.35)
        raise PatrolError("未取得 720×1280 的活动截图")

    def wait(self, recognizes, description, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            frame = self.shot()
            if recognizes(frame):
                return frame
            self.pause(.2)
        raise PatrolError(f"未确认{description}")

    def click(self, target):
        self.check_stop()
        # A very short tap can fall between two Unity input frames on MuMu.
        if not target or not self.controller.post_swipe(*target, *target, 75).wait().succeeded:
            raise PatrolError(f"点击未完成 {target}")
        self.pause(.45)

    def swipe(self, x1, y1, x2, y2, duration=480):
        self.check_stop()
        if not self.controller.post_swipe(x1, y1, x2, y2, duration).wait().succeeded:
            raise PatrolError("滑动未完成")
        self.pause(.65)

    def ocr(self, frame, roi):
        detail = self.context.run_recognition(
            OCR_NODE, frame, {OCR_NODE: {"roi": list(roi)}}
        )
        return getattr(detail, "all_results", None) or []

    def text(self, frame, roi, pattern):
        return next((r for r in self.ocr(frame, roi) if re.search(pattern, r.text)), None)

    def result_footer(self, frame):
        roi = (265, 1178, 190, 65)
        target = self.vision.hit(frame, "reward_close", roi, .78, shape=.85)
        if target:
            return target
        footer = self.text(frame, roi, r"点击\s*空白处\s*关闭")
        return self.center(footer) if footer else None

    def rewards(self, parent, *, required=True):
        seen = False
        end = time.monotonic() + 8
        while time.monotonic() < end:
            frame = self.shot()
            reward = self.vision.hit(frame, "reward_title", (245, 204, 230, 78))
            soul = self.vision.hit(frame, "soul_title", (245, 204, 230, 78))
            footer = self.result_footer(frame) if reward or soul else None
            if footer:
                self.click(footer)
                seen = True
            elif (seen or not required) and parent(frame):
                return frame
            else:
                self.pause(.2)
        if required:
            raise PatrolError("委托提交后的奖励或返回画面未确认")
        raise PatrolError("关闭树外详情后的奖励或地图未确认")

    def tree_step(self, before, direction):
        # A step stays shorter than the visible scan region, so adjacent views overlap.
        start, end = (480, 740) if direction > 0 else (740, 480)
        self.swipe(350, start, 350, end, duration=730)
        for _ in range(3):
            after = self.wait(self.vision.tree, "树内画面")
            delta = vertical_delta(before, after)
            self.pause(.35)
            stable = self.wait(self.vision.tree, "滑动后的树内主界面")
            settled = vertical_delta(after, stable)
            if settled is not None and abs(settled) <= 4 and delta is not None:
                if abs(delta) <= 5 or 0 < delta * direction <= 580:
                    return stable, abs(delta) > 5
        raise ScanError("无法确认树内滑动距离，仍有楼层未核实")

    def tabs(self, frame):
        close = self.vision.hit(frame, "room_close", (646, 737, 51, 155), .90, shape=.76)
        if not close:
            return None
        x, y = close
        results = self.ocr(frame, (173, y - 50, 375, 59))
        detail = next((r for r in results if "建筑详情" in r.text), None)
        upgrade = next((r for r in results if "建筑升级" in r.text), None)
        return (close, detail, upgrade) if detail and upgrade else None

    @staticmethod
    def center(result):
        x, y, w, h = result.box
        return x + w // 2, y + h // 2

    def close_room(self, frame):
        tabs = self.tabs(frame)
        if not tabs:
            raise PatrolError("设施详情关闭按钮未确认")
        self.click(tabs[0])
        return self.wait(self.vision.tree, "返回树内")

    def is_upgrade_result(self, frame):
        return self.vision.hit(frame, "upgrade_result", (248, 463, 226, 88), .94, shape=.85)

    def upgrade_result_footer(self, frame):
        return self.result_footer(frame) if self.is_upgrade_result(frame) else None

    def upgrade_refund_footer(self, frame):
        if self.vision.hit(frame, "upgrade_refund_title", (245, 204, 230, 78), .97, shape=.85):
            return self.result_footer(frame)
        return None

    def close_upgrade_result(self, frame):
        self.click(self.upgrade_result_footer(frame))
        frame = self.wait(
            lambda im: not self.is_upgrade_result(im) and (self.vision.tree(im) or self.tabs(im)),
            "关闭升级结果",
        )
        return frame if self.vision.tree(frame) else self.close_room(frame)

    def upgrade_room(self, frame):
        tabs = self.tabs(frame)
        self.click(self.center(tabs[2]))
        frame = self.wait(
            lambda im: self.tabs(im) and self.text(im, (75, 991, 165, 110), "资源产出效率|资源囤积上限"),
            "建筑升级面板",
        )
        ok = self.vision.hit(frame, "upgrade_confirm", (500, 1163, 150, 63), .985, shape=.94)
        if not ok:
            if not self.text(frame, (500, 1155, 150, 70), "确定|已满级") and not self.text(
                frame, (75, 875, 590, 370), "解锁|满级"
            ):
                raise PatrolError("建筑升级按钮状态未确认")
            logger.info("建木：该设施升级条件不足或已满级")
            return self.close_room(frame), False
        title = self.ocr(frame, (230, 871, 264, 56))
        self.click(ok)

        def interrupt(im):
            if self.vision.hit(im, "upgrade_interrupt", (78, 533, 555, 68), .93, shape=.85):
                return self.vision.hit(im, "upgrade_interrupt_ok", (433, 746, 154, 74), .93, shape=.85)
            return None

        def confirmation_or_result(im):
            return self.upgrade_result_footer(im) or self.upgrade_refund_footer(im) or interrupt(im)

        frame = self.wait(confirmation_or_result, "设施升级结果", timeout=10)
        for _ in range(2):
            confirm = interrupt(frame)
            if not confirm:
                break
            self.click(confirm)
            self.pause(1.7)
            frame = self.wait(confirmation_or_result, "设施升级完成", timeout=10)
        refund = self.upgrade_refund_footer(frame)
        if refund:
            self.click(refund)
            frame = self.wait(self.upgrade_result_footer, "返还材料后的设施升级结果", timeout=10)
        if not self.upgrade_result_footer(frame):
            raise PatrolError("二次确认后仍未出现设施升级结果")
        frame = self.close_upgrade_result(frame)
        logger.info("建木：设施升级完成 " + " ".join(r.text for r in title))
        return frame, True

    def plan_room(self, frame):
        if not self.text(frame, (52, 1038, 183, 47), "合成耗时.*00:00"):
            return self.close_room(frame)
        if not self.text(frame, (456, 960, 209, 48), "正常"):
            return self.close_room(frame)
        slider = self.vision.hit(frame, "plan_slider_start", (48, 1085, 39, 40), .92)
        if not slider:
            return self.close_room(frame)
        self.swipe(*slider, 650, slider[1], 440)
        frame = self.wait(self.tabs, "生产计划面板")
        ok = self.vision.hit(frame, "plan_confirm", (500, 1163, 150, 63), .985, shape=.94)
        output = self.text(frame, (405, 1040, 172, 48), r"预计产出[：:]?\s*[1-9]")
        if ok and output:
            self.click(ok)
            frame = self.wait(
                lambda im: self.vision.tree(im) or (self.tabs(im) and self.text(im, (45, 1017, 614, 213), "剩余|生产中|取消计划|终止")),
                "生产计划开始",
            )
            logger.info(f"建木：已设置生产计划 {output.text}")
        return frame if self.vision.tree(frame) else self.close_room(frame)

    def inspect_view(self, frame, mode, edge=None):
        handled = set()
        rois = [TREE]
        if edge == "top":
            rois.append((241, 101, 434, 169))
        if edge == "bottom":
            rois += [(48, 963, 274, 159), (221, 1104, 363, 175), (47, 1237, 628, 42)]
        for _ in range(30):
            targets = [p for roi in rois for p in self.vision.room_targets(frame, mode, roi)]
            target = next(((x, y) for x, y in targets if (x // 9, y // 9) not in handled), None)
            if not target:
                return frame
            x, y = target
            self.click((x, y) if mode == "upgrade" else (x, y - 34))
            frame = self.wait(self.tabs, "设施详情")
            if mode == "upgrade":
                frame, changed = self.upgrade_room(frame)
                if changed:
                    continue
            else:
                frame = self.plan_room(frame)
            handled.add((x // 9, y // 9))
        raise PatrolError("同一视窗设施处理达到上限")

    def traverse(self, frame, direction, mode=None):
        stationary = 0
        for _ in range(40):
            if mode:
                frame = self.inspect_view(frame, mode)
            frame, moved = self.tree_step(frame, direction)
            stationary = 0 if moved else stationary + 1
            if stationary == 2:
                return self.inspect_view(frame, mode, "top" if direction > 0 else "bottom") if mode else frame
        raise ScanError("未扫到树的边界，仍有楼层未核实")

    def close_map_panel(self, frame):
        self.click(self.vision.hit(frame, "panel_close", (650, 174, 54, 60), .88, shape=.77))
        return self.rewards(self.vision.outside, required=False)

    def panel(self, frame, marker):
        def opened(im):
            return not self.vision.outside(im) and self.vision.hit(
                im, "panel_close", (650, 174, 54, 60), .88, shape=.77
            )

        for _ in range(2):
            if not self.vision.outside(frame):
                raise PatrolError("地图标记的位置已失效")
            current = next((m for m in self.vision.markers(frame) if same_marker(marker, m)), None)
            if not current:
                logger.info(f"建木：地图标记 {marker} 已刷新，继续检查当前地图")
                return frame, False
            self.click(current[1])
            try:
                return self.wait(opened, "标记详情"), True
            except Stopped:
                raise
            except PatrolError:
                frame = self.shot()
                if opened(frame):
                    return frame, True
                if not self.vision.outside(frame):
                    raise
        logger.warning(f"建木：地图标记 {marker} 重试后仍未打开，留待下一次地图检查")
        return frame, False

    def order_button(self, frame, key):
        if not self.vision.hit(frame, "order_needs", (298, 530, 126, 54), .93):
            return None
        return self.vision.hit(frame, key, (268, 1018, 187, 66), .98, shape=.92)

    def scan_markers(self, frame):
        seen = []
        opened_count = 0
        candidates = []
        for _ in range(25):
            markers = self.vision.markers(frame)
            marker = next((m for m in markers if not any(same_marker(m, prior) for prior in seen)), None)
            if not marker:
                logger.info(f"建木：本次地图已打开 {opened_count}/{len(seen)} 个标记")
                return frame, candidates
            kind, point = marker
            frame, opened = self.panel(frame, marker)
            seen.append(marker)
            if not opened:
                continue
            opened_count += 1
            name = " ".join(r.text for r in self.ocr(frame, (170, 221, 380, 59))) or "标题未识别"
            is_order = self.vision.hit(frame, "order_needs", (298, 530, 126, 54), .93)
            mark = "!" if kind == "bang" else "?"
            category = "委托 " if is_order else ""
            logger.info(f"建木：地图 {mark} {point} · {category}{name}")
            if self.order_button(frame, "deliver_order"):
                candidates.append(marker)
            frame = self.close_map_panel(frame)
        raise PatrolError("地图标记未完成枚举")

    def deliver_marker(self, frame, candidates):
        for marker in candidates:
            frame, opened = self.panel(frame, marker)
            if not opened:
                continue
            deliver = self.order_button(frame, "deliver_order")
            if deliver:
                self.click(deliver)
                frame = self.rewards(
                    lambda im: self.vision.outside(im)
                    or self.vision.hit(im, "panel_close", (650, 174, 54, 60), .88, shape=.77)
                )
                if not self.vision.outside(frame):
                    frame = self.close_map_panel(frame)
                return frame, True
            frame = self.close_map_panel(frame)
        return frame, False

    def accepted(self, frame):
        return self.vision.hit(frame, "accepted_title", (28, 181, 173, 67), .95)

    def accepted_orders(self, frame):
        self.click(self.vision.hit(frame, "map_orders", (470, 1120, 88, 90), .94))
        frame = self.wait(self.accepted, "已接委托列表")
        deliver = self.vision.hit(frame, "deliver_list", (490, 277, 176, 774), .98, shape=.92)
        if deliver:
            self.click(deliver)
            frame = self.rewards(lambda im: self.accepted(im) or self.vision.outside(im))
            return (frame if self.vision.outside(frame) else self.close_map_panel(frame)), True
        return self.close_map_panel(frame), False

    def cycle(self):
        self.phase = "收取与派遣"
        frame = self.shot()
        if self.is_upgrade_result(frame):
            frame = self.wait(self.upgrade_result_footer, "上次升级结果的关闭提示")
            self.close_upgrade_result(frame)
        result = self.context.run_task("建木巡检-开始")
        if not result or not result.status.succeeded:
            raise PatrolError("收取与派遣未完成")
        self.phase = "逐层扫描设施升级"
        frame = self.wait(self.vision.tree, "树内主画面")
        frame = self.traverse(frame, 1)
        frame = self.inspect_view(frame, "upgrade", "top")
        frame = self.traverse(frame, -1, "upgrade")
        self.phase = "逐层扫描生产计划"
        frame = self.inspect_view(frame, "plan", "bottom")
        frame = self.traverse(frame, 1, "plan")
        self.phase = "树外地图与委托"
        self.click(self.vision.hit(frame, "outside_button", (590, 1118, 98, 106), .94))
        frame = self.wait(self.vision.outside, "树外地图")
        for _ in range(16):
            frame, candidates = self.scan_markers(frame)
            frame, delivered = self.deliver_marker(frame, candidates)
            if not delivered:
                frame, delivered = self.accepted_orders(frame)
            if not delivered:
                return
            logger.info("建木：交付完成，重新检查地图标记")


@AgentServer.custom_action("JianmuPatrol")
class JianmuPatrol(CustomAction):
    def run(self, context, argv):
        patrol = Patrol(context, Vision())
        interrupted = 0
        try:
            while True:
                patrol.check_stop()
                try:
                    patrol.cycle()
                except ScanError as error:
                    interrupted += 1
                    logger.warning(f"建木：{error}，本轮委托暂缓")
                    if interrupted >= 3:
                        raise
                else:
                    interrupted = 0
                patrol.phase = "等待下一轮"
                patrol.pause(60)
        except Stopped:
            return CustomAction.RunResult(success=True)
        except PatrolError as error:
            logger.warning(f"建木挂机停止在{patrol.phase}：{error}")
            return CustomAction.RunResult(success=False)
