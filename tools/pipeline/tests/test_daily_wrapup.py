"""Replay daily wrap-up routing with MaaFramework and no game or device."""

from collections import Counter
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from maa.controller import CustomController
from maa.resource import Resource
from maa.tasker import LoggingLevelEnum, Tasker


ASSETS = Path(os.environ.get("MAAYUAN_TEST_ASSETS", Path(__file__).resolve().parents[3] / "assets"))


class WrapUpController(CustomController):
    def __init__(self, close_blocked=False, auto_close=False):
        self.operations = []
        self.share_open = False
        self.close_blocked = close_blocked
        self.auto_close = auto_close
        super().__init__()

    def connect(self):
        return True

    def request_uuid(self):
        return "daily-wrapup-replay"

    def get_features(self):
        return 0

    def screencap(self):
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        if self.share_open:
            frame[:8, :8] = 255
        return frame

    def click(self, x, y):
        if 618 <= x < 680 and 36 <= y < 95:
            self.operations.append("close_share")
            if not self.close_blocked:
                self.share_open = False
            return True
        operation = {10: "share", 20: "gift", 30: "news", 40: "training"}[y]
        self.operations.append(operation)
        if operation == "share":
            self.share_open = not self.auto_close
        elif self.share_open:
            return False
        return True


class DailyWrapUpTests(unittest.TestCase):
    def replay(self, locale, news, training, upgrade=True, gift=True, share_visible=True,
               close_blocked=False, auto_close=False):
        folder = ASSETS / "resource" / locale / "pipeline"
        nodes = {}
        for name in ("wrap-up", "meetup", "house", "levelingup", "share", "giftgiving2", "special_training_reward"):
            nodes.update(json.loads((folder / "daily" / f"{name}.json").read_text(encoding="utf-8-sig")))
        nodes.update(json.loads((folder / "bird_food_2.json").read_text(encoding="utf-8-sig")))
        nodes.update(json.loads((folder / "base.json").read_text(encoding="utf-8-sig")))
        interface = json.loads((ASSETS / "interface.json").read_text(encoding="utf-8-sig"))
        task = next(item for item in interface["task"] if item["entry"] == "日常收尾启动")
        overrides = copy.deepcopy(task.get("pipeline_override", {}))
        choices = {
            "首位密探升级": "Yes" if upgrade else "No",
            "每日密探送礼": "送 1 次" if gift else "不送礼",
            "收尾前做小道消息": "Yes" if news else "No",
            "领取密探特训奖励": "Yes" if training else "No",
        }
        for option, choice in choices.items():
            case = next(item for item in interface["option"][option]["cases"] if item["name"] == choice)
            for name, values in case["pipeline_override"].items():
                overrides.setdefault(name, {}).update(values)

        # Replace successful UI work only at its boundary. Keep the real routing,
        # optional-node flags, hit limits, share handoff and StopTask unchanged.
        for name, next_node in {
            "相见互动启动": "相见互动结束",
            "家具互动启动": "等待家具互动结束",
            "密探升级启动": "退出升级界面",
            "赠送密探礼物启动": "赠礼结束",
            "start_小道消息": "end_小道消息循环开关",
            "领取密探特训奖励启动": "密探特训奖励提醒",
        }.items():
            nodes[name]["next"] = [next_node]
        operations = {"点击进行分享": 10, "赠送密探礼物启动": 20,
                      "start_小道消息": 30, "领取密探特训奖励启动": 40}
        retained = {
            "日常收尾启动", "相见互动启动", "相见互动结束", "家具互动启动",
            "等待家具互动结束", "密探升级启动", "退出升级界面", "每日分享启动",
            "进入分享popup", "进入分享界面", "点击进行分享", "确认分享结果",
            "赠送密探礼物启动", "赠礼结束", "小道消息收尾启动", "start_小道消息",
            "end_小道消息循环开关", "领取密探特训奖励启动", "密探特训奖励提醒", "stop",
        }
        retained.update(name for name in nodes if name.startswith("日常收尾-"))
        nodes = {name: nodes[name] for name in retained}
        # Unreached navigation alternatives still need resolvable resource names.
        nodes["进入界面-密探"] = {"enabled": False}
        nodes["退出送礼弹窗"] = {"enabled": False}
        nodes["进入界面-鸢报"] = {"enabled": False}
        for name, node in nodes.items():
            if name in {"日常收尾-关闭分享面板", "日常收尾-确认分享面板已关闭"}:
                # Only replace template recognition. Keep the real close target,
                # inverse check, routing and error handling intact.
                node["recognition"] = {
                    "type": "ColorMatch", "param": {
                        "roi": [0, 0, 8, 8], "lower": [255] * 3,
                        "upper": [255] * 3, "count": 64,
                    },
                }
                continue
            node["recognition"] = {"type": "DirectHit"}
            if name != "stop":
                node["action"] = {"type": "DoNothing"}
            if name in operations:
                node["action"] = {"type": "Click", "param": {"target": [1, operations[name], 1, 1]}}
            node.update(pre_delay=0, post_delay=0, rate_limit=1, timeout=50,
                        post_wait_freezes=0)
        if not share_visible:
            nodes["确认分享结果"]["recognition"] = {
                "type": "ColorMatch", "param": {
                    "roi": [0, 0, 8, 8], "lower": [128] * 3,
                    "upper": [128] * 3, "count": 64,
                },
            }
        for name, values in overrides.items():
            if name in nodes:
                nodes[name].update(values)
        # Only real task routing is overridden above; speed up its game delays.
        for node in nodes.values():
            node.update(pre_delay=0, post_delay=0, rate_limit=1, timeout=50,
                        post_wait_freezes=0)
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            (bundle / "pipeline").mkdir()
            (bundle / "pipeline/replay.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
            Tasker.set_log_dir(Path(tempfile.gettempdir()) / "maayuan-daily-wrapup-tests")
            Tasker.set_stdout_level(LoggingLevelEnum.Off)
            resource = Resource()
            self.assertTrue(resource.post_bundle(bundle).wait().status.succeeded)
            controller = WrapUpController(close_blocked=close_blocked, auto_close=auto_close)
            self.assertTrue(controller.post_connection().wait().status.succeeded)
            tasker = Tasker()
            self.assertTrue(tasker.bind(resource, controller))
            job = tasker.post_task("日常收尾启动")
            deadline = time.monotonic() + 3
            while not job.status.done and time.monotonic() < deadline:
                time.sleep(0.01)
            if not job.status.done:
                tasker.post_stop().wait()
                self.fail("Daily wrap-up did not terminate")
            return job.status.succeeded, Counter(controller.operations), controller.share_open

    def test_optional_steps_disabled(self):
        for locale in ("base", "zh_tw"):
            with self.subTest(locale=locale):
                succeeded, operations, share_open = self.replay(locale, False, False)
                self.assertTrue(succeeded)
                self.assertFalse(share_open)
                self.assertEqual(operations, Counter(share=1, close_share=1, gift=1))

    def test_enabled_optional_steps(self):
        for locale in ("base", "zh_tw"):
            for news, training in ((False, True), (True, False), (True, True)):
                with self.subTest(locale=locale, news=news, training=training):
                    succeeded, operations, share_open = self.replay(locale, news, training)
                    self.assertTrue(succeeded)
                    self.assertFalse(share_open)
                    self.assertEqual(operations, Counter(share=1, close_share=1, gift=1, news=int(news), training=int(training)))

    def test_no_upgrade_or_gift(self):
        for locale in ("base", "zh_tw"):
            with self.subTest(locale=locale):
                succeeded, operations, share_open = self.replay(locale, False, False, upgrade=False, gift=False)
                self.assertTrue(succeeded)
                self.assertFalse(share_open)
                self.assertEqual(operations, Counter(share=1, close_share=1))

    def test_unrecognized_share_result_still_fails(self):
        for locale in ("base", "zh_tw"):
            with self.subTest(locale=locale):
                succeeded, _, _ = self.replay(locale, False, False, upgrade=False, gift=False, share_visible=False)
                self.assertFalse(succeeded)

    def test_share_panel_must_close_before_success(self):
        for locale in ("base", "zh_tw"):
            for extra_steps in (False, True):
                with self.subTest(locale=locale, extra_steps=extra_steps):
                    succeeded, operations, share_open = self.replay(
                        locale, False, False, upgrade=extra_steps, gift=extra_steps,
                        close_blocked=True)
                    self.assertFalse(succeeded)
                    self.assertTrue(share_open)
                    self.assertGreaterEqual(operations["close_share"], 1)

    def test_share_panel_already_closed(self):
        for locale in ("base", "zh_tw"):
            with self.subTest(locale=locale):
                succeeded, operations, share_open = self.replay(
                    locale, False, False, upgrade=False, gift=False, auto_close=True)
                self.assertTrue(succeeded)
                self.assertFalse(share_open)
                self.assertEqual(operations, Counter(share=1))


if __name__ == "__main__":
    unittest.main()
