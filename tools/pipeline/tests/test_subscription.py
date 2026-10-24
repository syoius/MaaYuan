"""Replay monthly-card routing without a device, payment or real reward claim."""

import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from maa.controller import CustomController
from maa.resource import Resource
from maa.tasker import LoggingLevelEnum, Tasker


ROOT = Path(__file__).resolve().parents[3]


def marker(value, x):
    return {"type": "ColorMatch", "param": {
        "roi": [x, 0, 8, 8], "lower": [value] * 3,
        "upper": [value] * 3, "count": 64,
    }}


class MonthlyCardController(CustomController):
    def __init__(self, normal, fuchuan):
        self.cards = {"normal": normal, "fuchuan": fuchuan}
        self.tab = "fuchuan" if fuchuan != "absent" else "normal"
        self.reward = False
        self.claims = []
        super().__init__()

    def connect(self):
        return True

    def request_uuid(self):
        return "subscription-replay"

    def get_features(self):
        return 0

    def screencap(self):
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        def put(x, value, y=0):
            frame[y:y + 8, x:x + 8] = value
        if self.reward:
            put(340, 100, 1090)
            return frame
        put(0, 30)  # Mall header, visible on all welfare tabs.
        put(30, 50)  # Ordinary monthly-card tab.
        if self.cards["fuchuan"] != "absent":
            put(60, 60)
        if "unclaimed" in self.cards.values():
            put(90, 70)  # Welfare red dot may be caused by either card.
        put(100, 80 if self.tab == "normal" else 90)
        status = self.cards[self.tab]
        if status == "unclaimed":
            put(110, 75)
            frame[984:1020, 275:445] = [140, 217, 243]  # BGR, golden button.
        elif status == "claimed":
            put(120, 85)
            frame[984:1020, 275:445] = [217, 217, 217]
        return frame

    def click(self, x, y):
        if self.reward:
            if y > 1060 or (x > 600 and y < 100):
                self.reward = False
                return True
        elif y < 210:  # Open welfare; the game remembers the selected tab.
            return True
        elif y < 320:
            self.tab = "normal" if x < 120 else "fuchuan"
            return True
        elif 970 <= y <= 1035:
            if self.cards[self.tab] != "unclaimed":
                return False
            self.claims.append(self.tab)
            self.cards[self.tab] = "claimed"
            self.reward = True
            return True
        return False  # In particular, renewal/purchase buttons are never valid.


class SubscriptionTests(unittest.TestCase):
    def replay(self, locale, enabled, selection, normal, fuchuan, expected):
        source = ROOT / "assets/resource" / locale / "pipeline/daily/subscription.json"
        nodes = json.loads(source.read_text(encoding="utf-8"))
        # Keep routing, click targets, combined recognition and the golden-button
        # ColorMatch intact; only replace OCR/template recognition with markers.
        recognition = {
            "检查是否都领了": marker(30, 0),
            "领月卡商店20体": marker(150, 20),
            "领月卡60体": marker(70, 90),
            "月卡福利-切换普通月卡": marker(50, 30),
            "符传月卡-打开福利": marker(30, 0),
            "符传月卡-进入标签": marker(60, 60),
            "关闭60体获得界面": marker(100, 340),
            "符传月卡-关闭奖励": marker(100, 340),
        }
        for name in ("关闭60体获得界面", "符传月卡-关闭奖励"):
            recognition[name]["param"]["roi"] = [340, 1090, 8, 8]
        for name, reco in recognition.items():
            nodes[name]["recognition"] = reco
        for name, title in (("点击领取月卡60体", 80), ("月卡福利-普通月卡已领取", 80),
                            ("符传月卡-点击领取", 90), ("符传月卡-已领取", 90)):
            conditions = nodes[name]["recognition"]["param"]["all_of"]
            conditions[0]["recognition"] = marker(title, 100)
            conditions[1]["recognition"] = marker(85, 120) if name.endswith("已领取") else marker(75, 110)
        # A recognition box for the ordinary claim must point at its actual button.
        nodes["点击领取月卡60体"]["action"]["param"]["target"] = [330, 992, 55, 16]
        nodes["进入界面-商城"] = {"recognition": {"type": "DirectHit"}}
        nodes["stop"] = {"recognition": {"type": "DirectHit"}}
        for node in nodes.values():
            node.update(pre_delay=0, post_delay=0, rate_limit=0, timeout=150,
                        post_wait_freezes=0)
        interface = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
        for option, choice in (("领哪种", selection), ("领取符传月卡", "Yes" if enabled else "No")):
            case = next(c for c in interface["option"][option]["cases"] if c["name"] == choice)
            for name, override in case["pipeline_override"].items():
                nodes[name].update(override)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "pipeline").mkdir()
            (folder / "pipeline/test.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
            Tasker.set_log_dir(Path(tempfile.gettempdir()) / "maayuan-subscription-tests")
            Tasker.set_stdout_level(LoggingLevelEnum.Off)
            resource = Resource()
            self.assertTrue(resource.post_bundle(folder).wait().status.succeeded)
            controller = MonthlyCardController(normal, fuchuan)
            self.assertTrue(controller.post_connection().wait().status.succeeded)
            tasker = Tasker()
            self.assertTrue(tasker.bind(resource, controller))
            job = tasker.post_task("领体力启动")
            deadline = time.monotonic() + 5
            while not job.status.done and time.monotonic() < deadline:
                time.sleep(0.01)
            if not job.status.done:
                tasker.post_stop().wait()
                self.fail("Monthly-card routing did not terminate")
            result = job.get()
            self.assertEqual(controller.claims, expected)
            self.assertFalse(controller.reward)
            self.assertNotEqual(result.nodes[-1].name, "符传月卡-领取结果未确认")

    def test_card_combinations(self):
        cases = [
            (False, "都领", "claimed", "unclaimed", []),
            (True, "都领", "unclaimed", "unclaimed", ["normal", "fuchuan"]),
            (True, "都领", "claimed", "unclaimed", ["fuchuan"]),
            (True, "只领商店20体", "unclaimed", "unclaimed", ["fuchuan"]),
            (True, "都领", "claimed", "claimed", []),
            (True, "都领", "claimed", "not_owned", []),
            (True, "都领", "claimed", "absent", []),
            (False, "都领", "unclaimed", "unclaimed", ["normal"]),
        ]
        for locale in ("base", "zh_tw"):
            for case in cases:
                with self.subTest(locale=locale, case=case):
                    self.replay(locale, *case)


if __name__ == "__main__":
    unittest.main()
