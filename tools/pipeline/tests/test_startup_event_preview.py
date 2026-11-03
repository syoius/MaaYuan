"""Replay animated event previews through startup and daily navigation, without ADB."""

import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
from maa.controller import CustomController
from maa.resource import Resource
from maa.tasker import LoggingLevelEnum, Tasker

ROOT = Path(__file__).resolve().parents[3]
CLOSE = "关闭活动开屏预告"
OLD = "new今日不再提醒popup"
HOME = "已在界面-左半屏"


def color(value, roi):
    return {
        "type": "ColorMatch",
        "param": {
            "roi": roi,
            "lower": [value] * 3,
            "upper": [value] * 3,
            "count": roi[2] * roi[3],
        },
    }


def source(locale):
    nodes = {}
    for layer in ("base", "zh_tw") if locale == "zh_tw" else ("base",):
        for filename in ("start_up.json", "popups.json", "mainmap.json", "daily/meals.json"):
            nodes.update(json.loads((ROOT / f"assets/resource/{layer}/pipeline/{filename}").read_text("utf-8")))
    return nodes


class PreviewController(CustomController):
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario
        self.frames = 0
        self.clicks = []
        self.remaining = 2 if scenario == "consecutive" else 1
        self.icon = np.array(Image.open(ROOT / "assets/resource/base/image/login/event_preview_close.png").convert("RGB"))[:, :, ::-1]

    def connect(self): return True
    def request_uuid(self): return "offline-event-preview"
    def get_features(self): return 0

    def screencap(self):
        self.frames += 1
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        if not self.remaining:
            frame[1180:1188, 200:208] = 140
        elif self.scenario != "animation" or self.frames > 5:
            if self.scenario != "missing_footer":
                frame[1080:1088, 330:338] = 100
            if self.scenario != "missing_close":
                frame[75:122, 606:653] = self.icon
        return frame

    def click(self, x, y):
        self.clicks.append((x, y))
        # The original broad "不再" handler must never consume this preview.
        if 621 <= x < 638 and 90 <= y < 107:
            self.remaining -= 1
            return True
        return False


class EventPreviewTests(unittest.TestCase):
    def test_all_existing_popup_routes_prioritize_preview(self):
        for locale in ("base", "zh_tw"):
            nodes = source(locale)
            for name, node in nodes.items():
                for suffix in (OLD, "sub_弹窗关闭"):
                    for prefix in ("", "[JumpBack]"):
                        candidate = prefix + suffix
                        if candidate in node.get("next", []):
                            with self.subTest(locale=locale, entry=name):
                                self.assertLess(node["next"].index(prefix + CLOSE), node["next"].index(candidate))
            self.assertEqual(nodes[OLD]["next"][0], CLOSE)
            self.assertGreaterEqual(nodes["心纸君"]["timeout"], 20000)

    def replay(self, locale, entry, scenario):
        original = source(locale)
        nodes = {entry: copy.deepcopy(original[entry])}
        nodes[entry]["recognition"] = "DirectHit"
        nodes[entry].pop("on_error", None)
        for candidate in nodes[entry]["next"]:
            nodes[candidate.removeprefix("[JumpBack]")] = {"recognition": color(250, [0, 0, 2, 2])}
        nodes[CLOSE] = copy.deepcopy(original[CLOSE])
        # Only replace OCR (models aren't shipped in git); use the real close template,
        # click offset, next ordering and JumpBack behavior in MaaFramework.
        nodes[CLOSE]["recognition"]["param"]["all_of"][0]["recognition"] = color(100, [330, 1080, 8, 8])
        nodes[OLD] = copy.deepcopy(original[OLD])
        nodes[OLD]["recognition"] = color(100, [330, 1080, 8, 8])
        nodes["new退出今日不再提醒"] = {"recognition": color(250, [0, 0, 2, 2])}
        # Allow this isolated route to complete after the overlay disappears.
        nodes[entry]["next"].append(HOME)
        nodes[HOME] = {"recognition": color(140, [200, 1180, 8, 8])}
        for node in nodes.values():
            node.update(pre_delay=0, post_delay=0, timeout=300, rate_limit=5)
        nodes[entry]["enabled"] = True
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "pipeline").mkdir()
            (path / "pipeline/test.json").write_text(json.dumps(nodes, ensure_ascii=False), encoding="utf-8")
            image_dir = path / "image/login"
            image_dir.mkdir(parents=True)
            (image_dir / "event_preview_close.png").write_bytes((ROOT / "assets/resource/base/image/login/event_preview_close.png").read_bytes())
            Tasker.set_stdout_level(LoggingLevelEnum.Off)
            Tasker.set_log_dir(Path(tempfile.gettempdir()) / "maayuan-event-preview-tests")
            resource = Resource()
            self.assertTrue(resource.post_bundle(path).wait().status.succeeded)
            controller = PreviewController(scenario)
            self.assertTrue(controller.post_connection().wait().status.succeeded)
            tasker = Tasker()
            self.assertTrue(tasker.bind(resource, controller))
            job = tasker.post_task(entry)
            deadline = time.monotonic() + 10
            while not job.done and time.monotonic() < deadline:
                time.sleep(0.01)
            if not job.done:
                tasker.post_stop().wait()
                self.fail("Preview replay did not terminate")
            positive = scenario in ("ready", "consecutive", "animation")
            self.assertEqual(job.status.succeeded, positive)
            self.assertEqual(len(controller.clicks), (2 if scenario == "consecutive" else 1) if positive else 0)

    def test_startup_and_daily_navigation_resume_after_preview(self):
        for locale in ("base", "zh_tw"):
            for entry in ("心纸君", "返回主界面左半屏", "每日进膳启动"):
                for scenario in ("ready", "consecutive", "animation"):
                    with self.subTest(locale=locale, entry=entry, scenario=scenario):
                        self.replay(locale, entry, scenario)

    def test_close_requires_both_footer_and_icon(self):
        # Start with the guarded node directly: neither partial match may click.
        for scenario in ("missing_footer", "missing_close"):
            self.replay("base", "心纸君", scenario)


if __name__ == "__main__":
    unittest.main()
