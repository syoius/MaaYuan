import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np
from maa.controller import CustomController
from maa.resource import Resource
from maa.tasker import LoggingLevelEnum, Tasker


ROOT = Path(__file__).resolve().parents[3]
RESOURCES = ROOT / "assets/resource"
REFRESH = "启动超时-刷新游戏"
STOP = "启动超时-停止启动"
ENTRY = "如鸢官服"
SCREENSHOT = os.environ.get("MAAYUAN_STARTUP_SCREENSHOT")


def load_pipeline(locale, name):
    path = RESOURCES / locale / "pipeline" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def apply_ocr_replacements(text, params):
    for original, replacement in params.get("replace", []):
        text = text.replace(original, replacement)
    return text


class StartupTimeoutTests(unittest.TestCase):
    def test_checks_timeout_before_other_startup_actions(self):
        for locale in ("base", "zh_tw"):
            nodes = load_pipeline(locale, "start_up")
            nodes.update(load_pipeline(locale, "switch_account"))
            for entry in nodes["启动游戏版本"]["next"]:
                with self.subTest(locale=locale, entry=entry):
                    self.assertEqual(nodes[entry]["next"][:2], [f"[JumpBack]{REFRESH}", STOP])

    def test_recognizes_simplified_and_traditional_timeout_text(self):
        for locale in ("base", "zh_tw"):
            nodes = load_pipeline(locale, "start_up")
            title, button = nodes[REFRESH]["recognition"]["param"]["all_of"]
            self.assertEqual(title["recognition"], nodes[STOP]["recognition"])
            title_params = title["recognition"]["param"]
            button_params = button["recognition"]["param"]
            for message, label in (
                ("游戏启动超时，请检查网络设置", "刷新游戏"),
                ("遊戲啟動逾時，請檢查網路設定", "刷新遊戲"),
                ("遊戲啓動超時，請檢查網絡設置", "刷新遊戲"),
            ):
                with self.subTest(locale=locale, message=message):
                    self.assertRegex(apply_ocr_replacements(message, title_params), title_params["expected"])
                    self.assertRegex(apply_ocr_replacements(label, button_params), button_params["expected"])
                    self.assertNotRegex("继续等待", button_params["expected"])


class ReplayController(CustomController):
    def __init__(self, screenshot, recovered=None):
        super().__init__()
        self.screenshot = screenshot
        self.recovered = recovered
        self.clicks = []
        self.launches = 0

    def connect(self):
        return True

    def request_uuid(self):
        return "startup-timeout-replay"

    def get_features(self):
        return 0

    def screencap(self):
        return self.screenshot

    def start_app(self, intent):
        self.launches += 1
        return False

    def click(self, x, y):
        self.clicks.append((x, y))
        if self.recovered is not None:
            self.screenshot = self.recovered
        return True


@unittest.skipUnless(SCREENSHOT, "Set MAAYUAN_STARTUP_SCREENSHOT to replay the timeout dialog")
class StartupTimeoutReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pixels = np.fromfile(SCREENSHOT, dtype=np.uint8)
        screenshot = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
        cls.screenshot = cv2.resize(screenshot, (720, 1280), interpolation=cv2.INTER_AREA)
        Tasker.set_log_dir(Path(tempfile.gettempdir()) / "maayuan-startup-timeout-tests")
        Tasker.set_stdout_level(LoggingLevelEnum.Off)
        cls.resource = Resource()
        bundle = Path(os.environ.get("MAAYUAN_RESOURCES", RESOURCES)) / "base"
        if not (bundle / "model/ocr/rec.onnx").is_file():
            raise unittest.SkipTest("Set MAAYUAN_RESOURCES to the installed resource directory with OCR models")
        assert cls.resource.post_bundle(bundle).wait().status.succeeded

    def run_startup(self, screenshot, recovered=None):
        nodes = load_pipeline("base", "start_up")
        nodes[ENTRY]["enabled"] = True
        nodes["sub_启动如鸢官"].update(enabled=True, pre_delay=0, post_delay=0)
        for name in (REFRESH, STOP):
            nodes[name].update(pre_delay=0, post_delay=0)
        nodes["据点"] = {
            "recognition": "ColorMatch",
            "roi": [0, 0, 8, 8],
            "lower": [255, 255, 255],
            "upper": [255, 255, 255],
            "count": 64,
            "action": "DoNothing",
        }
        controller = ReplayController(screenshot, recovered)
        self.assertTrue(controller.post_connection().wait().status.succeeded)
        tasker = Tasker()
        self.assertTrue(tasker.bind(self.resource, controller))
        job = tasker.post_task(ENTRY, nodes)
        deadline = time.monotonic() + 15
        while not job.status.done and time.monotonic() < deadline:
            time.sleep(0.02)
        if not job.status.done:
            tasker.post_stop().wait()
            self.fail("Startup replay did not finish")
        self.assertEqual(controller.launches, 0)
        for x, y in controller.clicks:
            self.assertTrue(385 <= x <= 615 and 744 <= y <= 800, (x, y))
        return job.get(), controller.clicks

    def test_stops_after_two_refreshes_when_dialog_persists(self):
        result, clicks = self.run_startup(self.screenshot)
        self.assertEqual([node.name for node in result.nodes], [ENTRY, REFRESH, REFRESH, STOP])
        self.assertEqual(len(clicks), 2)

    def test_resumes_startup_when_refresh_clears_dialog(self):
        recovered = np.full_like(self.screenshot, 255)
        result, clicks = self.run_startup(self.screenshot, recovered)
        self.assertEqual([node.name for node in result.nodes], [ENTRY, REFRESH, "据点"])
        self.assertTrue(result.status.succeeded)
        self.assertEqual(len(clicks), 1)

    def test_does_not_click_when_refresh_button_is_absent(self):
        screenshot = self.screenshot.copy()
        screenshot[738:821, 375:630] = 255
        result, clicks = self.run_startup(screenshot)
        self.assertEqual([node.name for node in result.nodes], [ENTRY, STOP])
        self.assertEqual(clicks, [])

    def test_does_not_click_refresh_without_the_timeout_message(self):
        screenshot = self.screenshot.copy()
        screenshot[550:640, 145:575] = 255
        screenshot[:8, :8] = 255
        result, clicks = self.run_startup(screenshot)
        self.assertEqual([node.name for node in result.nodes], [ENTRY, "据点"])
        self.assertTrue(result.status.succeeded)
        self.assertEqual(clicks, [])


if __name__ == "__main__":
    unittest.main()
