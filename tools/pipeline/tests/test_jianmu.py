"""Replay the Jianmu production pass without connecting a device."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
from maa.controller import CustomController
from maa.tasker import LoggingLevelEnum, Tasker
from maa.resource import Resource


ASSETS = Path(__file__).resolve().parents[3] / "assets"
BASE = ASSETS / "resource/base"
ENTRY = "建木巡检-开始"
TEMPLATE_DIR = BASE / "image/jianmu"


def frame(*parts):
    image = Image.new("RGB", (720, 1280))
    for name, x, y in parts:
        with Image.open(TEMPLATE_DIR / f"{name}.png") as template:
            image.paste(template, (x, y))
    return np.asarray(image)[:, :, ::-1].copy()


class ReplayController(CustomController):
    def __init__(self, frames, targets):
        super().__init__()
        self.frames = frames
        self.targets = targets
        self.clicks = []

    def connect(self):
        return True

    def request_uuid(self):
        return "offline-jianmu"

    def get_features(self):
        return 0

    def screencap(self):
        return self.frames[min(len(self.clicks), len(self.frames) - 1)]

    def click(self, x, y):
        step = len(self.clicks)
        self.clicks.append((x, y))
        if step >= len(self.targets):
            return False
        left, top, right, bottom = self.targets[step]
        return left <= x < right and top <= y < bottom


class JianmuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        root = Path(temp.name)
        shutil.copytree(TEMPLATE_DIR, root / "image/jianmu")
        (root / "pipeline").mkdir()
        pipeline = json.loads((BASE / "pipeline/sp/jianmu.json").read_text("utf-8"))
        for node in pipeline.values():
            node.update(pre_delay=0, post_delay=0, timeout=300, rate_limit=10)
        (root / "pipeline/jianmu.json").write_text(json.dumps(pipeline), encoding="utf-8")
        Tasker.set_log_dir(Path(tempfile.gettempdir()) / "maayuan-jianmu-tests")
        Tasker.set_stdout_level(LoggingLevelEnum.Off)
        cls.resource = Resource()
        if not cls.resource.post_bundle(root).wait().succeeded:
            raise RuntimeError("Unable to load Jianmu resources")
        cls.tree = frame(("tree_title", 50, 154), ("collect", 606, 1016), ("dispatch", 501, 1016))
        cls.map = frame(("outside_title", 35, 147), ("return", 602, 1134))
        cls.reward = frame(("reward_title", 258, 218), ("reward_close", 293, 1199))
        cls.result = frame(("dispatch_title", 259, 218), ("dispatch_ok", 316, 1109))
        cls.collect = (621, 1028, 657, 1062)
        cls.dispatch = (516, 1028, 552, 1060)

    def replay(self, frames, targets, succeeded=True):
        controller = ReplayController(frames, targets)
        self.assertTrue(controller.post_connection().wait().succeeded)
        tasker = Tasker()
        self.assertTrue(tasker.bind(self.resource, controller))
        self.assertEqual(tasker.post_task(ENTRY).wait().succeeded, succeeded)
        self.assertEqual(len(controller.clicks), len(targets), controller.clicks)
        for (x, y), (left, top, right, bottom) in zip(controller.clicks, targets):
            self.assertTrue(left <= x < right and top <= y < bottom, (x, y))

    def test_map_to_tree_with_rewards_and_dispatch_result(self):
        self.replay(
            [self.map, self.tree, self.reward, self.tree, self.result, self.tree],
            [(622, 1150, 654, 1192), self.collect, (293, 1199, 427, 1223),
             self.dispatch, (316, 1109, 406, 1151)],
        )

    def test_no_result_dialogs(self):
        self.replay([self.tree], [self.collect, self.dispatch])

    def test_partial_tree_overlay_and_unrelated_dialog_do_not_click(self):
        partial = frame(("tree_title", 50, 154), ("dispatch", 501, 1016))
        wrong_reward = frame(("reward_title", 258, 218))
        for frames, targets in [([partial], []), ([self.tree, wrong_reward], [self.collect])]:
            with self.subTest(clicks_before_stop=len(targets)):
                self.replay(frames, targets, succeeded=False)

    def test_task_is_opt_in_and_uses_simplified_chinese_resources(self):
        interface = json.loads((ASSETS / "interface.json").read_text("utf-8"))
        task = next(task for task in interface["task"] if task["entry"] == "建木挂机-开始")
        self.assertFalse(task["default_check"])
        self.assertEqual(task["resource"], ["简中（游戏界面）"])


if __name__ == "__main__":
    unittest.main()
