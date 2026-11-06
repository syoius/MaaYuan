"""Replay exit outcomes through the real MaaFramework scheduler, without a device.

OCR is replaced with colored frames so these tests need no game screenshots or
OCR models. Node transitions, JumpBack handling and click actions use the actual
pipeline. Run with Python containing maafw and numpy.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from maa.controller import CustomController
from maa.resource import Resource
from maa.tasker import Tasker, LoggingLevelEnum


ROOT = Path(__file__).resolve().parents[3]
COLORS = {"scene": 40, "popup": 80, "map": 120, "dialogue": 160, "narration": 180}


class ReplayController(CustomController):
    def __init__(self, states):
        self.states = list(states)
        self.clicks = []
        super().__init__()

    def connect(self):
        return True

    def request_uuid(self):
        return "yecheng-exit-replay"

    def get_features(self):
        return 0

    def screencap(self):
        return np.full((1280, 720, 3), COLORS[self.states[0]], dtype=np.uint8)

    def click(self, x, y):
        self.clicks.append(self.states[0])
        if len(self.states) > 1:
            self.states.pop(0)
        return True


class YechengExitRecoveryTests(unittest.TestCase):
    def replay(self, locale, event, states):
        folder = ROOT / "assets/resource" / locale / "pipeline/sp/yecheng"
        source = {}
        for filename in ("yecheng.json", "utils.json"):
            source.update(json.loads((folder / filename).read_text(encoding="utf-8")))
        exit_node = f"邺城事件-{event}退出"
        confirm_node = exit_node + "确认"
        goal = "邺城事件-下一步"
        names = {exit_node, "邺城事件-确定", goal}
        if confirm_node in source:
            names.add(confirm_node)
        for name in (exit_node, confirm_node):
            if name not in source:
                continue
            names.update(item.removeprefix("[JumpBack]") for item in source[name].get("next", []))
        nodes = {name: copy.deepcopy(source.get(name, {})) for name in names}
        recognition_states = {
            exit_node: "scene", confirm_node: "popup", goal: "map",
            "邺城事件-确定": "popup", "邺城事件-检测对话": "dialogue",
            "邺城事件-跳过旁白": "narration",
        }
        for name, node in nodes.items():
            value = COLORS[recognition_states[name]] if name in recognition_states else 200
            node["recognition"] = {
                "type": "ColorMatch",
                "param": {"roi": [0, 0, 8, 8], "lower": [value] * 3,
                          "upper": [value] * 3, "count": 64},
            }
            node.update(pre_delay=0, post_delay=0, rate_limit=0, timeout=200)
            node["on_error"] = ["replay_failed"]
            if name not in {exit_node, confirm_node, "邺城事件-确定"}:
                node["next"] = []
        nodes[goal]["next"] = []
        nodes["replay_failed"] = {"recognition": {"type": "DirectHit"}}

        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp / "pipeline").mkdir()
            (temp / "pipeline/replay.json").write_text(
                json.dumps(nodes, ensure_ascii=False), encoding="utf-8"
            )
            Tasker.set_log_dir(Path(tempfile.gettempdir()) / "maayuan-exit-replay-logs")
            Tasker.set_stdout_level(LoggingLevelEnum.Off)
            resource = Resource()
            self.assertTrue(resource.post_bundle(temp).wait().status.succeeded)
            controller = ReplayController(states)
            self.assertTrue(controller.post_connection().wait().status.succeeded)
            tasker = Tasker()
            self.assertTrue(tasker.bind(resource, controller))
            result = tasker.post_task(exit_node).wait().get()
            self.assertEqual(result.nodes[-1].name, goal)
            self.assertEqual(controller.states, ["map"])
            self.assertEqual(controller.clicks, states[:-1])

    def test_exit_outcomes(self):
        for locale in ("base", "zh_tw"):
            for event in ("犒军", "点将台"):
                for states in (
                    ["scene", "popup", "map"],
                    ["scene", "map"],
                    ["scene", "popup", "scene", "popup", "map"],
                    ["scene", "popup", "popup", "map"],
                    ["scene", "popup", "dialogue", "map"],
                    ["scene", "narration", "popup", "map"],
                ):
                    with self.subTest(locale=locale, event=event, states=states):
                        self.replay(locale, event, states)


if __name__ == "__main__":
    unittest.main()
