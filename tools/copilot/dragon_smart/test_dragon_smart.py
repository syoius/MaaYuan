"""Offline vision, policy and dispatcher tests. No device controller is created."""
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
SOURCE = ROOT / "agent/custom/action"
package = ModuleType("dragon_test_package")
package.__path__ = [str(SOURCE)]
sys.modules[package.__name__] = package
from dragon_test_package.dragon_smart_vision import DragonVision, choose_action, read_image
from maa.agent.agent_server import AgentServer
with patch.dict(sys.modules, {"utils": SimpleNamespace(logger=logging.getLogger("test"))}):
    with patch.object(AgentServer, "custom_action", return_value=lambda cls: cls):
        from dragon_test_package.dragon_smart import DragonSmartAction, decode_params, parse_round_text


class Context:
    def __init__(self, image, battle=True):
        self.image = image
        self.battle = battle
        self.calls = []
        self.round_text = "8"
        self.tasker = SimpleNamespace(stopping=False, controller=SimpleNamespace(post_screencap=self.capture))

    def capture(self):
        return SimpleNamespace(wait=lambda: SimpleNamespace(get=lambda: self.image.copy()))

    def get_node_data(self, node):
        return {"name": node}

    def run_recognition(self, node, image):
        return SimpleNamespace(hit=self.battle, best_result=SimpleNamespace(text=self.round_text if self.battle else ""))

    def override_next(self, node, targets):
        self.calls.append((node, targets))
        return True


class TestDragonSmart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = DragonVision(ROOT / "assets/resource/base")
        cls.inside = read_image(HERE / "fixtures/inside.png")
        cls.outside = read_image(HERE / "fixtures/outside.png")

    def test_policy_all_counts(self):
        for qi in range(7):
            self.assertEqual(choose_action("outside", qi, True), "ultimate")
            self.assertEqual(choose_action("outside", qi, False), "normal")
            self.assertIsNone(choose_action("outside", qi, None))
            self.assertEqual(choose_action("inside", qi), "normal" if qi == 1 else "defense")
        self.assertIsNone(choose_action("unknown", 4))
        self.assertIsNone(choose_action("inside", None))
        self.assertEqual(choose_action("outside", None, False), "normal")

    def test_round_ocr_trailing_separator(self):
        for text, expected in (("9", "9"), ("9/", "9"), (" 9 / ", "9"), ("９／", "9"), ("10|", "10"), ("11\\", "11")):
            self.assertEqual(parse_round_text(text), expected)
        for text in ("abc9", "", "0", "9O", "回合9", "1 0", "9//"):
            self.assertIsNone(parse_round_text(text), text)

    def test_round_with_limit(self):
        for text, expected in (("1/10", "1"), ("9/10", "9"), ("10/10", "10"), (" １ ／ １０ ", "1"), ("12 / 20", "12")):
            self.assertEqual(parse_round_text(text), expected)
        for text in ("0/10", "1/0", "11/10", "1/10/20", "1//10", "1|10", "1/1O", "-1/10"):
            self.assertIsNone(parse_round_text(text), text)

    def test_limited_round_dispatch_and_mismatch(self):
        for current in range(1, 11):
            context = Context(self.outside)
            context.round_text = f"{current}/10"
            params = {"node": f"回合{current}行动1", "position": 1, "branches": {"normal": "a", "defense": "d", "ultimate": "u"}}
            with patch("dragon_test_package.dragon_smart.time.sleep"):
                self.assertTrue(DragonSmartAction().run(context, SimpleNamespace(custom_action_param=params)).success)
            self.assertEqual(context.calls, [(params["node"], ["u"])])
        # The denominator is a limit, not the current round.
        context = Context(self.outside)
        context.round_text = "1/10"
        with patch("dragon_test_package.dragon_smart.time.sleep"):
            self.assertFalse(DragonSmartAction().run(context, SimpleNamespace(custom_action_param=params)).success)
        self.assertEqual(context.calls, [])

    def test_logged_round_nine_slash_dispatches_skill(self):
        context = Context(self.outside)
        context.round_text = "9/"
        params = {"node": "回合9行动5", "position": 1, "branches": {"normal": "a", "defense": "d", "ultimate": "u"}}
        with patch("dragon_test_package.dragon_smart.time.sleep"):
            result = DragonSmartAction().run(context, SimpleNamespace(custom_action_param=json.dumps(params)))
        self.assertTrue(result.success)
        self.assertEqual(context.calls, [("回合9行动5", ["u"])])

    def test_wrong_round_is_not_accepted(self):
        context = Context(self.outside)
        context.round_text = "8/"
        params = {"node": "回合9行动5", "position": 1, "branches": {"normal": "a", "defense": "d", "ultimate": "u"}}
        with patch("dragon_test_package.dragon_smart.time.sleep"):
            with self.assertLogs("test", level="ERROR") as captured:
                result = DragonSmartAction().run(context, SimpleNamespace(custom_action_param=json.dumps(params)))
        self.assertFalse(result.success)
        self.assertEqual(context.calls, [])
        self.assertIn("回合不符", " ".join(captured.output))

    def test_missed_ocr_hit_cannot_supply_first_confirmation(self):
        context = Context(self.outside)
        hits = iter((False, True, False, True, False, True))
        context.run_recognition = lambda *args: SimpleNamespace(hit=next(hits), best_result=SimpleNamespace(text="8"))
        self.assertFalse(self.invoke(context).success)
        self.assertEqual(context.calls, [])

    def test_stop_during_recognition_never_dispatches(self):
        context = Context(self.outside)
        calls = []
        def recognition(*args):
            calls.append(1)
            context.tasker.stopping = len(calls) == 2
            return SimpleNamespace(hit=True, best_result=SimpleNamespace(text="8"))
        context.run_recognition = recognition
        self.assertFalse(self.invoke(context).success)
        self.assertEqual(context.calls, [])

    def test_table_inside_always_normal_does_not_require_qi(self):
        context = Context(self.inside)
        observation = {"vein": "inside", "qi": None, "action": None, "skill_ready": None}
        params = {"node": "回合6行动2", "branches": {"normal": "本表普攻", "defense": "本表普攻", "ultimate": "本表技能"}}
        context.round_text = "6"
        with patch.object(DragonVision, "inspect", return_value=observation), patch("dragon_test_package.dragon_smart.time.sleep"):
            self.assertTrue(DragonSmartAction().run(context, SimpleNamespace(custom_action_param=params)).success)
        self.assertEqual(context.calls, [("回合6行动2", ["本表普攻"])])
        observation["vein"] = "unknown"
        context.calls.clear()
        with patch.object(DragonVision, "inspect", return_value=observation), patch("dragon_test_package.dragon_smart.time.sleep"):
            self.assertFalse(DragonSmartAction().run(context, SimpleNamespace(custom_action_param=params)).success)
        self.assertEqual(context.calls, [])

    def test_user_log_describes_mapped_target_and_debug_keeps_evidence(self):
        context = Context(self.inside)  # default policy: defense
        params = {"node": "circle", "branches": {"normal": "本表普攻", "defense": "本表普攻", "ultimate": "本表技能"}}
        with self.assertLogs("test", level="DEBUG") as logs, patch("dragon_test_package.dragon_smart.time.sleep"):
            self.assertTrue(DragonSmartAction().run(context, SimpleNamespace(custom_action_param=params)).success)
        visible = "\n".join(r.getMessage() for r in logs.records if r.levelno >= logging.INFO)
        self.assertIn("转入「本表普攻」", visible)
        for raw in ("observation=", "defense", "selected=", "stable=", "None"):
            self.assertNotIn(raw, visible)
        self.assertTrue(any("observation=" in r.getMessage() and r.levelno == logging.DEBUG for r in logs.records))

    def test_exception_trace_is_debug_only(self):
        with self.assertLogs("test", level="DEBUG") as logs:
            result = DragonSmartAction().run(Context(self.outside), SimpleNamespace(custom_action_param="bad-json"))
        self.assertFalse(result.success)
        self.assertTrue(any(r.exc_info and r.levelno == logging.DEBUG for r in logs.records))
        self.assertFalse(any(r.exc_info for r in logs.records if r.levelno >= logging.INFO))

    def test_invalid_positions_stop_before_screenshot(self):
        for position in (0, 6, True, 1.0, "1"):
            context = Context(self.outside)
            params = {"node": "circle", "position": position, "branches": {"normal": "a", "defense": "d", "ultimate": "u"}}
            with patch.object(context.tasker.controller, "post_screencap") as capture:
                self.assertFalse(DragonSmartAction().run(context, SimpleNamespace(custom_action_param=params)).success)
                capture.assert_not_called()

    def test_dispatch_failure_stops(self):
        context = Context(self.outside)
        with patch.object(context, "override_next", return_value=False):
            self.assertFalse(self.invoke(context).success)

    def test_user_labelled_screenshots(self):
        for image, vein, qi, action in ((self.outside, "outside", 4, "ultimate"), (self.inside, "inside", 6, "defense")):
            result = self.vision.inspect(image)
            self.assertEqual((result["vein"], result["qi"], result["action"]), (vein, qi, action))
            print(vein, "qi=", result["qi"], "background_scores=", result["scores"])
        self.assertIs(self.vision.inspect(self.outside)["skill_ready"], True)
        self.assertIs(self.vision.inspect(self.inside)["skill_ready"], False)

    def test_native_capture_and_separate_followup(self):
        for name in ("inside-adb-20261101-185638.png", "inside-adb-validation.png"):
            image = read_image(HERE / "fixtures" / name)
            self.assertEqual(image.shape[:2], (1280, 720))
            result = self.vision.inspect(image)
            self.assertEqual((result["vein"], result["qi"], result["action"]), ("inside", 6, "defense"))
            self.assertIs(result["skill_ready"], False)

    def test_saved_caopi_outside_frame(self):
        result = self.vision.inspect(read_image(HERE / "fixtures/outside-caopi-native.png"))
        self.assertEqual((result["vein"], result["qi"], result["skill_ready"], result["action"]),
                         ("outside", 1, True, "ultimate"))

    def test_recorded_observation_sequences(self):
        # Replay recognition outputs, not screenshots or OCR. This checks the
        # dispatch contract against real flicker/retry sequences without claiming
        # a fresh vision accuracy measurement.
        cases = json.loads((HERE / "fixtures/observations-20261101.json").read_text(encoding="utf-8"))
        for case in cases:
            with self.subTest(node=case["node"], success=case["success"]):
                context = Context(self.inside)
                rounds = iter(s["round_text"] for s in case["samples"])
                context.run_recognition = lambda *args: SimpleNamespace(hit=True, best_result=SimpleNamespace(text=next(rounds)))
                samples = iter(s["observation"] for s in case["samples"])
                stability = iter(s["stable"] for s in case["samples"][1:])
                params = {"node": case["node"], "branches": {"normal": "普攻", "ultimate": "技能", "defense": "作业复核"}}
                with patch.object(DragonVision, "inspect", side_effect=lambda *args: next(samples)), \
                     patch.object(DragonVision, "stable_background", side_effect=lambda *args: next(stability)), \
                     patch("dragon_test_package.dragon_smart.time.sleep"):
                    result = DragonSmartAction().run(context, SimpleNamespace(custom_action_param=params))
                self.assertEqual(result.success, case["success"])
                self.assertEqual(len(context.calls), int(case["success"]))

    def test_native_full_energy_on_second_slot(self):
        for name in ("inside-adb-20261101-185638.png", "inside-adb-validation.png"):
            image = read_image(HERE / "fixtures" / name)
            ready, scores = self.vision.skill_readiness(image, 2)
            self.assertIs(ready, True, scores)

    def test_native_energy_observations_drive_outside_policy(self):
        image = read_image(HERE / "fixtures/inside-adb-validation.png")
        for position, action in ((1, "normal"), (2, "ultimate")):
            ready, _ = self.vision.skill_readiness(image, position)
            self.assertEqual(choose_action("outside", None, ready), action)

    def test_native_frame_dispatch(self):
        context = Context(read_image(HERE / "fixtures/inside-adb-validation.png"))
        self.assertTrue(self.invoke(context).success)
        self.assertEqual(context.calls, [("circle", ["d"])])

    def test_logged_double_encoded_parameters(self):
        params = {"node": "circle", "position": 1, "branches": {"normal": "a", "defense": "d", "ultimate": "u"}}
        for layers in (0, 1, 2, 3):
            wire = params
            for _ in range(layers):
                wire = json.dumps(wire, ensure_ascii=False)
            self.assertEqual(decode_params(wire), params)
            context = Context(read_image(HERE / "fixtures/inside-adb-validation.png"))
            with patch("dragon_test_package.dragon_smart.time.sleep"):
                result = DragonSmartAction().run(context, SimpleNamespace(custom_action_param=wire))
            self.assertTrue(result.success)
            self.assertEqual(context.calls, [("circle", ["d"])])

    def test_invalid_parameters_stop_before_capture(self):
        for wire in ("[]", "null", "bad-json", json.dumps(json.dumps(json.dumps(json.dumps({}))))):
            context = Context(self.outside)
            with patch.object(context.tasker.controller, "post_screencap") as capture:
                self.assertFalse(DragonSmartAction().run(context, SimpleNamespace(custom_action_param=wire)).success)
                capture.assert_not_called()
            self.assertEqual(context.calls, [])

    def test_boss_side_motion_does_not_block_stability(self):
        image = read_image(HERE / "fixtures/inside-adb-validation.png")
        changed = image.copy()
        changed[280:580, :65] = np.random.default_rng(9).integers(0, 255, (300, 65, 3), dtype=np.uint8)
        self.assertTrue(self.vision.stable_background(image, changed))
        context = Context(image)
        frames = iter((image, changed))
        context.tasker.controller.post_screencap = lambda: SimpleNamespace(wait=lambda: SimpleNamespace(get=lambda: next(frames)))
        self.assertTrue(self.invoke(context).success)
        self.assertEqual(context.calls, [("circle", ["d"])])

    def test_bottom_transition_is_not_stable(self):
        image = read_image(HERE / "fixtures/inside-adb-validation.png")
        changed = image.copy()
        changed[1220:1275] = 255
        self.assertFalse(self.vision.stable_background(image, changed))

    def outside_with_energy(self, count):
        image = cv2.resize(self.outside, (720, 1280))
        # Controlled fixtures built from actual lit/unlit patches. These are
        # synthetic counter states, not additional real-game observations.
        for index, (center, size) in enumerate(((828, 15), (847, 15), (864, 11), (879, 9))):
            if index < 4-count:
                patch = cv2.resize(self.vision.energy_unready, (size, size))
                half = size//2
                image[center-half:center+half+1, 16-half:16+half+1] = patch
        return image

    def test_outside_full_and_insufficient_energy(self):
        for count in range(5):
            result = self.vision.inspect(self.outside_with_energy(count))
            self.assertIs(result["skill_ready"], count == 4, result)
            self.assertEqual(result["action"], "ultimate" if count == 4 else "normal")

    def test_unknown_energy_does_not_cast_or_assume_empty(self):
        image = cv2.resize(self.outside, (720, 1280))
        image[805:920, 4:54] = 0
        result = self.vision.inspect(image)
        self.assertIsNone(result["skill_ready"])
        self.assertIsNone(result["action"])
        context = Context(image)
        self.assertFalse(self.invoke(context).success)
        self.assertEqual(context.calls, [])

    def test_energy_changes_between_frames_require_retry(self):
        context = Context(self.outside)
        frames = iter([self.outside_with_energy(i) for i in (4, 2, 4, 2, 4, 2)])
        context.capture = lambda: SimpleNamespace(wait=lambda: SimpleNamespace(get=lambda: next(frames)))
        context.tasker.controller.post_screencap = context.capture
        self.assertFalse(self.invoke(context).success)
        self.assertEqual(context.calls, [])

    def test_dimmed_full_energy_is_unknown_not_insufficient(self):
        image = cv2.resize(self.outside, (720, 1280))
        image[805:920, 4:35] = (image[805:920, 4:35] * 0.3).astype(np.uint8)
        result = self.vision.inspect(image)
        self.assertIsNone(result["skill_ready"])
        self.assertIsNone(result["action"])

    def test_inside_policy_does_not_require_energy_readiness(self):
        image = self.image_with_count(1)
        image[805:920, 4:54] = 0
        result = self.vision.inspect(image)
        self.assertIsNone(result["skill_ready"])
        self.assertEqual(result["action"], "normal")

    def image_with_count(self, count):
        image = cv2.resize(self.inside, (720, 1280))
        # Controlled counter fixture: textured background, 0..6 exact icons.
        image[990:1210, 5:55] = np.random.default_rng(42).integers(40, 140, (220, 50, 3), dtype=np.uint8)
        for n in range(count):
            y = 1000 + n * 29
            image[y:y+26, 10:36] = self.vision.icon
        return image

    def test_zero_one_and_multiple_icons(self):
        for count in range(7):
            result = self.vision.inspect(self.image_with_count(count))
            self.assertEqual(result["qi"], count)
            self.assertEqual(result["action"], "normal" if count == 1 else "defense")

    def test_black_and_wrong_orientation_rejected(self):
        self.assertIsNone(self.vision.inspect(np.zeros((1280, 720, 3), dtype=np.uint8))["action"])
        with self.assertRaises(ValueError):
            self.vision.inspect(np.zeros((720, 1280, 3), dtype=np.uint8))

    def invoke(self, context):
        params = {"node": "circle", "position": 1, "branches": {"normal": "a", "defense": "d", "ultimate": "u"}}
        with patch("dragon_test_package.dragon_smart.time.sleep"):
            return DragonSmartAction().run(context, SimpleNamespace(custom_action_param=json.dumps(params)))

    def test_dispatch_is_replaced_on_each_call(self):
        context = Context(self.outside)
        self.assertTrue(self.invoke(context).success)
        context.image = self.outside_with_energy(2)
        self.assertTrue(self.invoke(context).success)
        context.image = self.image_with_count(1)
        self.assertTrue(self.invoke(context).success)
        context.image = self.inside
        self.assertTrue(self.invoke(context).success)
        self.assertEqual(context.calls, [("circle", ["u"]), ("circle", ["a"]), ("circle", ["a"]), ("circle", ["d"])])

    def test_unknown_or_nonbattle_never_dispatches(self):
        for context in (Context(np.zeros((1280, 720, 3), np.uint8)), Context(self.inside, battle=False),
                        Context(read_image(HERE / "fixtures/nonbattle-cave-adb.png"), battle=False)):
            self.assertFalse(self.invoke(context).success)
            self.assertEqual(context.calls, [])

    def test_ambiguous_background_never_dispatches(self):
        image = cv2.resize(self.outside, (720, 1280))
        inside = cv2.resize(self.inside, (720, 1280))
        image[1220:1275, :220] = inside[1220:1275, :220]
        self.assertEqual(self.vision.inspect(image)["vein"], "unknown")
        context = Context(image)
        self.assertFalse(self.invoke(context).success)
        self.assertEqual(context.calls, [])

    def test_dispatcher_bundle_loads_in_native_engine(self):
        # A minimal contract fixture keeps this suite independent of a personal
        # client installation, external compiler skill and generated job cache.
        actions = {name: {"recognition": "DirectHit", "action": "DoNothing"} for name in ("普攻", "技能", "下拉")}
        for rnd in (8, 9, 10):
            key = f"回合{rnd}行动1"
            actions[key] = {"recognition": "DirectHit", "action": "Custom", "custom_action": "DragonSmartAction",
                            "custom_action_param": {"node": key, "position": 1,
                              "branches": {"normal": "普攻", "ultimate": "技能", "defense": "下拉"}}}
        doc = {"actions": actions}
        # Isolated hardlinks are read-only inputs; nothing in install is written.
        with tempfile.TemporaryDirectory(prefix="dragon-smart-", dir=HERE) as tmp:
            stage = Path(tmp).resolve()
            self.assertEqual(stage.parent, HERE.resolve())
            self.assertTrue(stage.name.startswith("dragon-smart-"))
            base = stage / "base"
            def ignore(directory, names):
                p = Path(directory)
                return [n for n in names if p.name == "copilot" and p.parent.name == "pipeline" and n.endswith(".json") and n != "copilot_config.json"]
            shutil.copytree(ROOT / "assets/resource/base", base, copy_function=os.link, ignore=ignore)
            (base / "pipeline/copilot/smart-preview.json").write_text(json.dumps(doc["actions"], ensure_ascii=False), encoding="utf-8")
            # AgentServer and the full framework use different DLL entrypoints;
            # validate the bundle in a separate, ordinary framework process.
            code = '''
import json,sys
from pathlib import Path
from maa.resource import Resource
base=Path(sys.argv[1])
doc=json.loads((base/'pipeline/copilot/smart-preview.json').read_text(encoding='utf-8'))
resource=Resource()
resource.post_bundle(base).wait()
assert resource.loaded
for rnd in (8,9,10):
 key='回合%d行动1'%rnd
 node=resource.get_node_data(key)
 assert node['action']['param']['custom_action']=='DragonSmartAction'
 params=doc[key]['custom_action_param']
 assert isinstance(params,dict)
 assert isinstance(node['action']['param']['custom_action_param'],dict)
 for branch in params['branches'].values():
  assert resource.get_node_data(branch) is not None
'''
            result = subprocess.run([sys.executable, "-", str(base)], input=code, text=True,
                                    encoding="utf-8", capture_output=True)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-3000:])


if __name__ == "__main__":
    unittest.main()
