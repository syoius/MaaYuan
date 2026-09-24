import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

_MODULE_PATH = AGENT_ROOT / "custom" / "action" / "star_backpack_capture_orchestration.py"
_SPEC = importlib.util.spec_from_file_location(
    "star_backpack_capture_orchestration_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

StarBackpackCaptureOrchestration = _MODULE.StarBackpackCaptureOrchestration
assemble_capture_batch = _MODULE.assemble_capture_batch
parse_params = _MODULE.parse_star_backpack_capture_orchestration_params


def _feedback():
    return {
        "max_micro_attempts": 2,
        "local_search_radius_px": 20,
        "diagnostic_safe_shift_px": [300, 600],
        "diagnostic_expected_shift_px": [250, 700],
        "diagnostic_physical_shift_px": [30, 700],
        "diagnostic_target_shift_px": [610, 630],
        "diagnostic_normal_safe_shift_px": [560, 650],
        "diagnostic_row_pitch_px": 166.5,
        "diagnostic_micro_trigger_shift_px": 300,
        "diagnostic_min_confidence": 0.2,
        "diagnostic_min_local_overlap_score": 0.6,
        "diagnostic_no_move_similarity": 0.97,
        "diagnostic_no_move_shift_px": 20,
    }


def _params(directory):
    return {
        "mode": "continuous_capture",
        "game_version": "如鸢",
        "debug_dir": directory,
        "support_tab_task": "星石背包-点击辅星",
        "experience_tab_task": "星石背包-点击经验星曜",
        "tab_settle_ms": 0,
        "compare_roi": [0, 0, 8, 8],
        "swipe": {"start": [1, 2], "end": [3, 4], "duration_ms": 1},
        "settle_ms": 0,
        "max_transitions": 2,
        "feedback": _feedback(),
    }


class _Controller:
    def __init__(self, events):
        self.events = events

    def post_screencap(self):
        self.events.append("screencap")
        return self

    def wait(self):
        return self

    def get(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _Context:
    def __init__(self):
        self.events = []
        self.tasker = SimpleNamespace(controller=_Controller(self.events))

    def run_task(self, name):
        self.events.append(f"task:{name}")
        return SimpleNamespace(status=SimpleNamespace(succeeded=True))


class _FakeProbe:
    def _capture(self, context):
        return context.tasker.controller.post_screencap().wait().get()

    def _run_continuous_capture(self, _context, _params, run_dir, _initial):
        section = run_dir.name
        return {
            "success": True,
            "stop_reason": "bottom_no_move",
            "retained_images": ["capture-00.png", "capture-01.png"],
            "retained_image_count": 2,
            "adjacent_relations": (
                [{"previous_image": "capture-00.png", "current_image": "capture-01.png", "relation": "overlap"}]
                if section == "main"
                else []
            ),
        }


class StarBackpackCaptureOrchestrationTests(unittest.TestCase):
    def test_happy_path_switches_each_tab_once_and_captures_experience_once(self):
        with tempfile.TemporaryDirectory() as directory:
            context = _Context()
            with mock.patch.object(_MODULE, "StarBackpackCaptureProbe", _FakeProbe), mock.patch.object(_MODULE, "upload_full_capture_batch", return_value=None) as upload:
                result = StarBackpackCaptureOrchestration().run(
                    context, SimpleNamespace(custom_action_param=_params(directory))
                )
            run_dir = next(Path(directory).iterdir())
            batch = json.loads((run_dir / "capture-batch.json").read_text(encoding="utf-8"))

        self.assertTrue(getattr(result, "success", False))
        upload.assert_called_once_with(context, run_dir)
        self.assertEqual(
            context.events,
            [
                "screencap",
                "task:星石背包-点击辅星",
                "screencap",
                "task:星石背包-点击经验星曜",
                "screencap",
            ],
        )
        self.assertEqual(list(batch["sections"]), ["main", "support", "experience"])
        self.assertEqual(batch["sections"]["experience"]["stopReason"], "single_capture")
        self.assertEqual(len(batch["sections"]["experience"]["images"]), 1)
        self.assertEqual(batch["sections"]["experience"]["adjacentRelations"], [])
        self.assertEqual(
            [
                image["sourceOrder"]
                for section in batch["sections"].values()
                for image in section["images"]
            ],
            [1, 2, 3, 4, 5],
        )

    def test_batch_ids_and_relations_remain_section_local(self):
        scroll_session = {
            "success": True,
            "retained_images": ["capture-00.png", "capture-01.png", "capture-02.png"],
            "adjacent_relations": [
                {"previous_image": "capture-00.png", "current_image": "capture-01.png", "relation": "overlap"}
            ],
        }
        support_session = {
            "success": True,
            "retained_images": ["capture-00.png", "capture-01.png"],
            "adjacent_relations": [
                {"previous_image": "capture-00.png", "current_image": "capture-01.png", "relation": "overlap"}
            ],
        }
        experience_session = {
            "success": True,
            "retained_images": ["capture-00.png"],
            "adjacent_relations": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            batch = assemble_capture_batch(
                Path(directory), "如鸢", scroll_session, support_session, experience_session
            )

        orders = [
            image["sourceOrder"]
            for section in batch["sections"].values()
            for image in section["images"]
        ]
        self.assertEqual(orders, [1, 2, 3, 4, 5, 6])
        main_ids = {image["sourceImageId"] for image in batch["sections"]["main"]["images"]}
        support_ids = {image["sourceImageId"] for image in batch["sections"]["support"]["images"]}
        self.assertTrue(all(set(relation.values()) - {"overlap"} <= main_ids for relation in batch["sections"]["main"]["adjacentRelations"]))
        self.assertTrue(all(set(relation.values()) - {"overlap"} <= support_ids for relation in batch["sections"]["support"]["adjacentRelations"]))
        self.assertEqual(batch["sections"]["experience"]["adjacentRelations"], [])

    def test_pipeline_uses_click_nodes_without_ocr_and_exposes_full_batch_transport(self):
        pipeline_path = (
            AGENT_ROOT.parent
            / "assets"
            / "resource"
            / "base"
            / "pipeline"
            / "star_backpack_capture_orchestration.json"
        )
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        params = pipeline["星石背包-正式三段采集"]["action"]["param"][
            "custom_action_param"
        ]
        interface = json.loads(
            (AGENT_ROOT.parent / "assets" / "interface.json").read_text(encoding="utf-8")
        )
        self.assertFalse(
            any(task["name"].startswith("开发调试｜星石") for task in interface["task"])
        )
        toolbox_mode = interface["option"]["百宝箱-模式"]
        star_capture_case = next(
            case for case in toolbox_mode["cases"] if case["name"] == "星石背包自动采集"
        )
        self.assertEqual(star_capture_case["option"], ["同步星石至YuanHub"])
        self.assertEqual(
            star_capture_case["pipeline_override"]["百宝箱启动"]["next"],
            ["星石背包-正式三段采集"],
        )
        self.assertEqual(params["support_tab_task"], "星石背包-点击辅星")
        self.assertEqual(params["experience_tab_task"], "星石背包-点击经验星曜")
        for task_name in (params["support_tab_task"], params["experience_tab_task"]):
            node = pipeline[task_name]
            self.assertEqual(node["recognition"]["type"], "DirectHit")
            self.assertEqual(node["action"]["type"], "Click")
        self.assertEqual(
            pipeline["星石背包-点击辅星"]["recognition"]["param"]["roi"],
            [320, 215, 80, 40],
        )
        self.assertEqual(
            pipeline["星石背包-点击经验星曜"]["recognition"]["param"]["roi"],
            [500, 215, 120, 40],
        )


if __name__ == "__main__":
    unittest.main()
