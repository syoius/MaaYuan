import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
sys.path.insert(0, str(AGENT_ROOT))

from custom.action.experience_book_stock_reporting import (  # noqa: E402
    EXPERIENCE_BOOK_SLOTS,
    ExperienceBookStockReporting,
    _save_and_upload_stock_snapshot,
    build_experience_book_snapshot_results,
    has_confirmed_consumption,
    recognize_experience_book_counts,
)


class _Job:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self

    def get(self):
        return self.value


class _Controller:
    def __init__(self, images):
        self.images = list(images)
        self.clicks = []

    def post_screencap(self):
        return _Job(self.images.pop(0))

    def post_click(self, x, y):
        self.clicks.append((x, y))
        return _Job()


def _context(controller):
    nodes = {
        "在线上传认证": {
            "attach": {
                "mode": "仅保存到本地",
                "base_url": "https://example.test",
            }
        }
    }
    return SimpleNamespace(
        tasker=SimpleNamespace(controller=controller),
        get_node_data=nodes.get,
    )


class ExperienceBookStockReportingTests(unittest.TestCase):
    def test_count_recognition_maps_fixed_slots_without_icon_recognition(self):
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        recognized = [
            (798, 0.91, "798", (0, 0, 1, 1)),
            (110, 0.92, "110", (0, 0, 1, 1)),
            (0, 0.93, "0", (0, 0, 1, 1)),
        ]
        with patch(
            "custom.action.experience_book_stock_reporting.recognize_count",
            side_effect=recognized,
        ) as recognize:
            counts = recognize_experience_book_counts(image)

        self.assertEqual(
            counts,
            {
                "bingshucanjuan": 798,
                "bingshuquanjuan": 110,
                "liutaobingshu": 0,
            },
        )
        self.assertEqual(recognize.call_count, 3)
        centers = [call.args[2] for call in recognize.call_args_list]
        self.assertEqual(centers, [slot.center for slot in EXPERIENCE_BOOK_SLOTS])

    def test_consumption_requires_a_decrease_and_no_increase(self):
        before = {
            "bingshucanjuan": 800,
            "bingshuquanjuan": 110,
            "liutaobingshu": 0,
        }
        self.assertTrue(
            has_confirmed_consumption(before, {**before, "bingshucanjuan": 798})
        )
        self.assertFalse(has_confirmed_consumption(before, dict(before)))
        self.assertFalse(
            has_confirmed_consumption(
                before,
                {**before, "bingshucanjuan": 798, "bingshuquanjuan": 111},
            )
        )

    def test_snapshot_keeps_all_three_books_including_zero(self):
        results = build_experience_book_snapshot_results(
            {
                "bingshucanjuan": 798,
                "bingshuquanjuan": 110,
                "liutaobingshu": 0,
            }
        )
        self.assertEqual(
            [(result["item_id"], result["count"]) for result in results],
            [
                ("bingshucanjuan", 798),
                ("bingshuquanjuan", 110),
                ("liutaobingshu", 0),
            ],
        )

    def test_local_report_is_a_listed_stock_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "StockReport-test.txt"
            context = _context(_Controller([]))

            actual_path = _save_and_upload_stock_snapshot(
                context,
                {"inventory_report_path": str(report_path)},
                {
                    "bingshucanjuan": 798,
                    "bingshuquanjuan": 110,
                    "liutaobingshu": 0,
                },
            )

            self.assertEqual(actual_path, report_path.resolve())
            report = report_path.read_text(encoding="utf-8-sig")
            self.assertIn("渠道：密探升级", report)
            self.assertIn("类型：库存快照", report)
            self.assertIn("兵书残卷 × 798", report)
            self.assertIn("兵书全卷 × 110", report)
            self.assertIn("六韬兵书 × 0", report)
            self.assertIn('"s":"listed"', report)

    @patch("custom.action.experience_book_stock_reporting.time.sleep")
    @patch("custom.action.experience_book_stock_reporting._save_and_upload_stock_snapshot")
    @patch("custom.action.experience_book_stock_reporting.recognize_experience_book_counts")
    def test_action_clicks_once_and_reports_only_after_confirmed_consumption(
        self, recognize, save, sleep
    ):
        recognize.side_effect = [
            {
                "bingshucanjuan": 800,
                "bingshuquanjuan": 110,
                "liutaobingshu": 0,
            },
            {
                "bingshucanjuan": 798,
                "bingshuquanjuan": 110,
                "liutaobingshu": 0,
            },
        ]
        save.return_value = Path("StockReport.txt")
        controller = _Controller([object(), object()])
        argv = SimpleNamespace(
            custom_action_param={},
            box=SimpleNamespace(x=473, y=1137, w=171, h=77),
        )

        result = ExperienceBookStockReporting().run(_context(controller), argv)

        self.assertTrue(result.success)
        self.assertEqual(controller.clicks, [(558, 1175)])
        sleep.assert_called_once_with(1.8)
        save.assert_called_once()
        self.assertEqual(save.call_args.args[2]["bingshucanjuan"], 798)

    @patch("custom.action.experience_book_stock_reporting.time.sleep")
    @patch("custom.action.experience_book_stock_reporting._save_and_upload_stock_snapshot")
    @patch("custom.action.experience_book_stock_reporting.recognize_experience_book_counts")
    def test_action_does_not_report_when_counts_are_unchanged(
        self, recognize, save, sleep
    ):
        counts = {
            "bingshucanjuan": 798,
            "bingshuquanjuan": 110,
            "liutaobingshu": 0,
        }
        recognize.side_effect = [counts, dict(counts)]
        controller = _Controller([object(), object()])
        argv = SimpleNamespace(
            custom_action_param={},
            box=SimpleNamespace(x=473, y=1137, w=171, h=77),
        )

        result = ExperienceBookStockReporting().run(_context(controller), argv)

        self.assertTrue(result.success)
        self.assertEqual(len(controller.clicks), 1)
        save.assert_not_called()

    @patch(
        "custom.action.experience_book_stock_reporting._save_and_upload_stock_snapshot",
        side_effect=RuntimeError("upload failed"),
    )
    @patch("custom.action.experience_book_stock_reporting.logger.exception")
    @patch("custom.action.experience_book_stock_reporting.time.sleep")
    @patch("custom.action.experience_book_stock_reporting.recognize_experience_book_counts")
    def test_reporting_failure_never_retries_the_level_up_click(
        self, recognize, sleep, exception, save
    ):
        recognize.side_effect = [
            {
                "bingshucanjuan": 800,
                "bingshuquanjuan": 110,
                "liutaobingshu": 0,
            },
            {
                "bingshucanjuan": 798,
                "bingshuquanjuan": 110,
                "liutaobingshu": 0,
            },
        ]
        controller = _Controller([object(), object()])
        argv = SimpleNamespace(
            custom_action_param={},
            box=SimpleNamespace(x=473, y=1137, w=171, h=77),
        )

        result = ExperienceBookStockReporting().run(_context(controller), argv)

        self.assertTrue(result.success)
        self.assertEqual(len(controller.clicks), 1)

    def test_interface_wires_reporting_into_daily_wrap_up(self):
        interface = json.loads(
            (REPO_ROOT / "assets" / "interface.json").read_text(encoding="utf-8")
        )
        task = next(task for task in interface["task"] if task["name"] == "🔶 日常收尾")
        self.assertIn("记录奖励内容及数量", task["option"])

        recording = interface["option"]["记录奖励内容及数量"]
        yes = next(case for case in recording["cases"] if case["name"] == "Yes")
        action = yes["pipeline_override"]["密探升级"]["action"]
        self.assertEqual(action["param"]["custom_action"], "ExperienceBookStockReporting")

        for resource_name in ("base", "zh_tw"):
            with self.subTest(resource=resource_name):
                pipeline = json.loads(
                    (
                        REPO_ROOT
                        / "assets"
                        / "resource"
                        / resource_name
                        / "pipeline"
                        / "daily"
                        / "levelingup.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(pipeline["密探升级"]["action"]["type"], "Click")


if __name__ == "__main__":
    unittest.main()
