import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from custom.action.paged_item_recognition import (  # noqa: E402
    SP_OPERATOR_IDS,
    TAB1_ITEM_IDS,
    TAB3_1_ITEM_IDS,
    SnapshotList,
    PageScan,
    _apply_snapshot_list,
    _filter_results,
    _has_snapshot_target_boundary,
    _load_tab3_2_ids,
    _parse_entity_type_filter,
    _parse_snapshot_list,
    _parse_swipe_rows,
    _snapshot_record_options,
    automatic_swipe,
)


class PagedItemRecognitionFilterTests(unittest.TestCase):
    def test_filter_is_disabled_by_default(self):
        self.assertIsNone(_parse_entity_type_filter(None))
        self.assertIsNone(_parse_entity_type_filter(""))
        results = [{"entity_type": "agent"}, {"entity_type": "item"}]
        self.assertIs(_filter_results(results, None), results)

    def test_filter_keeps_only_requested_entity_type(self):
        self.assertEqual(_parse_entity_type_filter(" Agent "), "agent")
        results = [
            {"entity_type": "agent", "item_name": "祢衡"},
            {"entity_type": "item", "item_name": "白金币"},
        ]
        self.assertEqual(
            _filter_results(results, "agent"),
            [{"entity_type": "agent", "item_name": "祢衡"}],
        )

    def test_filter_rejects_unknown_values(self):
        for value in ("mixed", [], 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "agent 或 item"):
                    _parse_entity_type_filter(value)


class PagedItemRecognitionSnapshotListTests(unittest.TestCase):
    def test_static_item_lists_are_disjoint_and_exclude_baijinbi(self):
        self.assertEqual(
            set(TAB1_ITEM_IDS), {"jizhi", "mazi", "sherou", "zhuyu"}
        )
        self.assertEqual(len(TAB3_1_ITEM_IDS), 53)
        self.assertFalse(set(TAB1_ITEM_IDS).intersection(TAB3_1_ITEM_IDS))
        self.assertNotIn("baijinbi", TAB3_1_ITEM_IDS)

    def test_tab3_2_follows_operator_catalog_and_excludes_sp(self):
        operators = [
            {"id": "char_001_normal"},
            {"id": "char_084_chendengsp"},
            {"id": "char_085_shizimiaosp"},
            {"id": "char_126_future"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operators.json"
            path.write_text(
                json.dumps({"OPERATORS": operators}), encoding="utf-8"
            )
            self.assertEqual(
                _load_tab3_2_ids(path),
                ("char_001_normal", "char_126_future"),
            )
        self.assertEqual(
            SP_OPERATOR_IDS,
            {"char_084_chendengsp", "char_085_shizimiaosp"},
        )

    def test_snapshot_list_implies_listed_stock_snapshot(self):
        snapshot_list = _parse_snapshot_list(" TAB1 ")
        self.assertIsNotNone(snapshot_list)
        self.assertEqual(snapshot_list.name, "tab1")
        self.assertEqual(
            _snapshot_record_options({}, snapshot_list),
            ("stock_snapshot", "listed"),
        )
        with self.assertRaisesRegex(ValueError, "stock_snapshot"):
            _snapshot_record_options(
                {"record_type": "reward_delta"}, snapshot_list
            )
        with self.assertRaisesRegex(ValueError, "listed"):
            _snapshot_record_options(
                {"snapshot_scope": "full"}, snapshot_list
            )

    def test_apply_snapshot_list_fills_zero_and_ignores_outside_ids(self):
        index = SimpleNamespace(
            entity_types=np.asarray(["item", "item", "item"]),
            agent_ids=np.asarray(["jizhi", "mazi", "baijinbi"]),
            operator_ids=np.asarray(["jizhi", "mazi", "baijinbi"]),
            operator_names=np.asarray(["鸡炙", "麻籽", "白金币"]),
        )
        snapshot_list = SnapshotList("test", "item", ("jizhi", "mazi"))
        completed, ignored, recognized_count = _apply_snapshot_list(
            [
                {"entity_type": "item", "item_id": "jizhi", "count": 17},
                {"entity_type": "item", "item_id": "baijinbi", "count": 8},
            ],
            snapshot_list,
            index,
        )
        self.assertEqual(recognized_count, 1)
        self.assertEqual(ignored, ["baijinbi"])
        self.assertEqual(
            [(entry["item_id"], entry["item_name"], entry["count"]) for entry in completed],
            [("jizhi", "鸡炙", 17), ("mazi", "麻籽", 0)],
        )

    def test_snapshot_list_requires_every_id_in_selected_index(self):
        index = SimpleNamespace(
            entity_types=np.asarray(["agent"]),
            agent_ids=np.asarray(["normal"]),
            operator_ids=np.asarray(["char_001_normal"]),
            operator_names=np.asarray(["普通密探"]),
        )
        with self.assertRaisesRegex(ValueError, "请先更新对应 NPZ"):
            _apply_snapshot_list(
                [],
                SnapshotList(
                    "tab3-2",
                    "agent",
                    ("char_001_normal", "char_126_future"),
                ),
                index,
            )

    def test_snapshot_list_keeps_screen_order_then_appends_zeroes(self):
        index = SimpleNamespace(
            entity_types=np.asarray(["item", "item", "item"]),
            agent_ids=np.asarray(["first", "second", "missing"]),
            operator_ids=np.asarray(["first", "second", "missing"]),
            operator_names=np.asarray(["第一项", "第二项", "未出现项"]),
        )
        snapshot_list = SnapshotList(
            "test", "item", ("first", "second", "missing")
        )
        completed, _, _ = _apply_snapshot_list(
            [
                {"entity_type": "item", "item_id": "second", "count": 2},
                {"entity_type": "item", "item_id": "first", "count": 1},
            ],
            snapshot_list,
            index,
        )
        self.assertEqual(
            [(entry["item_id"], entry["count"]) for entry in completed],
            [("second", 2), ("first", 1), ("missing", 0)],
        )

    def test_target_boundary_requires_a_complete_row_after_last_target(self):
        snapshot_list = SnapshotList("test", "item", ("target",))
        page = PageScan(
            results=[
                {
                    "entity_type": "item",
                    "item_id": "target",
                    "row": 1,
                }
            ],
            rejected=[],
            row_features=[
                {0: np.zeros(1), 1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)},
                {0: np.zeros(1), 1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)},
                {0: np.zeros(1), 1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)},
            ],
            column_count=4,
            layout={},
        )
        self.assertEqual(
            _has_snapshot_target_boundary(page, snapshot_list, False),
            (True, True),
        )

        page.row_features[2] = {0: np.zeros(1), 1: np.zeros(1)}
        self.assertEqual(
            _has_snapshot_target_boundary(page, snapshot_list, False),
            (False, True),
        )

    def test_non_target_gap_before_a_later_target_does_not_stop(self):
        snapshot_list = SnapshotList("test", "item", ("target",))
        page = PageScan(
            results=[
                {
                    "entity_type": "item",
                    "item_id": "outside",
                    "row": 0,
                },
                {
                    "entity_type": "item",
                    "item_id": "target",
                    "row": 1,
                },
            ],
            rejected=[],
            row_features=[
                {0: np.zeros(1), 1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)},
                {0: np.zeros(1), 1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)},
            ],
            column_count=4,
            layout={},
        )
        self.assertEqual(
            _has_snapshot_target_boundary(page, snapshot_list, False),
            (False, True),
        )

    def test_count_rejection_still_marks_a_target_row(self):
        snapshot_list = SnapshotList("test", "item", ("target",))
        page = PageScan(
            results=[],
            rejected=[
                {
                    "reason": "count-not-recognized",
                    "entity_type": "item",
                    "item_id": "target",
                    "row": 0,
                }
            ],
            row_features=[
                {0: np.zeros(1), 1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)},
                {0: np.zeros(1), 1: np.zeros(1), 2: np.zeros(1), 3: np.zeros(1)},
            ],
            column_count=4,
            layout={},
        )
        self.assertEqual(
            _has_snapshot_target_boundary(page, snapshot_list, False),
            (True, True),
        )


class PagedItemRecognitionSwipeTests(unittest.TestCase):
    def setUp(self):
        self.layout = {
            "column_centers": [118, 279, 440, 601],
            "row_centers": [329, 518, 707, 895, 1083],
            "circle_radius": [50, 60],
        }
        self.roi = (34, 245, 672, 940)

    def test_automatic_swipe_can_limit_distance_by_rows(self):
        self.assertEqual(
            automatic_swipe(self.layout, self.roi, 500, swipe_rows=2),
            (360, 1083, 360, 706, 500),
        )

    def test_automatic_swipe_preserves_full_page_default(self):
        self.assertEqual(
            automatic_swipe(self.layout, self.roi, 500),
            (360, 1083, 360, 329, 500),
        )

    def test_automatic_swipe_clamps_rows_to_visible_page(self):
        self.assertEqual(
            automatic_swipe(self.layout, self.roi, 500, swipe_rows=20),
            (360, 1083, 360, 329, 500),
        )

    def test_parse_swipe_rows(self):
        self.assertIsNone(_parse_swipe_rows({}))
        self.assertEqual(_parse_swipe_rows({"swipe_rows": "1.5"}), 1.5)
        for value in (0, -1, True, "bad", float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "swipe_rows"):
                    _parse_swipe_rows({"swipe_rows": value})


if __name__ == "__main__":
    unittest.main()
