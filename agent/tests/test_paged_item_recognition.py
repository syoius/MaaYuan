import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
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
    _uncertain_snapshot_ids,
    _filter_results,
    _has_snapshot_target_boundary,
    _load_tab3_2_ids,
    _locate_bag_detail_target,
    _bag_detail_identity,
    _resolve_bag_consensus,
    _parse_entity_type_filter,
    _parse_snapshot_list,
    _parse_swipe_rows,
    _resolve_bag_count,
    _resolve_inventory_report_context,
    _snapshot_record_options,
    automatic_swipe,
    find_row_overlap,
    overlap_candidate_scores,
    recognize_baijinbi_count,
)
from custom.reco.agent_item import (  # noqa: E402
    CountDigitCandidate,
    _bag_count_vote,
    _digit_candidates_to_result,
    _match_rejection_reason,
    recognize_count,
    recognize_count_digits,
)
from custom.action.toolbox_diagnostics import DiagnosticSession  # noqa: E402


class BagDetailCorrectionTests(unittest.TestCase):
    @patch("custom.action.paged_item_recognition._index_entries")
    def test_uncertain_lock_identity_comes_from_unique_popup_name(self, entries):
        fire = dict(entity_type="item", item_id="huoyuanjinsuo", item_name="火源金锁")
        earth = dict(entity_type="item", item_id="zaidijinsuo", item_name="载地金锁")
        entries.return_value = {("item", fire["item_id"]): fire, ("item", earth["item_id"]): earth}
        candidate = dict(fire, _uncertain_jinsuo=True)
        state = {"aliases": {}}
        earth_title = SimpleNamespace(text="载地金锁", score=0.99)
        self.assertEqual(_bag_detail_identity(candidate, [earth_title], None, state), earth)
        self.assertIsNone(_bag_detail_identity(candidate, [SimpleNamespace(text="铜镜", score=0.99)], None, state))
        self.assertIsNone(_bag_detail_identity(candidate, [earth_title, SimpleNamespace(text="火源金锁", score=0.99)], None, state))

    def test_detail_identity_overrides_unconfirmed_icon_candidate(self):
        base = dict(entity_type="item", item_center=(100, 500), _image_index=0)
        unresolved = dict(base, item_id="huoyuanjinsuo", count=None, match_score=0.99)
        detail = dict(base, item_id="zaidijinsuo", count=83, match_score=0.85, _count_source="detail")
        chosen, _ = _resolve_bag_consensus([unresolved, detail], (0, 0, 720, 1280), {})
        self.assertEqual((chosen["item_id"], chosen["count"]), ("zaidijinsuo", 83))

    def test_template_conflicts_require_detail_even_with_different_lengths(self):
        for first, second, template in ((2, 2, 27), (1, 1, 11), (12, 12, 17), (27, 27, 2)):
            with self.subTest(ocr=first, template=template):
                self.assertIsNone(_bag_count_vote(first, second, template))
        self.assertEqual(_bag_count_vote(27, 27, 27), 27)
        self.assertEqual(_bag_count_vote(27, 27, None), 27)
        self.assertIsNone(_bag_count_vote(27, 2, 27))

    @patch("custom.action.paged_item_recognition.recognize_agent_in_cell")
    @patch("custom.action.paged_item_recognition.detect_auto_layout")
    def test_relocation_uses_fresh_center_and_rejects_other_items(self, layout, recognize):
        candidate = dict(entity_type="item", item_id="tianfengjinsuo", item_center=(100, 500))
        layout.return_value = ([SimpleNamespace(center=(100, 490)), SimpleNamespace(center=(100, 440))], {})
        wrong = SimpleNamespace(entity_type="item", agent_id="huoyuanjinsuo", item_center=(100, 490))
        target = SimpleNamespace(entity_type="item", agent_id="tianfengjinsuo", item_center=(101, 439))
        recognize.side_effect = [(wrong, {}), (target, {})]
        result = _locate_bag_detail_target(None, candidate, None, (0, 0, 720, 1280), {})
        self.assertEqual(result.item_center, (101, 439))
        self.assertEqual(candidate["item_center"], (100, 500))
        self.assertEqual(recognize.call_args.args[1], (100, 440))
        recognize.side_effect = [(wrong, {}), (None, {})]
        self.assertIsNone(_locate_bag_detail_target(None, candidate, None, (0, 0, 720, 1280), {}))


class BaijinbiRecognitionTests(unittest.TestCase):
    def test_uses_configured_ocr_node_and_accepts_zero(self):
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        for raw, expected in (("1307", 1307), ("3177", 3177), ("0", 0)):
            with self.subTest(raw=raw):
                context = Mock()
                context.run_recognition.return_value = SimpleNamespace(
                    filtered_results=[SimpleNamespace(text=raw)]
                )
                self.assertEqual(recognize_baijinbi_count(context, image), expected)
                context.run_recognition.assert_called_once_with(
                    "背包-白金币识别", image
                )

    def test_failed_or_non_integer_ocr_is_not_reported_as_zero(self):
        for detail in (
            None,
            SimpleNamespace(filtered_results=[]),
            SimpleNamespace(filtered_results=[SimpleNamespace(text="1.3万")]),
        ):
            context = Mock()
            context.run_recognition.return_value = detail
            with self.assertRaises(ValueError):
                recognize_baijinbi_count(context, np.zeros((1, 1, 3), dtype=np.uint8))


class PagedItemRecognitionFilterTests(unittest.TestCase):
    def test_uncertain_candidate_is_omitted_and_absent_item_is_zero_filled(self):
        index = SimpleNamespace(
            entity_types=np.asarray(["item"] * 3),
            agent_ids=np.asarray(["earth", "fire", "absent"]),
            operator_ids=np.asarray(["earth", "fire", "absent"]),
            operator_names=np.asarray(["载地", "火源", "未出现"]),
        )
        results, _, count = _apply_snapshot_list(
            [{"item_id": "fire", "count": 177}],
            SnapshotList("test", "item", ("earth", "fire", "absent")),
            index, {"earth", "fire"},
        )
        self.assertEqual(count, 1)
        self.assertEqual(
            {r["item_id"]: r["count"] for r in results},
            {"fire": 177, "absent": 0},
        )

    @patch("custom.action.paged_item_recognition.logger.warning")
    def test_count_skipped_id_is_not_zero_filled_or_warned_again(self, warning):
        index = SimpleNamespace(
            entity_types=np.asarray(["item", "item", "item"]),
            agent_ids=np.asarray(["uncertain", "skipped", "absent"]),
            operator_ids=np.asarray(["uncertain", "skipped", "absent"]),
            operator_names=np.asarray(["候选未确认", "数量不可信", "未出现"]),
        )
        completed, ignored, count = _apply_snapshot_list(
            [], SnapshotList("test", "item", ("uncertain", "skipped", "absent")),
            index, {"uncertain", "skipped"}, count_skipped_ids={"skipped"},
        )
        self.assertEqual([(entry["item_id"], entry["count"]) for entry in completed], [("absent", 0)])
        self.assertEqual(ignored, [])
        self.assertEqual(count, 0)
        warning.assert_called_once_with("【广陵库房】以下候选未确认，本次不覆盖其库存：uncertain")

    @patch("custom.action.toolbox_diagnostics._BAG_WORKER")
    def test_finish_bag_treats_zero_count_talismans_as_missing(self, worker):
        session = SimpleNamespace(
            bag_capture_failed=False,
            images=[({"index": 0, "role": "bag_page", "roi": [0, 0, 1, 1]}, b"image")],
            event_id="00000000-0000-0000-0000-000000000001",
            run_id="00000000-0000-0000-0000-000000000002",
            occurred_at="2026-11-13T00:00:00+00:00",
            client={"resource_version": "unknown", "maafw_version": "unknown",
                    "resource": "base", "resolution": "720x1280"},
            secrets=[],
        )
        index_path = Mock()
        index_path.read_bytes.return_value = b"index"
        for counts, reason in (
            ((0, 0), "both_missing"),
            ((0, 1), "fuchuan_missing"),
            ((1, 0), "tianjifuchuan_missing"),
            ((1, 1), None),
        ):
            with self.subTest(counts=counts):
                worker.submit.reset_mock()
                DiagnosticSession.finish_bag(
                    session,
                    [{"item_id": item_id, "count": count}
                     for item_id, count in zip(("fuchuan", "tianjifuchuan"), counts)],
                    index_path,
                )
                worker.submit.assert_called_once()
                if reason is None:
                    worker.submit.assert_called_once_with(session)
                else:
                    manifest = worker.submit.call_args.args[1][0]
                    self.assertEqual(manifest["issues"], [{
                        "code": "bag_item_missing", "stage": "bag_tab1",
                        "reason": reason, "count": 1,
                    }])

    def test_refine_ambiguity_protects_close_candidates_only(self):
        index = SimpleNamespace(
            entity_types=np.asarray(["item"] * 3),
            agent_ids=np.asarray(["earth", "fire", "water"]),
            refine_groups=[SimpleNamespace(group_id="locks", min_margin=0.018)],
        )
        rejected = {
            "best_agent_id": "earth", "match_score": 0.93,
            "refined": True, "refine_group": "locks", "refine_score": 0.94,
            "refine_candidates": [
                {"item_id": "earth", "score": 0.94},
                {"item_id": "fire", "score": 0.93},
                {"item_id": "water", "score": 0.7},
            ],
        }
        self.assertEqual(_uncertain_snapshot_ids(rejected, index, {}), {"earth", "fire"})
        rejected["match_score"] = 0.4
        self.assertEqual(_uncertain_snapshot_ids(rejected, index, {}), set())

    def test_failed_count_protects_identified_item(self):
        self.assertEqual(
            _uncertain_snapshot_ids(
                {"entity_type": "item", "item_id": "earth", "reason": "count-not-recognized"},
                None, {},
            ),
            {"earth"},
        )

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
    def test_local_snapshot_custom_name_uses_stock_report_prefix(self):
        settings = SimpleNamespace(
            mode="仅保存到本地",
            report_filename="大号",
        )

        account, path = _resolve_inventory_report_context(
            {"inventory_report_path": "StockReport.txt"},
            SnapshotList("tab1", "item", ("jizhi",)),
            settings,
        )

        self.assertIsNone(account)
        self.assertEqual(path.name, "StockReport-大号.txt")

    def test_local_reward_custom_name_keeps_daily_rewards_prefix(self):
        settings = SimpleNamespace(
            mode="仅保存到本地",
            report_filename="大号",
        )

        _, path = _resolve_inventory_report_context({}, None, settings)

        self.assertEqual(path.name, "DailyRewards-大号.txt")

    @patch("custom.action.paged_item_recognition.logger.warning")
    @patch(
        "custom.action.paged_item_recognition.get_bound_account",
        side_effect=RuntimeError("offline"),
    )
    def test_auto_account_lookup_failure_falls_back_to_local_report(
        self, get_account, warning
    ):
        settings = SimpleNamespace(
            mode="自动上报",
            report_filename=None,
        )

        account, path = _resolve_inventory_report_context(
            {"inventory_report_path": "StockReport.txt"},
            SnapshotList("tab1", "item", ("jizhi",)),
            settings,
        )

        self.assertIsNone(account)
        self.assertEqual(path.name, "StockReport.txt")
        get_account.assert_called_once_with(settings)
        self.assertIn("继续扫描", warning.call_args.args[0])

    @patch("custom.action.paged_item_recognition.get_bound_account")
    def test_auto_account_lookup_keeps_account_specific_name(self, get_account):
        get_account.return_value = SimpleNamespace(id="acc_main", name="大号")
        settings = SimpleNamespace(
            mode="自动上报",
            report_filename=None,
        )

        account, path = _resolve_inventory_report_context(
            {"inventory_report_path": "StockReport.txt"},
            SnapshotList("tab1", "item", ("jizhi",)),
            settings,
        )

        self.assertEqual(account.id, "acc_main")
        self.assertEqual(path.name, "DailyRewards-大号-acc_main.txt")

    def test_static_item_lists_are_disjoint_and_exclude_baijinbi(self):
        self.assertEqual(
            set(TAB1_ITEM_IDS),
            {"fuchuan", "tianjifuchuan", "jizhi", "mazi", "sherou", "zhuyu"},
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

    def test_apply_snapshot_list_zero_fills_absent_and_ignores_outside_ids(self):
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

    @patch("custom.action.paged_item_recognition.logger.warning")
    def test_missing_new_agents_warn_in_chinese_and_are_not_zero_filled(self, warning):
        index = SimpleNamespace(
            entity_types=np.asarray(["agent", "agent"]),
            agent_ids=np.asarray(["yangxiu", "jiaxu"]),
            operator_ids=np.asarray(["char_001_yangxiu", "char_002_jiaxu"]),
            operator_names=np.asarray(["杨修", "贾诩"]),
        )
        completed, ignored, count = _apply_snapshot_list(
            [{"entity_type": "agent", "operator_id": "char_001_yangxiu", "count": 17}],
            SnapshotList("tab3-2", "agent", (
                "char_001_yangxiu", "char_002_jiaxu", "char_129_zhoutai", "char_130_chenlin",
            )),
            index,
        )
        warning.assert_called_once_with("暂未支持扫描新密探心纸：周泰, 陈琳")
        self.assertEqual(count, 1)
        self.assertEqual(ignored, [])
        self.assertEqual(
            [(entry["operator_id"], entry["count"]) for entry in completed],
            [("char_001_yangxiu", 17)],
        )

    def test_other_snapshot_lists_require_every_id_in_selected_index(self):
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
                    "test",
                    "agent",
                    ("char_001_normal", "char_126_future"),
                ),
                index,
            )

    def test_snapshot_list_keeps_screen_order_and_appends_absent_items_as_zero(self):
        index = SimpleNamespace(
            entity_types=np.asarray(["item", "item", "item", "item"]),
            agent_ids=np.asarray(["first", "second", "missing", "absent"]),
            operator_ids=np.asarray(["first", "second", "missing", "absent"]),
            operator_names=np.asarray(["第一项", "第二项", "未出现项", "缺失项"]),
        )
        snapshot_list = SnapshotList(
            "test", "item", ("absent", "first", "second", "missing")
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
            [("second", 2), ("first", 1), ("absent", 0), ("missing", 0)],
        )

    def test_duplicate_snapshot_result_reports_both_source_pages(self):
        index = SimpleNamespace(
            entity_types=np.asarray(["item"]),
            agent_ids=np.asarray(["target"]),
            operator_ids=np.asarray(["target"]),
            operator_names=np.asarray(["目标"]),
        )
        results = [
            {
                "entity_type": "item",
                "item_id": "target",
                "count": 1,
                "row": 3,
                "column": 0,
                "match_score": 0.98,
                "_source_page": 1,
                "_source_page_row": 3,
            },
            {
                "entity_type": "item",
                "item_id": "target",
                "count": 1,
                "row": 7,
                "column": 0,
                "match_score": 0.97,
                "_source_page": 2,
                "_source_page_row": 0,
            },
        ]
        with self.assertRaisesRegex(
            ValueError,
            r"first=page=1,.*second=page=2,",
        ):
            _apply_snapshot_list(
                results,
                SnapshotList("test", "item", ("target",)),
                index,
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

    def test_overlap_diagnostics_keep_failed_and_shorter_candidates(self):
        first = np.asarray([1.0, 0.0], dtype=np.float32)
        second = np.asarray([0.0, 1.0], dtype=np.float32)
        previous = [{0: first, 1: first}, {0: second, 1: second}]
        current = [{0: second, 1: second}, {0: first, 1: first}]

        self.assertEqual(
            overlap_candidate_scores(previous, current),
            [
                {
                    "count": 2,
                    "scores": [0.0, 0.0],
                    "identity_matches": [0, 0],
                    "identity_count_matches": [0, 0],
                },
                {
                    "count": 1,
                    "scores": [1.0],
                    "identity_matches": [0],
                    "identity_count_matches": [0],
                },
            ],
        )

    def test_two_matching_entity_ids_can_rescue_low_image_similarity(self):
        previous_feature = np.asarray([1.0, 0.0], dtype=np.float32)
        current_feature = np.asarray([0.8, 0.6], dtype=np.float32)
        previous = [{0: previous_feature, 1: previous_feature}]
        current = [{0: current_feature, 1: current_feature}]
        previous_entities = [
            {0: "agent:first", 1: "agent:second"}
        ]
        current_entities = [
            {0: "agent:first", 1: "agent:second"}
        ]

        overlap, scores = find_row_overlap(
            previous,
            current,
            0.88,
            previous_entities,
            current_entities,
        )

        self.assertEqual(overlap, 1)
        self.assertAlmostEqual(scores[0], 0.8, places=6)

    def test_identity_anchor_allows_an_adjacent_partial_row(self):
        first = np.asarray([1.0, 0.0], dtype=np.float32)
        weak = np.asarray([0.7, np.sqrt(0.51)], dtype=np.float32)
        previous = [
            {0: first, 1: first},
            {0: first, 1: first},
        ]
        current = [
            {0: weak, 1: weak},
            {0: weak, 1: weak},
        ]
        previous_entities = [
            {0: "agent:first", 1: "agent:second"},
            {},
        ]
        current_entities = [
            {0: "agent:first", 1: "agent:second"},
            {},
        ]

        overlap, _ = find_row_overlap(
            previous,
            current,
            0.88,
            previous_entities,
            current_entities,
        )

        self.assertEqual(overlap, 2)

    def test_one_matching_id_requires_near_threshold_image_score(self):
        previous_feature = np.asarray([1.0, 0.0], dtype=np.float32)
        near_feature = np.asarray([0.86, np.sqrt(1 - 0.86**2)], dtype=np.float32)
        weak_feature = np.asarray([0.7, np.sqrt(1 - 0.7**2)], dtype=np.float32)
        entities = [{0: "agent:only"}]

        self.assertEqual(
            find_row_overlap(
                [{0: previous_feature}],
                [{0: near_feature}],
                0.88,
                entities,
                entities,
            )[0],
            1,
        )
        self.assertEqual(
            find_row_overlap(
                [{0: previous_feature}],
                [{0: weak_feature}],
                0.88,
                entities,
                entities,
            )[0],
            0,
        )

    def test_matching_id_and_count_rescues_cropped_two_row_overlap(self):
        full = np.asarray([1.0, 0.0], dtype=np.float32)
        cropped = np.asarray([0.61, np.sqrt(1 - 0.61**2)], dtype=np.float32)
        partial = np.asarray([0.81, np.sqrt(1 - 0.81**2)], dtype=np.float32)
        previous = [{0: full, 1: full}, {0: full, 1: full}]
        current = [{0: cropped, 1: cropped}, {0: partial, 1: partial}]
        previous_entities = [{1: "agent:chendeng"}, {}]
        current_entities = [{1: "agent:chendeng"}, {}]
        previous_counts = [{1: 2}, {}]
        current_counts = [{1: 2}, {}]

        overlap, scores = find_row_overlap(
            previous,
            current,
            0.88,
            previous_entities,
            current_entities,
            previous_counts,
            current_counts,
        )

        self.assertEqual(overlap, 2)
        self.assertAlmostEqual(scores[0], 0.61, places=6)

    def test_matching_id_with_different_count_does_not_rescue_weak_row(self):
        full = np.asarray([1.0, 0.0], dtype=np.float32)
        weak = np.asarray([0.61, np.sqrt(1 - 0.61**2)], dtype=np.float32)
        entities = [{0: "agent:chendeng"}]

        overlap, _ = find_row_overlap(
            [{0: full}],
            [{0: weak}],
            0.88,
            entities,
            entities,
            [{0: 2}],
            [{0: 3}],
        )

        self.assertEqual(overlap, 0)


class BagCountResolutionTests(unittest.TestCase):
    def resolve(self, cell, page_counts, edge_pages=()):
        observations = {}
        for token in cell.split():
            page, reading = token.split(":")
            page = int(page[1:])
            count, raw, source, score = reading.split("/")
            observation = observations.setdefault(page, {
                "entity_type": "item", "item_id": "sample",
                "count": page_counts[page], "count_raw": str(page_counts[page]),
                "count_score": 0.9, "count_box": [100, 500, 95, 44],
                "match_score": 0.95, "item_center": [130, 1100 if page in edge_pages else 600],
                "_source_page": page + 1, "count_readings": [],
            })
            observation["count_readings"].append({
                "count": None if count == "undefined" else int(count),
                "raw": raw, "source": source, "score": float(score),
                "count_box": [100, 500, 95, 44],
            })
        return _resolve_bag_count(
            list(observations.values()), (34, 245, 672, 940),
            {"count_min_roi_bottom_distance": 70},
        )

    def test_300b6d_production_readings(self):
        # Original CELL readings from the .32 diagnostics; page_counts are the old per-page outputs.
        cases = {
            "mazi": (513, {0: 513, 1: 513},
                "p0:513/513/primary/0.794 p0:513/513/fallback/0.794 p0:513/513/fallback/0.751 "
                "p1:513/513/primary/0.875 p1:513/513/fallback/0.875 p1:13/13/fallback/0.882"),
            "jizhi": (574, {0: 574, 1: 574},
                "p0:574/574/primary/0.683 p0:574/574/fallback/0.683 p0:74/74/fallback/0.868 "
                "p1:574/574/primary/0.876 p1:574/574/fallback/0.876 p1:74/74/fallback/0.923"),
            "bingshuquanjuan": (7569, {0: 7569},
                "p0:undefined/7569/primary/0.398 p0:7569/7569/fallback/0.812 p0:69/69/fallback/0.858"),
            "qingtingyan": (2924, {0: 292, 1: 2924},
                "p0:292/292/primary/0.803 p0:2924/2924/fallback/0.827 p0:2924/2924/fallback/0.761 "
                "p1:2924/2924/primary/0.88 p1:2924/2924/fallback/0.88 p1:2924/2924/fallback/0.819"),
            "xianmenshan": (8275, {1: 8275},
                "p1:8275/8275/primary/0.843 p1:8275/8275/fallback/0.797 p1:48275/48275/fallback/0.456"),
            "bawanglei": (14855, {1: 1485},
                "p1:1485/1485/primary/0.796 p1:14855/14855/fallback/0.817 p1:14855/14855/fallback/0.851"),
            "baoshijing": (1161, {1: 1161},
                "p1:1161/1161/primary/0.868 p1:1161/1161/fallback/0.868 p1:161/161/fallback/0.905"),
            "liujinjing": (5271, {1: 5271},
                "p1:5271/5271/primary/0.89 p1:271/271/fallback/0.931 p1:271/271/fallback/0.816"),
            "jinsishan": (4905, {1: 4905, 2: 4905},
                "p1:4905/4905/primary/0.848 p1:4905/4905/fallback/0.774 p1:105/105/fallback/0.826 "
                "p2:4905/4905/primary/0.865 p2:4905/4905/fallback/0.822 p2:105/105/fallback/0.777"),
            "lingshanquan": (4361, {1: 4361, 2: 4361},
                "p1:4361/4361/primary/0.798 p1:361/361/fallback/0.834 p1:361/361/fallback/0.822 "
                "p2:4361/4361/primary/0.823 p2:361/361/fallback/0.804 p2:161/161/fallback/0.823"),
            "baimozhijiu": (5820, {1: 5820, 2: 5820},
                "p1:5820/5820/primary/0.819 p1:820/820/fallback/0.9 p1:820/820/fallback/0.867 "
                "p2:5820/5820/primary/0.854 p2:820/820/fallback/0.837 p2:820/820/fallback/0.814"),
            "juanshan": (2958, {2: 2958},
                "p2:2958/2958/primary/0.822 p2:2058/2058/fallback/0.742 p2:158/158/fallback/0.845"),
            "zhuojiu": (3380, {2: 33},
                "p2:33/33/primary/0.822 p2:3380/3380/fallback/0.797 p2:180/180/fallback/0.822"),
            "jieyangping": (302, {2: 302},
                "p2:302/302/primary/0.839 p2:302/302/fallback/0.789 p2:102/102/fallback/0.789"),
            "jiezheping": (400, {2: 400, 3: 400},
                "p2:4/4/primary/0.919 p2:400/400/fallback/0.844 p2:400/400/fallback/0.845 "
                "p3:400/400/primary/0.848 p3:400/400/fallback/0.908 p3:400/400/fallback/0.856"),
            "yangmingjinsuo": (622, {3: 622},
                "p3:622/622/primary/0.885 p3:622/622/fallback/0.835 p3:22/22/fallback/0.891"),
            "huoyuanjinsuo": (567, {3: 567},
                "p3:undefined/567/primary/0.334 p3:567/567/fallback/0.842 p3:67/67/fallback/0.837"),
            "shuixinjinsuo": (674, {3: 674},
                "p3:674/674/primary/0.837 p3:674/674/fallback/0.808 p3:74/74/fallback/0.93"),
            "zaidijinsuo": (967, {3: 967},
                "p3:967/967/primary/0.791 p3:967/967/fallback/0.77 p3:167/167/fallback/0.743"),
        }
        for entity_id, (expected, page_counts, readings) in cases.items():
            with self.subTest(entity_id=entity_id):
                chosen, suspect = self.resolve(readings, page_counts)
                self.assertEqual(chosen["count"], expected)
                self.assertEqual(chosen["count_raw"], str(expected))
                self.assertEqual(suspect["chosen"], expected)
                self.assertEqual(len(suspect["readings"]), len(readings.split()))
                if entity_id == "qingtingyan":
                    self.assertIn("page_conflict", suspect["reasons"])

    def test_other_production_ambiguities(self):
        cases = (
            (456, {0: 56},
             "p0:56/56/primary/0.897 p0:56/56/fallback/0.934 p0:456/456/fallback/0.884"),
            (388, {7: 388, 8: 388},
             "p7:3/3/primary/0.94 p7:3/3/fallback/0.91 p7:388/388/fallback/0.875 "
             "p8:3/3/primary/0.925 p8:3/3/fallback/0.823 p8:388/388/fallback/0.88"),
            (405, {2: 405},
             "p2:405/405/primary/0.859 p2:1405/1405/fallback/0.503 p2:1405/1405/fallback/0.503"),
            (384, {8: 4, 9: 384},
             "p8:4/4/primary/0.946 p8:4/4/fallback/0.939 p8:4/4/fallback/0.865 "
             "p9:384/384/primary/0.871 p9:384/384/fallback/0.847 p9:384/384/fallback/0.897"),
        )
        for expected, page_counts, readings in cases:
            with self.subTest(expected=expected):
                chosen, _ = self.resolve(readings, page_counts)
                self.assertEqual(chosen["count"], expected)

    def test_n16_production_readings_prefer_longest_grounded_value(self):
        cases = {
            "jieyangping": (3900, {2: 3900},
                "p2:3900/3900/primary/0.816 p2:100/100/fallback/0.81 p2:100/100/fallback/0.84"),
            "jinsishan": (6981, {1: 181, 2: 6981},
                "p1:181/181/primary/0.777 p1:181/181/fallback/0.743 p1:181/181/fallback/0.732 "
                "p2:6981/6981/primary/0.866 p2:181/181/fallback/0.831 p2:181/181/fallback/0.831"),
            "jincuodao": (1998, {1: 1998},
                "p1:1998/1998/primary/0.911 p1:1098/1098/fallback/0.776 p1:1098/1098/fallback/0.782"),
            "bingshucanjuan": (29898, {0: 29898},
                "p0:29898/29898/primary/0.852 p0:20898/20898/fallback/0.785 p0:20898/20898/fallback/0.676"),
            "bingshuquanjuan": (5062, {0: 5062},
                "p0:5062/5062/primary/0.908 p0:54162/54162/fallback/0.7 p0:54162/54162/fallback/0.661"),
            "xinghanjing": (4899, {1: 99, 2: 99},
                "p1:99/99/primary/0.847 p1:4899/4899/fallback/0.863 p1:4899/4899/fallback/0.811 "
                "p2:99/99/primary/0.908 p2:4899/4899/fallback/0.854 p2:4899/4899/fallback/0.826"),
            "baimozhijiu": (1981, {1: 1981, 2: 1981},
                "p1:1981/1981/primary/0.812 p1:181/181/fallback/0.787 p1:181/181/fallback/0.78 "
                "p2:1981/1981/primary/0.805 p2:181/181/fallback/0.8 p2:181/181/fallback/0.845"),
            "jizhi": (1196, {0: 196, 1: 1196},
                "p0:196/196/primary/0.883 p0:196/196/fallback/0.883 p0:196/196/fallback/0.826 "
                "p1:1196/1196/primary/0.9 p1:1196/1196/fallback/0.9 p1:1196/1196/fallback/0.885"),
        }
        for entity_id, (expected, page_counts, readings) in cases.items():
            with self.subTest(entity_id=entity_id):
                chosen, suspect = self.resolve(readings, page_counts)
                self.assertEqual(chosen["count"], expected)
                self.assertEqual(chosen["count_raw"], str(expected))
                self.assertEqual(suspect["chosen"], expected)
                self.assertEqual(len(suspect["readings"]), len(readings.split()))
                if entity_id in ("jizhi", "jinsishan"):
                    self.assertIn("page_conflict", suspect["reasons"])
                    self.assertEqual(chosen["_source_page"], max(page_counts) + 1)

    def test_length_then_primary_frequency_then_non_edge_then_frequency_then_score(self):
        cases = (
            (12, "p0:12/12/primary/0.8 p0:12/12/fallback/0.8 "
             "p1:34/34/fallback/0.95 p1:34/34/fallback/0.95", (0,)),
            (34, "p0:12/12/primary/0.95 p0:12/12/fallback/0.95 "
             "p1:34/34/primary/0.8 p1:34/34/fallback/0.8", (0,)),
            (345, "p0:12/12/primary/0.95 p0:12/12/fallback/0.95 "
             "p1:345/345/primary/0.8 p1:345/345/fallback/0.8", ()),
            (34, "p0:12/12/primary/0.8 p0:12/12/fallback/0.8 "
             "p1:34/34/primary/0.95 p1:34/34/fallback/0.95", ()),
            (345, "p0:345/345/primary/0.8 p0:345/345/fallback/0.8 "
             "p1:12/12/primary/0.95 p1:12/12/primary/0.95", (0,)),
            (34, "p0:12/12/primary/0.95 p0:12/12/fallback/0.95 p0:12/12/fallback/0.95 "
             "p1:34/34/primary/0.8 p1:34/34/fallback/0.8", (0,)),
            (12, "p0:12/12/primary/0.8 p0:12/12/fallback/0.8 p0:12/12/fallback/0.8 "
             "p1:34/34/primary/0.95 p1:34/34/fallback/0.95", ()),
        )
        for expected, readings, edge_pages in cases:
            with self.subTest(expected=expected, edge_pages=edge_pages):
                chosen, suspect = self.resolve(readings, {0: 12, 1: expected}, edge_pages)
                self.assertEqual(chosen["count"], expected)
                if edge_pages:
                    self.assertNotIn("edge_zone", suspect["reasons"])

    def test_no_qualified_primary_keeps_previous_support_rule(self):
        chosen, suspect = self.resolve(
            "p1:4/4/primary/0.573 p1:1/1/fallback/0.881 p1:1/1/fallback/0.987", {1: 4},
        )
        self.assertEqual(chosen["count"], 1)
        self.assertEqual(chosen["count_score"], 0.987)
        self.assertEqual(suspect["reasons"], ["threshold_disagree"])

    def test_unconfirmed_or_disagreeing_low_score_candidates_are_omitted(self):
        cases = (
            "p0:undefined/9/primary/0.4 p0:12/12/fallback/0.8",
            "p0:undefined//primary/0.4 p0:12/12/fallback/0.8",
            "p0:12/12/primary/0.699 p0:34/34/fallback/0.699",
        )
        for readings in cases:
            with self.subTest(readings=readings):
                chosen, _ = self.resolve(readings, {0: 12})
                self.assertIsNone(chosen)

    def test_low_score_consensus_requires_a_nonempty_primary(self):
        cases = (
            (574, "p0:574/574/primary/0.65 p0:574/574/fallback/0.66 "
             "p0:574/574/fallback/0.68"),
            (None, "p0:574/574/primary/0.65 p0:74/74/fallback/0.66"),
            (574, "p0:574/574/primary/0.65"),
            (574, "p0:574/574/primary/0.65 p0:undefined/74/fallback/0.66"),
            (None, "p0:undefined/574/primary/0.65 p0:574/574/fallback/0.66"),
            (None, "p0:undefined//primary/0.65 p0:undefined//fallback/0.66"),
        )
        for expected, readings in cases:
            with self.subTest(readings=readings):
                chosen, _ = self.resolve(readings, {0: 574})
                self.assertEqual(chosen["count"] if chosen is not None else None, expected)

    def test_single_primary_is_accepted_at_score_boundary_and_keeps_edge_reason(self):
        chosen, suspect = self.resolve("p0:0/0/primary/0.70", {0: 0}, (0,))
        self.assertEqual(chosen["count"], 0)
        self.assertEqual(suspect["reasons"], ["edge_zone"])


class BagCountDigitExtractionTests(unittest.TestCase):
    def recognize_fixture(self, name, threshold):
        path = Path(__file__).parent / "fixtures" / "bag_count" / f"{name}.png"
        image = cv2.imread(str(path))
        self.assertIsNotNone(image, str(path))
        # The fixture is the exact standard count crop, without its surrounding frame.
        return recognize_count_digits(
            image, (35, -19),
            {"count_binary_threshold": threshold, "count_badge_threshold": 200},
        )

    def test_earth_lock_keeps_six_before_narrow_one_and_excludes_element_badge(self):
        for threshold in (165, 170, 175):
            with self.subTest(threshold=threshold):
                count, _, raw, box = self.recognize_fixture("zaidijinsuo-614", threshold)
                self.assertEqual((count, raw, box), (614, "614", (0, 0, 95, 44)))

    def test_wine_keeps_both_leading_ones(self):
        count, _, raw, _ = self.recognize_fixture("baimozhijiu-11981", 165)
        self.assertEqual((count, raw), (11981, "11981"))

    def test_gold_fan_keeps_prefix_before_narrow_one(self):
        for threshold in (165, 170):
            with self.subTest(threshold=threshold):
                count, _, raw, _ = self.recognize_fixture("jinsishan-15150", threshold)
                self.assertEqual((count, raw), (15150, "15150"))

    def test_chicken_keeps_both_leading_ones_at_fallback_threshold(self):
        count, _, raw, _ = self.recognize_fixture("jizhi-1196", 175)
        self.assertEqual((count, raw), (1196, "1196"))

    def test_fire_lock_still_excludes_icon_and_element_badge(self):
        for threshold in (165, 170, 175):
            with self.subTest(threshold=threshold):
                count, _, raw, _ = self.recognize_fixture("huoyuanjinsuo-704", threshold)
                self.assertEqual((count, raw), (704, "704"))


class CountBinaryFallbackTests(unittest.TestCase):
    def test_ambiguous_leading_zero_uses_close_nonzero_match(self):
        first_scores = {value: 0.0 for value in "0123456789"}
        first_scores.update({"0": 0.8013926, "6": 0.7622426})
        zero_scores = {value: 0.0 for value in "0123456789"}
        zero_scores["0"] = 0.93

        result = _digit_candidates_to_result(
            [
                CountDigitCandidate(45, 18, 10, 15, first_scores),
                CountDigitCandidate(57, 17, 11, 16, zero_scores),
                CountDigitCandidate(69, 16, 11, 17, zero_scores),
            ],
            (251, 823, 95, 44),
            {"count_digit_threshold": 0.45},
        )

        self.assertEqual(result[:3], (600, 0.7622426, "600"))

    def test_unambiguous_nonzero_leading_digit_is_unchanged(self):
        def candidate(x, digit, score):
            scores = {value: 0.0 for value in "0123456789"}
            scores[digit] = score
            return CountDigitCandidate(x, 17, 10, 16, scores)

        result = _digit_candidates_to_result(
            [
                candidate(45, "1", 0.95),
                candidate(57, "0", 0.93),
                candidate(69, "0", 0.94),
            ],
            (0, 0, 95, 44),
            {"count_digit_threshold": 0.45},
        )

        self.assertEqual(result[:3], (100, 0.93, "100"))

    def test_wide_low_confidence_blob_before_three_digits_is_discarded(self):
        def candidate(x, width, digit, score):
            scores = {value: 0.0 for value in "0123456789"}
            scores[digit] = score
            return CountDigitCandidate(x, 17, width, 16, scores)

        result = _digit_candidates_to_result(
            [
                candidate(24, 18, "5", 0.33),
                candidate(45, 10, "5", 0.96),
                candidate(57, 11, "8", 0.94),
                candidate(69, 11, "8", 0.93),
            ],
            (0, 0, 95, 44),
            {"count_digit_threshold": 0.45},
        )

        self.assertEqual(result[:3], (588, 0.93, "588"))

    def test_narrow_low_confidence_leading_digit_still_rejects_count(self):
        scores = {value: 0.0 for value in "0123456789"}
        scores["5"] = 0.33
        result = _digit_candidates_to_result(
            [CountDigitCandidate(45, 17, 10, 16, scores)],
            (0, 0, 95, 44),
            {"count_digit_threshold": 0.45},
        )

        self.assertIsNone(result[0])

    @patch("custom.reco.agent_item.recognize_count_digits")
    def test_fallback_probes_all_thresholds_but_keeps_first_success(self, recognize_digits):
        recognize_digits.side_effect = [
            (None, 0.0, "", (1, 2, 3, 4)),
            (54, 0.9, "54", (1, 2, 3, 4)),
            (541, 0.95, "541", (1, 2, 3, 4)),
        ]
        image = np.zeros((1, 1, 3), dtype=np.uint8)

        result = recognize_count(
            None,
            image,
            (0.0, 0.0),
            {
                "count_binary_threshold": 165,
                "count_binary_fallback_thresholds": [170, 175],
            },
        )

        self.assertEqual(result[0], 54)
        self.assertEqual(recognize_digits.call_count, 3)
        self.assertEqual(
            recognize_digits.call_args_list[0].args[2][
                "count_binary_threshold"
            ],
            165,
        )
        self.assertEqual(
            recognize_digits.call_args_list[1].args[2][
                "count_binary_threshold"
            ],
            170,
        )

    @patch("custom.reco.agent_item.recognize_count_digits")
    def test_primary_success_keeps_value_and_records_fallbacks(self, recognize_digits):
        recognize_digits.side_effect = [
            (623, 0.8, "623", (1, 2, 3, 4)),
            (62, 0.9, "62", (1, 2, 3, 4)),
            (6231, 0.95, "6231", (1, 2, 3, 4)),
        ]
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        readings = []

        result = recognize_count(
            None,
            image,
            (0.0, 0.0),
            {"count_binary_threshold": 165},
            readings=readings,
        )

        self.assertEqual(result[0], 623)
        self.assertEqual(recognize_digits.call_count, 3)
        self.assertEqual([reading["count"] for reading in readings], [623, 62, 6231])
        self.assertEqual([reading["source"] for reading in readings], ["primary", "fallback", "fallback"])

    @patch("custom.reco.agent_item.recognize_count_digits")
    def test_single_digit_result_can_be_replaced_by_longer_fallback(
        self, recognize_digits
    ):
        recognize_digits.side_effect = [
            (3, 0.89, "3", (1, 2, 3, 4)),
            (156, 0.80, "156", (1, 2, 3, 4)),
            (356, 0.835, "356", (1, 2, 3, 4)),
        ]

        result = recognize_count(
            None,
            np.zeros((1, 1, 3), dtype=np.uint8),
            (0.0, 0.0),
            {"count_binary_fallback_thresholds": [170, 175]},
        )

        self.assertEqual(result[:3], (356, 0.835, "356"))
        self.assertEqual(recognize_digits.call_count, 3)

    @patch("custom.reco.agent_item.recognize_count_digits")
    def test_low_score_longer_fallback_does_not_replace_primary(
        self, recognize_digits
    ):
        recognize_digits.side_effect = [
            (1, 0.99, "1", (1, 2, 3, 4)),
            (14197, 0.63, "14197", (1, 2, 3, 4)),
        ]

        result = recognize_count(
            None,
            np.zeros((1, 1, 3), dtype=np.uint8),
            (0.0, 0.0),
            {"count_binary_fallback_thresholds": [175]},
        )

        self.assertEqual(result[:3], (1, 0.99, "1"))


class AdaptiveMatchThresholdTests(unittest.TestCase):
    PARAMS = {
        "match_threshold": 0.90,
        "match_low_threshold": 0.85,
        "match_min_margin": 0.08,
    }

    def test_strong_match_does_not_require_margin(self):
        self.assertIsNone(_match_rejection_reason(0.91, 0.0, self.PARAMS))

    def test_relaxed_match_requires_sufficient_margin(self):
        self.assertIsNone(_match_rejection_reason(0.88, 0.10, self.PARAMS))
        self.assertEqual(
            _match_rejection_reason(0.88, 0.07, self.PARAMS),
            "match-margin-below-threshold",
        )

    def test_score_below_relaxed_threshold_is_rejected(self):
        self.assertEqual(
            _match_rejection_reason(0.84, 0.50, self.PARAMS),
            "match-score-below-threshold",
        )

    def test_missing_runner_up_rejects_only_relaxed_match(self):
        self.assertEqual(
            _match_rejection_reason(0.88, None, self.PARAMS),
            "match-margin-below-threshold",
        )

    def test_omitting_low_threshold_preserves_hard_threshold(self):
        self.assertEqual(
            _match_rejection_reason(0.895, 0.50, {"match_threshold": 0.90}),
            "match-score-below-threshold",
        )

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "match_low_threshold"):
            _match_rejection_reason(
                0.90,
                0.10,
                {"match_threshold": 0.85, "match_low_threshold": 0.90},
            )


if __name__ == "__main__":
    unittest.main()
