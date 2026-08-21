import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    _resolve_inventory_report_context,
    _snapshot_record_options,
    automatic_swipe,
    find_row_overlap,
    overlap_candidate_scores,
)
from custom.reco.agent_item import (  # noqa: E402
    CountDigitCandidate,
    _digit_candidates_to_result,
    _match_rejection_reason,
    recognize_count,
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
    def test_fallback_runs_only_after_primary_failure(self, recognize_digits):
        recognize_digits.side_effect = [
            (None, 0.0, "", (1, 2, 3, 4)),
            (54, 0.9, "54", (1, 2, 3, 4)),
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
        self.assertEqual(recognize_digits.call_count, 2)
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
    def test_primary_success_skips_fallback(self, recognize_digits):
        recognize_digits.return_value = (623, 0.8, "623", (1, 2, 3, 4))
        image = np.zeros((1, 1, 3), dtype=np.uint8)

        result = recognize_count(
            None,
            image,
            (0.0, 0.0),
            {"count_binary_threshold": 165},
        )

        self.assertEqual(result[0], 623)
        recognize_digits.assert_called_once()

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
