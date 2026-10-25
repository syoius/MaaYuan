import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_scan_live import run_scenario


class Reader:
    def __init__(self, roster=("a", "b", "c"), position=0):
        self.roster, self.position = roster, position
        self.disc = "命盘一"
        self.single_disc = False
        self.moves, self.clicks, self.toggles = [], [], []
        self._roi_config = {"clicks": {"disc_entry": "enter", "disc_back": "back"}}

    def _require_page(self, page):
        return page

    def _read_main(self, image):
        return {"operator_id": self.roster[self.position], "name_raw": "observed"}

    def _operator_arrow_visible(self, image, direction):
        target = self.position + (1 if direction == "next" else -1)
        return 0 <= target < len(self.roster)

    def _move_operator(self, direction, main):
        self.moves.append(direction)
        if not self._operator_arrow_visible(None, direction):
            return False
        self.position += 1 if direction == "next" else -1
        return True

    def ocr_text(self, image, roi):
        return self.disc

    def _toggle_disc(self, previous):
        self.toggles.append(previous)
        if self.single_disc:
            return False
        self.disc = "命盘二" if self.disc == "命盘一" else "命盘一"
        return True

    def click(self, point, image, settle_ms=0):
        self.clicks.append(point)

    def collect_current_from_main(self, main):
        return {**main, "stats": {"life": None}}


class LiveScenarioTests(unittest.TestCase):
    def run_case(self, reader, scenario, state=None):
        state = state if state is not None else {}
        run_scenario(reader, scenario, 2, state, lambda: None, str.strip)
        return state

    def test_navigation_learns_account_order_and_restores_any_start(self):
        for size in (1, 2, 4):
            for roster in itertools.permutations("abcd", size):
                for start in range(size):
                    reader = Reader(roster, start)
                    state = self.run_case(reader, "navigation")
                    self.assertEqual(reader.position, start)
                    self.assertTrue(state["restored"])
                    if size == 1:
                        self.assertIsNone(state["scenario_passed"])
                    else:
                        self.assertTrue(state["scenario_passed"])

    def test_unknown_initial_identity_prevents_all_actions(self):
        reader = Reader((None,))
        with self.assertRaisesRegex(RuntimeError, "身份未确认"):
            self.run_case(reader, "navigation")
        self.assertEqual(reader.moves + reader.clicks + reader.toggles, [])

    def test_uncertain_forward_movement_does_not_guess_return_clicks(self):
        reader = Reader()
        reader._move_operator = Mock(side_effect=RuntimeError("click timeout"))
        state = {}
        with self.assertRaisesRegex(RuntimeError, "click timeout"):
            self.run_case(reader, "navigation", state)
        reader._move_operator.assert_called_once()
        self.assertIsNone(state["restored"])

    def test_wrong_return_identity_stops_restoration(self):
        reader = Reader()
        reader._read_main = Mock(side_effect=[{"operator_id": identity} for identity in ("a", "b", "c", "wrong")])
        state = {}
        with self.assertRaisesRegex(RuntimeError, "返回路径身份不符"):
            self.run_case(reader, "navigation", state)
        self.assertEqual(reader.moves, ["next", "next", "previous"])
        self.assertIsNone(state["restored"])

    def test_disc_returns_to_original_selection_and_main(self):
        for label in ("命盘一", "命盘二"):
            reader = Reader()
            reader.disc = label
            state = self.run_case(reader, "disc")
            self.assertEqual(reader.disc, label)
            self.assertTrue(state["restored"])
            self.assertEqual(reader.clicks, ["enter", "back"])
            self.assertEqual(len(reader.toggles), 2)

    def test_single_disc_is_skipped_not_double_switch_pass(self):
        reader = Reader()
        reader.single_disc = True
        state = self.run_case(reader, "disc")
        self.assertTrue(state["restored"])
        self.assertIsNone(state["scenario_passed"])
        self.assertEqual(reader.toggles, ["命盘一"])

    def test_disc_failure_after_switch_still_restores_known_label(self):
        reader = Reader()
        original = reader._toggle_disc

        def fail_once(label):
            result = original(label)
            if len(reader.toggles) == 1:
                raise RuntimeError("switch confirmation failed")
            return result

        reader._toggle_disc = fail_once
        state = {}
        with self.assertRaisesRegex(RuntimeError, "switch confirmation failed"):
            self.run_case(reader, "disc", state)
        self.assertTrue(state["restored"])
        self.assertEqual(reader.disc, "命盘一")

    def test_unreadable_disc_never_triggers_a_blind_toggle(self):
        reader = Reader()
        reader.disc = "unknown"
        state = {}
        with self.assertRaisesRegex(RuntimeError, "标签未确认"):
            self.run_case(reader, "disc", state)
        self.assertEqual(reader.toggles, [])
        self.assertIsNone(state["restored"])

    def test_current_does_not_upload_or_move_and_does_not_claim_accuracy(self):
        reader = Reader()
        state = self.run_case(reader, "current")
        self.assertTrue(state["restored"])
        self.assertIsNone(state["accuracy_passed"])
        self.assertEqual(reader.moves, [])
        self.assertIsNone(state["record"]["stats"]["life"])

    def test_smoke_finds_observed_neighbor_with_two_discs_and_restores_start(self):
        for roster in (("a", "b", "c"), ("c", "b", "a")):
            reader = Reader(roster)
            original_toggle = reader._toggle_disc
            reader._toggle_disc = lambda label: original_toggle(label) if roster[reader.position] == "b" else False
            state = self.run_case(reader, "smoke")
            self.assertEqual(reader.position, 0)
            self.assertTrue(state["restored"])
            self.assertTrue(state["scenario_passed"])
            search = state["checks"]["disc-search"]
            self.assertEqual(search["tested_operator_id"], "b")
            self.assertEqual(search["observed_ids"], list(roster[:2]))
            self.assertEqual(len(search["attempts"]), 2)
            self.assertEqual(reader.disc, "命盘一")

    def test_smoke_does_not_claim_complete_coverage_when_all_discs_are_single(self):
        reader = Reader()
        reader.single_disc = True
        state = self.run_case(reader, "smoke")
        self.assertTrue(state["restored"])
        self.assertIsNone(state["scenario_passed"])
        self.assertEqual(state["skipped_checks"], ["disc-search"])
        self.assertEqual(len(state["checks"]["disc-search"]["attempts"]), 3)
        self.assertEqual(reader.position, 0)

    def test_disc_search_does_not_return_through_an_unconfirmed_disc_page(self):
        reader = Reader()
        reader._toggle_disc = Mock(side_effect=RuntimeError("failed"))
        reader.ocr_text = Mock(side_effect=["命盘一", "unknown"])
        state = {}
        with self.assertRaisesRegex(RuntimeError, "标签未确认"):
            self.run_case(reader, "disc-search", state)
        self.assertIsNone(state["restored"])
        self.assertEqual(reader.moves, [])


if __name__ == "__main__":
    unittest.main()
