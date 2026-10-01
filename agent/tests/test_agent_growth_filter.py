import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.action.agent_info_collector import AgentInfoCollector, _AgentInfoReader
from custom.action.operator_growth_exchange import read_growth_states


class GrowthFilterTests(unittest.TestCase):
    def reader(self, single=False):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.params = {
            "active_only": True, "scan_single": single, "resume": False,
            "growth_states": {"a": "graduated", "b": "active", "c": "skip"},
        }
        reader.max_operators = 10
        reader.context = Mock()
        reader._require_page = Mock()
        reader.game = "代号鸢"
        reader.screenshot = Mock(return_value=object())
        reader._read_main = Mock(side_effect=[
            {"operator_id": key, "name": key, "_name_exact": True} for key in ["a", "b", "c", "a"]
        ])
        reader.collect_current_from_main = Mock(side_effect=lambda main: main)
        reader._publish_checkpoint = Mock(side_effect=lambda records, record: records.append(record))
        reader.click = Mock()
        reader._wait_for_text_change = Mock(return_value="changed")
        return reader

    def test_skips_graduated_and_skip_but_continues_carousel(self):
        reader = self.reader()
        self.assertTrue(reader.run())
        reader.collect_current_from_main.assert_called_once_with({"operator_id": "b", "name": "b", "_name_exact": True})
        reader._publish_checkpoint.assert_called_once()
        self.assertEqual(reader.click.call_count, 3)

    def test_single_excluded_operator_is_successful_noop(self):
        reader = self.reader(single=True)
        self.assertTrue(reader.run())
        reader.collect_current_from_main.assert_not_called()
        reader._publish_checkpoint.assert_not_called()
        reader.click.assert_not_called()

    def test_full_scan_collects_both_forms_and_restores_before_next(self):
        pairs = (
            ("char_084_chendengsp", "char_013_chendeng"),
            ("char_085_shizimiaosp", "char_023_shizimiao"),
        )
        for pair in pairs:
            for original_id, other_id in (pair, tuple(reversed(pair))):
                with self.subTest(original_id=original_id):
                    reader = self.reader()
                    reader.params["active_only"] = False
                    reader.max_operators = 1
                    main = {"operator_id": original_id, "name": original_id, "_name_exact": True}
                    other = {"operator_id": other_id, "name": other_id, "_name_exact": True}
                    reader._read_main = Mock(return_value=main)
                    events = []
                    reader._run_task = Mock(return_value=True)
                    reader._should_stop = Mock(return_value=False)
                    reader._wait_for_operator_form = Mock(side_effect=lambda operator_id: (
                        events.append(operator_id) or (other if operator_id == other_id else main)
                    ))
                    reader.click = Mock(side_effect=lambda *args, **kwargs: events.append("next"))

                    self.assertTrue(reader.run())

                    self.assertEqual(events, [other_id, original_id, "next"])
                    self.assertEqual(
                        [call.args[1]["operator_id"] for call in reader._publish_checkpoint.call_args_list],
                        [original_id, other_id],
                    )
                    self.assertEqual(reader._run_task.call_count, 2)

    def test_form_filter_is_independent_when_original_sp_is_excluded(self):
        reader = self.reader()
        reader.max_operators = 1
        reader.params["growth_states"] = {"char_084_chendengsp": "skip"}
        main = {"operator_id": "char_084_chendengsp", "name": "陈登·黍王", "_name_exact": True}
        other = {"operator_id": "char_013_chendeng", "name": "陈登", "_name_exact": True}
        reader._read_main = Mock(return_value=main)
        reader._run_task = Mock(return_value=True)
        reader._should_stop = Mock(return_value=False)
        reader._wait_for_operator_form = Mock(side_effect=[other, main])
        self.assertTrue(reader.run())
        reader.collect_current_from_main.assert_called_once_with(other)
        reader._publish_checkpoint.assert_called_once()

    def test_other_form_restores_after_collection_error(self):
        reader = self.reader()
        main = {"operator_id": "char_084_chendengsp"}
        other = {"operator_id": "char_013_chendeng"}
        reader._run_task = Mock(return_value=True)
        reader._should_stop = Mock(return_value=False)
        reader._wait_for_operator_form = Mock(side_effect=[other, main])
        reader._collect_and_publish = Mock(side_effect=RuntimeError("capture failed"))
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            reader._collect_other_form(main, [])
        self.assertEqual(reader._run_task.call_count, 2)
        self.assertEqual(reader._wait_for_operator_form.call_args.args[0], "char_084_chendengsp")

    def test_form_restore_failure_stops_before_next_operator(self):
        reader = self.reader()
        reader.params["active_only"] = False
        main = {"operator_id": "char_084_chendengsp", "name": "陈登·黍王", "_name_exact": True}
        reader._read_main = Mock(return_value=main)
        reader._run_task = Mock(side_effect=[True, False])
        reader._should_stop = Mock(return_value=False)
        reader._wait_for_operator_form = Mock(return_value={"operator_id": "char_013_chendeng"})
        with self.assertRaisesRegex(RuntimeError, "恢复原密探形态"):
            reader.run()
        reader.click.assert_not_called()

    def test_single_sp_scan_does_not_switch_form(self):
        reader = self.reader(single=True)
        main = {"operator_id": "char_084_chendengsp", "name": "陈登·黍王", "_name_exact": True}
        reader._read_main = Mock(return_value=main)
        reader._run_task = Mock()
        self.assertTrue(reader.run())
        reader._run_task.assert_not_called()
        reader._publish_checkpoint.assert_called_once()

    def test_wait_for_form_rejects_wrong_identity(self):
        reader = self.reader()
        reader.transition_timeout_ms = 0
        reader._sleep_checked = Mock()
        reader._read_main = Mock(return_value={"operator_id": "wrong", "_name_exact": True})
        with self.assertRaisesRegex(RuntimeError, "身份未确认"):
            reader._wait_for_operator_form("char_013_chendeng")

    def test_cancellation_does_not_restore_by_clicking(self):
        reader = self.reader()
        reader._run_task = Mock(return_value=True)
        reader._should_stop = Mock(return_value=True)
        reader._wait_for_operator_form = Mock(side_effect=InterruptedError("stopped"))
        with self.assertRaises(InterruptedError):
            reader._collect_other_form({"operator_id": "char_084_chendengsp"}, [])
        reader._run_task.assert_called_once()

    def test_default_active_and_unconfirmed_identity(self):
        reader = self.reader()
        self.assertTrue(reader._should_collect({"operator_id": "unmarked"}))
        self.assertFalse(reader._should_collect({"operator_id": None}))

    def test_corrected_identity_is_checked_before_publish(self):
        reader = self.reader(single=True)
        reader._read_main = Mock(return_value={"operator_id": "b", "name": "b", "_name_exact": True})
        reader.collect_current_from_main = Mock(return_value={"operator_id": "a", "name": "a"})
        self.assertTrue(reader.run())
        reader._publish_checkpoint.assert_not_called()

    def test_query_failure_stops_before_game_actions(self):
        context = Mock()
        nodes = {
            "密探采集上报配置": {"attach": {"upload": True}},
            "密探采集养成筛选配置": {"attach": {"active_only": True}},
        }
        context.get_node_data.side_effect = lambda key: nodes.get(key)
        with (
            patch("custom.action.agent_info_collector.read_upload_settings", return_value=SimpleNamespace(base_url="https://example.test", token="secret")),
            patch("custom.action.agent_info_collector.get_bound_account", return_value=SimpleNamespace(id="acc1", name="test")),
            patch("custom.action.agent_info_collector.read_growth_states", side_effect=RuntimeError("unavailable")),
            patch("custom.action.agent_info_collector._AgentInfoReader") as reader,
        ):
            self.assertFalse(AgentInfoCollector().run(context, SimpleNamespace(custom_action_param={})).success)
            reader.assert_not_called()

    def test_uncertain_name_checks_discs_before_details_and_reuses_them(self):
        for initial_id in (None, "a"):
            with self.subTest(initial_id=initial_id):
                reader = self.reader(single=True)
                main = {"operator_id": initial_id, "name": "partial", "name_raw": "partial", "_name_exact": False}
                reader._read_main = Mock(return_value=main)
                reader.collect_current_from_main = _AgentInfoReader.collect_current_from_main.__get__(reader)
                events = []
                configs = [{"label": "one", "slots": []}]
                reader._collect_discs = Mock(side_effect=lambda probe: events.append("discs") or configs)
                reader._confirm_operator_from_discs = Mock(return_value={"id": "b", "name": "b"})
                reader._resolve_locked_disc_names = Mock()
                reader._collect_details = Mock(side_effect=lambda record: events.append("details") or {})
                reader._collect_huaji = Mock(side_effect=lambda record: events.append("huaji") or {})

                self.assertTrue(reader.run())
                self.assertEqual(events, ["discs", "details", "huaji"])
                reader._collect_discs.assert_called_once()
                self.assertIsNone(reader._collect_discs.call_args.args[0]["operator_id"])
                record = reader._publish_checkpoint.call_args.args[1]
                self.assertEqual(record["operator_id"], "b")
                self.assertIs(record["disc_configs"], configs)
                self.assertEqual(record["collection_debug"]["name_match_operator_id"], initial_id)
                self.assertNotIn("_prefetched_discs", record)
                reader.click.assert_not_called()

    def test_uncertain_name_skips_remaining_collection_after_disc_check(self):
        for operator in (None, {"id": "a", "name": "a"}, {"id": "c", "name": "c"}):
            with self.subTest(operator=operator):
                reader = self.reader(single=True)
                reader._read_main = Mock(return_value={"operator_id": "b", "name": "partial", "_name_exact": False})
                reader._collect_discs = Mock(return_value=[])
                reader._confirm_operator_from_discs = Mock(return_value=operator)
                reader._resolve_locked_disc_names = Mock()
                self.assertTrue(reader.run())
                reader._collect_discs.assert_called_once()
                reader.collect_current_from_main.assert_not_called()
                reader._publish_checkpoint.assert_not_called()
                reader.click.assert_not_called()

    def test_exact_excluded_name_does_not_read_discs(self):
        reader = self.reader(single=True)
        reader._collect_discs = Mock()
        self.assertTrue(reader.run())
        reader._collect_discs.assert_not_called()

    def test_annotation_response_and_request_contract(self):
        payload = {"data": {"account_id": "acc1", "items": [
            {"operator_id": "a", "growth_state": "graduated"},
            {"operator_id": "b", "growth_state": "active"},
        ]}}
        response = Mock()
        response.read.return_value = json.dumps(payload).encode()
        with patch("custom.action.operator_growth_exchange.urllib_request.urlopen") as open_url:
            open_url.return_value.__enter__.return_value = response
            self.assertEqual(read_growth_states("https://example.test", "secret", "acc1"), {"a": "graduated", "b": "active"})
            request = open_url.call_args.args[0]
            self.assertEqual(request.full_url, "https://example.test/open-api/operator/annotations")
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.get_header("Authorization"), "Bearer secret")
            for invalid in [None, {"data": []}, {"data": {"account_id": "other", "items": []}},
                            {"data": {"account_id": "acc1", "items": [{"operator_id": "a", "growth_state": "unknown"}]}}]:
                response.read.return_value = json.dumps(invalid).encode()
                with self.assertRaises(ValueError):
                    read_growth_states("https://example.test", "secret", "acc1")

    def test_http_error_has_actionable_message_without_token(self):
        with patch("custom.action.operator_growth_exchange.urllib_request.urlopen", side_effect=HTTPError("https://example.test", 403, "Forbidden", {}, None)):
            with self.assertRaisesRegex(RuntimeError, "operator:read") as error:
                read_growth_states("https://example.test", "secret", "acc1")
            self.assertNotIn("secret", str(error.exception))


if __name__ == "__main__":
    unittest.main()
