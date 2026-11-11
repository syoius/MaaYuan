import json
from itertools import permutations
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.action.agent_info_collector import AgentInfoCollector, CollectorError, _AgentInfoReader
from custom.action.operator_growth_exchange import read_growth_states
from jsonschema import Draft202012Validator


class GrowthFilterTests(unittest.TestCase):
    def reader(self, single=False):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.params = {
            "active_only": True, "scan_single": single, "resume": False,
            "growth_states": {"a": "graduated", "b": "active", "c": "skip", "d": "discarded"},
        }
        reader.max_operators = 10
        reader.context = Mock()
        reader.context.run_recognition.return_value = None
        reader._should_stop = Mock(return_value=False)
        reader.transition_timeout_ms = 1000
        reader.poll_interval_ms = 10
        reader._sleep_checked = Mock()
        reader._require_page = Mock(return_value=np.zeros((1280, 720, 3), dtype=np.uint8))
        reader.game = "代号鸢"
        reader.screenshot = Mock(return_value=np.full((1280, 720, 3), 100, dtype=np.uint8))
        reader._read_main = Mock(side_effect=[
            {"operator_id": key, "name": key, "_name_exact": True} for key in ["a", "b", "c", "d", "a"]
        ])
        reader.collect_current_from_main = Mock(side_effect=lambda main: main)
        reader._publish_checkpoint = Mock(side_effect=lambda records, record: records.append(record))
        reader.click = Mock()
        reader._wait_for_text_change = Mock(return_value="changed")
        reader._seek_first_operator = Mock()
        reader._move_operator = Mock(side_effect=[True, True, True, False])
        return reader

    def test_scan_start_respects_current_option_and_single_mode(self):
        for single in (False, True):
            for start_from_current in (None, False, True):
                with self.subTest(single=single, start_from_current=start_from_current):
                    reader = self.reader(single=single)
                    reader.params["active_only"] = False
                    if start_from_current is not None:
                        reader.params["start_from_current"] = start_from_current
                    self.assertTrue(reader.run())
                    if single or start_from_current:
                        reader._seek_first_operator.assert_not_called()
                    else:
                        reader._seek_first_operator.assert_called_once()
                    self.assertEqual(
                        [call.args[0]["operator_id"] for call in reader.collect_current_from_main.call_args_list],
                        ["a"] if single else ["a", "b", "c", "d"],
                    )
                    self.assertEqual(reader._move_operator.call_count, 0 if single else 4)

    def test_scan_start_option_wires_resources_and_runtime(self):
        root = Path(__file__).resolve().parents[2]
        interface = json.loads((root / "assets/interface.json").read_text())
        collection = next(case for case in interface["option"]["百宝箱-模式"]["cases"]
                          if case["name"] == "采集密探信息")
        self.assertEqual(collection["option"][collection["option"].index("仅扫描一位") + 1], "从当前密探开始")
        option = interface["option"]["从当前密探开始"]
        self.assertEqual(option["type"], "switch")
        self.assertEqual(option["default_case"], "No")
        for resource in ("base", "zh_tw"):
            pipeline = json.loads((root / f"assets/resource/{resource}/pipeline/agent_info_collector.json").read_text())
            self.assertEqual(pipeline["密探采集起点配置"]["attach"], {"start_from_current": False})
        for case in option["cases"]:
            enabled = case["name"] == "Yes"
            node = case["pipeline_override"]["密探采集起点配置"]
            self.assertEqual(node["attach"], {"start_from_current": enabled})
            context = SimpleNamespace(get_node_data=lambda name: node if name == "密探采集起点配置" else None)
            with patch("custom.action.agent_info_collector._AgentInfoReader") as reader:
                reader.return_value.run.return_value = True
                result = AgentInfoCollector().run(context, SimpleNamespace(custom_action_param={"start_from_current": not enabled}))
                self.assertTrue(result.success)
                self.assertEqual(reader.call_args.args[1]["start_from_current"], enabled)

    def test_skips_graduated_skip_and_discarded_but_continues_carousel(self):
        reader = self.reader()
        self.assertTrue(reader.run())
        reader.collect_current_from_main.assert_called_once_with({"operator_id": "b", "name": "b", "_name_exact": True})
        reader._publish_checkpoint.assert_called_once()
        self.assertEqual(reader._move_operator.call_count, 4)

    def test_single_excluded_operator_is_successful_noop(self):
        reader = self.reader(single=True)
        self.assertTrue(reader.run())
        reader.collect_current_from_main.assert_not_called()
        reader._publish_checkpoint.assert_not_called()
        reader.click.assert_not_called()
        reader._seek_first_operator.assert_not_called()

    def test_next_operator_timeout_is_failure_even_with_saved_records(self):
        reader = self.reader()
        reader.params["active_only"] = False
        reader._move_operator.side_effect = RuntimeError("密探翻页未确认")
        self.assertFalse(reader.run())
        reader._publish_checkpoint.assert_called_once()

    def test_full_scan_follows_observed_order_and_account_size(self):
        for size in (1, 2, 4):
            for order in permutations(("a", "b", "c", "d"), size):
                with self.subTest(order=order):
                    reader = self.reader()
                    reader.params["active_only"] = False
                    reader._read_main = Mock(side_effect=[
                        {"operator_id": key, "name": key, "_name_exact": True}
                        for key in order
                    ])
                    reader._move_operator = Mock(side_effect=[True] * (size - 1) + [False])
                    self.assertTrue(reader.run())
                    self.assertEqual(
                        [call.args[0]["operator_id"] for call in reader.collect_current_from_main.call_args_list],
                        list(order),
                    )
                    self.assertEqual(reader._publish_checkpoint.call_count, size)
                    self.assertEqual(reader._move_operator.call_count, size)
                    reader._seek_first_operator.assert_called_once()

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
                    if other_id.endswith("sp"):
                        reader.context.run_recognition.side_effect = lambda node, image: SimpleNamespace(
                            hit=node == "密探信息采集-主界面就绪"
                        )
                        reader._read_main = Mock(side_effect=[main, other])
                    reader._move_operator = Mock(side_effect=lambda *args, **kwargs: events.append("next") or True)

                    self.assertTrue(reader.run())

                    self.assertEqual(events, ([other_id] if not other_id.endswith("sp") else []) + [original_id, "next"])
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

    def test_locked_sp_closes_description_and_verifies_original_identity(self):
        reader = self.reader()
        reader.context.run_recognition.return_value = SimpleNamespace(hit=True)
        reader._run_task = Mock(return_value=True)
        reader._close_prompt_stably = Mock()
        reader._wait_for_operator_form = Mock()
        reader._collect_and_publish = Mock()
        reader._collect_other_form({"operator_id": "char_013_chendeng"}, [])
        reader._run_task.assert_called_once_with("密探信息采集-切换形态")
        reader._close_prompt_stably.assert_called_once_with(
            "密探信息采集-未解锁形态提示", "form_tip_close_failed", "无法关闭未解锁形态提示"
        )
        reader._wait_for_operator_form.assert_called_once_with("char_013_chendeng")
        reader._collect_and_publish.assert_not_called()

    def test_locked_sp_prompt_close_failure_stops_without_collecting_other_form(self):
        reader = self.reader()
        reader.context.run_recognition.return_value = SimpleNamespace(hit=True)
        reader._run_task = Mock(return_value=True)
        reader._close_prompt_stably = Mock(side_effect=CollectorError("无法关闭未解锁形态提示", "form_tip_close_failed"))
        reader._collect_and_publish = Mock()
        with self.assertRaisesRegex(CollectorError, "无法关闭未解锁形态提示") as error:
            reader._collect_other_form({"operator_id": "char_013_chendeng"}, [])
        self.assertEqual(error.exception.reason, "form_tip_close_failed")
        reader._collect_and_publish.assert_not_called()

    def test_locked_sp_prompt_appearing_on_second_poll_is_closed(self):
        reader = self.reader()
        first = reader._require_page.return_value.copy()
        second = reader.screenshot.return_value
        reader.screenshot.side_effect = [first, second]
        reader.context.run_recognition.side_effect = [
            SimpleNamespace(hit=False), SimpleNamespace(hit=True), SimpleNamespace(hit=True),
        ]
        reader._read_main = Mock(return_value={"operator_id": "char_013_chendeng", "_name_exact": True})
        reader._prepare_growth_filter = Mock()
        reader._run_task = Mock(return_value=True)
        reader._close_prompt_stably = Mock()
        reader._wait_for_operator_form = Mock()
        reader._collect_and_publish = Mock()
        with (
            patch("custom.action.agent_info_collector.diagnostic_call") as diagnostic,
            patch("custom.action.agent_info_collector.logger.info") as info,
        ):
            reader._collect_other_form({"operator_id": "char_013_chendeng"}, [])
        self.assertEqual([call.args for call in reader.context.run_recognition.call_args_list], [
            ("密探信息采集-未解锁形态提示", first),
            ("密探信息采集-主界面就绪", first),
            ("密探信息采集-未解锁形态提示", second),
        ])
        reader._run_task.assert_called_once_with("密探信息采集-切换形态")
        reader._close_prompt_stably.assert_called_once_with(
            "密探信息采集-未解锁形态提示", "form_tip_close_failed", "无法关闭未解锁形态提示"
        )
        reader._sleep_checked.assert_called_once_with(reader.poll_interval_ms)
        reader._wait_for_operator_form.assert_called_once_with("char_013_chendeng")
        reader._collect_and_publish.assert_not_called()
        reader._read_main.assert_not_called()
        reader._prepare_growth_filter.assert_not_called()
        diagnostic.assert_not_called()
        self.assertIn("SP 形态未解锁，继续扫描", info.call_args.args[0])

    def test_unchanged_sp_name_retries_three_clicks_then_skips(self):
        reader = self.reader()
        # Exercise the scaled name ROI; changes outside it must not confirm a switch.
        baseline = np.zeros((640, 360, 3), dtype=np.uint8)
        frame = baseline.copy()
        frame[500:] = 255
        reader._require_page.return_value = baseline
        reader.screenshot.return_value = frame
        reader.context.run_recognition.side_effect = lambda node, image: SimpleNamespace(
            hit=node == "密探信息采集-主界面就绪"
        )
        now = [0.0]
        clicks = []
        reader._sleep_checked.side_effect = lambda ms: now.__setitem__(0, now[0] + 0.5)
        reader._run_task = Mock(side_effect=lambda node: clicks.append(now[0]) or True)
        reader._prepare_growth_filter = Mock()
        reader._collect_and_publish = Mock()
        reader._wait_for_operator_form = Mock()
        with (
            patch("custom.action.agent_info_collector.time.monotonic", side_effect=lambda: now[0]),
            patch("custom.action.agent_info_collector.diagnostic_call") as diagnostic,
            patch("custom.action.agent_info_collector.logger.warning") as warning,
        ):
            reader._collect_other_form({"operator_id": "char_023_shizimiao"}, [])
        self.assertEqual(clicks, [0.0, 1.5, 3.0])
        self.assertEqual(now[0], 4.5)
        self.assertTrue(all(call.args == ("密探信息采集-切换形态",) for call in reader._run_task.call_args_list))
        reader._read_main.assert_not_called()
        reader._prepare_growth_filter.assert_not_called()
        reader._collect_and_publish.assert_not_called()
        reader._wait_for_operator_form.assert_not_called()
        diagnostic.assert_not_called()
        self.assertIn("SP 形态切换无响应，跳过", warning.call_args.args[0])

    def test_changed_sp_name_gets_separate_identity_budget_and_collects(self):
        reader = self.reader()
        baseline = reader._require_page.return_value
        changed = baseline.copy()
        changed[221:447, 32:138] = 100
        reader.screenshot.side_effect = [baseline.copy(), changed, changed]
        reader.context.run_recognition.side_effect = lambda node, image: SimpleNamespace(
            hit=node == "密探信息采集-主界面就绪"
        )
        original = {"operator_id": "char_023_shizimiao", "name": "史子眇", "_name_exact": True}
        other = {"operator_id": "char_085_shizimiaosp", "name": "史子眇·之影", "_name_exact": True}
        reader._read_main = Mock(side_effect=[{"operator_id": None, "_name_exact": False}, other])
        reader._prepare_growth_filter = Mock(return_value={"operator_id": None})
        reader._run_task = Mock(return_value=True)
        reader._wait_for_operator_form = Mock(return_value=original)
        reader._collect_and_publish = Mock()
        now = [0.0]
        reader._sleep_checked.side_effect = lambda ms: now.__setitem__(0, now[0] + 0.9)
        with patch("custom.action.agent_info_collector.time.monotonic", side_effect=lambda: now[0]):
            reader._collect_other_form(original, [])
        reader._prepare_growth_filter.assert_called_once()
        reader._collect_and_publish.assert_called_once_with(other, [])
        reader._wait_for_operator_form.assert_called_once_with(original["operator_id"])
        self.assertEqual(reader._run_task.call_count, 2)

    def test_changed_sp_name_confirmed_as_original_skips(self):
        for exact in (False, True):
            with self.subTest(exact=exact):
                reader = self.reader()
                original = {"operator_id": "char_023_shizimiao", "name": "史子眇", "_name_exact": True}
                reader.context.run_recognition.side_effect = lambda node, image: SimpleNamespace(
                    hit=node == "密探信息采集-主界面就绪"
                )
                reader._read_main = Mock(return_value=dict(original, _name_exact=exact))
                reader._prepare_growth_filter = Mock(return_value=original)
                reader._run_task = Mock(return_value=True)
                reader._collect_and_publish = Mock()
                with patch("custom.action.agent_info_collector.diagnostic_call") as diagnostic:
                    reader._collect_other_form(original, [])
                self.assertEqual(reader._prepare_growth_filter.call_count, 0 if exact else 1)
                reader._collect_and_publish.assert_not_called()
                reader._run_task.assert_called_once_with("密探信息采集-切换形态")
                diagnostic.assert_not_called()

    def test_locked_sp_dismissal_does_not_skip_identity_confirmation_failure(self):
        reader = self.reader()
        reader.context.run_recognition.return_value = SimpleNamespace(hit=True)
        reader._run_task = Mock(return_value=True)
        reader._close_prompt_stably = Mock()
        reader._wait_for_operator_form = Mock(side_effect=RuntimeError("身份未确认"))
        with self.assertRaisesRegex(RuntimeError, "身份未确认"):
            reader._collect_other_form({"operator_id": "char_013_chendeng"}, [])

    def test_form_prompt_recognition_error_still_confirms_and_restores_form(self):
        reader = self.reader()
        main = {"operator_id": "char_013_chendeng"}
        other = {"operator_id": "char_084_chendengsp", "name": "陈登·黍王", "_name_exact": True}
        def recognize(node, image):
            if node == "密探信息采集-未解锁形态提示":
                raise RuntimeError("recognition failed")
            return SimpleNamespace(hit=True)
        reader.context.run_recognition.side_effect = recognize
        reader._read_main = Mock(return_value=other)
        reader._run_task = Mock(return_value=True)
        reader._should_stop = Mock(return_value=False)
        reader._wait_for_operator_form = Mock(return_value=main)
        reader._collect_and_publish = Mock()
        reader._collect_other_form(main, [])
        reader._collect_and_publish.assert_called_once_with(other, [])
        self.assertEqual(reader._run_task.call_count, 2)
        reader._wait_for_operator_form.assert_called_once_with(main["operator_id"])

    def test_cancel_during_form_prompt_recognition_does_not_click_again(self):
        reader = self.reader()
        reader.context.run_recognition.side_effect = InterruptedError("stopped")
        reader._run_task = Mock(return_value=True)
        reader._wait_for_operator_form = Mock()
        with self.assertRaises(InterruptedError):
            reader._collect_other_form({"operator_id": "char_013_chendeng"}, [])
        reader._run_task.assert_called_once()
        reader._wait_for_operator_form.assert_not_called()

    def test_sp_poll_timeout_rejects_missing_page_and_wrong_identity(self):
        for ready, message, reason in (
            (False, "等待 main 页面超时", "page_timeout"),
            (True, "身份未确认", "form_identity_unconfirmed"),
        ):
            with self.subTest(ready=ready):
                reader = self.reader()
                reader.transition_timeout_ms = 0
                reader.context.run_recognition.side_effect = [
                    SimpleNamespace(hit=False), SimpleNamespace(hit=ready),
                ]
                reader._read_main = Mock(return_value={"operator_id": "wrong", "name": "wrong", "_name_exact": True})
                reader._run_task = Mock(return_value=True)
                reader._collect_and_publish = Mock()
                with self.assertRaisesRegex(RuntimeError, message) as error:
                    reader._collect_other_form({"operator_id": "char_013_chendeng"}, [])
                self.assertEqual(error.exception.reason, reason)
                reader._collect_and_publish.assert_not_called()

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
        self.assertFalse(reader._should_collect({"operator_id": "d"}))
        reader.params["active_only"] = False
        self.assertTrue(reader._should_collect({"operator_id": "d"}))

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
            {"operator_id": "d", "growth_state": "discarded"},
        ]}}
        response = Mock()
        response.read.return_value = json.dumps(payload).encode()
        with patch("custom.action.operator_growth_exchange.urllib_request.urlopen") as open_url:
            open_url.return_value.__enter__.return_value = response
            self.assertEqual(read_growth_states("https://example.test", "secret", "acc1"), {"a": "graduated", "b": "active", "d": "discarded"})
            request = open_url.call_args.args[0]
            self.assertEqual(request.full_url, "https://example.test/open-api/operator/annotations")
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.get_header("Authorization"), "Bearer secret")
            for invalid in [None, {"data": []}, {"data": {"account_id": "other", "items": []}},
                            {"data": {"account_id": "acc1", "items": [{"operator_id": "a", "growth_state": "unknown"}]}}]:
                response.read.return_value = json.dumps(invalid).encode()
                with self.assertRaises(ValueError):
                    read_growth_states("https://example.test", "secret", "acc1")

    def test_discarded_annotation_schema_extension_for_listed_and_full(self):
        schema = json.loads((Path(__file__).resolve().parents[2] / "docs/schemas/operator-growth-exchange-v3.schema.json").read_text())
        self.assertEqual(schema["$defs"]["growthState"]["enum"], ["active", "graduated", "skip", "discarded"])
        for scope in ("listed", "full"):
            with self.subTest(scope=scope):
                document = {
                    "format": "myshare-operator-exchange", "version": 3,
                    "exported_at": "2026-10-03T00:00:00Z",
                    "producer": {"platform": "test", "version": "1"},
                    "accounts": [{"id": "test-account", "name": "测试账号"}],
                    "records": [{
                        "account_id": "test-account", "record_id": "test:discarded",
                        "record_type": "operator_annotation_snapshot", "game": "如鸢",
                        "effective_at": "2026-10-03T00:00:00Z", "snapshot_scope": scope,
                        "source_kind": "backup", "entries": [{
                            "operator_id": "op", "growth_state": "discarded",
                            "favorite": True, "note": "保留备注", "targets": None,
                        }],
                    }],
                }
                Draft202012Validator(schema).validate(document)

    def test_http_error_has_actionable_message_without_token(self):
        with patch("custom.action.operator_growth_exchange.urllib_request.urlopen", side_effect=HTTPError("https://example.test", 403, "Forbidden", {}, None)):
            with self.assertRaisesRegex(RuntimeError, "operator:read") as error:
                read_growth_states("https://example.test", "secret", "acc1")
            self.assertNotIn("secret", str(error.exception))


if __name__ == "__main__":
    unittest.main()
