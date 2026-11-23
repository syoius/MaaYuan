"""Frame lifetime, OCR disagreement and navigation failure regressions."""
import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom.action.agent_info_collector import CollectorError, _AgentInfoReader, MAIN_STAT_ROIS, OCR_NODE, PAGE_NODES


def result(text):
    return SimpleNamespace(filtered_results=[SimpleNamespace(text=text)])


class ScanReliabilityTests(unittest.TestCase):
    def test_prompt_recognizes_dialog_body_in_both_languages_and_only_cancels(self):
        root = Path(__file__).resolve().parents[2]
        for language in ("base", "zh_tw"):
            data = json.loads((root / f"assets/resource/{language}/pipeline/agent_info_collector.json").read_text())
            node = data["密探信息采集-命盘单套提示"]
            pattern = node["recognition"]["param"]["expected"]
            self.assertRegex("解锁命盘二需要安期信赖值达到8000", pattern)
            self.assertRegex("解鎖命盤二需要安期信賴值達到8000", pattern)
            self.assertIsNone(re.search(pattern, "定军山生效中"))
            x, y, w, h = node["action"]["param"]["target"]
            self.assertTrue(90 <= x < x + w <= 330 and 740 <= y < y + h <= 800)

    def reader(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader._ensure_running = Mock()
        reader._sleep_checked = Mock()
        reader.current_page = "main"
        reader._page_image = None
        reader.transition_timeout_ms = 1000
        reader.poll_interval_ms = 100
        reader.context = SimpleNamespace(tasker=SimpleNamespace(controller=Mock()), run_task=Mock())
        reader.operators = {}
        reader._horizontal_name_readings = Mock(return_value=[])
        reader.screenshot = Mock(return_value=np.zeros((1280, 720, 3), dtype=np.uint8))
        return reader

    def test_reuses_frame_until_click_or_task_even_if_task_fails(self):
        reader = self.reader()
        frame = reader._require_page("main")
        self.assertIs(reader._require_page("main"), frame)
        self.assertEqual(reader.screenshot.call_count, 1)
        reader.click((10, 10), frame, settle_ms=0)
        reader._require_page("main")
        self.assertEqual(reader.screenshot.call_count, 2)
        reader.context.run_task.side_effect = RuntimeError("partial action failure")
        self.assertIsNone(reader._run_task("test"))
        reader._require_page("main")
        self.assertEqual(reader.screenshot.call_count, 3)

    def test_new_page_does_not_reuse_old_page_frame(self):
        reader = self.reader()
        reader._require_page("main")
        reader._run_task = Mock(return_value=True)
        reader._require_page("detail")
        self.assertEqual(reader.screenshot.call_count, 2)

    def test_page_click_retries_only_on_source_and_returns_required_page_frame(self):
        reader = self.reader()
        reader.click = Mock()
        settled_frame = object()
        def require_page(page):
            self.assertIsNone(reader.current_page)
            self.assertIsNone(reader._page_image)
            return settled_frame
        reader._require_page = Mock(side_effect=require_page)
        now = [0.0]
        reader._sleep_checked.side_effect = lambda ms: now.__setitem__(0, now[0] + ms / 1000)
        def recognize(node, frame):
            return SimpleNamespace(hit=node == PAGE_NODES["detail"] or reader.click.call_count == 2)
        reader.context.run_recognition = Mock(side_effect=recognize)
        frame = reader.screenshot.return_value
        reader._page_image = frame
        with patch("custom.action.agent_info_collector.time.monotonic", side_effect=lambda: now[0]):
            self.assertIs(reader._click_until_page("detail_close", frame, "main", "detail"), settled_frame)
        self.assertEqual(reader.click.call_count, 2)
        self.assertEqual(reader.click.call_args_list[0], reader.click.call_args_list[1])
        reader._require_page.assert_called_once_with("main")

    def test_collect_details_reads_oddities_from_required_page_frame(self):
        reader = self.reader()
        reader.click = Mock()
        reader._page_image = object()
        transition_detail, settled_detail, transition_main, settled_main = (object() for _ in range(4))
        reader.screenshot.side_effect = [transition_detail, settled_detail, transition_main, settled_main]
        reader.context.run_recognition = Mock(return_value=SimpleNamespace(hit=True))
        reader._run_task = Mock(return_value=True)
        reader._require_page = Mock(wraps=reader._require_page)
        reader._read_oddities = Mock(return_value={"攻击力": {"current": 1, "max": 2}})

        self.assertEqual(reader._collect_details({}), reader._read_oddities.return_value)

        reader._read_oddities.assert_called_once_with(settled_detail)
        self.assertEqual([call.args[0] for call in reader._require_page.call_args_list], ["main", "detail", "main"])
        self.assertEqual([call.args[0] for call in reader._run_task.call_args_list], [PAGE_NODES["detail"], PAGE_NODES["main"]])
        self.assertIs(reader._page_image, settled_main)

    def test_page_click_falls_back_after_three_attempts_or_unknown_source(self):
        for source_visible, attempts in ((True, 3), (False, 1)):
            with self.subTest(source_visible=source_visible):
                reader = self.reader()
                reader.click = Mock()
                reader._require_page = Mock(side_effect=CollectorError("timeout", "page_timeout"))
                reader.context.run_recognition = Mock(side_effect=lambda node, frame: SimpleNamespace(
                    hit=source_visible and node == PAGE_NODES["main"]))
                now = [0.0]
                reader._sleep_checked.side_effect = lambda ms: now.__setitem__(0, now[0] + ms / 1000)
                with patch("custom.action.agent_info_collector.time.monotonic", side_effect=lambda: now[0]):
                    with self.assertRaises(CollectorError) as caught:
                        reader._click_until_page("detail_entry", reader.screenshot.return_value, "detail", "main")
                self.assertEqual(caught.exception.reason, "page_timeout")
                self.assertEqual(reader.click.call_count, attempts)
                reader._require_page.assert_called_once_with("detail")

    def test_page_click_cancellation_prevents_retry(self):
        reader = self.reader()
        reader.click = Mock()
        reader._sleep_checked.side_effect = InterruptedError("stopped")
        with self.assertRaises(InterruptedError):
            reader._click_until_page("detail_entry", reader.screenshot.return_value, "detail", "main")
        reader.click.assert_called_once()

    def test_name_change_requires_two_consecutive_nonempty_matches(self):
        reader = self.reader()
        reader.ocr_text = Mock(side_effect=["旧", "动", "", "新", "新"])
        self.assertEqual(reader._wait_for_text_change((0, 0, 100, 100), "旧", "姓名"), "新")
        self.assertEqual(reader.screenshot.call_count, 5)
        self.assertIs(reader._require_page("main"), reader.screenshot.return_value)
        self.assertEqual(reader.screenshot.call_count, 5)

    def test_numeric_detector_and_signed_cultivation(self):
        for key, text, expected in (("life", "20757", 20757), ("cultivation", "+17", 17), ("cultivation", "0", 0)):
            with self.subTest(key=key):
                reader = self.reader()
                reader.ocr_text = Mock(return_value=text)
                reader.ocr = Mock(return_value=result(text))
                self.assertEqual(reader._read_stat(reader.screenshot(), key, MAIN_STAT_ROIS[key]), expected)
                reader.ocr.assert_not_called()

    def test_cultivation_detector_is_not_vetoed_by_unreliable_direct_ocr(self):
        for detected, direct, expected in (("+1", "+U", 1), ("+7", "十乙", 7), ("+11", "+1", 11), ("十1", "十U", 1), ("十17", "", 17)):
            with self.subTest(detected=detected):
                reader = self.reader()
                reader.ocr_text = Mock(return_value=detected)
                reader.ocr = Mock(return_value=result(direct))
                self.assertEqual(reader._read_stat(reader.screenshot(), "cultivation", MAIN_STAT_ROIS["cultivation"]), expected)
                reader.ocr.assert_not_called()

    def test_cultivation_does_not_extract_digits_from_garbage(self):
        for detected in ("+U", "十乙", "修为+7", "-1", "1O"):
            with self.subTest(detected=detected):
                reader = self.reader()
                reader.ocr_text = Mock(return_value=detected)
                reader.ocr = Mock(return_value=result(detected))
                self.assertIsNone(reader._read_stat(reader.screenshot(), "cultivation", MAIN_STAT_ROIS["cultivation"]))

    def test_empty_vertical_name_can_confirm_transition_with_unique_horizontal_name(self):
        reader = self.reader()
        reader.operators = {"史子眇": {"id": "char_023_shizimiao", "name": "史子眇"}}
        reader.ocr_text = Mock(return_value="")
        reader._horizontal_name_readings.return_value = ["史子眇", "史子眇"]
        self.assertEqual(reader._wait_for_text_change((32, 221, 106, 226), "程显", "下一位密探"), "史子眇")
        self.assertEqual(reader.screenshot.call_count, 2)

    def test_conflicting_horizontal_names_do_not_confirm_transition(self):
        for raw in ("", "4"):
            with self.subTest(raw=raw):
                reader = self.reader()
                reader.operators = {name: {"name": name} for name in ("陈登", "史子眇")}
                reader.ocr_text = Mock(return_value=raw)
                reader._horizontal_name_readings.return_value = ["陈登", "史子眇"]
                with patch("custom.action.agent_info_collector.time.monotonic", side_effect=[0, .2, 2]):
                    self.assertIsNone(reader._wait_for_text_change((32, 221, 106, 226), "程显", "下一位密探"))

    def test_same_operator_is_not_a_transition_when_name_reading_method_changes(self):
        for raw, name in (("王粲觉醒", "王粲"), ("夏候静", "夏侯惇"), ("陈登·黍王觉醒", "陈登黍王")):
            with self.subTest(raw=raw):
                reader = self.reader()
                reader.operators = {name: {"name": name}}
                reader.ocr_text = Mock(return_value="")
                reader._horizontal_name_readings.return_value = [name, name]
                with patch("custom.action.agent_info_collector.time.monotonic", side_effect=[0, .2, 2]):
                    self.assertIsNone(reader._wait_for_text_change((32, 221, 106, 226), raw, "下一位密探", previous_readings=(name,)))

    def test_unmatched_horizontal_reading_can_signal_movement_but_old_readings_cannot(self):
        reader = self.reader()
        reader.ocr_text = Mock(return_value="")
        reader._horizontal_name_readings.return_value = ["手吵", "史子吵"]
        self.assertEqual(reader._wait_for_text_change((32, 221, 106, 226), "程显", "下一位密探", previous_readings=("程昱",)), "史子吵")
        with patch("custom.action.agent_info_collector.time.monotonic", side_effect=[0, .2, 2]):
            self.assertIsNone(reader._wait_for_text_change((32, 221, 106, 226), "", "下一位密探", previous_readings=("史子眇", "手吵", "史子吵")))

    def test_unstable_vertical_noise_uses_stable_horizontal_identity(self):
        reader = self.reader()
        reader.operators = {"甘宁": {"name": "甘宁"}}
        reader.ocr_text = Mock(side_effect=["4", "", "二", "H"])
        reader._horizontal_name_readings.return_value = ["甘宁", "未匹配"]
        self.assertEqual(reader._wait_for_text_change((32, 221, 106, 226), "贾诩", "下一位密探"), "甘宁")
        self.assertEqual(reader.screenshot.call_count, 2)

    def test_transition_uses_full_charset_when_default_name_readings_are_unstable(self):
        reader = self.reader()
        reader.operators = {"干吉": {"name": "干吉"}}
        reader.ocr_text = Mock(side_effect=["平吉", "十吉"])
        reader._horizontal_name_readings = Mock(side_effect=lambda image, full_charset=False:
            ["干吉", "干吉"] if full_charset else ["天", ""])
        self.assertEqual(reader._wait_for_text_change((32, 221, 106, 226), "葛洪", "下一位密探"), "干吉")
        self.assertEqual(reader.screenshot.call_count, 2)

    def test_unmatched_full_charset_noise_does_not_hide_stable_default_reading(self):
        reader = self.reader()
        reader.ocr_text = Mock(side_effect=["春", "长"])
        full_readings = iter((["史子彤", "史吵", "小"], ["子彤", "吵"]))
        reader._horizontal_name_readings = Mock(side_effect=lambda image, full_charset=False:
            next(full_readings) if full_charset else ["子痧", "史子渺"])
        self.assertEqual(reader._wait_for_text_change((32, 221, 106, 226), "程显", "下一位密探"), "史子渺")
        self.assertEqual(reader.screenshot.call_count, 2)

    def test_numeric_detector_is_not_vetoed_by_direct_suffix_noise(self):
        reader = self.reader()
        reader.ocr_text = Mock(return_value="1424")
        reader.ocr = Mock(return_value=result("1424°"))
        self.assertEqual(reader._read_stat(reader.screenshot(), "life", MAIN_STAT_ROIS["life"]), 1424)
        reader.ocr.assert_not_called()

    def test_level_removes_background_before_first_read(self):
        reader = self.reader()
        frame = np.full((1280, 720, 3), 120, dtype=np.uint8)
        frame[1090:1130, 100:110] = 240
        reader.ocr_text = Mock(return_value="2")
        reader.ocr = Mock(return_value=result("7"))
        self.assertEqual(reader._read_stat(frame, "level", MAIN_STAT_ROIS["level"]), 2)
        cleaned = reader.ocr_text.call_args.args[0]
        self.assertEqual(set(np.unique(cleaned)), {0, 255})
        self.assertTrue(np.all(cleaned[0] == 255))
        reader.ocr_text.assert_called_once()
        reader.ocr.assert_not_called()

    def test_level_without_visible_digits_stays_missing(self):
        reader = self.reader()
        reader.ocr_text = Mock(return_value="2")
        self.assertIsNone(reader._read_stat(reader.screenshot(), "level", MAIN_STAT_ROIS["level"]))
        reader.ocr_text.assert_not_called()

    def test_numeric_garbage_or_nonpositive_value_is_missing(self):
        frame = np.full((1280, 720, 3), 240, dtype=np.uint8)
        for key in ("level", "life", "attack"):
            for detected in ("1O0", "等级100", "0", "-17", "1424°", "十1"):
                with self.subTest(key=key, text=detected):
                    reader = self.reader()
                    reader.ocr_text = Mock(return_value=detected)
                    self.assertIsNone(reader._read_stat(frame, key, MAIN_STAT_ROIS[key]))

    def test_only_rec_uses_pixel_crop_and_resets_mode_for_detector(self):
        reader = self.reader()
        reader.context.run_recognition = Mock(return_value=result("100"))
        frame = np.zeros((2560, 1440, 3), np.uint8)
        reader.ocr(frame, MAIN_STAT_ROIS["level"], only_rec=True)
        _, crop, override = reader.context.run_recognition.call_args.args
        self.assertEqual(crop.shape[:2], (130, 186))
        self.assertEqual(override[OCR_NODE]["recognition"]["param"]["roi"], [0, 0, 186, 130])
        reader.ocr(frame, MAIN_STAT_ROIS["level"])
        self.assertFalse(reader.context.run_recognition.call_args.args[2][OCR_NODE]["recognition"]["param"]["only_rec"])

    def disc_reader(self):
        reader = self.reader()
        reader.current_page = "disc"
        reader._page_image = reader.screenshot.return_value
        reader.context.run_recognition = Mock(return_value=SimpleNamespace(hit=False))
        return reader

    def test_disc_change_requires_stable_new_label_and_reuses_frame(self):
        reader = self.disc_reader()
        reader.ocr_text = Mock(side_effect=["命盘一", "命盘二", "", "命盘二", "命盘二"])
        self.assertTrue(reader._toggle_disc("命盘一"))
        reader._require_page("disc")
        self.assertEqual(reader.screenshot.call_count, 5)
        reader.context.tasker.controller.post_click.assert_called_once()

    def test_single_disc_closes_prompt_and_rechecks_page(self):
        for interval, recognized_frames in ((300, 3), (600, 2)):
            with self.subTest(interval=interval):
                reader = self.disc_reader()
                reader.poll_interval_ms = interval
                clicks = []
                reader.context.run_recognition.side_effect = lambda node, image: SimpleNamespace(
                    hit=node == "密探信息采集-命盘界面就绪" or not reader._run_task.called
                )
                with self.prompt_clock(reader) as clock:
                    reader._run_task = Mock(side_effect=lambda node: clicks.append(
                        (clock.return_value, reader.context.run_recognition.call_count)
                    ) or True)
                    self.assertFalse(reader._toggle_disc("命盘一"))
                self.assertEqual(clicks, [(0.6, recognized_frames)])
                reader._run_task.assert_called_once_with("密探信息采集-命盘单套提示")
                self.assertEqual(reader.context.run_recognition.call_args.args[0], "密探信息采集-命盘界面就绪")
                self.assertEqual(reader.screenshot.call_count, recognized_frames + 3)

    def prompt_clock(self, reader):
        clock = Mock(return_value=0.0)
        reader._sleep_checked.side_effect = lambda ms: setattr(
            clock, "return_value", round(clock.return_value + ms / 1000.0, 3)
        )
        return patch("custom.action.agent_info_collector.time.monotonic", clock)

    def test_prompt_disappearing_before_stable_returns_without_click(self):
        reader = self.disc_reader()
        reader.context.run_recognition.side_effect = [
            SimpleNamespace(hit=hit) for hit in (False, True, False, False)
        ]
        reader._run_task = Mock()
        with self.prompt_clock(reader) as clock, patch("custom.action.agent_info_collector.diagnostic_call") as diagnostic:
            reader._close_prompt_stably("提示", "disc_tip_close_failed", "无法关闭命盘单套提示")
            self.assertEqual(clock.return_value, 0.4)
        reader._run_task.assert_not_called()
        diagnostic.assert_not_called()

    def test_prompt_retries_when_first_close_is_ignored(self):
        reader = self.disc_reader()
        clicks = []
        reader.context.run_recognition.side_effect = lambda node, image: SimpleNamespace(
            hit=node == "密探信息采集-命盘界面就绪" or reader._run_task.call_count < 2
        )
        with self.prompt_clock(reader) as clock:
            reader._run_task = Mock(side_effect=lambda node: clicks.append(clock.return_value) or True)
            self.assertFalse(reader._toggle_disc("命盘一"))
        self.assertEqual(clicks, [0.6, 2.1])

    def test_prompt_unknown_screenshots_do_not_confirm_disappearance(self):
        reader = self.disc_reader()
        reader.context.run_recognition.side_effect = lambda node, image: SimpleNamespace(hit=not reader._run_task.called)
        def close(node):
            reader.screenshot.side_effect = [None, object(), None, object()]
            return True
        reader._run_task = Mock(side_effect=close)
        with self.prompt_clock(reader) as clock:
            reader._close_prompt_stably("提示", "form_tip_close_failed", "无法关闭未解锁形态提示")
            self.assertEqual(clock.return_value, 1.0)
        self.assertEqual(reader.context.run_recognition.call_count, 8)

    def test_prompt_must_hit_consecutive_frames_before_close(self):
        reader = self.disc_reader()
        reader.poll_interval_ms = 200
        reader.context.run_recognition.side_effect = [
            SimpleNamespace(hit=hit) for hit in (False, True, False, True, True, False, False)
        ]
        clicks = []
        with self.prompt_clock(reader) as clock:
            reader._run_task = Mock(side_effect=lambda node: clicks.append(clock.return_value) or True)
            reader._close_prompt_stably("提示", "form_tip_close_failed", "无法关闭未解锁形态提示")
        self.assertEqual(clicks, [1.0])

    def test_prompt_still_visible_after_close_times_out(self):
        for reason, message in (
            ("disc_tip_close_failed", "无法关闭命盘单套提示"),
            ("form_tip_close_failed", "无法关闭未解锁形态提示"),
        ):
            for task_succeeded in (True, False):
                with self.subTest(reason=reason, task_succeeded=task_succeeded):
                    reader = self.disc_reader()
                    reader.context.run_recognition.return_value.hit = True
                    reader._run_task = Mock(return_value=task_succeeded)
                    with self.prompt_clock(reader), patch("custom.action.agent_info_collector.diagnostic_call") as diagnostic:
                        with self.assertRaisesRegex(CollectorError, message) as error:
                            reader._close_prompt_stably("提示", reason, message)
                    self.assertEqual(error.exception.reason, reason)
                    self.assertEqual(reader._run_task.call_count, 3)
                    diagnostic.assert_called_once_with(None, "issue", "navigation_failed", "navigation", reason=reason)

    def test_cancel_during_prompt_close_does_not_retry(self):
        reader = self.disc_reader()
        reader.context.run_recognition.return_value.hit = True
        reader._run_task = Mock(side_effect=InterruptedError("stopped"))
        with self.prompt_clock(reader):
            with self.assertRaises(InterruptedError):
                reader._close_prompt_stably("提示", "disc_tip_close_failed", "无法关闭命盘单套提示")
        reader._run_task.assert_called_once()

    def test_unchanged_disc_times_out_without_second_click(self):
        reader = self.disc_reader()
        reader.click = Mock()
        reader.ocr_text = Mock(return_value="命盘一")
        with patch("custom.action.agent_info_collector.time.monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(RuntimeError, "切换未确认"):
                reader._toggle_disc("命盘一")
        reader.click.assert_called_once()

    def test_unknown_disc_label_does_not_click(self):
        reader = self.disc_reader()
        with self.assertRaisesRegex(RuntimeError, "无法确认当前"):
            reader._toggle_disc("")
        reader.context.tasker.controller.post_click.assert_not_called()

    def test_task_cancellation_is_not_swallowed(self):
        reader = self.reader()
        reader.context.run_task.side_effect = InterruptedError("stopped")
        with self.assertRaises(InterruptedError):
            reader._run_task("test")

    def test_second_disc_read_failure_still_restores_first(self):
        reader = self.disc_reader()
        reader._click_until_page = Mock()
        reader._operator_by_id = Mock(return_value=None)
        reader._require_page = Mock(return_value=reader.screenshot.return_value)
        reader._should_stop = Mock(return_value=False)
        reader._scan_disc_config = Mock(side_effect=[
            {"label": "命盘一", "signature": (), "slots": []}, RuntimeError("read failed")
        ])
        reader._toggle_disc = Mock(return_value=True)
        reader.ocr_text = Mock(side_effect=["命盘二", "命盘一", "命盘一"])
        with self.assertRaisesRegex(RuntimeError, "read failed"):
            reader._collect_discs({})
        self.assertEqual([call.args[0] for call in reader._toggle_disc.call_args_list], ["命盘一", "命盘二"])

    def test_wrong_restore_stops_collection(self):
        reader = self.disc_reader()
        reader._click_until_page = Mock()
        reader._operator_by_id = Mock(return_value=None)
        reader._require_page = Mock(return_value=reader.screenshot.return_value)
        reader._should_stop = Mock(return_value=False)
        reader._scan_disc_config = Mock(side_effect=[
            {"label": "命盘一", "signature": (), "slots": []},
            {"label": "命盘二", "signature": (("a", "active", "test"),), "slots": []},
        ])
        reader._toggle_disc = Mock(return_value=True)
        reader.ocr_text = Mock(return_value="命盘二")
        with self.assertRaisesRegex(RuntimeError, "恢复结果未确认"):
            reader._collect_discs({})

    def test_restore_label_requires_consecutive_fresh_reads(self):
        for labels, expected_reads in ((["加一", "加一"], 2), (["加", "加一", "加一"], 3)):
            with self.subTest(labels=labels):
                reader = self.disc_reader()
                reader.current_page = "main"
                reader._run_task = Mock(return_value=True)
                reader._click_until_page = Mock()
                reader._operator_by_id = Mock(return_value=None)
                reader._should_stop = Mock(return_value=False)
                reader._scan_disc_config = Mock(side_effect=[
                    {"label": "加一", "signature": (), "slots": []},
                    {"label": "命盘二", "signature": (("a", "active", "test"),), "slots": []},
                ])
                reader._toggle_disc = Mock(return_value=True)
                reader.ocr_text = Mock(side_effect=["命盘二", *labels])
                reader._collect_discs({})
                self.assertEqual(reader.ocr_text.call_count, expected_reads + 1)
                self.assertEqual(reader.screenshot.call_count, expected_reads + 1)
                self.assertEqual([call.args[0] for call in reader._sleep_checked.call_args_list], [300] * (expected_reads - 1))
                self.assertEqual(reader._toggle_disc.call_count, 2)

    def test_restore_failure_logs_bounded_cells_and_does_not_accept_third_configuration(self):
        for same_label in (False, True):
            with self.subTest(same_label=same_label):
                reader = self.disc_reader()
                reader._click_until_page = Mock()
                reader._operator_by_id = Mock(return_value=None)
                reader._require_page = Mock(return_value=reader.screenshot.return_value)
                reader._should_stop = Mock(return_value=False)
                first_cells, second_cells, current_cells = (tuple([char * 300]) for char in ("甲", "乙", "丙"))
                reader._scan_disc_config = Mock(side_effect=[
                    {"label": "同名" if same_label else "命盘一", "slots": [], "cells_fingerprint": first_cells},
                    {"label": "同名" if same_label else "命盘二", "slots": [], "cells_fingerprint": second_cells},
                ])
                reader._toggle_disc = Mock(return_value=True)
                reader.ocr_text = Mock(return_value="同名" if same_label else "命盘三")
                reader._disc_cells_fingerprint = Mock(return_value=current_cells)
                with patch("custom.action.agent_info_collector.logger") as log:
                    with self.assertRaises(CollectorError) as error:
                        reader._collect_discs({})
                self.assertEqual(error.exception.reason, "disc_restore_failed")
                log.bind.assert_called_with(console=False)
                message = log.bind.return_value.warning.call_args.args[0]
                for field in ("first_label=", "second_label=", "restored_label=", "original_cells=", "current_cells=", "cells_match=False"):
                    self.assertIn(field, message)
                self.assertNotIn("甲" * 201, message)
                self.assertNotIn("丙" * 201, message)
                reader._click_until_page.assert_called_once()

    def test_unstable_restore_label_is_not_accepted(self):
        reader = self.disc_reader()
        reader._click_until_page = Mock()
        reader._operator_by_id = Mock(return_value=None)
        reader._require_page = Mock(return_value=reader.screenshot.return_value)
        reader._should_stop = Mock(return_value=False)
        reader._scan_disc_config = Mock(side_effect=[
            {"label": "加一", "slots": []}, {"label": "命盘二", "slots": []},
        ])
        reader._toggle_disc = Mock(return_value=True)
        reader.ocr_text = Mock(side_effect=["命盘二", "加", "加二", "加一", "加一"])
        reader._disc_cells_fingerprint = Mock(return_value=("原内容",))
        with self.assertRaisesRegex(CollectorError, "恢复结果未确认"):
            reader._collect_discs({})
        reader._click_until_page.assert_called_once()

    def test_cancellation_during_second_disc_does_not_restore_or_click(self):
        reader = self.disc_reader()
        reader._click_until_page = Mock()
        reader._operator_by_id = Mock(return_value=None)
        reader._require_page = Mock(return_value=reader.screenshot.return_value)
        reader._should_stop = Mock(return_value=True)
        reader._scan_disc_config = Mock(side_effect=[
            {"label": "命盘一", "signature": (), "slots": []}, InterruptedError("stopped")
        ])
        reader._toggle_disc = Mock(return_value=True)
        with self.assertRaises(InterruptedError):
            reader._collect_discs({})
        reader._toggle_disc.assert_called_once_with("命盘一", None)

    def numeric_retry_reader(self, identity="a"):
        reader = self.reader()
        reader._collect_details = Mock(return_value={})
        reader._collect_discs = Mock(return_value=[])
        reader._confirm_operator_from_discs = Mock(return_value=None)
        reader._collect_huaji = Mock(return_value={})
        reader._require_page = Mock(return_value=reader.screenshot.return_value)
        reader._read_main = Mock(return_value={
            "operator_id": identity,
            "stats": {"life": 1424, "attack": 999, "level": 2, "cultivation": 1},
        })
        return reader

    def test_normal_mode_preserves_missing_stats_without_extra_read(self):
        reader = self.numeric_retry_reader()
        reader.scan_mode = "normal"
        record = reader.collect_current_from_main({"operator_id": "a", "name": "任意密探", "stats": {"life": None}})
        self.assertIsNone(record["stats"]["life"])
        self.assertEqual(record["collection_debug"]["scan_mode"], "normal")
        reader._read_main.assert_not_called()

    def test_normal_mode_uses_original_level_image_and_strict_cultivation(self):
        reader = self.reader()
        reader.scan_mode = "normal"
        image = reader.screenshot()
        reader.ocr_text = Mock(return_value="2")
        self.assertEqual(reader._read_stat(image, "level", MAIN_STAT_ROIS["level"]), 2)
        self.assertIs(reader.ocr_text.call_args.args[0], image)
        for text, expected in (("+17", 17), ("十1", None), ("杂字17", None)):
            reader.ocr_text.return_value = text
            self.assertEqual(reader._read_stat(image, "cultivation", MAIN_STAT_ROIS["cultivation"]), expected)

    def test_missing_stats_use_fresh_same_identity_frame_without_overwriting_valid_values(self):
        for field in ("life", "cultivation"):
            with self.subTest(field=field):
                reader = self.numeric_retry_reader()
                original = {"life": 1424, "attack": 222, "level": 2, "cultivation": 1}
                original[field] = None
                main = {"operator_id": "a", "name": "任意密探", "stats": original}
                record = reader.collect_current_from_main(main)
                self.assertEqual(record["stats"][field], 1424 if field == "life" else 1)
                self.assertEqual(record["stats"]["attack"], 222)
                self.assertIsNone(original[field])
                reader._read_main.assert_called_once()
                self.assertIsNone(reader._page_image)

    def test_missing_stats_retry_rejects_wrong_identity(self):
        reader = self.numeric_retry_reader("b")
        with self.assertRaisesRegex(RuntimeError, "数值补读.*身份"):
            reader.collect_current_from_main({"operator_id": "a", "name": "任意密探", "stats": {"life": None}})

    def test_missing_stats_retry_keeps_missing_for_unconfirmed_identity(self):
        reader = self.numeric_retry_reader(None)
        record = reader.collect_current_from_main({"operator_id": "a", "name": "任意密探", "stats": {"life": None}})
        self.assertIsNone(record["stats"]["life"])

    def test_missing_stats_are_not_guessed_when_fresh_read_is_still_missing(self):
        reader = self.numeric_retry_reader()
        reader._read_main.return_value["stats"]["life"] = None
        record = reader.collect_current_from_main({"operator_id": "a", "name": "任意密探", "stats": {"life": None}})
        self.assertIsNone(record["stats"]["life"])

    def test_numeric_retry_skips_complete_or_unconfirmed_records(self):
        for identity, stats in (("a", {"life": 1424}), (None, {"life": None})):
            with self.subTest(identity=identity):
                reader = self.numeric_retry_reader()
                reader.collect_current_from_main({"operator_id": identity, "name": "任意密探", "stats": stats})
                reader._read_main.assert_not_called()


class ScanRetryTests(unittest.TestCase):
    def reader(self, **params):
        with patch.object(_AgentInfoReader, "_load_operators", return_value={}):
            reader = _AgentInfoReader(SimpleNamespace(run_task=Mock()), {
                "resume": False, "start_from_current": True, **params,
            })
        reader._require_page = Mock(return_value=object())
        reader._ensure_running = Mock()
        reader._sleep_checked = Mock()
        reader._recover_to_main = Mock(return_value=True)
        reader._read_main = Mock(return_value={"name": "甲", "operator_id": "char_a"})
        reader._prepare_growth_filter = Mock(side_effect=lambda main: main)
        reader._collect_other_form = Mock()
        reader._move_operator = Mock(return_value=False)
        reader._publish_checkpoint = Mock(side_effect=lambda records, record: records.append(record))
        reader.collect_current_from_main = Mock(side_effect=lambda main: main)
        return reader

    def test_retry_option_wires_both_resources_and_runtime(self):
        from custom.action.agent_info_collector import AgentInfoCollector
        from custom.action.toolbox_diagnostics import REASONS

        root = Path(__file__).resolve().parents[2]
        interface = json.loads((root / "assets/interface.json").read_text())
        option = interface["option"]["最大重试次数"]
        self.assertEqual(option["type"], "select")
        self.assertEqual(option["default_case"], "2")
        self.assertEqual([case["name"] for case in option["cases"]], list("012345"))
        collection = next(case for case in interface["option"]["百宝箱-模式"]["cases"]
                          if case["name"] == "采集密探信息")
        self.assertIn("高级设置", collection["option"])
        self.assertNotIn("最大重试次数", collection["option"])
        advanced = interface["option"]["高级设置"]
        self.assertEqual(advanced["type"], "switch")
        self.assertEqual(advanced["default_case"], "No")
        cases = {case["name"]: case for case in advanced["cases"]}
        self.assertNotIn("option", cases["No"])
        self.assertEqual(cases["No"]["pipeline_override"], {
            "密探采集扫描模式配置": {"attach": {"scan_mode": "enhanced"}},
            "密探采集重试配置": {"attach": {"max_retries": 2}},
        })
        self.assertEqual(cases["Yes"]["option"], ["密探采集扫描模式", "最大重试次数"])
        self.assertNotIn("pipeline_override", cases["Yes"])
        self.assertIn("retry_recovered", REASONS)
        for language in ("base", "zh_tw"):
            pipeline = json.loads((root / f"assets/resource/{language}/pipeline/agent_info_collector.json").read_text())
            self.assertEqual(pipeline["密探采集重试配置"]["attach"], {"max_retries": 2})
        for count, case in enumerate(option["cases"]):
            node = case["pipeline_override"]["密探采集重试配置"]
            self.assertEqual(node["attach"], {"max_retries": count})
            context = SimpleNamespace(get_node_data=lambda name: node if name == "密探采集重试配置" else None)
            with patch("custom.action.agent_info_collector._AgentInfoReader") as reader:
                reader.return_value.run.return_value = True
                self.assertTrue(AgentInfoCollector().run(context, SimpleNamespace(custom_action_param={})).success)
                self.assertEqual(reader.call_args.args[1]["max_retries"], count)

    def test_retry_configuration_clamps_and_defaults(self):
        for value, expected in ((None, 2), ("bad", 2), ([], 2), (float("inf"), 2), (-1, 0), (9, 5), ("3", 3)):
            with self.subTest(value=value):
                reader = self.reader(max_retries=value)
                self.assertEqual(reader.max_retries, expected)
                self.assertEqual((reader.retry_budget, reader.retries_used), (10, 0))
        self.assertEqual(self.reader().max_retries, 2)

    def test_collection_recovers_and_reports_original_reason_and_recovery(self):
        reader = self.reader()
        error = CollectorError("timeout", "page_timeout")
        main = reader._read_main.return_value
        reader.collect_current_from_main.side_effect = [error, main]
        with patch("custom.action.agent_info_collector.diagnostic_call") as diagnostic:
            self.assertTrue(reader.run())
        reader._recover_to_main.assert_called_once()
        reader._publish_checkpoint.assert_called_once()
        reader._collect_other_form.assert_called_once()
        self.assertEqual(reader.retries_used, 1)
        self.assertEqual([call.kwargs.get("reason") for call in diagnostic.call_args_list], ["page_timeout", "retry_recovered"])

    def test_collection_exhaustion_preserves_original_exception(self):
        reader = self.reader()
        error = CollectorError("timeout", "page_timeout")
        reader.collect_current_from_main.side_effect = error
        with self.assertRaises(CollectorError) as caught:
            reader.run()
        self.assertIs(caught.exception, error)
        self.assertEqual(reader.collect_current_from_main.call_count, 3)
        self.assertEqual(reader._recover_to_main.call_count, 2)
        reader._move_operator.assert_not_called()

    def test_fuzzy_name_retry_reuses_disc_confirmed_identity_without_rereading(self):
        for stage in ("collection", "navigation"):
            with self.subTest(stage=stage):
                reader = self.reader()
                raw = {"name": "史子吵", "operator_id": None, "_name_exact": False}
                confirmed = {"name": "史子眇", "operator_id": "char_023_shizimiao"}
                reader._read_main.side_effect = [dict(raw), dict(raw)]
                reader._prepare_growth_filter = Mock(wraps=_AgentInfoReader._prepare_growth_filter.__get__(reader))
                reader._collect_discs = Mock(return_value=[])
                reader._confirm_operator_from_discs = Mock(return_value={
                    "id": confirmed["operator_id"], "name": confirmed["name"],
                })
                reader._resolve_locked_disc_names = Mock()
                if stage == "collection":
                    reader.collect_current_from_main.side_effect = [CollectorError("scan", "ocr_failed"), confirmed]
                else:
                    reader._move_operator.side_effect = [CollectorError("turn", "page_turn_unconfirmed"), False]

                with patch("custom.action.agent_info_collector.diagnostic_call") as diagnostic:
                    self.assertTrue(reader.run())

                self.assertEqual(reader._read_main.call_count, 2)
                self.assertEqual(reader._collect_discs.call_count, 2)
                self.assertEqual(reader.collect_current_from_main.call_count, 2 if stage == "collection" else 1)
                self.assertEqual(reader.retries_used, 1)
                self.assertIn("retry_recovered", [call.kwargs.get("reason") for call in diagnostic.call_args_list])
                if stage == "collection":
                    retried_main = reader.collect_current_from_main.call_args.args[0]
                else:
                    retried_main = reader._move_operator.call_args.args[1]
                self.assertEqual(retried_main["operator_id"], confirmed["operator_id"])
                self.assertIn("_prefetched_discs", retried_main)

    def test_zero_budget_and_nonretryable_errors_do_not_recover(self):
        for params, error in (({"max_retries": 0}, CollectorError("timeout", "page_timeout")),
                              ({}, CollectorError("boundary", "list_boundary_unverified")),
                              ({}, RuntimeError("unexpected")), ({}, InterruptedError("stopped"))):
            with self.subTest(params=params, error=error):
                reader = self.reader(**params)
                reader.collect_current_from_main.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    reader.run()
                self.assertIs(caught.exception, error)
                reader._recover_to_main.assert_not_called()

    def test_stop_and_failed_recovery_preserve_original_error(self):
        for stopped, recovery in ((True, True), (False, False), (False, CollectorError("recovery", "ocr_failed"))):
            with self.subTest(stopped=stopped, recovery=recovery):
                reader = self.reader()
                error = CollectorError("timeout", "page_timeout")
                reader.collect_current_from_main.side_effect = error
                reader._should_stop = Mock(return_value=stopped)
                if isinstance(recovery, Exception):
                    reader._recover_to_main.side_effect = recovery
                else:
                    reader._recover_to_main.return_value = recovery
                with self.assertRaises(CollectorError) as caught:
                    reader.run()
                self.assertIs(caught.exception, error)
                self.assertEqual(reader._recover_to_main.call_count, 0 if stopped else 1)

    def test_changed_identity_after_recovery_stops_without_collecting_another_operator(self):
        reader = self.reader()
        reader._read_main.side_effect = [{"name": key, "operator_id": key} for key in ("char_a", "char_b")]
        error = CollectorError("timeout", "page_timeout")
        reader.collect_current_from_main.side_effect = error
        with self.assertRaises(CollectorError) as caught:
            reader.run()
        self.assertIs(caught.exception, error)
        reader.collect_current_from_main.assert_called_once()
        reader._recover_to_main.assert_called_once()

    def test_other_form_identity_is_allowed_during_retry(self):
        reader = self.reader()
        base = {"name": "陈登", "operator_id": "char_013_chendeng"}
        other = {"name": "陈登黍王", "operator_id": "char_084_chendengsp"}
        reader._read_main.side_effect = [base, other]
        reader.collect_current_from_main.side_effect = [CollectorError("form", "form_switch_failed"), other]
        self.assertTrue(reader.run())
        self.assertEqual(reader._publish_checkpoint.call_args.args[1], other)

    def test_other_form_failure_retries_whole_operator_and_success_reports_recovery(self):
        reader = self.reader()
        reader._collect_other_form.side_effect = [CollectorError("form", "form_switch_failed"), None]
        with patch("custom.action.agent_info_collector.diagnostic_call") as diagnostic:
            self.assertTrue(reader.run())
        self.assertEqual(reader.collect_current_from_main.call_count, 2)
        self.assertEqual(reader._collect_other_form.call_count, 2)
        self.assertIn("retry_recovered", [call.kwargs.get("reason") for call in diagnostic.call_args_list])

    def test_unresolved_identity_exhaustion_returns_false_with_original_diagnostic(self):
        reader = self.reader()
        reader._read_main.return_value = {"operator_id": "char_a", "name": ""}
        with patch("custom.action.agent_info_collector.diagnostic_call") as diagnostic:
            self.assertFalse(reader.run())
        self.assertEqual(reader._recover_to_main.call_count, 2)
        self.assertEqual(sum(call.kwargs.get("reason") == "name_and_disc_unresolved"
                             and call.args[2] == "identity_unresolved" for call in diagnostic.call_args_list), 3)
        reader.collect_current_from_main.assert_not_called()

    def test_total_retry_budget_is_shared_across_operators(self):
        reader = self.reader()
        state = {"index": 0}
        reader._read_main.side_effect = lambda image: {"name": str(state["index"]), "operator_id": f"char_{state['index']}"}
        failures = set()
        def collect(main):
            key = main["operator_id"]
            if key not in failures:
                failures.add(key)
                raise CollectorError("timeout", "page_timeout")
            return main
        reader.collect_current_from_main.side_effect = collect
        reader._move_operator.side_effect = lambda *args: (state.__setitem__("index", state["index"] + 1) or True)
        with self.assertRaises(CollectorError):
            reader.run()
        self.assertEqual(reader.retries_used, 10)
        self.assertEqual(reader._recover_to_main.call_count, 10)
        self.assertEqual(reader._publish_checkpoint.call_count, 10)

    def test_page_turn_already_completed_does_not_skip_or_duplicate_next_operator(self):
        reader = self.reader()
        first = reader._read_main.return_value
        second = {"name": "乙", "operator_id": "char_b"}
        reader._read_main.side_effect = [first, second, second]
        reader._move_operator.side_effect = [CollectorError("turn", "page_turn_unconfirmed"), False]
        self.assertTrue(reader.run())
        self.assertEqual([call.args[0]["operator_id"] for call in reader.collect_current_from_main.call_args_list], ["char_a", "char_b"])
        self.assertEqual(reader._move_operator.call_count, 2)

    def test_collection_and_page_turn_share_per_operator_budget(self):
        reader = self.reader(max_retries=2)
        first = reader._read_main.return_value
        reader.collect_current_from_main.side_effect = [CollectorError("scan", "ocr_failed"), first]
        reader._move_operator.side_effect = CollectorError("turn", "page_turn_unconfirmed")
        self.assertFalse(reader.run())
        self.assertEqual(reader._recover_to_main.call_count, 2)
        self.assertEqual(reader._move_operator.call_count, 2)
        self.assertEqual(reader.collect_current_from_main.call_count, 2)

    def test_unverified_boundary_does_not_retry(self):
        reader = self.reader()
        reader._move_operator.side_effect = CollectorError("boundary", "list_boundary_unverified")
        self.assertFalse(reader.run())
        reader._recover_to_main.assert_not_called()

    def test_recovery_uses_only_recognized_page_actions_and_caches_main_frame(self):
        from custom.action.agent_info_collector import DISC_PROMPT_NODE

        for source, click_key in (("detail", "detail_close"), ("huaji", "huaji_back"), ("disc", "disc_back"),
                                  (DISC_PROMPT_NODE, None), ("密探信息采集-未解锁形态提示", None)):
            with self.subTest(source=source):
                reader = self.reader()
                frame, fresh_frame, main_frame = object(), object(), object()
                reader.screenshot = Mock(side_effect=[frame, fresh_frame, main_frame] if click_key else [frame, main_frame])
                reader._require_page.return_value = main_frame
                reader.context.run_recognition = Mock(side_effect=lambda node, image: SimpleNamespace(
                    hit=node == (PAGE_NODES["main"] if image is main_frame else PAGE_NODES.get(source, source))))
                reader.click = Mock()
                reader._run_task = Mock()
                self.assertTrue(_AgentInfoReader._recover_to_main(reader))
                if click_key:
                    reader.click.assert_called_once_with(reader._roi_config["clicks"][click_key], fresh_frame, settle_ms=0)
                    reader._run_task.assert_not_called()
                    reader._sleep_checked.assert_called_once_with(reader.poll_interval_ms)
                else:
                    reader.click.assert_not_called()
                    reader._run_task.assert_called_once_with(source)
                    reader._sleep_checked.assert_called_once_with(500)
                self.assertEqual(reader.current_page, "main")
                self.assertIs(reader._page_image, main_frame)

    def test_recovery_rechecks_fresh_source_and_never_clicks_main(self):
        reader = self.reader()
        old_frame, main_frame = object(), object()
        reader.screenshot = Mock(side_effect=[old_frame, main_frame, main_frame])
        reader.context.run_recognition = Mock(side_effect=lambda node, image: SimpleNamespace(
            hit=node == PAGE_NODES["disc" if image is old_frame else "main"]))
        reader.click = Mock()
        reader._click_until_page = Mock()
        self.assertTrue(_AgentInfoReader._recover_to_main(reader))
        reader.click.assert_not_called()
        reader._click_until_page.assert_not_called()
        self.assertIs(reader._page_image, main_frame)

    def test_recovery_main_on_second_confirmation_frame_does_not_repeat_return(self):
        reader = self.reader()
        reader.screenshot = Mock(return_value=object())
        reader.click = Mock()
        now = [0.0]
        reader._sleep_checked.side_effect = lambda ms: now.__setitem__(0, now[0] + ms / 1000)
        reader.context.run_recognition = Mock(side_effect=lambda node, image: SimpleNamespace(
            hit=node == PAGE_NODES["main" if reader.click.called and now[0] >= 4.0 else "disc"]))
        with patch("custom.action.agent_info_collector.time.monotonic", side_effect=lambda: now[0]):
            self.assertTrue(_AgentInfoReader._recover_to_main(reader))
        reader.click.assert_called_once()
        self.assertEqual([call.args[0] for call in reader._sleep_checked.call_args_list].count(600), 2)

    def test_recovery_page_timeout_is_caught_for_next_round(self):
        reader = self.reader()
        old_frame, main_frame = object(), object()
        reader.screenshot = Mock(side_effect=[old_frame, old_frame, main_frame])
        reader.context.run_recognition = Mock(side_effect=lambda node, image: SimpleNamespace(
            hit=node == PAGE_NODES["disc" if image is old_frame else "main"]))
        reader._click_until_page = Mock(side_effect=CollectorError("timeout", "page_timeout"))
        self.assertTrue(_AgentInfoReader._recover_to_main(reader))
        reader._click_until_page.assert_called_once_with("disc_back", old_frame, "main", "disc")
        self.assertIs(reader._page_image, main_frame)

    def test_recovery_accepts_confirmed_main_on_fourth_round(self):
        reader = self.reader()
        source_frame, main_frame = object(), object()
        reader.screenshot = Mock(return_value=source_frame)
        reader.context.run_recognition = Mock(side_effect=lambda node, image: SimpleNamespace(hit=node == PAGE_NODES["disc"]))
        reader._click_until_page = Mock(side_effect=[CollectorError("timeout", "page_timeout")] * 3 + [main_frame])
        self.assertTrue(_AgentInfoReader._recover_to_main(reader))
        self.assertEqual(reader._click_until_page.call_count, 4)
        self.assertEqual(reader.screenshot.call_count, 8)
        self.assertEqual(reader._sleep_checked.call_count, 3)
        self.assertEqual(reader.current_page, "main")
        self.assertIs(reader._page_image, main_frame)

    def test_recovery_unknown_page_never_clicks_and_stops_after_four_rounds(self):
        reader = self.reader()
        reader.screenshot = Mock(return_value=object())
        reader.context.run_recognition = Mock(return_value=SimpleNamespace(hit=False))
        reader.click = Mock()
        self.assertFalse(_AgentInfoReader._recover_to_main(reader))
        self.assertEqual(reader.screenshot.call_count, 4)
        self.assertEqual(reader._sleep_checked.call_count, 3)
        reader.click.assert_not_called()
        self.assertIsNone(reader.current_page)
        self.assertIsNone(reader._page_image)

    def test_recovery_cancellation_propagates(self):
        reader = self.reader()
        reader._ensure_running.side_effect = InterruptedError("stop")
        reader.screenshot = Mock()
        with self.assertRaises(InterruptedError):
            _AgentInfoReader._recover_to_main(reader)
        reader.screenshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
