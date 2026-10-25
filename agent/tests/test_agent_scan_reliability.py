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
from custom.action.agent_info_collector import _AgentInfoReader, MAIN_STAT_ROIS, OCR_NODE


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
        reader = self.disc_reader()
        reader.context.run_recognition.side_effect = [
            SimpleNamespace(hit=True), SimpleNamespace(hit=False), SimpleNamespace(hit=True)
        ]
        reader._run_task = Mock(return_value=True)
        self.assertFalse(reader._toggle_disc("命盘一"))
        self.assertEqual(reader._run_task.call_args_list[0].args[0], "密探信息采集-命盘单套提示")
        self.assertEqual(reader.context.run_recognition.call_args.args[0], "密探信息采集-命盘界面就绪")
        self.assertEqual(reader.screenshot.call_count, 2)

    def test_prompt_still_visible_after_close_times_out(self):
        reader = self.disc_reader()
        reader.click = Mock()
        reader.context.run_recognition.return_value.hit = True
        reader._run_task = Mock(return_value=True)
        with patch("custom.action.agent_info_collector.time.monotonic", side_effect=[0, .5, 2]):
            with self.assertRaisesRegex(RuntimeError, "提示关闭未确认"):
                reader._toggle_disc("命盘一")
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
        reader._operator_by_id = Mock(return_value=None)
        reader._require_page = Mock(return_value=reader.screenshot.return_value)
        reader._should_stop = Mock(return_value=False)
        reader._scan_disc_config = Mock(side_effect=[
            {"label": "命盘一", "signature": (), "slots": []}, RuntimeError("read failed")
        ])
        reader._toggle_disc = Mock(return_value=True)
        reader.ocr_text = Mock(side_effect=["命盘二", "命盘一"])
        with self.assertRaisesRegex(RuntimeError, "read failed"):
            reader._collect_discs({})
        self.assertEqual([call.args[0] for call in reader._toggle_disc.call_args_list], ["命盘一", "命盘二"])

    def test_wrong_restore_stops_collection(self):
        reader = self.disc_reader()
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

    def test_cancellation_during_second_disc_does_not_restore_or_click(self):
        reader = self.disc_reader()
        reader._operator_by_id = Mock(return_value=None)
        reader._require_page = Mock(return_value=reader.screenshot.return_value)
        reader._should_stop = Mock(return_value=True)
        reader._scan_disc_config = Mock(side_effect=[
            {"label": "命盘一", "signature": (), "slots": []}, InterruptedError("stopped")
        ])
        reader._toggle_disc = Mock(return_value=True)
        with self.assertRaises(InterruptedError):
            reader._collect_discs({})
        reader._toggle_disc.assert_called_once_with("命盘一")

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


if __name__ == "__main__":
    unittest.main()
