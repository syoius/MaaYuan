"""Name fallback decision tests, independent of OCR models and game screenshots."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom.action.agent_info_collector import _AgentInfoReader, MAIN_NAME_ROI


class HorizontalNameFallbackTests(unittest.TestCase):
    def reader(self, raw="土燮"):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.operators = {
            name: {"id": name, "name": name}
            for name in ("士燮", "程昱", "程普", "陈登", "史子眇", "李脱")
        }
        reader.roi = {"main_name": MAIN_NAME_ROI, "main_stats": {}}
        reader.ocr_text = Mock(return_value=raw)
        reader._ensure_running = Mock()
        reader.context = Mock()
        return reader

    def test_normal_mode_skips_horizontal_ocr_and_preserves_unconfirmed_identity(self):
        reader = self.reader()
        reader.scan_mode = "normal"
        record = reader._read_main(np.zeros((1280, 720, 3), dtype=np.uint8))
        self.assertIsNone(record["operator_id"])
        reader.context.run_recognition.assert_not_called()
        self.assertEqual(reader._horizontal_name_readings(None, full_charset=True), [])

    def test_exact_name_skips_retry_and_keeps_awakened_badge(self):
        reader = self.reader("陈登觉醒")
        reader._horizontal_name_readings = Mock()
        result = reader._read_main(np.zeros((1280, 720, 3), dtype=np.uint8))
        reader._horizontal_name_readings.assert_not_called()
        self.assertEqual(result["name_raw"], "陈登觉醒")
        self.assertEqual(result["operator_id"], "陈登")

    def test_unique_exact_retry_accepts_and_preserves_original_text(self):
        for readings in (["士燮", "士燮"], ["无匹配", "士燮"]):
            with self.subTest(readings=readings):
                reader = self.reader()
                reader._horizontal_name_readings = Mock(return_value=readings)
                result = reader._read_main(None)
                self.assertEqual(result["name_raw"], "土燮")
                self.assertEqual(result["operator_id"], "士燮")
                self.assertTrue(result["_name_exact"])
                self.assertEqual(result["_name_ocr_fallback"], readings)

    def test_conflicting_candidates_remain_unconfirmed(self):
        reader = self.reader("程量")
        reader._horizontal_name_readings = Mock(return_value=["程昱", "程普"])
        result = reader._read_main(None)
        self.assertIsNone(result["operator_id"])
        self.assertFalse(result["_name_exact"])
        self.assertEqual(result["_operator_match"], "unconfirmed")

    def test_no_substring_or_nearest_name_acceptance(self):
        reader = self.reader("陈登杂字")
        reader._horizontal_name_readings = Mock(return_value=["李悦", "1士燮"])
        self.assertIsNone(reader._read_main(None)["operator_id"])

    def test_traditional_retry_is_normalized(self):
        reader = self.reader("李悦")
        reader._horizontal_name_readings = Mock(return_value=["李脫", "李脱"])
        self.assertEqual(reader._read_main(None)["operator_id"], "李脱")

    def test_full_charset_retry_matches_rare_character_without_guessing(self):
        reader = self.reader("徐")
        reader.operators["徐稺"] = {"id": "xuzhi", "name": "徐稺"}
        reader._horizontal_name_readings = Mock(side_effect=[
            ["徐裤", "徐"], ["徐裤", "徐稺", "徐", "徐稺"],
        ])
        main = reader._read_main(None)
        self.assertEqual(main["operator_id"], "xuzhi")
        self.assertTrue(main["_name_exact"])
        self.assertEqual(reader._horizontal_name_readings.call_args.kwargs, {"full_charset": True})

    def test_conflicting_full_charset_names_remain_unconfirmed(self):
        reader = self.reader("程量")
        reader._horizontal_name_readings = Mock(side_effect=[
            ["程量", "程量"], ["程昱", "程普", "程昱", "程普"],
        ])
        self.assertIsNone(reader._read_main(None)["operator_id"])

    def test_full_charset_model_uses_original_and_binary_variants(self):
        reader = self.reader()
        reader.context.run_recognition.return_value = SimpleNamespace(filtered_results=[])
        reader._horizontal_name_readings(np.zeros((1280, 720, 3), dtype=np.uint8), full_charset=True)
        calls = reader.context.run_recognition.call_args_list
        self.assertEqual(len(calls), 8)
        for call in calls:
            node, frame, override = call.args
            self.assertEqual(override[node]["recognition"]["param"]["model"], "operator_names")
        self.assertTrue(np.all(calls[1].args[1] == 255))

    def test_full_charset_checks_both_columns_when_red_costume_looks_like_badge(self):
        reader = self.reader()
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        crop = image[221:447, 32:138]
        crop[88:215, :35] = 80
        crop[88:215, 35:72] = 160
        crop[150:215, 72:106] = (30, 70, 220)
        reader.context.run_recognition.return_value = SimpleNamespace(filtered_results=[])
        reader._horizontal_name_readings(image, full_charset=True)
        calls = reader.context.run_recognition.call_args_list
        self.assertEqual(int(calls[0].args[1][33, 10, 0]), 80)
        self.assertEqual(int(calls[4].args[1][33, 10, 0]), 160)

    def test_approximate_horizontal_name_needs_unique_disc_evidence(self):
        reader = self.reader("")
        reader.operators["史子眇"]["discs"] = [{"ot_name": "技能提供减伤"}]
        reader._horizontal_name_readings = Mock(return_value=["手吵", "史子吵"])
        main = reader._read_main(None)
        self.assertIsNone(main["operator_id"])
        self.assertFalse(main["_name_exact"])
        self.assertIn("史子眇", [item["operator_id"] for item in main["_operator_candidates"]])
        self.assertIsNone(reader._confirm_operator_from_discs(main, [{"slots": []}]))
        confirmed = reader._confirm_operator_from_discs(main, [{"slots": [{"state": "inactive", "name": "技能提供减伤"}]}])
        self.assertEqual(confirmed["id"], "史子眇")
        reader.operators["陈登"]["discs"] = [{"ot_name": "技能提供减伤"}]
        self.assertIsNone(reader._confirm_operator_from_discs(main, [{"slots": [{"state": "inactive", "name": "技能提供减伤"}]}]))

    def test_reorders_upright_cells_and_uses_pixel_roi_at_scaled_resolution(self):
        for scale in (1, 2):
            with self.subTest(scale=scale):
                reader = self.reader()
                image = np.zeros((1280 * scale, 720 * scale, 3), dtype=np.uint8)
                for top, bottom, value in ((28, 88, 40), (88, 149, 100), (149, 215, 200)):
                    image[(221 + top)*scale:(221 + bottom)*scale, 67*scale:138*scale] = value
                reader.context.run_recognition.return_value = SimpleNamespace(
                    filtered_results=[SimpleNamespace(text="士燮")]
                )
                self.assertEqual(reader._horizontal_name_readings(image), ["士燮", "士燮"])
                calls = reader.context.run_recognition.call_args_list
                self.assertEqual(len(calls), 2)
                for call, values in zip(calls, ((100, 200), (40, 100, 200))):
                    node, horizontal, override = call.args
                    self.assertEqual(horizontal.shape, (66, 71 * len(values), 3))
                    for index, value in enumerate(values):
                        self.assertEqual(int(horizontal[33, 71 * index + 35, 0]), value)
                    param = override[node]["recognition"]["param"]
                    self.assertEqual(param["roi"], [0, 0, 71 * len(values), 66])
                    self.assertTrue(param["only_rec"])
                    self.assertEqual(param["replace"], [])

    def test_awakened_badge_moves_name_crop_to_left(self):
        reader = self.reader()
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        name = image[221:447, 32:138]
        name[88:215, :35] = 80
        name[88:215, 35:72] = 160
        name[150:215, 72:106] = (30, 70, 220)
        reader.context.run_recognition.return_value = SimpleNamespace(
            filtered_results=[SimpleNamespace(text="王粲")]
        )

        self.assertEqual(reader._horizontal_name_readings(image), ["王粲", "王粲"])
        horizontal = reader.context.run_recognition.call_args_list[0].args[1]
        self.assertEqual(int(horizontal[33, 10, 0]), 80)
        self.assertEqual(int(horizontal[33, 50, 0]), 160)
        self.assertEqual(int(horizontal[33, 70, 0]), 160)

    def test_exception_on_second_hypothesis_discards_partial_success(self):
        reader = self.reader()
        reader.context.run_recognition.side_effect = [
            SimpleNamespace(filtered_results=[SimpleNamespace(text="士燮")]),
            RuntimeError("recognition unavailable"),
        ]
        self.assertEqual(reader._horizontal_name_readings(np.zeros((1280,720,3),np.uint8)), [])

    def test_cancellation_propagates(self):
        reader = self.reader()
        reader.context.run_recognition.side_effect = InterruptedError("stopped")
        with self.assertRaises(InterruptedError):
            reader._horizontal_name_readings(np.zeros((1280,720,3),np.uint8))


if __name__ == "__main__":
    unittest.main()
