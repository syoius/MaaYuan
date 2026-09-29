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
