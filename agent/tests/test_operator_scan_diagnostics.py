import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom.action import agent_info_collector as collector
from custom.action.operator_scan_diagnostics import ScanFailureDiagnostics


class ScanDiagnosticsTests(unittest.TestCase):
    def test_shipped_config_is_off_at_both_layers(self):
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "assets/interface.json").read_text())
        self.assertEqual(config["telemetry"]["sentry"], {
            "dsn": "", "tracing": False, "traces_sample_rate": 0, "failure_attachments_sample_rate": 0})
        for language in ("base", "zh_tw"):
            pipeline = json.loads((root / f"assets/resource/{language}/pipeline/agent_info_collector.json").read_text())
            node = pipeline["密探采集诊断配置"]
            self.assertEqual(node["attach"], {"failure_diagnostics": False})
            context = SimpleNamespace(get_node_data=lambda name: node if name == "密探采集诊断配置" else None)
            with patch.object(collector, "_AgentInfoReader") as reader:
                reader.return_value.run.return_value = True
                result = collector.AgentInfoCollector().run(context, SimpleNamespace(custom_action_param={"failure_diagnostics": True}))
                self.assertTrue(result.success)
                self.assertIs(reader.call_args.args[1]["failure_diagnostics"], False)

    def test_only_explicit_boolean_true_constructs_diagnostics(self):
        with patch.object(collector._AgentInfoReader, "_load_operators", return_value={}), \
             patch.object(collector, "ScanFailureDiagnostics") as diagnostics:
            for value in (None, False, "false", "true", 1):
                reader = collector._AgentInfoReader(Mock(), {"failure_diagnostics": value})
                self.assertIsNone(reader._failure_diagnostics)
            diagnostics.assert_not_called()
            reader = collector._AgentInfoReader(Mock(), {"failure_diagnostics": True})
            diagnostics.assert_called_once_with("enhanced", "代号鸢")
            self.assertIs(reader._failure_diagnostics, diagnostics.return_value)

    def test_disabled_path_keeps_single_screenshot_and_no_new_writer(self):
        frame = np.zeros((1280, 720, 3), np.uint8)
        context = Mock()
        context.stop = False
        context.tasker.stopping = False
        context.tasker.controller.post_screencap.return_value.wait.return_value.get.return_value = frame
        with patch.object(collector._AgentInfoReader, "_load_operators", return_value={}), \
             patch.object(collector, "ScanFailureDiagnostics") as diagnostics:
            reader = collector._AgentInfoReader(context, {})
            self.assertIs(reader.screenshot(), frame)
            context.tasker.controller.post_screencap.assert_called_once()
            diagnostics.assert_not_called()

    def test_observation_does_not_write_until_failure_and_deduplicates_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "on_error/operator_scan"
            diagnostics = ScanFailureDiagnostics("enhanced", "代号鸢", root)
            frame = np.zeros((1280, 720, 3), np.uint8)
            frame[:200] = 255
            diagnostics.last_frame = frame
            diagnostics.observe_main(frame, {"life": None, "attack": 99, "level": 2, "cultivation": 0}, True)
            self.assertFalse(root.exists())
            diagnostics.save_failure("task_failed")
            diagnostics.save_failure("exception")
            images = list(root.glob("*.jpg"))
            self.assertEqual(len(images), 1)
            self.assertEqual(len(list(root.glob("*.log"))), 1)
            masked = cv2.imread(str(images[0]))
            self.assertEqual(masked.shape, (1280, 720, 3))
            self.assertEqual(int(masked.max()), 0)
            self.assertTrue(np.all(frame[:200] == 255))
            metadata = json.loads(next(root.glob("*.log")).read_text())
            self.assertEqual(metadata["missing_fields"], ["life"])
            self.assertEqual(metadata["images"][0]["masked_top"], 200)
            self.assertEqual(metadata["image_source"], "previously_observed_frames_not_fresh_failure_capture")

    def test_oversized_images_are_omitted_and_metadata_is_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostics = ScanFailureDiagnostics("secret-token", "private-account", root)
            diagnostics.stage = "account/path"
            diagnostics.last_frame = np.zeros((1280, 720, 3), np.uint8)
            with patch("custom.action.operator_scan_diagnostics.cv2.imencode",
                       return_value=(True, np.zeros(ScanFailureDiagnostics.MAX_IMAGE_BYTES + 1, np.uint8))):
                diagnostics.save_failure("connection-code")
            text = next(root.glob("*.log")).read_text()
            for secret in ("secret-token", "private-account", "account/path", "connection-code"):
                self.assertNotIn(secret, text)
            self.assertEqual(list(root.glob("*.jpg")), [])
            self.assertEqual(json.loads(text)["omitted_images"], ["last"])

    def test_disk_failure_never_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = ScanFailureDiagnostics("normal", "如鸢", Path(directory))
            with patch.object(Path, "mkdir", side_effect=PermissionError("read only")):
                diagnostics.save_failure("task_failed")

    def test_normal_failure_and_exception_save_but_success_and_stop_do_not(self):
        for outcome, stopped, expected_reason in (
            (False, False, "task_failed"), (RuntimeError("failed"), False, "exception"),
            (True, False, None), (InterruptedError("cancel"), False, None), (False, True, None),
        ):
            with self.subTest(outcome=outcome, stopped=stopped):
                diagnostics = Mock()
                reader = Mock(_failure_diagnostics=diagnostics)
                reader._should_stop.return_value = stopped
                if isinstance(outcome, Exception):
                    reader.run.side_effect = outcome
                else:
                    reader.run.return_value = outcome
                context = SimpleNamespace(get_node_data=lambda name: None)
                with patch.object(collector, "_AgentInfoReader", return_value=reader):
                    result = collector.AgentInfoCollector().run(context, SimpleNamespace(custom_action_param={}))
                self.assertEqual(result.success, outcome is True)
                if expected_reason:
                    diagnostics.save_failure.assert_called_once_with(expected_reason)
                else:
                    diagnostics.save_failure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
