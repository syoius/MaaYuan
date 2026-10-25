import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import operator_scan_debug as debug


class OfflineReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "frame.png"
        self.image.write_bytes(b"fixture")
        self.manifest = self.root / "cases.json"
        self.resource = self.root / "resource"
        (self.resource / "pipeline").mkdir(parents=True)
        (self.resource / "pipeline/agent_info_collector.json").write_text("{}")

    def cases(self, expected=None):
        case = {"image": "frame.png"}
        if expected is not None:
            case["expected"] = expected
        debug.write_json(self.manifest, {"cases": [case]})

    def test_relative_paths_and_optional_partial_truth(self):
        self.cases({"life": 123, "cultivation": 0})
        case = debug.load_cases(self.manifest, "stats")[0]
        self.assertEqual(Path(case["image"]), self.image.resolve())
        self.assertEqual(case["sha256"], debug.digest(self.image))
        self.cases()
        self.assertEqual(debug.load_cases(self.manifest, "main")[0]["expected"], {})

    def test_invalid_truth_and_scope_are_rejected(self):
        for expected in ({"life": None}, {"level": True}, {"attack": 0},
                         {"level": -1}, {"typo": 1}, {"operator_id": ""}):
            with self.subTest(expected=expected):
                self.cases(expected)
                with self.assertRaises(ValueError):
                    debug.load_cases(self.manifest, "main")
        self.cases({"operator_id": "confirmed-id"})
        with self.assertRaises(ValueError):
            debug.load_cases(self.manifest, "stats")

    def test_empty_missing_and_duplicate_images_are_rejected(self):
        for cases in ([], [{"image": "absent.png"}],
                      [{"image": "frame.png"}, {"image": "./frame.png"}]):
            debug.write_json(self.manifest, {"cases": cases})
            with self.assertRaises(ValueError):
                debug.load_cases(self.manifest, "stats")

    def test_accuracy_only_counts_labeled_fields(self):
        rows = [{"mode": "enhanced", "seconds": 1, "ocr": [],
                 "verdicts": debug.assess({"life": 1, "attack": None, "level": 7},
                                          {"life": 1, "attack": 3, "level": 2})},
                {"mode": "normal", "seconds": 1, "ocr": [], "verdicts": {}}]
        result = debug.summarize(rows, ["enhanced", "normal"])
        self.assertEqual(result["enhanced"]["labeled_fields"], 3)
        for key in ("correct", "missing", "wrong"):
            self.assertEqual(result["enhanced"][key], 1)
        self.assertFalse(result["enhanced"]["accuracy_passed"])
        self.assertIsNone(result["normal"]["accuracy_passed"])
        self.assertEqual(debug.summarize(list(reversed(rows)), ["enhanced", "normal"]), result)

    def test_mode_disagreement_reports_values_without_claiming_ground_truth(self):
        rows = [{"image": "one.png", "mode": "normal", "actual": {"level": 7, "life": 100}},
                {"image": "one.png", "mode": "enhanced", "actual": {"level": 2, "life": 100}},
                {"image": "unpaired.png", "mode": "enhanced", "actual": {"level": 1}}]
        differences = debug.mode_differences(rows)
        self.assertEqual(differences, [{"image": "one.png", "differences": {
            "level": {"normal": 7, "enhanced": 2}}}])
        self.assertNotIn("expected", differences[0])

    def invoke(self, output, action):
        self.cases()
        with patch.object(debug, "supervise", side_effect=action), \
             patch.object(debug, "fingerprints", return_value={"source": "hash"}), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return debug.main(["--manifest", str(self.manifest), "--resource", str(self.resource),
                               "--output", str(output)])

    def test_failed_callback_is_not_success_even_if_process_exits_zero(self):
        output = self.root / "run"

        def worker(*args):
            debug.write_json(output / "worker-result.json", {
                "execution_success": False, "error": "callback failed", "summary": {}})
            return 0

        self.assertEqual(self.invoke(output, worker), 2)
        self.assertFalse(json.loads((output / "summary.json").read_text())["execution_success"])

    def test_timeout_is_recorded_as_execution_failure(self):
        output = self.root / "run"
        self.assertEqual(self.invoke(output, subprocess.TimeoutExpired("worker", 1)), 2)
        self.assertFalse(json.loads((output / "summary.json").read_text())["execution_success"])

    def test_accuracy_failure_has_distinct_exit_code(self):
        output = self.root / "run"

        def worker(*args):
            debug.write_json(output / "worker-result.json", {
                "execution_success": True, "summary": {"normal": {"accuracy_passed": False}}})
            return 0

        self.assertEqual(self.invoke(output, worker), 1)
        self.assertTrue(json.loads((output / "summary.json").read_text())["execution_success"])

    def test_existing_output_is_not_overwritten(self):
        output = self.root / "run"
        output.mkdir()
        sentinel = output / "summary.json"
        sentinel.write_text("keep")
        self.assertEqual(self.invoke(output, AssertionError("must not run")), 2)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_input_change_invalidates_an_otherwise_successful_run(self):
        output = self.root / "run"

        def worker(*args):
            self.image.write_bytes(b"changed during replay")
            debug.write_json(output / "worker-result.json", {
                "execution_success": True, "summary": {"enhanced": {"accuracy_passed": True}}})
            return 0

        self.assertEqual(self.invoke(output, worker), 2)
        self.assertFalse(json.loads((output / "summary.json").read_text())["execution_success"])

    def test_unlabeled_success_does_not_claim_accuracy(self):
        output = self.root / "run"

        def worker(*args):
            debug.write_json(output / "worker-result.json", {
                "execution_success": True, "summary": {"enhanced": {"accuracy_passed": None}}})
            return 0

        self.assertEqual(self.invoke(output, worker), 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertIsNone(summary["summary"]["enhanced"]["accuracy_passed"])

    def test_live_requires_explicit_device_and_single_mode(self):
        for arguments in (["--live", "navigation"],
                          ["--live", "navigation", "--adb-path", str(self.image), "--serial", "device", "--mode", "both"],
                          ["--live", "navigation", "--adb-path", str(self.image), "--serial", "device", "--steps", "6"],
                          ["--live", "current", "--manifest", str(self.manifest)]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    debug.main(arguments)
                self.assertEqual(raised.exception.code, 2)

    def test_live_defaults_to_enhanced_and_preserves_failed_restoration_evidence(self):
        output = self.root / "live-run"

        def worker(*args):
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["modes"], ["enhanced"])
            self.assertEqual(manifest["cases"], [])
            debug.write_json(output / "live-result.json", {"restored": None, "phase": "restore_disc"})
            debug.write_json(output / "worker-result.json", {
                "execution_success": False, "error": "restore failed", "summary": {}})
            return 2

        with patch.object(debug, "supervise", side_effect=worker), \
             patch.object(debug, "fingerprints", return_value={}), \
             contextlib.redirect_stderr(io.StringIO()):
            status = debug.main(["--live", "disc", "--adb-path", str(self.image), "--serial", "device",
                                 "--resource", str(self.resource), "--output", str(output)])
        self.assertEqual(status, 2)
        summary = json.loads((output / "summary.json").read_text())
        self.assertIsNone(summary["live"]["restored"])
        self.assertEqual(summary["live"]["phase"], "restore_disc")

    def test_replay_manifest_preserves_game_and_resource_without_device_settings(self):
        self.cases()
        document = json.loads(self.manifest.read_text())
        document.update(game="如鸢", resource="resource")
        debug.write_json(self.manifest, document)
        output = self.root / "replay"

        def worker(*args):
            config = json.loads((output / "manifest.json").read_text())
            self.assertEqual(config["game"], "如鸢")
            self.assertEqual(Path(config["resource"]), self.resource.resolve())
            self.assertNotIn("serial", config)
            self.assertEqual(config["cases"][0]["expected"], {})
            debug.write_json(output / "worker-result.json", {
                "execution_success": True, "summary": {"enhanced": {"accuracy_passed": None}}})
            return 0

        with patch.object(debug, "supervise", side_effect=worker), \
             patch.object(debug, "fingerprints", return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            status = debug.main(["--manifest", str(self.manifest), "--output", str(output)])
        self.assertEqual(status, 0)

    @unittest.skipIf(os.name == "nt", "POSIX process-group cleanup")
    def test_timeout_kills_real_worker_process(self):
        pidfile = self.root / "pid"
        code = "import os,time,pathlib;pathlib.Path(%r).write_text(str(os.getpid()));time.sleep(30)" % str(pidfile)
        with (self.root / "log").open("w") as log:
            with self.assertRaises(subprocess.TimeoutExpired):
                debug.supervise([sys.executable, "-c", code], log, 1, os.environ.copy())
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)


if __name__ == "__main__":
    unittest.main()
