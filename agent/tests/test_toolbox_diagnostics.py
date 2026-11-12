import copy
import base64
import gzip
import hashlib
from datetime import datetime, timedelta, timezone
from email.message import Message
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom.action import agent_info_collector as collector
from custom.action import inventory_reporting
from custom.action import operator_growth_exchange as exchange
from custom.action import toolbox_diagnostics as diagnostics
from custom.action import toolbox_diagnostics_log as diagnostic_log
from custom.action import toolbox_diagnostics_transport as transport
from custom.action.inventory_reporting import BoundAccount, UploadSettings

TOKEN = "private-connection-secret"
PRIVATE = [TOKEN, r"C:\Users\alice", "测试账号", "acc_private", "alice@example.com", "Bearer", "secret-report.json"]


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        token_store = patch.object(
            inventory_reporting, "TOKEN_STORE_PATH", Path(directory.name) / "yuanhub_connection.json"
        )
        token_store.start()
        self.addCleanup(token_store.stop)
        self.worker = Mock()
        self.worker.configure.return_value = 1
        self.guard = patch.object(diagnostics, "_WORKER", self.worker)
        self.guard.start()
        self.addCleanup(self.guard.stop)
        log_guard = patch.object(diagnostics, "collect_log", return_value=None)
        self.collect_log = log_guard.start()
        self.addCleanup(log_guard.stop)
        diagnostics.COUNTERS.clear()
        self.session = diagnostics.DiagnosticSession(TOKEN, transport.TRUSTED_BASE, "summary", "base")
        self.worker.reset_mock()

    def package(self):
        calls = [call for call in self.worker.submit.call_args_list if len(call.args) == 2]
        self.assertEqual(len(calls), 1)
        return calls[0].args[1][:2]

    def scan_disc(self, frame):
        reader = collector._AgentInfoReader.__new__(collector._AgentInfoReader)
        reader._toolbox_diagnostics = self.session
        reader._require_page = Mock(return_value=frame)
        reader.ocr_text = Mock(return_value="")
        return reader._scan_disc_config(None)

    def assert_private(self, raw):
        raw = raw.decode() if isinstance(raw, bytes) else raw
        for value in PRIVATE:
            self.assertNotIn(value, raw)
            self.assertNotIn(json.dumps(value)[1:-1], raw)

    def context(self, mode="summary", token=TOKEN):
        nodes = {"在线上传认证": {"attach": {"token": token, "base_url": transport.TRUSTED_BASE, "mode": "自动上报"}},
                 "密探采集诊断配置": {"attach": {"toolbox_diagnostics": mode}}}
        return SimpleNamespace(get_node_data=lambda key: nodes.get(key), stop=False,
                               tasker=SimpleNamespace(stopping=False, running=True))

    def action(self, run=None, constructor=None, **params):
        reader = Mock(_failure_diagnostics=None, _console_progress={})
        reader._should_stop.return_value = False
        reader.run.side_effect = run if isinstance(run, BaseException) or callable(run) else None
        reader.run.return_value = True if run is None else run
        with (patch.object(collector, "_AgentInfoReader", side_effect=constructor, return_value=reader),
              patch.object(collector, "read_upload_settings", return_value=UploadSettings("自动上报", TOKEN, transport.TRUSTED_BASE)),
              patch.object(collector, "get_bound_account", return_value=BoundAccount("acc_private", "测试账号"))):
            return collector.AgentInfoCollector().run(self.context(), SimpleNamespace(custom_action_param={"upload": True, **params}))

    def test_sample_a_exception_is_sanitized_and_does_not_change_failure(self):
        def fail():
            raise RuntimeError(" ".join(PRIVATE))
        result = self.action(run=fail)
        self.assertFalse(result.success)
        manifest, files = self.package()
        self.assertEqual(manifest["outcome"], "failed")
        issue = manifest["issues"][0]
        self.assertEqual(issue["code"], "collector_exception")
        self.assertEqual(issue["error_type"], "RuntimeError")
        self.assertTrue(any(frame.startswith("agent/custom/action/agent_info_collector.py:run:") for frame in issue["stack"]))
        self.assertEqual(files, [])
        self.assert_private(json.dumps(manifest, ensure_ascii=False))

    def test_sample_b_rejection_survives_business_catch_and_merges_by_field(self):
        response = Mock(status=200)
        response.read.return_value = json.dumps({"data": {"rejected": 1, "items": [{
            "status": "rejected", "blocking_errors": [
                {"code": "invalid_disc_loadout", "field": "disc_loadouts[0].discs[0].ot_name", "message": " ".join(PRIVATE)},
                {"code": "invalid_level", "field": "level", "message": " ".join(PRIVATE)}]}]}}).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        reader = collector._AgentInfoReader.__new__(collector._AgentInfoReader)
        reader.params = {"upload": True}
        reader.context = self.context()
        reader._toolbox_diagnostics = self.session
        self.session.operator_id = "char_001_yangxiu"
        with (patch.object(exchange.urllib_request, "urlopen", return_value=response),
              patch.object(exchange, "validate_v3_document"), patch.object(exchange, "scrub_v3_equipment")):
            self.assertIsNone(reader._upload_v3_if_enabled({"records": [{"record_id": "private", "entries": [{"operator_id": "char_001_yangxiu"}]}]}))
            self.session.operator_id = "char_002_test"
            reader._upload_v3_if_enabled({"records": [{"record_id": "private", "entries": [{"operator_id": "char_002_test"}]}]})
        self.session.finish(True)
        manifest, _ = self.package()
        self.assertEqual(manifest["outcome"], "upload_failed")
        self.assertEqual(len(manifest["issues"]), 2)
        for issue in manifest["issues"]:
            self.assertEqual(issue["code"], "upload_rejected")
            self.assertEqual(issue["count"], 2)
            self.assertEqual(issue["stage"], "preview")
            self.assertEqual(len(issue["operator_ids"]), 2)
        self.assert_private(json.dumps(manifest, ensure_ascii=False))

    def test_pre_reader_account_annotations_and_catalog_failures(self):
        for target, code, extra in (
            ("get_bound_account", "account_lookup_failed", {}),
            ("read_growth_states", "annotations_failed", {"active_only": True}),
            ("refresh_operator_catalog", "catalog_unavailable", {}),
        ):
            with self.subTest(target=target):
                self.worker.reset_mock()
                with (patch.object(collector, "read_upload_settings", return_value=UploadSettings("自动上报", TOKEN, transport.TRUSTED_BASE)),
                      patch.object(collector, "get_bound_account", return_value=BoundAccount("acc_private", "测试账号")),
                      patch.object(collector, target, side_effect=RuntimeError(" ".join(PRIVATE)))):
                    result = collector.AgentInfoCollector().run(self.context(), SimpleNamespace(custom_action_param={"upload": True, **extra}))
                self.assertFalse(result.success)
                manifest, _ = self.package()
                self.assertIn(code, [issue["code"] for issue in manifest["issues"]])
                self.assert_private(json.dumps(manifest, ensure_ascii=False))

    def test_stopped_local_only_no_token_and_full_success_do_not_report(self):
        self.assertFalse(self.action(run=InterruptedError()).success)
        self.assertFalse([call for call in self.worker.submit.call_args_list if len(call.args) == 2])
        for params in ({"upload": False}, {"upload": True, "name_compare": True}, {"upload": True, "disc_compare": True}):
            self.assertIsNone(diagnostics.start_diagnostics(self.context(), params))
        self.assertIsNone(diagnostics.start_diagnostics(self.context(token=""), {"upload": True}))
        self.worker.reset_mock()
        self.assertTrue(self.action().success)
        self.assertFalse([call for call in self.worker.submit.call_args_list if len(call.args) == 2])
        self.session.issue("data_review", "checkpoint")
        self.session.finish(False, stopped=True)
        self.assertFalse([call for call in self.worker.submit.call_args_list if len(call.args) == 2])

    def test_missing_auth_node_silently_skips_diagnostics(self):
        context = SimpleNamespace(get_node_data=Mock(side_effect=KeyError("missing")))
        with patch.object(diagnostics.logger, "warning") as warning:
            self.assertIsNone(diagnostics.start_diagnostics(context, {"upload": True}))
            warning.assert_not_called()

    def test_log_is_collected_only_when_reporting_and_queued_as_bytes(self):
        self.session.finish(True)
        self.session.finish(False, stopped=True)
        self.collect_log.assert_not_called()
        self.session.protect("测试账号")
        log = gzip.compress(b"redacted log", mtime=0)
        self.collect_log.return_value = log
        self.session.finish(False)
        self.collect_log.assert_called_once_with(self.session.log_directory, self.session.started_at, self.session.secrets)
        self.assertEqual(self.worker.submit.call_args.args[1][2], log)
        self.assertEqual(self.session.log_directory, Path.cwd() / "debug/custom")

    def test_aggregation_keeps_statuses_and_machine_error_codes_across_operators(self):
        self.session.issue("upload_rejected", "commit", "char_001_yangxiu", errors=[{"code": "INVALID_DISC", "field": "entries[0].disc_loadouts[0].discs[0].ot_name"}], statuses={"basic": "partial"})
        self.session.issue("upload_rejected", "commit", "char_002_test", errors=[{"code": "invalid_catalog", "field": "entries[0].disc_loadouts[0].discs[0].ot_name"}], statuses={"disc_loadouts": "review"})
        self.session.finish(True)
        manifest, _ = self.package()
        issue = manifest["issues"][0]
        self.assertEqual(issue["count"], 2)
        self.assertEqual(len(issue["operator_ids"]), 2)
        self.assertEqual(len(issue["blocking_errors"]), 2)
        self.assertEqual(issue["section_status"], {"basic": "partial", "disc_loadouts": "review"})

    def test_review_and_identity_issues_keep_operator_evidence_separate(self):
        operators = [("char_001_yangxiu", "杨修", "攻击提升", "basic"),
                     ("char_002_test", "张飞", "生命提升", "disc_loadouts")]
        self.session.catalog_info({"OPERATORS": [
            {"name": name, "discs": [{"ot_name": sample}]} for _, name, sample, _ in operators]})
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        for oid, name, sample, section in operators:
            self.session.observe_main(frame, collector.ROI_CONFIGS["代号鸢"], oid)
            self.session.observe_disc(sample)
            self.session.issue("identity_unresolved", "identity", oid, sample=name,
                               errors=[{"code": "invalid_" + section, "field": "identity.operator_id"}],
                               statuses={section: "review"})
        self.session.checkpoint({"records": [{"entries": [
            {"operator_id": oid, "section_status": {section: "review"}}
            for oid, _, _, section in operators]}]})
        self.session.finish(True)
        manifest, files = self.package()
        self.assertEqual(len(manifest["issues"]), 4)
        for oid, name, sample, section in operators:
            issues = {item["code"]: item for item in manifest["issues"] if item["operator_ids"] == [oid]}
            self.assertEqual(set(issues), {"identity_unresolved", "data_review"})
            for item in issues.values():
                self.assertEqual(item["count"], 1)
                self.assertEqual(item["section_status"], {section: "review"})
            self.assertEqual(issues["identity_unresolved"]["ocr_samples"], [name])
            self.assertEqual(issues["identity_unresolved"]["blocking_errors"], [{"code": "invalid_" + section, "field": "identity.operator_id"}])
            self.assertEqual(issues["data_review"]["ocr_samples"], [sample])
        self.assertEqual(files, [])
        self.assertEqual(self.session.image_cache, {})

    def test_section_review_prioritizes_private_filtered_raw_disc_names(self):
        oid = "char_071_caozhi"
        self.session.catalog_info({"OPERATORS": [{"discs": [{"ot_name": "初始能量+3"}]}]})
        self.session.known_seen[oid] = ["初始能量+3"]
        self.session.protect("隐私原文")
        entry = {"operator_id": oid, "section_status": {"disc_loadouts": "partial"}}
        document = {"records": [{"entries": [entry]}]}
        self.session.checkpoint(document)
        entry["diagnostics"] = {"disc_name_review": [
            "音义远", "音义远", "隐私原文", TOKEN, "超过八个汉字的原文内容", "音义远3", "",
            "青衿远", "君不见", "无法确认", "额外原文",
        ]}
        self.session.checkpoint(document)
        self.session.finish(True)
        manifest, files = self.package()
        issue = manifest["issues"][0]
        self.assertEqual(issue["reason"], "section_needs_review")
        self.assertEqual(issue["ocr_samples"], ["音义远", "青衿远", "君不见", "无法确认"])
        self.assertEqual(files, [])
        raw = diagnostics.encode_manifest(manifest, (*PRIVATE, "隐私原文"))
        self.assert_private(raw)
        self.assertNotIn("隐私原文", raw.decode())

    def test_failed_diagnostic_adapter_cannot_change_action_return(self):
        broken = Mock()
        broken.finish.side_effect = RuntimeError("diagnostic failure")
        broken.protect.side_effect = RuntimeError("diagnostic failure")
        with patch.object(collector, "start_diagnostics", return_value=broken):
            self.assertTrue(self.action().success)
            self.assertFalse(self.action(run=RuntimeError("business failure")).success)
        self.worker.submit.side_effect = RuntimeError("queue failure")
        self.session.issue("data_review", "checkpoint")
        self.assertIsNone(self.session.finish(True))

    def test_checkpoint_covers_statuses_and_unmatched_without_serializing_report(self):
        document = {"accounts": PRIVATE, "records": [{"entries": [
            {"operator_id": "char_001_yangxiu", "name": "测试账号", "section_status": {
                "basic": "partial", "disc_loadouts": "review", "oddities": "unavailable", "combat_stats": "partial"}, "diagnostics": PRIVATE}],
            "unmatched": [{"raw_name": " ".join(PRIVATE)}]}]}
        self.session.checkpoint(document)
        self.session.finish(True)
        manifest, _ = self.package()
        self.assertEqual({item["code"] for item in manifest["issues"]}, {"data_review", "identity_unresolved"})
        self.assert_private(json.dumps(manifest, ensure_ascii=False))

    def test_server_review_only_reports_excess_over_current_checkpoint(self):
        entries = [
            {"operator_id": "char_001_yangxiu", "section_status": {"basic": "partial", "disc_loadouts": "review"}},
            {"operator_id": "char_002_test", "section_status": {"basic": "ready"}},
        ]
        for client_count, checkpoint_entries in ((1, entries), (0, entries[1:])):
            self.session.checkpoint({"records": [{"entries": checkpoint_entries}]})
            for partial, review in ((0, 0), (1, 0), (0, 1), (1, 1)):
                for stage in ("preview", "commit"):
                    with self.subTest(client_count=client_count, partial=partial, review=review, stage=stage):
                        self.session.issues.clear()
                        self.session.upload_response({"data": {"accepted": 1, "partial": partial, "review": review}}, stage)
                        issues = [item for item in self.session.issues.values() if item.get("reason") == "server_marked_review"]
                        self.assertEqual(len(issues), int(partial + review > client_count))

    def test_upload_network_http_commit_and_unverified_paths(self):
        for stage in ("preview", "commit"):
            for code in ("upload_network_failed", "upload_http_failed", "upload_rejected", "upload_response_unverified"):
                with self.subTest(stage=stage, code=code):
                    self.worker.reset_mock()
                    self.session.issues.clear()
                    reader = collector._AgentInfoReader.__new__(collector._AgentInfoReader)
                    reader.params = {"upload": True}
                    reader.context = self.context()
                    reader._toolbox_diagnostics = self.session
                    kwargs = {"return_value": {"raw": " ".join(PRIVATE)}} if code.endswith("unverified") else {"side_effect": exchange.ExchangeApiError(" ".join(PRIVATE), code)}
                    with (patch.object(collector, "preview_v3_document", return_value={"data": {"accepted": 1}}),
                          patch.object(collector, "commit_v3_document", return_value={"data": {"accepted": 1}}),
                          patch.object(collector, stage + "_v3_document", **kwargs)):
                        reader._upload_v3_if_enabled({"records": [{"record_id": "private", "entries": [{"operator_id": "char_001_yangxiu"}]}]})
                    self.session.finish(True)
                    manifest, _ = self.package()
                    self.assertIn((code, stage), [(item["code"], item["stage"]) for item in manifest["issues"]])
                    self.assert_private(json.dumps(manifest, ensure_ascii=False))

    def test_explicit_navigation_screenshot_ocr_and_identity_failures(self):
        reader = collector._AgentInfoReader.__new__(collector._AgentInfoReader)
        reader._toolbox_diagnostics = self.session
        reader.context = SimpleNamespace(tasker=Mock(), run_task=Mock())
        reader._ensure_running = Mock()
        reader.context.tasker.controller.post_screencap.side_effect = OSError(" ".join(PRIVATE))
        self.assertIsNone(reader.screenshot())
        reader.context.run_recognition = Mock(side_effect=RuntimeError(" ".join(PRIVATE)))
        self.assertIsNone(reader.ocr(np.zeros((1280, 720, 3), dtype=np.uint8), (0, 0, 20, 20)))
        reader._require_page = Mock(side_effect=RuntimeError("enter failed"))
        reader.params = {"scan_single": True}
        self.assertFalse(reader.run())
        reader._require_page = Mock(return_value=np.zeros((1280, 720, 3), dtype=np.uint8))
        reader._read_main = Mock(return_value={"name": ""})
        reader._prepare_growth_filter = Mock(return_value={"name": ""})
        reader._load_records = Mock(return_value=[])
        reader.max_operators = 1
        self.assertFalse(reader.run())
        self.assertTrue({"navigation_failed", "screenshot_or_ocr_failed", "identity_unresolved"}.issubset({item["code"] for item in self.session.issues.values()}))

    def test_images_are_opt_in_scaled_crops_and_encoded_without_screen_header(self):
        frame = np.zeros((2560, 1440, 3), dtype=np.uint8)
        frame[:400] = 255
        frame[1666:1728, 266:438] = 127
        self.session.observe_main(frame, collector.ROI_CONFIGS["代号鸢"], "char_001_yangxiu")
        self.assertEqual(self.session.images, [])
        self.assertEqual(self.session.image_cache, {})
        self.session.mode = "images"
        self.session.observe_main(frame, collector.ROI_CONFIGS["代号鸢"], "char_001_yangxiu")
        self.scan_disc(frame)
        self.scan_disc(frame)
        self.session.issue("data_review", "checkpoint")
        self.session.finish(True)
        manifest, files = self.package()
        self.assertEqual(len(files), 3)
        self.assertEqual(manifest["client"]["resolution"], "1440x2560")
        for meta, data in zip(manifest["images"], files):
            self.assertLessEqual(len(data), transport.MAX_IMAGE)
            crop = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(crop.shape[:2], (meta["roi"][3], meta["roi"][2]))
            self.assertLess(crop.shape[0] * crop.shape[1], frame.shape[0] * frame.shape[1])
            self.assertLess(crop.mean(), 250)
        body, _ = transport.multipart(diagnostics.encode_manifest(manifest, (TOKEN,)), files)
        self.assert_private(body.decode("latin1"))

    def test_images_are_allocated_per_problem_operator_with_relevant_roles(self):
        self.session.mode = "images"
        operators = ["char_001_yangxiu", "char_002_test", "char_003_test"]
        self.session.observe_main(np.zeros((1280, 720, 3), dtype=np.uint8), collector.ROI_CONFIGS["代号鸢"], None)
        self.session.issue("identity_unresolved", "identity")
        for index, oid in enumerate(operators):
            frame = np.full((1280, 720, 3), 50 * (index + 1), dtype=np.uint8)
            self.session.observe_main(frame, collector.ROI_CONFIGS["代号鸢"], None if index == 2 else oid)
            self.scan_disc(frame)
            self.scan_disc(frame)
            if index == 2:
                self.session.resolve_operator(oid)
            if index == 0:
                self.session.issue("collector_exception", "disc", reason="disc_plan_unknown")
            elif index == 1:
                self.session.issue("identity_unresolved", "identity")
            else:
                self.session.issue("data_review", "checkpoint", statuses={"disc_loadouts": "review"})
            self.assertEqual(len(self.session.image_cache[oid]), 3)
            self.assertTrue(all(isinstance(data, bytes) for _, data in self.session.image_cache[oid]))
        self.session.finish(True)
        manifest, files = self.package()
        self.assertEqual([meta["operator_id"] for meta in manifest["images"]], operators)
        self.assertEqual([meta["role"] for meta in manifest["images"]], ["disc_cells", "name_area", "disc_cells"])
        self.assertEqual([meta["index"] for meta in manifest["images"]], [0, 1, 2])
        self.assertEqual(len(files), 3)
        for index, data in enumerate(files):
            crop = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertAlmostEqual(crop.mean(), 50 * (index + 1), delta=1)

    def test_disc_scan_crops_one_scaled_bounding_rectangle_per_config(self):
        self.session.mode = "images"
        frame = np.full((2560, 1440, 3), 80, dtype=np.uint8)
        oid = "char_001_yangxiu"
        self.session.observe_main(frame, collector.ROI_CONFIGS["代号鸢"], oid)
        for index in (1, 2):
            with patch.object(self.session, "crop", wraps=self.session.crop) as crop:
                self.scan_disc(frame)
            crop.assert_called_once()
            self.assertEqual(crop.call_args.args[1:], ((14, 203, 692, 691), "disc_cells"))
            self.assertEqual(len(self.session.image_cache[oid]), 1 + index)
        self.session.issue("data_review", "disc", statuses={"disc_loadouts": "review"})
        self.session.finish(True)
        manifest, files = self.package()
        self.assertEqual([meta["role"] for meta in manifest["images"]], ["disc_cells", "disc_cells", "name_area"])
        self.assertEqual(manifest["images"][0]["roi"], [28, 406, 1384, 1382])
        decoded = cv2.imdecode(np.frombuffer(files[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (1382, 1384))

    def test_failed_manifest_prioritizes_last_full_frame_with_matching_roi(self):
        self.session.mode = "images"
        frame = np.random.default_rng(0).integers(0, 256, (2560, 1440, 3), dtype=np.uint8)
        oid = "char_001_yangxiu"
        self.session.observe_main(frame, collector.ROI_CONFIGS["代号鸢"], oid)
        self.scan_disc(np.zeros_like(frame))
        self.scan_disc(np.zeros_like(frame))
        reader = collector._AgentInfoReader.__new__(collector._AgentInfoReader)
        reader._toolbox_diagnostics = self.session
        reader._ensure_running = Mock()
        controller = Mock()
        controller.post_screencap.return_value.wait.return_value.get.return_value = frame
        reader.context = SimpleNamespace(tasker=SimpleNamespace(controller=controller))
        self.assertIs(reader.screenshot(), frame)
        controller.post_screencap.side_effect = OSError("screenshot failed")
        self.assertIsNone(reader.screenshot())
        self.session.issue("collector_exception", "disc", reason="disc_plan_unknown")
        self.session.finish(False)
        manifest, files = self.package()
        self.assertEqual(manifest["outcome"], "failed")
        self.assertEqual(len(files), 3)
        self.assertEqual([meta["role"] for meta in manifest["images"]], ["last_frame", "disc_cells", "disc_cells"])
        meta = manifest["images"][0]
        self.assertEqual(meta, {"index": 0, "role": "last_frame", "roi": [0, 0, 1440, 2560], "operator_id": oid})
        self.assertLessEqual(len(files[0]), transport.MAX_IMAGE)
        decoded = cv2.imdecode(np.frombuffer(files[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, frame.shape)
        diagnostics.encode_manifest(manifest, (TOKEN,))

    def test_last_frame_is_omitted_for_summary_success_and_review(self):
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        self.session.observe_frame(frame)
        self.assertIsNone(self.session.last_frame)
        self.session.mode = "images"
        self.session.observe_frame(frame)
        self.session.finish(True)
        self.worker.submit.assert_not_called()
        self.session.issue("data_review", "disc")
        self.session.finish(True)
        manifest, files = self.package()
        self.assertEqual(manifest["outcome"], "data_review")
        self.assertEqual(files, [])

    def test_last_frame_encoding_failure_does_not_drop_diagnostics(self):
        self.session.mode = "images"
        self.session.observe_frame(np.zeros((1280, 720, 3), dtype=np.uint8))
        with patch.object(cv2, "imencode", side_effect=RuntimeError("encoding failed")):
            self.session.finish(False)
        manifest, files = self.package()
        self.assertEqual(manifest["outcome"], "failed")
        self.assertEqual(files, [])

    def test_known_ocr_only_and_final_secret_check_drops_whole_package(self):
        self.session.catalog_info({"OPERATORS": [{"name": "杨修", "discs": [{"ot_name": "攻击提升"}]}]}, {"source": "remote", "version": "v1"})
        for value in PRIVATE + ["任意账号名", "杨修", "攻击提升"]:
            self.session.issue("data_review", "main", sample=value)
        self.assertEqual(next(iter(self.session.issues.values()))["ocr_samples"], ["杨修", "攻击提升"])
        self.session.client["resource_version"] = TOKEN
        self.session.finish(True)
        self.worker.submit.assert_not_called()
        self.assertEqual(diagnostics.COUNTERS["diagnostics_privacy_dropped"], 1)

    def test_strict_schema_and_budget_reject_unknown_and_private_fields(self):
        self.session.issue("data_review", "checkpoint")
        self.session.finish(True)
        manifest, _ = self.package()
        for location in ("top", "client", "issue", "error", "catalog"):
            invalid = copy.deepcopy(manifest)
            target = invalid if location == "top" else invalid[location] if location in {"client", "catalog"} else invalid["issues"][0]
            if location == "error":
                target["blocking_errors"] = [{"code": "invalid_disc", "field": "alice@example.com", "message": TOKEN}]
            else:
                target["params"] = PRIVATE
            with self.assertRaises((ValueError, TypeError)):
                diagnostics.encode_manifest(invalid, (TOKEN,))
        with self.assertRaises(ValueError):
            transport.multipart(b"a" * (transport.MAX_MANIFEST + 1), [])
        with self.assertRaises(ValueError):
            transport.multipart(b"{}", [b"\xff\xd8" + b"x" * transport.MAX_IMAGE])

    def test_interface_and_both_resources_share_consent_defaults(self):
        root = Path(__file__).resolve().parents[2]
        interface = json.loads((root / "assets/interface.json").read_text())
        option = interface["option"]["帮助改进百宝箱"]
        self.assertEqual(option["default_case"], "发送脱敏诊断")
        self.assertIn("姓名区、整盘命盘，以及失败时的报错画面（整屏，可能包含游戏界面信息）", option["description"])
        collection = next(item for item in interface["option"]["百宝箱-模式"]["cases"] if item["name"] == "采集密探信息")
        self.assertNotIn("帮助改进百宝箱", collection["option"])
        self.assertNotIn("仅更新养成中的密探", collection["option"])
        self.assertIn("同步至YuanHub", collection["option"])
        sync = {case["name"]: case for case in interface["option"]["同步至YuanHub"]["cases"]}
        self.assertEqual(sync["No"]["option"], ["密探采集本地文件名"])
        self.assertEqual(sync["Yes"]["option"], ["连接码", "仅更新养成中的密探", "帮助改进百宝箱"])
        for resource in ("base", "zh_tw"):
            pipeline = json.loads((root / f"assets/resource/{resource}/pipeline/agent_info_collector.json").read_text())
            self.assertEqual(pipeline["密探采集诊断配置"]["attach"]["toolbox_diagnostics"], "summary")
            for case, mode in zip(option["cases"], ("off", "summary", "images")):
                self.assertEqual(case["pipeline_override"]["密探采集诊断配置"]["attach"]["toolbox_diagnostics"], mode)


class WorkerTests(unittest.TestCase):
    package = DiagnosticTests.package
    assert_private = DiagnosticTests.assert_private

    def setUp(self):
        DiagnosticTests.setUp(self)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.real_worker = diagnostics.DiagnosticWorker(Path(self.directory.name))
        self.real_worker.mode = "summary"
        self.real_worker.epoch = 1
        self.session.issue("collector_exception", "main", exception=RuntimeError(" ".join(PRIVATE)))
        self.session.finish(False)
        self.manifest, self.files = self.package()
        self.job = (1, TOKEN, transport.TRUSTED_BASE, (self.manifest, self.files, None))
        self.worker.reset_mock()

    def test_queue_is_nonblocking_and_bounded(self):
        for _ in range(10):
            self.real_worker.submit(self.session, (self.manifest, [], None))
        self.assertEqual(self.real_worker.queue.qsize(), 4)
        self.assertEqual(diagnostics.COUNTERS["diagnostics_queue_dropped"], 6)

    def test_offline_persists_sanitized_package_then_retries_same_event(self):
        with patch.object(diagnostics, "deliver", return_value=("pending", 0)) as send:
            self.real_worker._process(self.job)
        files = list(Path(self.directory.name).glob("*.json"))
        self.assertEqual(len(files), 1)
        self.assert_private(files[0].read_text())
        self.assertEqual(json.loads(send.call_args.args[0])["event_id"], self.manifest["event_id"])
        with patch.object(diagnostics, "deliver", return_value=("accepted", 0)) as send:
            self.real_worker._process((1, "different-token", transport.TRUSTED_BASE, None))
            send.assert_not_called()
            self.real_worker._process((1, TOKEN, transport.TRUSTED_BASE, None))
            self.assertEqual(json.loads(send.call_args.args[0])["event_id"], self.manifest["event_id"])
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_log_is_redacted_before_disk_and_retried_without_reading_files(self):
        raw = diagnostic_log.redact_log(f"{TOKEN} 测试账号 /Users/alice/debug\n", [TOKEN, "测试账号"])
        log = gzip.compress(raw.encode(), mtime=0)
        job = (1, TOKEN, transport.TRUSTED_BASE, (self.manifest, [], log))
        with patch.object(diagnostics, "deliver", return_value=("pending", 0)) as send:
            self.real_worker._process(job)
            self.assertEqual(send.call_args.kwargs["log"], log)
        path = next(Path(self.directory.name).glob("*.json"))
        envelope = json.loads(path.read_text())
        self.assertEqual(base64.b64decode(envelope["log"]), log)
        self.assert_private(gzip.decompress(base64.b64decode(envelope["log"])))
        self.real_worker.strip_images.set()
        self.real_worker._tick()
        self.assertEqual(json.loads(path.read_text())["log"], envelope["log"])
        with patch.object(diagnostics, "deliver", return_value=("accepted", 0)) as send:
            self.real_worker._process((1, TOKEN, transport.TRUSTED_BASE, None))
            self.assertEqual(send.call_args.kwargs["log"], log)
        self.collect_log.assert_called_once()
        self.assertFalse(path.exists())

    def test_invalid_pending_log_does_not_block_summary(self):
        self.real_worker._save(self.manifest, [], hashlib.sha256(TOKEN.encode()).hexdigest(), 0)
        path = next(Path(self.directory.name).glob("*.json"))
        for value in ("not base64!", base64.b64encode(b"not gzip").decode()):
            envelope = json.loads(path.read_text())
            envelope["log"] = value
            path.write_text(json.dumps(envelope))
            with patch.object(diagnostics, "deliver", return_value=("pending", 0)) as send:
                self.real_worker._process((1, TOKEN, transport.TRUSTED_BASE, None))
                self.assertIsNone(send.call_args.kwargs["log"])
            self.assertNotIn("log", json.loads(path.read_text()))

    def test_pending_count_bytes_expiry_and_disable(self):
        for _ in range(23):
            self.real_worker._save(self.manifest, [], "binding", 0)
        self.assertEqual(len(list(Path(self.directory.name).glob("*.json"))), 20)
        self.assertEqual(diagnostics.COUNTERS["diagnostics_pending_dropped"], 3)
        path = next(Path(self.directory.name).glob("*.json"))
        os.utime(path, (time.time() - 8 * 86400,) * 2)
        self.assertEqual(len(self.real_worker._prune()), 19)
        self.real_worker.MAX_BYTES = path.stat().st_size * 2 if path.exists() else len(json.dumps(self.manifest)) * 3
        self.real_worker._prune()
        self.assertLessEqual(sum(path.stat().st_size for path in Path(self.directory.name).glob("*.json")), self.real_worker.MAX_BYTES)
        self.real_worker.mode = "off"
        self.real_worker.purge.set()
        self.real_worker.queue.put_nowait(self.job)
        self.real_worker._tick()
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_disk_corruption_and_network_failure_stay_inside_boundary(self):
        Path(self.directory.name, "invalid.json").write_text('{"token":"private-connection-secret"}')
        self.real_worker.queue.put_nowait((1, TOKEN, transport.TRUSTED_BASE, None))
        with patch.object(diagnostics, "deliver") as send:
            self.real_worker._tick()
            send.assert_not_called()
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        self.real_worker.queue.put_nowait(self.job)
        with patch.object(self.real_worker, "_save", side_effect=OSError("disk full")):
            self.assertIsNone(self.real_worker._tick())
        self.assertEqual(self.real_worker.queue.unfinished_tasks, 0)

    def test_retry_after_future_and_permanent_failure(self):
        with patch.object(diagnostics, "deliver", return_value=("pending", time.time() + 3600)):
            self.real_worker._process(self.job)
        with patch.object(diagnostics, "deliver") as send:
            self.real_worker._process((1, TOKEN, transport.TRUSTED_BASE, None))
            send.assert_not_called()
        path = next(Path(self.directory.name).glob("*.json"))
        envelope = json.loads(path.read_text())
        envelope["retry_at"] = 0
        path.write_text(json.dumps(envelope))
        with patch.object(diagnostics, "deliver", return_value=("permanent", 0)):
            self.real_worker._process((1, TOKEN, transport.TRUSTED_BASE, None))
        self.assertFalse(path.exists())

    def test_retry_and_image_withdrawal_preserve_original_expiry(self):
        self.real_worker._save(self.manifest, [], hashlib.sha256(TOKEN.encode()).hexdigest(), 0)
        path = next(Path(self.directory.name).glob("*.json"))
        old = time.time() - 6 * 86400
        os.utime(path, (old, old))
        with patch.object(diagnostics, "deliver", return_value=("pending", 0)):
            self.real_worker._process((1, TOKEN, transport.TRUSTED_BASE, None))
        self.assertAlmostEqual(path.stat().st_mtime, old, places=3)
        self.real_worker.strip_images.set()
        self.real_worker._tick()
        self.assertAlmostEqual(path.stat().st_mtime, old, places=3)

    def test_only_one_daemon_is_started_and_submit_does_not_wait(self):
        thread = Mock()
        thread.is_alive.return_value = True
        with patch.object(diagnostics.threading, "Thread", return_value=thread) as constructor:
            self.real_worker.configure("images")
            self.real_worker.configure("images")
            self.assertEqual(constructor.call_count, 1)
            self.assertTrue(constructor.call_args.kwargs["daemon"])
            thread.start.assert_called_once()

    def test_summary_removes_pending_images_and_old_jobs_are_revoked(self):
        self.real_worker._save({**self.manifest, "images": [{"index": 0, "role": "disc_cells", "roi": [14, 203, 172, 170]}]}, [b"\xff\xd8crop"], "binding", 0)
        self.real_worker.strip_images.set()
        self.real_worker._tick()
        envelope = json.loads(next(Path(self.directory.name).glob("*.json")).read_text())
        self.assertEqual(envelope["files"], [])
        self.assertEqual(envelope["manifest"]["images"], [])
        self.real_worker.epoch = 2
        with patch.object(diagnostics, "deliver") as send:
            self.real_worker._process(self.job)
            send.assert_not_called()


class LogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.started = datetime(2026, 1, 2, 23, 59, 59, 500000)

    def test_timestamp_filter_includes_continuations_and_previous_day_zip(self):
        with zipfile.ZipFile(self.root / "2026-01-02.log.zip", "w") as archive:
            archive.writestr("2026-01-02.log", (
                "orphan before first entry\n"
                "2026-01-02 23:59:59.499 | INFO | old\nold continuation\n"
                "2026-01-02 23:59:59.500 | ERROR | first\nTraceback:\n  File test.py\n"))
        (self.root / "2026-01-03.log").write_text(
            "2026-01-03 00:00:00.000 | INFO | second\nnew continuation\n", encoding="utf-8")
        raw = gzip.decompress(diagnostic_log.collect_log(self.root, self.started, [])).decode()
        self.assertNotIn("old", raw)
        self.assertNotIn("orphan", raw)
        self.assertIn("first\nTraceback:\n  File test.py\n", raw)
        self.assertIn("second\nnew continuation\n", raw)
        self.assertLess(raw.index("first"), raw.index("second"))

    def test_redaction_paths_variables_credentials_long_strings_and_email(self):
        text = (
            r"C:\Users\yangp\agent C:/Users/Win User/file /Users/macuser/file /home/linuxuser/file" + "\n"
            + r"C:\\Users\\escaped\\agent" + "\n"
            "  │ secret variable\n  └ more variable\n┌ another variable\n╰ final variable\n"
            "RuntimeError: keep this message\n"
            "private-long private 测试账号 acc_private\n"
            'token=short-token access_token: other-secret 连接码：中文连接码\n'
            'authorization: Bearer short-auth\nBearer short-bearer\n'
            '{"token": "quoted secret", "连接码": "another secret"}\n'
            "abcdef0123456789abcdef0123 QUJDREVGR0hJSktMTU5PUFFSU1Q= alice@example.com\n")
        raw = diagnostic_log.redact_log(text, ["private", "private-long", "测试账号", "acc_private"])
        for secret in ("yangp", "Win User", "macuser", "linuxuser", "escaped", "secret variable", "more variable",
                       "another variable", "final variable", "private", "-long", "测试账号", "acc_private",
                       "short-token", "other-secret", "中文连接码", "short-auth", "short-bearer", "quoted secret",
                       "another secret", "abcdef0123456789abcdef0123", "QUJDREVGR0hJSktMTU5PUFFSU1Q=", "alice@example.com"):
            self.assertNotIn(secret, raw)
        self.assertIn(r"C:\Users\***\agent", raw)
        self.assertIn("/Users/***/file", raw)
        self.assertIn("/home/***/file", raw)
        self.assertIn("RuntimeError: keep this message", raw)

    def test_opaque_redaction_preserves_function_names_and_masks_mixed_tokens(self):
        names = ("ensure_linux_venv_and_relaunch_if_needed", "_resolve_locked_by_elimination",
                 "check_and_install_dependencies")
        lines = [f"agent.utils:{name}:123" for name in names]
        tokens = ("9rSv2Jm5Qx8Lk1Np4Td7Wz0B", "M8k2_N4p6-Q0r8+S2t4/U6v8=")
        text = "\n".join([*lines, *(f"[{value}]" for value in tokens)])
        self.assertEqual(diagnostic_log.redact_log(text, []), "\n".join([*lines, "[***]", "[***]"]))
        self.assertEqual(diagnostic_log.redact_log(names[0], [names[0]]), "***")

    def test_tail_is_200_kib_valid_utf8_with_truncation_notice(self):
        text = "2026-01-02 23:59:59.500 | ERROR | start\n" + ("日志内容 " * 80 + "\n") * 1000 + "日志末尾\n"
        (self.root / "2026-01-02.log").write_text(text, encoding="utf-8")
        log = diagnostic_log.collect_log(self.root, self.started, [])
        raw = gzip.decompress(log)
        prefix = "（前面的日志已截断）\n".encode()
        expected = text.encode()[-diagnostic_log.MAX_LOG_TEXT:].decode("utf-8", errors="ignore").encode()
        self.assertEqual(raw, prefix + expected)
        self.assertTrue(raw.decode().endswith("日志末尾\n"))
        self.assertLessEqual(len(log), diagnostic_log.MAX_LOG)

    def test_long_lines_are_truncated_without_losing_following_entries(self):
        line = "2026-01-02 23:59:59.500 | ERROR | " + "日志" * 800
        following = "2026-01-03 00:00:00.000 | INFO | following\n"
        (self.root / "2026-01-02.log").write_text(line + "\n" + following, encoding="utf-8")
        raw = gzip.decompress(diagnostic_log.collect_log(self.root, self.started, [])).decode()
        self.assertEqual(raw, line[:1000] + "…（本行已截断）\n" + following)

    def test_missing_unreadable_and_empty_logs_produce_failure_summary(self):
        raw = gzip.decompress(diagnostic_log.collect_log(self.root, self.started, [])).decode()
        self.assertIn("（日志收集失败）", raw)
        self.assertIn("失败原因=没有匹配的日志条目", raw)
        self.assertIn("异常类型=无", raw)
        (self.root / "2026-01-02.log").write_text("2026-01-02 23:59:59.499 | INFO | old\n", encoding="utf-8")
        raw = gzip.decompress(diagnostic_log.collect_log(self.root, self.started, [])).decode()
        self.assertIn("失败原因=没有匹配的日志条目", raw)
        self.assertNotIn("| INFO | old", raw)
        with patch.object(Path, "open", side_effect=OSError("unreadable")):
            log = diagnostic_log.collect_log(self.root, self.started, [])
        self.assertTrue(diagnostic_log.valid_log(log))
        raw = gzip.decompress(log).decode()
        self.assertIn("失败原因=读取或处理异常(OSError)", raw)
        self.assertIn("异常类型=OSError", raw)
        self.assertIn("unreadable", raw)

    def test_log_size_gzip_and_utf8_validation(self):
        self.assertTrue(diagnostic_log.valid_log(gzip.compress("运行日志".encode())))
        for data in (b"", b"not gzip", b"x" * (diagnostic_log.MAX_LOG + 1),
                     gzip.compress(b"\xff"), gzip.compress(b"x" * (diagnostic_log.MAX_EXPANDED_LOG + 1)),
                     gzip.compress(b"valid")[:-3]):
            self.assertFalse(diagnostic_log.valid_log(data))


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.dumps({"event_id": "test-event"}).encode()

    def test_log_intent_put_and_complete_without_images(self):
        log = gzip.compress("诊断日志".encode(), mtime=0)
        md5 = base64.b64encode(hashlib.md5(log, usedforsecurity=False).digest()).decode()
        upload = {"method": "PUT", "url": "https://bucket.myqcloud.com/log?signature=private",
                  "headers": {"Content-Type": "application/gzip", "Content-MD5": md5, "Content-Length": str(len(log))}}
        accepted = {"status": 200, "payload": {"event_id": "test-event", "status": "accepted", "log_upload": upload}}
        with patch.object(transport, "_isolated_once", side_effect=[accepted, {"status": 200}, {"status": 200}]) as send:
            self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN, direct=True, log=log)[0], "accepted")
        intent, put, complete = send.call_args_list
        body = json.loads(intent.args[2])
        self.assertEqual(body["log"], {"size": len(log), "content_md5": md5})
        self.assertEqual(body["uploads"], [])
        self.assertEqual(intent.kwargs["path"], "/open-api/maayuan/diagnostics/intents")
        self.assertEqual(put.args[:3], ("", "", log))
        self.assertEqual(put.kwargs["upload"], {"url": upload["url"], "headers": upload["headers"]})
        self.assertEqual(complete.args[2], b"{}")
        self.assertEqual(complete.kwargs["path"], "/open-api/maayuan/diagnostics/test-event/complete")

    def test_images_and_log_share_isolated_put_validation(self):
        log = gzip.compress(b"log", mtime=0)
        image = b"\xff\xd8crop"
        upload = {"method": "PUT", "url": "https://bucket.myqcloud.com/image", "headers": {"Content-Type": "image/jpeg"}}
        payload = {"event_id": "test-event", "status": "accepted", "uploads": [{**upload, "index": 0}],
                   "log_upload": {**upload, "url": "https://bucket.myqcloud.com/log", "headers": {"Content-Type": "application/gzip"}}}
        raw = json.dumps({"event_id": "test-event", "images": [{}]}).encode()
        with patch.object(transport, "_isolated_once", side_effect=[{"status": 200, "payload": payload}, {"status": 200}, {"status": 200}, {"status": 200}]) as send:
            self.assertEqual(transport.deliver(raw, [image], transport.TRUSTED_BASE, TOKEN, direct=True, log=log)[0], "accepted")
        self.assertEqual([call.args[2] for call in send.call_args_list[1:]], [image, log, b"{}"])
        for bad in ({**upload, "url": "https://evil.test/log"}, {**upload, "method": "POST"},
                    {**upload, "headers": {"x": "value\r\nAuthorization: bad"}}, None):
            with patch.object(transport, "_isolated_once") as send:
                self.assertFalse(transport._upload_one(bad, log))
                send.assert_not_called()

    def test_400_retries_intent_once_without_log(self):
        log = gzip.compress(b"log", mtime=0)
        accepted = {"status": 200, "payload": {"event_id": "test-event", "status": "accepted"}}
        with patch.object(transport, "_isolated_once", side_effect=[{"status": 400}, accepted]) as send:
            self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN, direct=True, log=log)[0], "accepted")
        first, second = send.call_args_list
        self.assertIn("log", json.loads(first.args[2]))
        self.assertNotIn("log", json.loads(second.args[2]))
        self.assertEqual(first.kwargs, second.kwargs)
        with patch.object(transport, "_isolated_once", return_value={"status": 400}) as send:
            self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN, direct=True, log=log)[0], "permanent")
            self.assertEqual(send.call_count, 2)

    def test_multipart_fallback_does_not_include_log(self):
        log = gzip.compress(b"log", mtime=0)
        accepted = {"status": 200, "payload": {"event_id": "test-event", "status": "accepted"}}
        for status in (404, 405, 409):
            with patch.object(transport, "_isolated_once", side_effect=[{"status": status}, accepted]) as send:
                self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN, direct=True, log=log)[0], "accepted")
            fallback = send.call_args_list[1]
            self.assertTrue(fallback.args[3].startswith("multipart/form-data"))
            self.assertNotIn(log, fallback.args[2])
            self.assertNotIn(b'name="log"', fallback.args[2])

    def test_invalid_log_and_budget_rejection_do_not_block_event(self):
        for state in ("accepted_without_images", "duplicate"):
            with patch.object(transport, "_isolated_once", return_value={"status": 200, "payload": {"event_id": "test-event", "status": state}}) as send:
                self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN, direct=True, log=gzip.compress(b"log"))[0], "accepted")
                self.assertEqual(send.call_count, 1)
        for log in (b"not gzip", b"x" * (diagnostic_log.MAX_LOG + 1), gzip.compress(b"\xff")):
            with patch.object(transport, "_isolated_once", return_value={"status": 200, "payload": {"event_id": "test-event", "status": "accepted"}}) as send:
                self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN, direct=True, log=log)[0], "accepted")
                self.assertNotIn("log", json.loads(send.call_args.args[2]))
                self.assertEqual(send.call_count, 1)

    def test_acceptance_requires_http_200_status_and_matching_event(self):
        for state in ("accepted", "duplicate", "accepted_without_images"):
            with patch.object(transport, "_isolated_once", return_value={"status": 200, "payload": {"status": state, "event_id": "test-event"}}):
                self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN)[0], "accepted")
        for payload in ({}, {"status": "accepted", "event_id": "wrong"}, [], {"status": "rejected"}):
            with patch.object(transport, "_isolated_once", return_value={"status": 200, "payload": payload}) as send:
                self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN)[0], "permanent")
                self.assertEqual(send.call_count, 1)

    def test_retry_policy_deadline_and_retry_after(self):
        for status in (0, 408, 429, 500, 503):
            with (patch.object(transport, "_isolated_once", return_value={"status": status}),
                  patch.object(transport.time, "sleep")):
                self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN)[0], "pending")
        for status in (200, 301, 400, 401, 403, 413, 422):
            with patch.object(transport, "_isolated_once", return_value={"status": status}) as send:
                self.assertEqual(transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN)[0], "permanent")
                self.assertEqual(send.call_count, 1)
        with (patch.object(transport, "_isolated_once", return_value={"status": 429, "retry_after": "60"}) as send,
              patch.object(transport.time, "sleep") as sleep):
            state, retry_at = transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN)
            self.assertEqual(state, "pending")
            self.assertGreaterEqual(retry_at, time.time() + 59)
            self.assertEqual(send.call_count, 1)
            sleep.assert_not_called()
        with (patch.object(transport.time, "monotonic", side_effect=[0, 0, 5, 5, 8]),
              patch.object(transport.time, "sleep"),
              patch.object(transport, "_isolated_once", return_value={"status": 0}) as send):
            transport.deliver(self.raw, [], transport.TRUSTED_BASE, TOKEN)
            self.assertEqual([call.args[-1] for call in send.call_args_list], [5, 3])
        http_date = (datetime.now(timezone.utc) + timedelta(seconds=60)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        self.assertGreater(transport.retry_delay(http_date), 58)

    def test_trusted_https_only_no_redirect_and_credentials_only_in_header(self):
        for base in ("http://api-hub.maayuan.com", "https://evil.test", "https://api-hub.maayuan.com.evil.test", "https://user:pass@api-hub.maayuan.com", "https://api-hub.maayuan.com/path", "https://api-hub.maayuan.com?token=x"):
            with patch.object(transport, "_isolated_once") as send:
                self.assertEqual(transport.deliver(self.raw, [], base, TOKEN)[0], "permanent")
                send.assert_not_called()
        self.assertIsNone(transport._NoRedirect().redirect_request(Mock(), None, 302, "", {}, "https://evil.test"))
        response = Mock(status=200)
        response.read.return_value = b'{"status":"accepted","event_id":"test-event"}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch.object(transport.request, "build_opener", return_value=opener):
            transport._http_once(transport.TRUSTED_BASE, TOKEN, self.raw, "multipart/form-data", 5)
        req = opener.open.call_args.args[0]
        self.assertEqual(req.get_header("Authorization"), "Bearer " + TOKEN)
        self.assertNotIn(TOKEN.encode(), req.data)
        response.read.assert_called_once_with(transport.MAX_RESPONSE + 1)

    def test_http_error_limits_and_timeout_process_is_killable(self):
        opener = Mock()
        headers = Message()
        headers["Retry-After"] = "60"
        body = Mock()
        opener.open.side_effect = HTTPError(transport.TRUSTED_BASE, 429, "private", headers, body)
        with patch.object(transport.request, "build_opener", return_value=opener):
            self.assertEqual(transport._http_once(transport.TRUSTED_BASE, TOKEN, b"{}", "x", 5), {"status": 429, "retry_after": "60"})
        body.read.assert_not_called()
        with patch.object(transport.subprocess, "run", side_effect=subprocess.TimeoutExpired("child", 5)) as run:
            self.assertEqual(transport._isolated_once(transport.TRUSTED_BASE, TOKEN, b"{}", "x", 5), {"status": 0})
            self.assertEqual(run.call_args.kwargs["timeout"], 5)
            self.assertNotIn(TOKEN, " ".join(run.call_args.args[0]))
            self.assertIn(TOKEN.encode(), run.call_args.kwargs["input"])
        for error in (URLError("offline"), OSError("offline")):
            opener.open.side_effect = error
            with patch.object(transport.request, "build_opener", return_value=opener):
                self.assertEqual(transport._http_once(transport.TRUSTED_BASE, TOKEN, b"{}", "x", 5), {"status": 0})

    def test_transport_child_hides_console_only_on_windows(self):
        for platform in ("win32", "linux", "darwin"):
            with (self.subTest(platform=platform),
                  patch.object(transport.sys, "platform", platform),
                  patch.object(transport.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
                  patch.object(transport.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=b'{"status":400}')) as run):
                self.assertEqual(transport._isolated_once("https://evil.test", TOKEN, b"{}", "x", 5), {"status": 400})
                if platform == "win32":
                    self.assertEqual(run.call_args.kwargs["creationflags"], subprocess.CREATE_NO_WINDOW)
                else:
                    self.assertNotIn("creationflags", run.call_args.kwargs)

    def test_real_transport_child_rejects_untrusted_host_without_network(self):
        self.assertEqual(transport._isolated_once("https://evil.test", TOKEN, b"{}", "x", 5), {"status": 400})


if __name__ == "__main__":
    unittest.main()
