"""Exercise the real upload worker and HTTP transport without a game or cloud writes."""
import copy
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread, get_ident
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom.action import agent_info_collector as collector
from custom.action import operator_scan_upload as uploads
from custom.action.inventory_reporting import UploadSettings
from custom.action.operator_growth_exchange import build_v3_document


def source_record():
    return {
        "operator_id": "char_095_zhangyan", "name": "张燕", "name_raw": "张燕",
        "stats": {}, "oddities": {}, "huaji": {}, "disc_configs": None,
        "collection_debug": {"operator_match": "name"},
    }


def document(scan_id):
    return build_v3_document([source_record()], scan_id)


@contextmanager
def local_api(*, hold=None, fail=None, hold_record="scan:first"):
    calls, entered, release = [], Event(), Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            stage = self.path.rsplit("/", 1)[-1]
            record_id = payload["records"][0]["record_id"]
            calls.append((stage, record_id, payload))
            status, response = 200, {"data": {"accepted": 1}}
            if record_id == hold_record and stage == hold:
                entered.set()
                if not release.wait(5):
                    status, response = 503, {"error": "test gate timed out"}
            if record_id.endswith("first") and stage == fail:
                status, response = 500, {"error": "test-secret"}
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    serving.start()
    settings = UploadSettings("自动上报", "test-secret", f"http://127.0.0.1:{server.server_port}")
    try:
        yield SimpleNamespace(settings=settings, calls=calls, entered=entered, release=release)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)


@contextmanager
def upload_queue(api, *, commit=True):
    queue = uploads.OperatorScanUploadQueue(api.settings, commit=commit)
    try:
        yield queue
    finally:
        api.release.set()
        queue.finish(lambda: False)


class OperatorScanUploadTests(unittest.TestCase):
    def test_next_checkpoint_is_saved_while_first_upload_is_blocked(self):
        with local_api(hold="preview", hold_record="scan:batch-0001") as api, tempfile.TemporaryDirectory() as directory:
            reader = collector._AgentInfoReader.__new__(collector._AgentInfoReader)
            output = Path(directory) / "report.json"
            reader.params = {"upload": True, "output": str(output)}
            reader._upload_settings = api.settings
            reader.scan_id, reader.publish_sequence = "batch", 0
            reader.scan_started_at, reader.game = "2026-11-17T15:00:00+08:00", "代号鸢"
            reader.context = SimpleNamespace(
                get_node_data=Mock(side_effect=AssertionError("Upload settings must stay frozen"))
            )
            owner = get_ident()
            responses = []

            def upload_response(payload, stage, **kwargs):
                self.assertEqual(get_ident(), owner)
                responses.append((stage, kwargs["operator_id"], kwargs["client_review_count"]))

            reader._toolbox_diagnostics = SimpleNamespace(checkpoint=Mock(), upload_response=upload_response)
            records = []
            try:
                reader._publish_checkpoint(records, source_record())
                self.assertTrue(api.entered.wait(2))
                second = source_record()
                second.update(operator_id="char_084_chendengsp", name="陈登·黍王", name_raw="陈登黍王")
                reader._publish_checkpoint(records, second)
                saved = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(len(saved["records"]), 2)
                self.assertEqual(len(api.calls), 1)
                self.assertEqual(saved["records"][0], api.calls[0][2]["records"][0])
                self.assertFalse(reader._console_progress["char_095_zhangyan"][0])
                self.assertFalse(reader._console_progress["char_084_chendengsp"][0])
            finally:
                api.release.set()
                reader.finish_uploads()
            self.assertEqual([(stage, record_id) for stage, record_id, _ in api.calls],
                             [("preview", "scan:batch-0001"), ("commit", "scan:batch-0001"),
                              ("preview", "scan:batch-0002"), ("commit", "scan:batch-0002")])
            reader.context.get_node_data.assert_not_called()
            self.assertEqual([item[:2] for item in responses],
                             [("preview", "char_095_zhangyan"), ("commit", "char_095_zhangyan"),
                              ("preview", "char_084_chendengsp"), ("commit", "char_084_chendengsp")])
            self.assertTrue(all(status[0] for status in reader._console_progress.values()))

    def test_queue_owns_snapshot_and_preserves_fifo_preview_commit_pairs(self):
        with local_api(hold="preview") as api, upload_queue(api) as queue:
            first, second = document("first"), document("second")
            expected_first, expected_second = copy.deepcopy(first), copy.deepcopy(second)
            queue.enqueue(first)
            self.assertTrue(api.entered.wait(2))
            queue.enqueue(second)
            first["records"][0]["entries"][0]["name"] = "changed first"
            second["records"][0]["entries"][0]["name"] = "changed second"
            self.assertEqual(len(api.calls), 1)
            api.release.set()
            queue.finish(lambda: False)
            self.assertFalse(queue._worker.is_alive())
            self.assertEqual([(stage, record_id) for stage, record_id, _ in api.calls],
                             [("preview", "scan:first"), ("commit", "scan:first"),
                              ("preview", "scan:second"), ("commit", "scan:second")])
            self.assertEqual([payload for _, _, payload in api.calls],
                             [expected_first, expected_first, expected_second, expected_second])

    def test_stop_skips_pending_uploads_and_commit_after_inflight_preview(self):
        for held_stage in ("preview", "commit"):
            for initial_wait in (False, True):
                with self.subTest(stage=held_stage, wait=initial_wait), local_api(hold=held_stage) as api:
                    with upload_queue(api) as queue:
                        queue.enqueue(document("first"))
                        self.assertTrue(api.entered.wait(2))
                        queue.enqueue(document("second"))

                        def release_when_cancelled():
                            queue._cancel.wait(2)
                            api.release.set()

                        releaser = Thread(target=release_when_cancelled, daemon=True)
                        releaser.start()
                        polls = iter([False, True] if initial_wait else [True])
                        queue.finish(lambda: next(polls, True))
                        releaser.join(timeout=2)
                        self.assertFalse(queue._worker.is_alive())
                        expected = [("preview", "scan:first")]
                        if held_stage == "commit":
                            expected.append(("commit", "scan:first"))
                        self.assertEqual([(stage, record_id) for stage, record_id, _ in api.calls], expected)
                        self.assertEqual(queue._skipped, 2 if held_stage == "preview" else 1)

    def test_http_failure_does_not_block_later_documents_and_redacts_secret(self):
        for failed_stage in ("preview", "commit"):
            with self.subTest(stage=failed_stage), local_api(fail=failed_stage) as api:
                with patch.object(uploads, "logger") as log, upload_queue(api) as queue:
                    queue.enqueue(document("first"))
                    queue.enqueue(document("second"))
                    queue.finish(lambda: False)
                    self.assertEqual([(stage, record_id) for stage, record_id, _ in api.calls][-2:],
                                     [("preview", "scan:second"), ("commit", "scan:second")])
                    warning = log.bind.return_value.warning
                    warning.assert_called_once()
                    self.assertNotIn("test-secret", str(warning.call_args))
                    self.assertIn("<redacted>", str(warning.call_args))

    def test_old_upload_result_does_not_mark_newer_checkpoint_synced(self):
        with local_api(hold="preview", hold_record="scan:second") as api:
            reader = collector._AgentInfoReader.__new__(collector._AgentInfoReader)
            reader.context = SimpleNamespace()
            reader.params = {"upload": True}
            reader._upload_settings = api.settings
            reader._console_progress = {"char_095_zhangyan": (False, False)}
            try:
                reader._queue_upload_if_enabled(document("first"))
                reader._queue_upload_if_enabled(document("second"))
                self.assertTrue(api.entered.wait(2))
                reader._drain_upload_results()
                self.assertFalse(reader._console_progress["char_095_zhangyan"][0])
            finally:
                api.release.set()
                reader.finish_uploads()
            self.assertTrue(reader._console_progress["char_095_zhangyan"][0])

    def test_action_summary_waits_for_background_results(self):
        with local_api(hold="preview", hold_record="scan:batch-0001") as api, tempfile.TemporaryDirectory() as directory:
            class Reader(collector._AgentInfoReader):
                def _load_operators(self):
                    return {}

                def run(self):
                    self.scan_id = "batch"
                    self._publish_checkpoint([], source_record())
                    return True

                def finish_uploads(self):
                    api.release.set()
                    super().finish_uploads()

            context = SimpleNamespace(get_node_data=lambda name: {"attach": {"upload": True}} if name == "密探采集上报配置" else {})
            account = SimpleNamespace(id="acc_test", name="test")
            with (patch.object(collector, "_AgentInfoReader", Reader),
                  patch.object(collector, "read_upload_settings", return_value=api.settings),
                  patch.object(collector, "get_bound_account", return_value=account),
                  patch.object(collector._AgentInfoReader, "_output_path", return_value=Path(directory) / "report.json"),
                  patch.object(collector, "logger") as log):
                result = collector.AgentInfoCollector().run(context, SimpleNamespace(custom_action_param={}))
            self.assertTrue(result.success)
            self.assertIn("已同步 1 位", "\n".join(str(call.args[0]) for call in log.info.call_args_list))
            self.assertEqual([stage for stage, _, _ in api.calls], ["preview", "commit"])

    def test_preview_only_and_closed_queue(self):
        with local_api() as api, upload_queue(api, commit=False) as queue:
            queue.enqueue(document("first"))
            queue.finish(lambda: False)
            self.assertEqual([stage for stage, _, _ in api.calls], ["preview"])
            with self.assertRaisesRegex(RuntimeError, "队列已关闭"):
                queue.enqueue(document("second"))

    def test_local_only_does_not_create_worker_or_read_credentials(self):
        reader = collector._AgentInfoReader.__new__(collector._AgentInfoReader)
        reader.params = {"upload": False}
        reader.context = Mock()
        with patch.object(collector, "OperatorScanUploadQueue") as factory:
            reader._queue_upload_if_enabled(document("first"))
            reader.finish_uploads()
        factory.assert_not_called()
        reader.context.get_node_data.assert_not_called()

    def test_comparison_readers_do_not_require_upload_authentication(self):
        for mode in ("name_compare", "disc_compare"):
            with self.subTest(mode=mode), patch.object(collector._AgentInfoReader, "_load_operators", return_value={}):
                reader = collector._AgentInfoReader(Mock(), {"upload": True, mode: True})
                self.assertIsNone(reader._upload_queue)
                self.assertIsNone(reader._upload_settings)

    def test_action_always_finishes_queue_on_success_error_and_stop(self):
        for outcome in (True, RuntimeError("scan failed"), InterruptedError("stopped")):
            with self.subTest(outcome=outcome):
                run = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
                reader = SimpleNamespace(run=run, finish_uploads=Mock(), _failure_diagnostics=None)
                with patch.object(collector, "_AgentInfoReader", return_value=reader):
                    result = collector.AgentInfoCollector().run(
                        SimpleNamespace(get_node_data=lambda name: {}), SimpleNamespace(custom_action_param={})
                    )
                self.assertEqual(result.success, outcome is True)
                reader.finish_uploads.assert_called_once()


if __name__ == "__main__":
    unittest.main()
