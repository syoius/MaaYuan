import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom.action.operator_catalog_sync import refresh_operator_catalog
from custom.action.operator_growth_exchange import _operator_catalog_by_id, set_operator_catalog
from custom.action.agent_info_collector import _AgentInfoReader


class OperatorCatalogSyncTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "operators.json"
        self.old = {"OPERATORS": [{"id": "old", "name": "旧密探"}]}
        self.new = {"OPERATORS": [{"id": "new", "name": "新密探"}]}
        self.path.write_text(json.dumps(self.old), encoding="utf-8")
        self.addCleanup(_operator_catalog_by_id.cache_clear)

    def response(self, payload):
        response = Mock(status=200)
        response.read.return_value = json.dumps(payload).encode()
        response.headers.get_content_charset.return_value = "utf-8"
        response.headers.get.return_value = None
        return response

    def test_checks_each_scan_despite_recent_sync_and_updates_file(self):
        self.path.with_name("operators.sync.meta").write_text(json.dumps({"last_check_ts": int(time.time()), "etag": "old-etag"}))
        with patch("custom.action.autoformation.urllib_request.urlopen") as fetch:
            fetch.return_value.__enter__.return_value = self.response(self.new)
            self.assertEqual(refresh_operator_catalog(self.path), self.new)
            self.assertEqual(refresh_operator_catalog(self.path), self.new)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(fetch.call_args.kwargs["timeout"], 8)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), self.new)

    def test_not_modified_and_network_failure_preserve_local(self):
        for error in (HTTPError("https://example.test", 304, "Not Modified", {}, None), URLError("offline")):
            with self.subTest(error=error), patch("custom.action.autoformation.urllib_request.urlopen", side_effect=error):
                self.assertEqual(refresh_operator_catalog(self.path), self.old)
                self.assertEqual(json.loads(self.path.read_text()), self.old)

    def test_invalid_remote_does_not_replace_local(self):
        for payload in ({"OPERATORS": []}, {"OPERATORS": [{"name": "缺少ID"}]}, {"OPERATORS": [None]}):
            with self.subTest(payload=payload), patch("custom.action.autoformation.urllib_request.urlopen") as fetch:
                fetch.return_value.__enter__.return_value = self.response(payload)
                self.assertEqual(refresh_operator_catalog(self.path), self.old)
                self.assertEqual(json.loads(self.path.read_text()), self.old)

    def test_no_usable_catalog_stops(self):
        self.path.unlink()
        with patch("custom.action.autoformation.urllib_request.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(RuntimeError, "停止密探采集"):
                refresh_operator_catalog(self.path)

    def test_reader_and_exchange_share_remote_data_even_if_write_fails(self):
        with patch("custom.action.autoformation.urllib_request.urlopen") as fetch, patch(
            "custom.action.operator_catalog_sync._CollectorCatalogSync._write_operators_file", return_value=False
        ):
            fetch.return_value.__enter__.return_value = self.response(self.new)
            data = refresh_operator_catalog(self.path)
        set_operator_catalog(self.old)
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader._ensure_running = Mock()
        with patch("custom.action.agent_info_collector.refresh_operator_catalog", return_value=data):
            operators = reader._load_operators()
        self.assertEqual(operators["新密探"]["id"], "new")
        self.assertIn("new", _operator_catalog_by_id())
        self.assertNotIn("old", _operator_catalog_by_id())


if __name__ == "__main__":
    unittest.main()
