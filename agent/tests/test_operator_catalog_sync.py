import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom.action.autoformation import AutoFormation
from custom.action.operator_catalog_sync import refresh_operator_catalog
from custom.action.operator_growth_exchange import _operator_catalog_by_id, set_operator_catalog
from custom.action.agent_info_collector import _AgentInfoReader


class OperatorCatalogSyncTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "operators.json"
        self.remote_path = self.path.with_name("operators.remote.json")
        self.meta_path = self.path.with_name("operators.sync.meta")
        self.old = {"OPERATORS": [{"id": "old", "name": "旧密探", "rarity": 5, "discs": [{"ot_name": "旧命盘"}]}]}
        self.new = {"OPERATORS": [{"id": "new", "name": "新密探", "rarity": 5, "discs": [{"ot_name": "新命盘"}]}]}
        self.path.write_text(json.dumps(self.old), encoding="utf-8")
        self.addCleanup(_operator_catalog_by_id.cache_clear)

    def response(self, payload, headers=None):
        response = Mock(status=200)
        response.read.return_value = json.dumps(payload).encode()
        response.geturl.return_value = AutoFormation.OPERATORS_REMOTE_URL
        response.headers.get_content_charset.return_value = "utf-8"
        response.headers.get.side_effect = (headers or {}).get
        return response

    def test_checks_each_scan_despite_recent_sync_and_updates_file(self):
        original_bytes = self.path.read_bytes()
        self.meta_path.write_text(json.dumps({
            "last_check_ts": int(time.time()),
            "source_url": AutoFormation.OPERATORS_REMOTE_URL,
            "remote_hash": AutoFormation._operators_hash(self.old),
            "etag": '"old-etag"',
            "last_modified": "Mon, 09 Nov 2026 00:00:00 GMT",
        }))
        with patch("custom.action.autoformation.urllib_request.urlopen") as fetch:
            fetch.return_value.__enter__.return_value = self.response(self.new, {"ETag": '"new-etag"'})
            self.assertEqual(refresh_operator_catalog(self.path), self.new)
            first_request = fetch.call_args.args[0]
            self.assertIsNone(first_request.get_header("If-none-match"))
            self.assertIsNone(first_request.get_header("If-modified-since"))
            self.assertEqual(refresh_operator_catalog(self.path), self.new)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(fetch.call_args.kwargs["timeout"], 8)
            self.assertEqual(fetch.call_args.args[0].get_header("If-none-match"), '"new-etag"')
        self.assertEqual(json.loads(self.remote_path.read_text(encoding="utf-8")), self.new)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        meta = json.loads(self.meta_path.read_text())
        self.assertEqual(meta["source_url"], AutoFormation.OPERATORS_REMOTE_URL)
        self.assertEqual(meta["etag"], '"new-etag"')
        self.assertEqual(meta["last_modified"], "")
        self.assertEqual(meta["remote_hash"], AutoFormation._operators_hash(self.new))

    def test_alias_updates_reach_collection_and_export_on_the_next_scan(self):
        from custom.action import operator_growth_exchange as exchange

        remote = {"OPERATORS": [{"id": "char_074_xunyou", "name": "荀攸", "rarity": 5, "discs": [
            {"ot_name": "于无声处", "aliases": ["第一版错读"]},
        ]}]}
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader._ensure_running = Mock()
        for alias in ("第一版错读", "新增错读"):
            remote["OPERATORS"][0]["discs"][0]["aliases"] = [alias]
            with patch("custom.action.autoformation.urllib_request.urlopen") as fetch:
                fetch.return_value.__enter__.return_value = self.response(remote)
                with patch("custom.action.agent_info_collector.refresh_operator_catalog",
                           side_effect=lambda _: refresh_operator_catalog(self.path)):
                    reader.operators = reader._load_operators()
            self.assertEqual(json.loads(self.remote_path.read_text()), remote)
            self.assertEqual(json.loads(self.path.read_text()), self.old)
            self.assertEqual(reader._known_disc_name(alias), "于无声处")
            self.assertEqual(exchange._canonical_disc_name("char_074_xunyou", alias), "于无声处")
        self.assertIsNone(reader._known_disc_name("第一版错读"))
        self.assertIsNone(exchange._canonical_disc_name("char_074_xunyou", "第一版错读"))
        with patch("custom.action.autoformation.urllib_request.urlopen", side_effect=URLError("offline")):
            self.assertEqual(refresh_operator_catalog(self.path), remote)

    def test_not_modified_and_network_failure_preserve_local(self):
        original_bytes = self.path.read_bytes()
        for cache_state in ("valid", "missing", "corrupt"):
            if cache_state == "valid":
                self.remote_path.write_text(json.dumps(self.new), encoding="utf-8")
            elif cache_state == "missing":
                self.remote_path.unlink()
            else:
                self.remote_path.write_text("invalid JSON", encoding="utf-8")
            meta = {
                "source_url": AutoFormation.OPERATORS_REMOTE_URL,
                "remote_hash": AutoFormation._operators_hash(self.new),
                "etag": '"cached-etag"',
                "last_modified": "Mon, 09 Nov 2026 00:00:00 GMT",
            }
            errors = [URLError("offline")]
            if cache_state == "valid":
                errors.insert(0, HTTPError(AutoFormation.OPERATORS_REMOTE_URL, 304, "Not Modified", {}, None))
            for error in errors:
                self.meta_path.write_text(json.dumps(meta), encoding="utf-8")
                with self.subTest(cache=cache_state, error=error), patch(
                    "custom.action.autoformation.urllib_request.urlopen", side_effect=error
                ) as fetch:
                    expected = self.new if cache_state == "valid" else self.old
                    self.assertEqual(refresh_operator_catalog(self.path), expected)
                    request = fetch.call_args.args[0]
                    if cache_state == "valid":
                        self.assertEqual(request.get_header("If-none-match"), meta["etag"])
                        self.assertEqual(request.get_header("If-modified-since"), meta["last_modified"])
                        self.assertEqual(json.loads(self.remote_path.read_text()), self.new)
                        saved_meta = json.loads(self.meta_path.read_text())
                        for key in ("etag", "last_modified", "remote_hash"):
                            self.assertEqual(saved_meta[key], meta[key])
                        if isinstance(error, HTTPError) and error.code == 304:
                            self.assertIn("last_success_ts", saved_meta)
                    else:
                        self.assertIsNone(request.get_header("If-none-match"))
                        self.assertIsNone(request.get_header("If-modified-since"))
                    self.assertEqual(self.path.read_bytes(), original_bytes)

    def test_local_placeholders_do_not_block_scan_when_remote_is_unavailable(self):
        placeholders = [
            {"id": f"char_{number}_unknown", "name": "", "rarity": 5, "discs": [{"ot_name": "占位命盘"}]}
            for number in (133, 134, 135)
        ]
        local = {"OPERATORS": [*self.old["OPERATORS"], *placeholders], "PROFESSIONS": []}
        self.path.write_text(json.dumps(local), encoding="utf-8")
        original_bytes = self.path.read_bytes()
        for error in (
            HTTPError(AutoFormation.OPERATORS_REMOTE_URL, 404, "Not Found", {}, None),
            URLError("offline"),
        ):
            with self.subTest(error=error), patch("custom.action.autoformation.urllib_request.urlopen", side_effect=error):
                data = refresh_operator_catalog(self.path)
            self.assertEqual(data, {**local, "OPERATORS": self.old["OPERATORS"]})
            self.assertEqual(self.path.read_bytes(), original_bytes)
            reader = _AgentInfoReader.__new__(_AgentInfoReader)
            reader._ensure_running = Mock()
            with patch("custom.action.agent_info_collector.refresh_operator_catalog", return_value=data):
                operators = reader._load_operators()
            self.assertEqual(operators["旧密探"]["id"], "old")
            self.assertEqual(set(_operator_catalog_by_id()), {"old"})

    def test_remote_placeholders_are_preserved_on_disk_and_skipped_in_scan(self):
        remote = {
            "OPERATORS": [*self.new["OPERATORS"], {
                "id": "char_133_unknown", "name": "", "rarity": 5, "discs": [{"ot_name": "占位命盘"}],
            }],
            "PROFESSIONS": [],
        }
        with patch("custom.action.autoformation.urllib_request.urlopen") as fetch:
            fetch.return_value.__enter__.return_value = self.response(remote)
            self.assertEqual(refresh_operator_catalog(self.path), {**remote, "OPERATORS": self.new["OPERATORS"]})
        self.assertEqual(json.loads(self.remote_path.read_text(encoding="utf-8")), remote)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), self.old)

    def test_catalog_with_only_placeholders_cannot_start_scan(self):
        self.path.write_text(json.dumps({"OPERATORS": [{
            "id": "char_133_unknown", "name": "", "rarity": 5, "discs": [{"ot_name": "占位命盘"}],
        }]}))
        with patch("custom.action.autoformation.urllib_request.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(RuntimeError, "停止密探采集"):
                refresh_operator_catalog(self.path)

    def test_invalid_remote_does_not_replace_local(self):
        for payload in (
            {"OPERATORS": []},
            {"OPERATORS": [{"name": "缺少ID", "rarity": 5, "discs": [{"ot_name": "命盘"}]}]},
            {"OPERATORS": [None]},
            {"OPERATORS": [{"id": "missing-rarity", "name": "缺少稀有度", "discs": [{"ot_name": "命盘"}]}]},
            {"OPERATORS": [{"id": "missing-discs", "name": "缺少命盘", "rarity": 5}]},
        ):
            with self.subTest(payload=payload), patch("custom.action.autoformation.urllib_request.urlopen") as fetch:
                fetch.return_value.__enter__.return_value = self.response(payload)
                self.assertEqual(refresh_operator_catalog(self.path), self.old)
                self.assertEqual(json.loads(self.path.read_text()), self.old)
                self.assertFalse(self.remote_path.exists())

    def test_placeholders_do_not_hide_invalid_records_in_remote_catalog(self):
        fields = {"rarity": 5, "discs": [{"ot_name": "命盘"}]}
        placeholder = {**fields, "id": "char_133_unknown", "name": ""}
        for invalid in (
            {**fields, "name": "缺少ID"},
            {**fields, "id": "unnamed", "name": ""},
            {**fields, "id": "unnamed", "name": " "},
            {**fields, "id": "char_134_unknown", "name": None},
            {**fields, "id": "char_134_unknown"},
            {**fields, "id": "new", "name": "重复ID"},
            placeholder,
            None,
        ):
            payload = {"OPERATORS": [*self.new["OPERATORS"], placeholder, invalid]}
            with self.subTest(invalid=invalid), patch("custom.action.autoformation.urllib_request.urlopen") as fetch:
                fetch.return_value.__enter__.return_value = self.response(payload)
                self.assertEqual(refresh_operator_catalog(self.path), self.old)
                self.assertEqual(json.loads(self.path.read_text()), self.old)
                self.assertFalse(self.remote_path.exists())

    def test_no_usable_catalog_stops(self):
        self.path.unlink()
        with patch("custom.action.autoformation.urllib_request.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(RuntimeError, "停止密探采集"):
                refresh_operator_catalog(self.path)

    def test_reader_and_exchange_share_remote_data_even_if_write_fails(self):
        for cached in (False, True):
            meta = {"source_url": AutoFormation.OPERATORS_REMOTE_URL}
            if cached:
                self.remote_path.write_text(json.dumps(self.old), encoding="utf-8")
                meta.update({
                    "etag": '"old-etag"',
                    "last_modified": "Mon, 09 Nov 2026 00:00:00 GMT",
                    "remote_hash": AutoFormation._operators_hash(self.old),
                })
            self.meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.subTest(cached=cached), patch("custom.action.autoformation.urllib_request.urlopen") as fetch, patch(
                "custom.action.operator_catalog_sync._CollectorCatalogSync._write_operators_file", return_value=False
            ):
                fetch.return_value.__enter__.return_value = self.response(self.new, {
                    "ETag": '"new-etag"',
                    "Last-Modified": "Tue, 10 Nov 2026 00:00:00 GMT",
                })
                data = refresh_operator_catalog(self.path)
                self.assertEqual(data, self.new)
                saved_meta = json.loads(self.meta_path.read_text())
                for key in ("etag", "last_modified", "remote_hash"):
                    if key in meta:
                        self.assertEqual(saved_meta[key], meta[key])
                    else:
                        self.assertNotIn(key, saved_meta)
                if cached:
                    self.assertEqual(json.loads(self.remote_path.read_text()), self.old)
                else:
                    self.assertFalse(self.remote_path.exists())
                self.assertEqual(json.loads(self.path.read_text()), self.old)
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
