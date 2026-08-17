import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest.mock import patch

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "custom" / "action" / "inventory_reporting.py"
)
SPEC = importlib.util.spec_from_file_location("inventory_reporting", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载测试模块: {MODULE_PATH}")
inventory_reporting = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory_reporting
SPEC.loader.exec_module(inventory_reporting)


class _Context:
    def __init__(self, attach):
        self.attach = attach

    def get_node_data(self, name):
        if name != inventory_reporting.AUTH_NODE_NAME:
            return None
        return {"attach": self.attach}


class _Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self.payload


def _agent_result(name, operator_id, count):
    return {
        "entity_type": "agent",
        "item_id": name,
        "item_name": name,
        "operator_id": operator_id,
        "operator_name": name,
        "count": count,
    }


def _item_result(name, item_id, count):
    return {
        "entity_type": "item",
        "item_id": item_id,
        "item_name": name,
        "operator_id": item_id,
        "operator_name": name,
        "count": count,
    }


class InventoryReportingTests(unittest.TestCase):
    def setUp(self):
        inventory_reporting._ACCOUNT_CACHE_KEY = None
        inventory_reporting._ACCOUNT_CACHE_VALUE = None

    def test_reads_runtime_attach_and_requires_token_for_auto_upload(self):
        settings = inventory_reporting.read_upload_settings(
            _Context(
                {
                    "mode": "自动上报",
                    "token": "runtime-token",
                    "base_url": "http://127.0.0.1:8080/",
                    "inventory_report_filename": "大号",
                }
            )
        )

        self.assertEqual(settings.mode, "自动上报")
        self.assertEqual(settings.token, "runtime-token")
        self.assertEqual(settings.base_url, "http://127.0.0.1:8080")
        self.assertEqual(settings.report_filename, "大号")
        self.assertNotIn("runtime-token", repr(settings))
        with self.assertRaisesRegex(ValueError, "需要填写 token"):
            inventory_reporting.read_upload_settings(
                _Context(
                    {
                        "mode": "自动上报",
                        "token": "",
                    }
                )
            )
        local = inventory_reporting.read_upload_settings(
            _Context({"mode": "仅保存到本地"})
        )
        self.assertEqual(local.token, "")
        self.assertIsNone(local.report_filename)

    def test_custom_report_filename_builds_account_specific_txt_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = inventory_reporting.resolve_inventory_report_destination(
                None, "大号", root
            )
            self.assertEqual(path, (root / "DailyRewards-大号.txt").resolve())

            with self.assertRaisesRegex(ValueError, "Windows 不允许"):
                inventory_reporting.resolve_inventory_report_destination(
                    None, "大号/小号", root
                )
            with self.assertRaisesRegex(ValueError, "不能为空"):
                inventory_reporting.resolve_inventory_report_destination(
                    None, "   ", root
                )

    def test_bound_account_generates_safe_unique_report_filename(self):
        account = inventory_reporting.BoundAccount(
            id="acc_main", name='大/小:鸟? "双修"'
        )
        part = inventory_reporting.bound_account_report_filename(account)
        self.assertEqual(part, "大_小_鸟_ _双修_-acc_main")
        with tempfile.TemporaryDirectory() as directory:
            path = inventory_reporting.resolve_inventory_report_destination(
                None, part, Path(directory)
            )
            self.assertEqual(
                path,
                (Path(directory) / "DailyRewards-大_小_鸟_ _双修_-acc_main.txt").resolve(),
            )

        unnamed = inventory_reporting.BoundAccount(id="acc_fallback", name="")
        self.assertEqual(
            inventory_reporting.bound_account_report_filename(unnamed),
            "acc_fallback",
        )

    def test_bound_account_is_cached_and_token_change_requeries(self):
        first = inventory_reporting.UploadSettings(
            mode="自动上报",
            token="token-one",
            base_url="http://127.0.0.1:8080",
        )
        second = inventory_reporting.UploadSettings(
            mode="自动上报",
            token="token-two",
            base_url="http://127.0.0.1:8080",
        )
        responses = [
            _Response(json.dumps({"status_code": 200, "data": {"id": "acc_one", "name": "一号"}}).encode()),
            _Response(json.dumps({"status_code": 200, "data": {"id": "acc_two", "name": "二号"}}).encode()),
        ]
        with patch.object(
            inventory_reporting.urllib_request, "urlopen", side_effect=responses
        ) as urlopen:
            self.assertEqual(inventory_reporting.get_bound_account(first).id, "acc_one")
            self.assertEqual(inventory_reporting.get_bound_account(first).id, "acc_one")
            self.assertEqual(inventory_reporting.get_bound_account(second).id, "acc_two")

        self.assertEqual(urlopen.call_count, 2)
        for call in urlopen.call_args_list:
            request = call.args[0]
            self.assertEqual(
                request.full_url,
                "http://127.0.0.1:8080/open-api/inventory/account",
            )
        self.assertEqual(
            urlopen.call_args_list[1].args[0].get_header("Authorization"),
            "Bearer token-two",
        )

    def test_bound_account_reports_invalid_token_without_retry(self):
        settings = inventory_reporting.UploadSettings(
            mode="自动上报",
            token="invalid-token",
            base_url="http://127.0.0.1:8080",
        )
        unauthorized = HTTPError(
            "http://127.0.0.1:8080/open-api/inventory/account",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"code":"unauthorized"}}'),
        )
        with patch.object(
            inventory_reporting.urllib_request, "urlopen", side_effect=unauthorized
        ) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "Token 无效"):
                inventory_reporting.get_bound_account(settings)
        self.assertEqual(urlopen.call_count, 1)

    def test_reward_document_uses_operator_id_and_merges_duplicate_rewards(self):
        document = inventory_reporting.build_exchange_document(
            [
                _agent_result("司马徽", "char_105_simahui", 2),
                _agent_result("司马徽", "char_105_simahui", 3),
                _agent_result("绣球", "char_029_xiuqiu", 1),
            ],
            "invocation",
            "2026-08-17T10:00:00+08:00",
            "2026-08-17T10:01:00+08:00",
            "派遣",
            "reward_delta",
            None,
            "acc_main",
        )

        record = document["records"][0]
        self.assertEqual(document["version"], 2)
        self.assertNotIn("accounts", document)
        self.assertEqual(record["account_id"], "acc_main")
        self.assertEqual(record["record_id"], "myshare:invocation")
        self.assertEqual(record["entity_type"], "agent")
        self.assertEqual(
            record["entries"],
            [
                {"id": "char_105_simahui", "name": "司马徽", "count": 5},
                {"id": "char_029_xiuqiu", "name": "绣球", "count": 1},
            ],
        )
        self.assertNotIn("snapshot_scope", record)

    def test_mixed_reward_document_splits_agent_and_item_records(self):
        document = inventory_reporting.build_exchange_document(
            [
                _agent_result("司马徽", "char_105_simahui", 1),
                _item_result("茱萸", "zhuyu", 2),
                _item_result("茱萸", "zhuyu", 3),
            ],
            "mixed",
            "2026-08-17T10:00:00+08:00",
            "2026-08-17T10:01:00+08:00",
            "派遣",
            "reward_delta",
            None,
            "acc_main",
        )

        self.assertEqual(
            [record["record_id"] for record in document["records"]],
            ["myshare:mixed:agent", "myshare:mixed:item"],
        )
        self.assertEqual(
            [record["entity_type"] for record in document["records"]],
            ["agent", "item"],
        )
        self.assertEqual(
            document["records"][1]["entries"],
            [{"id": "zhuyu", "name": "茱萸", "count": 5}],
        )

    def test_mixed_record_sections_follow_first_screen_occurrence(self):
        document = inventory_reporting.build_exchange_document(
            [
                _item_result("茱萸", "zhuyu", 2),
                _agent_result("司马徽", "char_105_simahui", 1),
            ],
            "mixed-order",
            "2026-08-17T10:00:00+08:00",
            "2026-08-17T10:01:00+08:00",
            "派遣",
            "reward_delta",
            None,
            "acc_main",
        )

        self.assertEqual(
            [record["entity_type"] for record in document["records"]],
            ["item", "agent"],
        )
        self.assertEqual(
            [record["record_id"] for record in document["records"]],
            ["myshare:mixed-order:item", "myshare:mixed-order:agent"],
        )

    def test_snapshot_uses_item_id_and_rejects_duplicate_absolute_values(self):
        document = inventory_reporting.build_exchange_document(
            [_item_result("白金币", "baijinbi", 103)],
            "stock",
            "2026-08-17T10:00:00+08:00",
            "2026-08-17T10:01:00+08:00",
            "背包",
            "stock_snapshot",
            "full",
            "acc_main",
        )
        record = document["records"][0]
        self.assertEqual(record["snapshot_scope"], "full")
        self.assertEqual(record["entries"][0]["id"], "baijinbi")

        with self.assertRaisesRegex(ValueError, "重复对象"):
            inventory_reporting.build_exchange_document(
                [
                    _item_result("白金币", "baijinbi", 103),
                    _item_result("白金币", "baijinbi", 104),
                ],
                "stock",
                "2026-08-17T10:00:00+08:00",
                "2026-08-17T10:01:00+08:00",
                "背包",
                "stock_snapshot",
                "full",
                "acc_main",
            )

    def test_report_is_human_readable_and_contains_compact_reference(self):
        document = inventory_reporting.build_exchange_document(
            [
                _agent_result("司马徽", "char_105_simahui", 1),
                _item_result("白金币", "baijinbi", 103),
            ],
            "stock",
            "2026-08-17T10:00:00+08:00",
            "2026-08-17T10:01:00+08:00",
            "背包",
            "stock_snapshot",
            "listed",
            "acc_main",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DailyRewards.txt"
            inventory_reporting.append_inventory_report(path, document, "等待自动上报")
            inventory_reporting.update_upload_status(
                path,
                ["myshare:stock:agent", "myshare:stock:item"],
                "自动上报失败（HTTP 401）",
            )

            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            text = path.read_text(encoding="utf-8-sig")
            self.assertIn("时间：2026-08-17T10:00:00+08:00", text)
            self.assertIn("渠道：背包", text)
            self.assertIn("子账号ID：acc_main", text)
            self.assertIn("密探：", text)
            self.assertIn("  司马徽 × 1", text)
            self.assertIn("道具：", text)
            self.assertIn("  白金币 × 103", text)
            self.assertNotIn("上报状态：等待自动上报", text)
            self.assertNotIn("上报状态更新：", text)
            self.assertEqual(text.count("上报状态："), 1)
            self.assertIn("自动上报失败（HTTP 401）", text)
            machine_line = next(
                line
                for line in text.splitlines()
                if line.startswith(inventory_reporting.MACHINE_MARKER)
            )
            reference = json.loads(
                machine_line[len(inventory_reporting.MACHINE_MARKER) :]
            )
            self.assertEqual(
                reference,
                {
                    "a": "acc_main",
                    "r": ["myshare:stock:agent", "myshare:stock:item"],
                    "s": "listed",
                },
            )

    def test_upload_posts_bearer_document_and_reads_import_result(self):
        document = {"format": "myshare-inventory-exchange", "records": []}
        settings = inventory_reporting.UploadSettings(
            mode="自动上报",
            token="secret-token",
            base_url="http://127.0.0.1:8080",
        )
        response = _Response(
            json.dumps({"data": {"accepted": 1, "duplicates": 0}}).encode("utf-8")
        )

        with patch.object(
            inventory_reporting.urllib_request, "urlopen", return_value=response
        ) as urlopen:
            result = inventory_reporting.upload_inventory_document(
                document, settings, max_attempts=1
            )

        self.assertTrue(result.success)
        self.assertIn("accepted=1", result.message)
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8080/open-api/inventory/import",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertEqual(json.loads(request.data.decode("utf-8")), document)

    def test_token_binding_populates_every_record_before_upload(self):
        settings = inventory_reporting.read_upload_settings(
            _Context(
                {
                    "mode": "自动上报",
                    "token": "bound-token",
                    "base_url": "http://127.0.0.1:8080",
                }
            )
        )
        account_response = _Response(
            json.dumps(
                {"status_code": 200, "data": {"id": "acc_bound", "name": "绑定账号"}}
            ).encode("utf-8")
        )
        import_response = _Response(
            json.dumps({"data": {"accepted": 2, "duplicates": 0}}).encode("utf-8")
        )
        with patch.object(
            inventory_reporting.urllib_request,
            "urlopen",
            side_effect=[account_response, import_response],
        ) as urlopen:
            account = inventory_reporting.get_bound_account(settings)
            document = inventory_reporting.build_exchange_document(
                [
                    _agent_result("司马徽", "char_105_simahui", 1),
                    _item_result("茱萸", "zhuyu", 2),
                ],
                "bound-flow",
                "2026-08-17T10:00:00+08:00",
                "2026-08-17T10:01:00+08:00",
                "派遣",
                "reward_delta",
                None,
                account.id,
            )
            result = inventory_reporting.upload_inventory_document(
                document, settings, max_attempts=1
            )

        self.assertTrue(result.success)
        self.assertTrue(document["records"])
        self.assertEqual(
            {record["account_id"] for record in document["records"]},
            {"acc_bound"},
        )
        uploaded = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(uploaded, document)

    def test_upload_retries_network_failure_but_not_http_401(self):
        settings = inventory_reporting.UploadSettings(
            mode="自动上报",
            token="secret-token",
            base_url="http://127.0.0.1:8080",
        )
        response = _Response(b"{}")
        with (
            patch.object(
                inventory_reporting.urllib_request,
                "urlopen",
                side_effect=[URLError("offline"), response],
            ) as urlopen,
            patch.object(inventory_reporting, "_retry_wait"),
        ):
            result = inventory_reporting.upload_inventory_document(
                {}, settings, max_attempts=2
            )
        self.assertTrue(result.success)
        self.assertEqual(urlopen.call_count, 2)

        unauthorized = HTTPError(
            "http://127.0.0.1:8080/open-api/inventory/import",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"code":"unauthorized","message":"bad token"}}'),
        )
        with patch.object(
            inventory_reporting.urllib_request,
            "urlopen",
            side_effect=unauthorized,
        ) as urlopen:
            result = inventory_reporting.upload_inventory_document(
                {}, settings, max_attempts=3
            )
        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 401)
        self.assertIn("Token 无效", result.message)
        self.assertEqual(urlopen.call_count, 1)

        mismatch = HTTPError(
            "http://127.0.0.1:8080/open-api/inventory/import",
            403,
            "Forbidden",
            {},
            io.BytesIO(
                b'{"error":{"code":"account_scope_mismatch","message":"wrong account"}}'
            ),
        )
        with patch.object(
            inventory_reporting.urllib_request, "urlopen", side_effect=mismatch
        ):
            result = inventory_reporting.upload_inventory_document(
                {}, settings, max_attempts=1
            )
        self.assertFalse(result.success)
        self.assertIn("Token 绑定账号与库存记录不一致", result.message)

    def test_openapi_token_current_and_export_have_no_account_selectors(self):
        spec_path = MODULE_PATH.parents[3] / "tools" / "analytics" / "backend-openapi.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        paths = spec["paths"]
        self.assertIn("/open-api/inventory/account", paths)
        current_names = {
            item["name"]
            for item in paths["/open-api/inventory/current"]["get"].get("parameters", [])
        }
        export_names = {
            item["name"]
            for item in paths["/open-api/inventory/export"]["get"].get("parameters", [])
        }
        self.assertNotIn("account_id", current_names)
        self.assertNotIn("account_id", export_names)
        self.assertNotIn("scope", export_names)


if __name__ == "__main__":
    unittest.main()
