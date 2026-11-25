import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))

from custom.action import inventory_maintenance as task  # noqa: E402
from custom.action.inventory_reporting import (  # noqa: E402
    LOCAL_ONLY_MODE,
    UploadResult,
    UploadSettings,
    append_inventory_report,
    upload_inventory_document,
)
from custom.action.maintenance_stock import (  # noqa: E402
    Stock,
    cloud_stock,
    local_stock,
    report_records,
    timestamp,
)
from custom.action.maintenance_training import BOOK_XP, load_goals  # noqa: E402


BASE = {"wind_fire_sets": 1, "scan_if_missing": False, "stock_source": "local_report"}
DATE = "2026-01-01T10:00:00+00:00"
AUTH = UploadSettings(LOCAL_ONLY_MODE, "demo", "https://example.test")


class Game:
    """Replay child task outcomes and OCR while keeping real planning, journal and reports."""

    def __init__(self, test):
        self.test = test
        self.tasker = SimpleNamespace(stopping=False, controller=self)
        self.max_count = 6
        self.quota = {}
        self.energy = 50
        self.submitted = []
        self.scans = []
        self.scan_counts = {}
        self.attempts = []
        self.locked = set()
        self.focus = []
        self.fail_after_submit = False
        self.bad_digit = False
        self.scanner_ok = True
        self.filename = ""

    def clone(self):
        return self

    def get_node_data(self, node):
        if node == task.AUTH_NODE_NAME:
            return {
                "attach": {
                    "mode": LOCAL_ONLY_MODE,
                    "token": "demo",
                    "inventory_report_filename": self.filename,
                }
            }
        return {
            "action": {"param": {"custom_action_param": {"snapshot_list": "tab3-1"}}}
        }

    def post_screencap(self):
        return SimpleNamespace(wait=lambda: SimpleNamespace(get=lambda: self.count))

    def run_recognition(self, node, image, overrides=None):
        if node == task.COUNTER:
            digit = "?" if self.bad_digit else str(self.count)
            expected = overrides[node]["recognition"]["param"]["expected"][0]
            return SimpleNamespace(
                hit=re.fullmatch(expected, digit) is not None,
                all_results=[SimpleNamespace(text=digit)],
                filtered_results=[SimpleNamespace(text=digit)],
            )
        return SimpleNamespace(hit=(node == task.START or self.count > self.energy))

    def run_task(self, entry, overrides):
        end = entry
        if entry == "库存保持-进度":
            self.focus.append(overrides[entry]["focus"])
        elif entry == "背包识别-start":
            params = overrides[task.SCAN]["action"]["param"]["custom_action_param"]
            self.scans.append(params["snapshot_list"])
            if self.scanner_ok and self.scan_counts:
                doc = task.document(
                    self.scan_counts,
                    self.test.names,
                    "背包-道具",
                    "stock_snapshot",
                    None,
                    task.now().isoformat(),
                    "listed",
                )
                append_inventory_report(
                    Path(params["inventory_report_path"]), doc, LOCAL_ONLY_MODE
                )
            if not self.scanner_ok:
                return None
            end = task.SCAN
        elif entry in ("刷修为历练启动", "刷经验历练启动"):
            kind = entry[1:3]
            opt = overrides["历练奖励配置"]["attach"]
            self.channel = opt["acquisition_channel"]
            stage = overrides[f"找到{kind}历练具体层数"]["recognition"]["param"]
            level = stage.get("expected", stage.get("template"))
            self.attempts.append(level)
            end = f"想扫荡几次{kind}历练"
            self.test.assertEqual(overrides[task.MAXIMUM]["next"], [end])
            self.test.assertEqual(overrides[end]["next"], [])
            self.count = min(self.max_count, self.quota.setdefault(self.channel, 6))
            if level in self.locked:
                end = task.LOCKED
            elif self.count == 0:
                end = task.QUOTA
        elif entry == task.RED_COST:
            self.count -= 1
        elif entry == task.START:
            self.test.assertEqual(
                overrides[task.SETTLED]["action"]["type"], "DoNothing"
            )
            self.submitted.append((self.channel, self.count))
            self.energy -= self.count
            self.quota[self.channel] -= self.count
            if self.fail_after_submit:
                return None
            end = task.SETTLED
        elif entry != task.HOME:
            raise AssertionError(entry)
        return SimpleNamespace(
            status=SimpleNamespace(succeeded=True),
            nodes=[SimpleNamespace(name=end, completed=True)],
        )


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "agent").mkdir()
        shutil.copy(ROOT / "agent/items.json", self.root / "agent/items.json")
        shutil.copy(ROOT / "assets/interface.json", self.root / "interface.json")
        self.names = task._item_names(self.root / "agent/items.json")
        self.goal = load_goals(self.root, BASE)[0]
        self.wind = dict(self.goal.targets)
        self.g = Game(self)
        upload = patch.object(
            task,
            "upload_inventory_document",
            return_value=UploadResult(False, 403, "HTTP 403"),
        )
        self.upload = upload.start()
        self.addCleanup(upload.stop)

    def write(self, counts, path="StockReport.txt", owner=None, at=DATE):
        doc = task.document(
            counts, self.names, "背包-道具", "stock_snapshot", owner, at, "listed"
        )
        append_inventory_report(self.root / path, doc, LOCAL_ONLY_MODE)
        return doc

    def start(self, **config):
        task.execute(self.g, {**BASE, **config}, self.root)

    def deficit(self):
        return {
            **self.wind,
            "cuishan": self.wind["cuishan"] - 70,
            "jinsishan": self.wind["jinsishan"] - 80,
        }

    def test_disabled_and_met_do_not_navigate(self):
        self.start(wind_fire_sets=0)
        self.assertFalse((self.root / "data").exists())
        self.write({**self.wind, "liutaobingshu": 239})
        self.start(experience_target=2386300, scan_if_missing=True)
        self.assertEqual((self.g.scans, self.g.attempts), ([], []))

    def test_native_options_resolve_presets_in_both_resource_packs(self):
        interface = json.loads((self.root / "interface.json").read_text("utf-8"))
        task_ui = next(t for t in interface["task"] if t["entry"] == task.ENTRY)
        self.assertFalse(task_ui["default_check"])
        self.assertEqual(task_ui["label"], "📦 库存保持")
        source = interface["option"]["库存保持-库存来源"]
        local = next(case for case in source["cases"] if case["name"] == "本地库存报告")
        self.assertFalse(local.get("option"))
        for locale in ("base", "zh_tw"):
            nodes = json.loads(
                (
                    ROOT
                    / "assets/resource"
                    / locale
                    / "pipeline/daily/inventory-maintenance.json"
                ).read_text("utf-8")
            )
            self.assertEqual(
                nodes[task.ENTRY]["action"]["param"]["custom_action"],
                "InventoryMaintenance",
            )
            config = nodes[task.ENTRY]["attach"].copy()
            self.assertEqual(load_goals(self.root, config), [])
            on = []
            for key in task_ui["option"][2:]:
                switch = interface["option"][key]
                self.assertEqual(switch["default_case"], "No")
                self.assertTrue((ROOT / "assets" / switch["icon"]).is_file())
                no = next(c for c in switch["cases"] if c["name"] == "No")
                yes = next(c for c in switch["cases"] if c["name"] == "Yes")
                self.assertEqual(no["option"], yes["option"])
                quantity = interface["option"][no["option"][0]]
                self.assertEqual(len(quantity["cases"]), 3)
                config.update(no["pipeline_override"][task.ENTRY]["attach"])
                config.update(
                    quantity["cases"][1]["pipeline_override"][task.ENTRY]["attach"]
                )
                on.append(yes["pipeline_override"][task.ENTRY]["attach"])
            self.assertEqual(load_goals(self.root, config), [])
            config.update(on[0])
            self.assertEqual(
                [g.label for g in load_goals(self.root, config)], ["风火历练"]
            )
            for enabled in on[1:]:
                config.update(enabled)
            goals = load_goals(self.root, config)
            self.assertEqual(
                [g.label for g in goals],
                ["风火历练", "地水历练", "阴阳历练", "经验历练"],
            )
            self.assertEqual(goals[0].targets["cuishan"], 1140)
            self.assertEqual(goals[-1].experience, 4772600)

    def test_shared_drop_three_sweeps_reported_once_across_restarts(self):
        self.write(self.deficit())
        self.start()
        self.start()
        self.assertEqual(self.g.attempts, ["六"])
        self.assertEqual(self.g.submitted, [("历练-风火", 3)])
        self.upload.assert_not_called()
        target, stock = local_stock(self.root, AUTH, self.names)
        report = target.rewards
        contents = report.read_bytes()
        with report.open("ab") as file:
            file.write(contents.decode("utf-8-sig").encode())
        self.assertEqual(
            local_stock(self.root, AUTH, self.names)[1].counts, stock.counts
        )
        self.assertEqual(stock.counts["jinsishan"], self.wind["jinsishan"] + 10)

    def test_game_maximum_can_exceed_the_planned_batch(self):
        for displayed, wanted, energy, expected in (
            (7, 6, 50, 6),
            (7, 6, 4, 4),
            (12, 3, 50, 3),
        ):
            with self.subTest(displayed=displayed, wanted=wanted, energy=energy):
                self.g.count = displayed
                self.g.energy = energy
                self.assertEqual(task.affordable_count(self.g, wanted), expected)
                self.assertEqual(self.g.count, expected)
                self.assertEqual(self.g.submitted, [])

    def test_total_estimate_includes_later_stages(self):
        counts = self.wind.copy()
        counts["juanshan"] -= 31
        counts["beihuifengshan"] -= 46
        self.write(counts)
        with patch.object(task, "logger") as log:
            self.start()
        messages = "\n".join(call.args[0] for call in log.info.call_args_list)
        self.assertIn("预计还需 4 次，本批计划第四层 2 次", messages)
        self.assertIn("预计还需 2 次，本批计划第十二层 2 次", messages)
        self.assertIn("库存已达标，还需 0 次，本轮已刷 4 次", messages)

    def test_remaining_quota_only_stops_this_group(self):
        self.write(
            {**self.deficit(), **dict.fromkeys(BOOK_XP, 0), "liutaobingshu": 230}
        )
        self.g.quota["历练-风火"] = 2
        with patch.object(task, "logger") as log:
            self.start(experience_target=2386300)
        self.assertEqual(self.g.submitted, [("历练-风火", 2), ("历练-经验", 5)])
        messages = "\n".join(call.args[0] for call in log.info.call_args_list)
        self.assertIn("库存来源：本地库存报告", messages)
        self.assertIn("翠扇 500/570、金丝扇 1360/1440", messages)
        self.assertIn(
            "还缺翠扇 70、金丝扇 80。按可扫关卡预计还需 3 次，本批计划第六层 3 次",
            messages,
        )
        self.assertIn("界面已确认 2 次", messages)
        self.assertIn(
            "计划 3 次，实际结算 2 次，本轮已刷 2 次。记账后库存：翠扇 580、金丝扇 1420",
            messages,
        )
        self.assertIn("折合经验 2300000，目标 2386300", messages)
        self.assertTrue(any("金丝扇 20" in s for s in self.g.focus))

    def test_stamina_stops_all_groups(self):
        self.write({**self.deficit(), **dict.fromkeys(BOOK_XP, 0)})
        self.g.energy = 1
        self.start(experience_target=2386300)
        self.assertEqual(self.g.submitted, [("历练-风火", 1)])
        self.assertTrue(any("体力不足" in s for s in self.g.focus))

    def test_locked_stage_replans_and_quota_caps_zero_stock(self):
        self.write(dict.fromkeys(BOOK_XP, 0))
        self.g.locked.add("绝境历练")
        self.start(wind_fire_sets=0, experience_target=2386300)
        self.assertEqual(self.g.attempts, ["绝境历练", "aozhan.png"])
        self.assertEqual(self.g.submitted, [("历练-经验", 6)])

    def test_partial_data_with_and_without_scan(self):
        self.write(self.deficit())
        self.start(experience_target=2386300, earth_water_sets=1)
        self.assertEqual(self.g.scans, [])
        self.assertEqual(self.g.submitted, [("历练-风火", 3)])
        self.g.scan_counts = {
            **self.wind,
            **dict.fromkeys(BOOK_XP, 0),
            "liutaobingshu": 238,
        }
        self.start(experience_target=2386300, earth_water_sets=1, scan_if_missing=True)
        self.assertEqual(self.g.scans, ["tab3-1"])
        self.assertEqual(self.g.submitted[-1], ("历练-经验", 1))
        self.assertTrue(any("地水历练库存未知" in s for s in self.g.focus))

    def test_failed_scan_keeps_known_other_group(self):
        self.write(self.deficit())
        self.g.scanner_ok = False
        self.start(experience_target=2386300, scan_if_missing=True)
        self.assertEqual(self.g.scans, ["tab3-1"])
        self.assertEqual(self.g.submitted, [("历练-风火", 3)])

    def test_bad_ocr_never_clicks(self):
        self.write(self.deficit())
        self.g.bad_digit = True
        with self.assertRaisesRegex(ValueError, r"无法确认扫荡次数.*计划 3 次.*'\?'"):
            self.start()
        self.assertEqual(self.g.submitted, [])

    def test_uncertain_settlement_requires_new_snapshot(self):
        self.write(self.deficit())
        self.g.fail_after_submit = True
        with self.assertRaisesRegex(RuntimeError, "扫荡结算未确认"):
            self.start()
        self.assertFalse(any("实际结算" in s for s in self.g.focus))
        self.g.fail_after_submit = False
        self.start()
        self.assertEqual(len(self.g.submitted), 1)
        self.assertFalse((self.root / "DailyRewards.txt").exists())
        self.g.scan_counts = self.wind
        self.start(scan_if_missing=True)
        self.assertEqual(len(self.g.submitted), 1)
        self.assertEqual(self.g.scans, ["tab3-1"])

    def test_accounts_and_named_local_files_are_isolated(self):
        self.write(self.wind, "StockReport-A.txt", "acc_a")
        self.write(self.deficit(), "StockReport-B.txt", "acc_b")
        self.start()
        self.assertFalse(self.g.attempts)
        self.g.filename = "B"
        self.start()
        self.assertEqual(self.g.submitted, [("历练-风火", 3)])
        self.assertTrue((self.root / "DailyRewards-B.txt").exists())
        self.assertFalse((self.root / "DailyRewards-A.txt").exists())

    def test_local_source_auto_finds_named_report_and_scans_once_if_ambiguous(self):
        self.write(self.deficit(), "StockReport-A.txt", "acc_a")
        self.start(scan_if_missing=True)
        self.assertEqual(self.g.scans, [])
        self.assertEqual(self.g.submitted, [("历练-风火", 3)])
        first_rewards = (self.root / "DailyRewards-A.txt").read_bytes()

        self.write(self.deficit(), "StockReport-B.txt", "acc_b")
        self.g.scan_counts = self.wind
        self.start(scan_if_missing=True)
        self.assertEqual(self.g.scans, ["tab3-1"])
        self.assertEqual(self.g.submitted, [("历练-风火", 3)])
        self.assertEqual((self.root / "DailyRewards-A.txt").read_bytes(), first_rewards)
        self.assertFalse((self.root / "DailyRewards-B.txt").exists())

    def test_cloud_api_scope_and_baselines(self):
        rows = [
            {
                "account_id": "acc_a",
                "entity_type": "item",
                "full_baseline_at": None,
                "entries": {
                    "a": {"count": 5},
                    "b": {"count": 0, "listed_baseline_at": DATE},
                },
            }
        ]

        def http(req, timeout):
            self.assertTrue(
                req.full_url.endswith("/open-api/inventory/current?entity_type=item")
            )
            self.assertEqual(req.get_header("Authorization"), "Bearer demo")
            self.assertEqual(req.get_method(), "GET")
            return io.BytesIO(json.dumps({"statusCode": 200, "data": rows}).encode())

        with patch("custom.action.maintenance_stock.request.urlopen", side_effect=http):
            names = dict.fromkeys("abc", "item")
            self.assertEqual(cloud_stock("acc_a", AUTH, names).counts, {"b": 0})
            rows[0]["full_baseline_at"] = DATE
            self.assertEqual(
                cloud_stock("acc_a", AUTH, names).counts, {"a": 5, "b": 0, "c": 0}
            )
            with self.assertRaisesRegex(ValueError, "账号"):
                cloud_stock("acc_b", AUTH, names)

    def test_read_only_cloud_replay_and_permission_denied_fallback(self):
        account = SimpleNamespace(id="acc_a", name="A")
        values = self.deficit()

        def current(*_):
            return Stock(values.copy(), dict.fromkeys(values, timestamp(DATE)))

        with (
            patch.object(task, "get_bound_account", return_value=account),
            patch.object(task, "cloud_stock", side_effect=current),
        ):
            self.start(stock_source="yuanhub")
            self.start(stock_source="yuanhub")
            self.assertEqual(self.g.submitted, [("历练-风火", 3)])
            self.upload.assert_called_once()
            report = self.root / "DailyRewards-A-acc_a.txt"
            self.assertIn("自动上报失败（HTTP 403）", report.read_text("utf-8-sig"))
        denied = HTTPError("http://example.test", 403, "Forbidden", {}, None)
        with (
            patch.object(task, "get_bound_account", return_value=account),
            patch.object(task, "cloud_stock", side_effect=denied),
        ):
            self.start(stock_source="yuanhub")
            self.assertFalse(self.g.scans)
            self.g.scan_counts = self.wind
            self.start(stock_source="yuanhub", scan_if_missing=True)
            self.assertEqual(self.g.scans, ["tab3-1"])
        self.assertEqual(len(self.g.submitted), 1)

    def test_account_lookup_recovers_after_unassigned_scan_and_sweep(self):
        self.g.scan_counts = self.deficit()
        with patch.object(
            task, "get_bound_account", side_effect=RuntimeError("连接码查询失败")
        ):
            self.start(stock_source="yuanhub", scan_if_missing=True)
        self.assertEqual(self.g.submitted, [("历练-风火", 3)])
        self.upload.assert_not_called()
        account = SimpleNamespace(id="acc_a", name="A")
        old = Stock(self.deficit(), dict.fromkeys(self.wind, timestamp(DATE)))
        with (
            patch.object(task, "get_bound_account", return_value=account),
            patch.object(task, "cloud_stock", return_value=old),
        ):
            self.start(stock_source="yuanhub")
        self.assertEqual(len(self.g.submitted), 1)
        self.assertFalse((self.root / "DailyRewards-A-acc_a.txt").exists())

    def test_partial_cloud_changes_do_not_double_count_local_receipts(self):
        account = SimpleNamespace(id="acc_a", name="A")
        values = self.deficit()

        def current(*_):
            return Stock(values.copy(), dict.fromkeys(values, timestamp(DATE)))

        with (
            patch.object(task, "get_bound_account", return_value=account),
            patch.object(task, "cloud_stock", side_effect=current),
        ):
            self.start(stock_source="yuanhub")
            values["cuishan"] += 7
            self.start(stock_source="yuanhub")
            self.assertTrue(any("库存未知" in msg for msg in self.g.focus))
            values["cuishan"] += 113
            values["jinsishan"] += 90
            self.start(stock_source="yuanhub")
            self.assertIn("已达标", self.g.focus[-1])
        self.assertEqual(self.g.submitted, [("历练-风火", 3)])

    def test_successful_cloud_upload_does_not_mask_later_stock_decrease(self):
        account = SimpleNamespace(id="acc_a", name="A")
        values = self.deficit()

        def current(*_):
            return Stock(values.copy(), dict.fromkeys(values, timestamp(DATE)))

        with (
            patch.object(task, "get_bound_account", return_value=account),
            patch.object(task, "cloud_stock", side_effect=current),
        ):
            self.upload.return_value = UploadResult(True, 200, "accepted=1")
            self.start(stock_source="yuanhub")
            # The next cloud reading equals the old count after the user spends
            # the uploaded rewards. It must not receive the same batch again.
            self.g.quota.clear()
            self.start(stock_source="yuanhub")
        self.assertEqual(self.g.submitted, [("历练-风火", 3), ("历练-风火", 3)])

    def test_unconfirmed_cloud_settlement_does_not_upload(self):
        account = SimpleNamespace(id="acc_a", name="A")
        values = self.deficit()
        stock = Stock(values, dict.fromkeys(values, timestamp(DATE)))
        self.g.fail_after_submit = True
        with (
            patch.object(task, "get_bound_account", return_value=account),
            patch.object(task, "cloud_stock", return_value=stock),
            self.assertRaisesRegex(RuntimeError, "扫荡结算未确认"),
        ):
            self.start(stock_source="yuanhub")
        self.upload.assert_not_called()
        self.assertFalse((self.root / "DailyRewards-A-acc_a.txt").exists())

    def test_cloud_sweeps_upload_same_record_and_refresh_current_stock(self):
        requests = []
        submitted_records = []
        entries = {
            key: {"count": n, "listed_baseline_at": DATE}
            for key, n in self.deficit().items()
        }

        class Hub(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                requests.append(("GET", self.path, self.headers.get("Authorization")))
                if self.path == "/open-api/inventory/account":
                    data = {"id": "acc_test", "name": "测试"}
                elif self.path == "/open-api/inventory/current?entity_type=item":
                    data = [
                        {
                            "account_id": "acc_test",
                            "entity_type": "item",
                            "entries": entries,
                        }
                    ]
                else:
                    self.send_error(404)
                    return
                self.respond(data)

            def do_POST(self):
                requests.append(("POST", self.path, self.headers.get("Authorization")))
                (record,) = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )["records"]
                if record in submitted_records:
                    self.respond({"accepted": 0, "duplicates": 1})
                    return
                submitted_records.append(record)
                for item in record["entries"]:
                    entries[item["id"]]["count"] += item["count"]
                # Simulate a failed response after the server has already applied the record.
                self.send_error(503)

            def respond(self, data):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"statusCode": 200, "data": data}).encode())

        with HTTPServer(("127.0.0.1", 0), Hub) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            settings = UploadSettings(
                LOCAL_ONLY_MODE,
                "read-and-write",
                f"http://127.0.0.1:{server.server_port}",
            )
            try:
                self.upload.side_effect = upload_inventory_document
                with patch.object(task, "read_upload_settings", return_value=settings):
                    self.start(stock_source="yuanhub")
                    self.start(stock_source="yuanhub")
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(
            requests,
            [
                (method, path, "Bearer read-and-write")
                for method, path in (
                    ("GET", "/open-api/inventory/account"),
                    ("GET", "/open-api/inventory/current?entity_type=item"),
                    ("POST", "/open-api/inventory/import"),
                    ("POST", "/open-api/inventory/import"),
                    ("GET", "/open-api/inventory/current?entity_type=item"),
                )
            ],
        )
        self.assertEqual(self.g.submitted, [("历练-风火", 3)])
        self.assertEqual(len(submitted_records), 1)
        path = self.root / "DailyRewards-测试-acc_test.txt"
        (saved,) = report_records(path, self.names)
        self.assertEqual(saved.id, submitted_records[0]["record_id"])
        self.assertEqual(submitted_records[0]["account_id"], "acc_test")
        self.assertEqual(saved.counts, {"cuishan": 120, "jinsishan": 90})
        self.assertIn(
            "自动上报成功（HTTP 200，accepted=0，duplicates=1）",
            path.read_text("utf-8-sig"),
        )
        self.assertEqual(entries["cuishan"]["count"], self.wind["cuishan"] + 50)
