import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from custom.action.agent_info_collector import (  # noqa: E402
    DISC_CELLS,
    HUAJI_MAX_ROI,
    ODDITY_ROWS,
    PAGE_NODES,
    ROI_CONFIGS,
    AgentInfoCollector,
    _AgentInfoReader,
    _clean_operator_name,
    _has_awakened_badge,
    _huaji_nodes,
    _is_sp_huaji_layout,
    _is_pending_awaken_layout,
    _is_max_huaji_layout,
    _normalise,
    _operator_name_key,
    _regular_huaji_advance_state,
    _ratio,
    _star_stone_from_parts,
)


class AgentInfoCollectorParsingTests(unittest.TestCase):
    def test_game_click_coordinates_are_independent(self):
        base_clicks = ROI_CONFIGS["代号鸢"]["clicks"]
        ruyuan_clicks = ROI_CONFIGS["如鸢"]["clicks"]

        self.assertIsNot(base_clicks, ruyuan_clicks)
        self.assertEqual(
            set(base_clicks),
            {
                "detail_entry",
                "detail_close",
                "huaji_entry",
                "huaji_back",
                "disc_entry",
                "disc_back",
                "disc_switch",
                "next_operator",
            },
        )
        self.assertEqual(set(base_clicks), set(ruyuan_clicks))

    def test_detail_navigation_uses_selected_game_click_config(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.roi = {
            "clicks": {
                "detail_entry": (101, 202),
                "detail_close": (303, 404),
            }
        }
        image = object()
        pages = []
        clicks = []
        reader._require_page = lambda page: pages.append(page) or image
        reader.click = lambda point, current, settle_ms=0: clicks.append(point)
        reader._read_oddities = lambda current: {"攻击力": {"current": 1}}

        result = reader._collect_details({})

        self.assertEqual(pages, ["main", "detail", "main"])
        self.assertEqual(clicks, [(101, 202), (303, 404)])
        self.assertEqual(result, {"攻击力": {"current": 1}})

    def test_reader_initializes_batch_and_traversal_state(self):
        reader = _AgentInfoReader(
            SimpleNamespace(),
            {"resource": "base", "max_operators": 999, "scan_id": "batch-id"},
        )

        self.assertEqual(reader.max_operators, 300)
        self.assertEqual(reader.scan_id, "batch-id")
        self.assertEqual(reader.publish_sequence, 0)
        self.assertTrue(reader.operators)

    def test_each_checkpoint_gets_new_id_and_shares_frozen_document(self):
        import custom.action.agent_info_collector as collector

        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.publish_sequence = 0
        reader.scan_id = "batch-id"
        reader.scan_started_at = "2026-08-21T12:00:00+08:00"
        reader.game = "代号鸢"
        written = []
        uploaded = []
        exchange_records = []

        with tempfile.TemporaryDirectory() as directory:
            reader.params = {"output": str(Path(directory) / "report.json")}

            def build(records, scan_id, game, effective_at):
                return {
                    "records": [
                        {
                            "record_id": f"scan:{scan_id}",
                            "record_type": "operator_snapshot",
                            "entries": records,
                            "unmatched": [],
                        }
                    ],
                    "game": game,
                    "effective_at": effective_at,
                }

            with (
                patch.object(collector, "build_v3_document", side_effect=build),
                patch.object(collector, "discover_v3_schema", return_value=None),
                patch.object(collector, "validate_v3_document"),
                patch.object(
                    collector,
                    "write_v3_document",
                    side_effect=lambda document, path: written.append(copy.deepcopy(document)) or path,
                ),
                patch.object(
                    reader,
                    "_upload_v3_if_enabled",
                    side_effect=lambda document: uploaded.append(document),
                ),
            ):
                reader._publish_checkpoint(
                    exchange_records,
                    {"operator_id": "char_001", "name": "first"},
                )
                reader._publish_checkpoint(
                    exchange_records,
                    {"operator_id": "char_002", "name": "second"},
                )

        self.assertEqual(
            [item["records"][-1]["record_id"] for item in written],
            ["scan:batch-id-0001", "scan:batch-id-0002"],
        )
        self.assertEqual(len(written[0]["records"]), 1)
        self.assertEqual(len(written[1]["records"]), 2)
        self.assertEqual(len(uploaded[0]["records"]), 1)
        self.assertEqual(len(uploaded[1]["records"]), 1)
        self.assertEqual(written[0]["records"][0], uploaded[0]["records"][0])
        self.assertEqual(written[1]["records"][1], uploaded[1]["records"][0])
        self.assertEqual(len(written[0]["records"][0]["entries"]), 1)
        self.assertEqual(len(written[1]["records"][1]["entries"]), 1)

    def test_real_checkpoint_writes_and_resumes_only_v3_file(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.publish_sequence = 0
        reader.scan_id = "batch-id"
        reader.scan_started_at = "2026-08-21T12:00:00+08:00"
        reader.game = "代号鸢"
        records = []

        def source_record(operator_id, name):
            return {
                "operator_id": operator_id,
                "name": name,
                "name_raw": name,
                "stats": {},
                "oddities": {},
                "huaji": {},
                "disc_configs": None,
                "collection_debug": {"operator_match": "name"},
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            reader.params = {"output": str(output), "upload": False}

            reader._publish_checkpoint(
                records,
                source_record("char_095_zhangyan", "张燕"),
            )
            reader._publish_checkpoint(
                records,
                source_record("char_084_chendengsp", "陈登·黍王"),
            )

            document = json.loads(output.read_text(encoding="utf-8"))
            resumed = reader._load_records()

            self.assertEqual(document["format"], "myshare-operator-exchange")
            self.assertEqual(len(document["records"]), 2)
            self.assertEqual(len(resumed), 2)
            self.assertFalse((Path(directory) / "report.raw.json").exists())

    def test_commit_log_includes_redacted_response_details(self):
        import custom.action.agent_info_collector as collector

        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.params = {"upload": True, "commit": True}
        reader.context = SimpleNamespace(
            get_node_data=lambda name: {
                "attach": {
                    "token": "secret-token",
                    "base_url": "http://example.test",
                }
            }
        )
        document = {"records": [{"record_id": "scan:test"}]}

        with (
            patch.object(collector, "preview_v3_document", return_value={"data": {"accepted": 1}}),
            patch.object(
                collector,
                "commit_v3_document",
                return_value={"data": {"accepted": 1, "warnings": ["secret-token"]}},
            ),
            patch.object(collector.logger, "info") as info,
        ):
            reader._upload_v3_if_enabled(document)

        messages = "\n".join(str(call.args[0]) for call in info.call_args_list)
        self.assertIn('"accepted":1', messages)
        self.assertIn("<redacted>", messages)
        self.assertNotIn("secret-token", messages)

    def test_external_stop_is_raised_cooperatively(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.context = SimpleNamespace(
            stop=False,
            tasker=SimpleNamespace(stopping=True, running=True),
        )
        with self.assertRaises(InterruptedError):
            reader._ensure_running()

    def test_star_slot_templates_distinguish_empty_and_equipped(self):
        import cv2

        root = Path(__file__).resolve().parents[2]
        empty_image = cv2.imread(str(root / "debug/dhy/base/agent-info-disc-1.png"))
        equipped_image = cv2.imread(str(root / "debug/dhy/base/agent-info-disc-2.png"))
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader._star_templates = None
        rois = {
            "main": (245, 1005, 170, 155),
            "support": (430, 1005, 180, 155),
        }
        self.assertTrue(reader._star_slot_has_placeholder(empty_image, rois["main"]))
        self.assertTrue(reader._star_slot_has_placeholder(empty_image, rois["support"]))
        self.assertFalse(reader._star_slot_has_placeholder(equipped_image, rois["main"]))
        self.assertFalse(reader._star_slot_has_placeholder(equipped_image, rois["support"]))

    def test_record_key_prefers_operator_id_and_cleans_name(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        self.assertEqual(
            reader._record_key({"operator_id": "char_001", "name": "错误"}),
            "char_001",
        )
        self.assertEqual(reader._record_key({"name": "王粲觉醒"}), "王粲")

    def test_load_records_keeps_valid_checkpoint_records(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        with tempfile.TemporaryDirectory() as directory:
            reader.params = {"output": str(Path(directory) / "report.json")}
            output = Path(directory) / "report.json"
            output.write_text(
                '{"format":"myshare-operator-exchange","version":3,"records":['
                '{"record_type":"operator_snapshot","entries":['
                '{"operator_id":"char_001"}],"unmatched":[]},null,{}]}',
                encoding="utf-8",
            )
            self.assertEqual(
                reader._load_records(),
                [
                    {
                        "record_type": "operator_snapshot",
                        "entries": [{"operator_id": "char_001"}],
                        "unmatched": [],
                    }
                ],
            )

    def test_custom_output_is_the_only_checkpoint_file(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.params = {"output": "MyOperators.json"}

        self.assertEqual(reader._output_path().name, "MyOperators.json")
        self.assertFalse(hasattr(reader, "_checkpoint_path"))

    def test_local_filename_option_uses_separate_config_node(self):
        import json

        root = Path(__file__).resolve().parents[2]
        interface = json.loads((root / "assets/interface.json").read_text(encoding="utf-8"))
        no_case = next(
            case
            for case in interface["option"]["同步至YuanHub"]["cases"]
            if case["name"] == "No"
        )
        yes_case = next(
            case
            for case in interface["option"]["同步至YuanHub"]["cases"]
            if case["name"] == "Yes"
        )
        filename_option = interface["option"]["密探采集本地文件名"]

        self.assertIn("密探采集本地文件名", no_case["option"])
        self.assertEqual(
            filename_option["pipeline_override"]["密探采集本地文件配置"]["attach"]["output"],
            "{operator_report_filename}",
        )
        self.assertEqual(
            yes_case["pipeline_override"]["密探采集本地文件配置"]["attach"]["output"],
            "YuanHubMyBox.json",
        )
        for resource in ("base", "zh_tw"):
            pipeline = json.loads(
                (root / f"assets/resource/{resource}/pipeline/agent_info_collector.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                pipeline["密探采集本地文件配置"]["attach"]["output"],
                "AgentInfoReport.json",
            )

    def test_custom_action_merges_separate_runtime_config_nodes(self):
        import custom.action.agent_info_collector as collector

        captured = {}

        class Reader:
            def __init__(self, context, params):
                captured.update(params)

            def run(self):
                return True

        nodes = {
            "密探采集游戏版本配置": {"attach": {"game": "如鸢"}},
            "密探采集上报配置": {"attach": {"upload": False, "commit": True}},
            "密探采集本地文件配置": {"attach": {"output": "MyOperators.json"}},
        }
        context = SimpleNamespace(get_node_data=lambda name: nodes[name])
        argv = SimpleNamespace(custom_action_param={"resource": "zh_tw"})

        with patch.object(collector, "_AgentInfoReader", Reader):
            result = AgentInfoCollector().run(context, argv)

        self.assertTrue(result.success)
        self.assertEqual(
            captured,
            {
                "resource": "zh_tw",
                "game": "如鸢",
                "upload": False,
                "commit": True,
                "output": "MyOperators.json",
            },
        )

    def test_upload_uses_token_bound_account_report_name(self):
        import custom.action.agent_info_collector as collector

        captured = {}

        class Reader:
            def __init__(self, context, params):
                captured.update(params)

            def run(self):
                return True

        nodes = {
            "密探采集游戏版本配置": {"attach": {"game": "代号鸢"}},
            "密探采集上报配置": {"attach": {"upload": True, "commit": True}},
            "密探采集本地文件配置": {"attach": {"output": "shared.json"}},
        }
        context = SimpleNamespace(get_node_data=lambda name: nodes[name])
        argv = SimpleNamespace(custom_action_param={"resource": "base"})
        account = SimpleNamespace(id="acc_bird", name='大/小:鸟')

        with (
            patch.object(collector, "_AgentInfoReader", Reader),
            patch.object(collector, "read_upload_settings", return_value=object()),
            patch.object(collector, "get_bound_account", return_value=account),
        ):
            result = AgentInfoCollector().run(context, argv)

        self.assertTrue(result.success)
        self.assertEqual(
            captured["output"],
            "YuanHubMyBox-大_小_鸟-acc_bird.json",
        )

    def test_upsert_exchange_record_replaces_checkpoint_in_place(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        records = [
            {
                "record_id": "scan:old-1",
                "record_type": "operator_snapshot",
                "entries": [{"operator_id": "char_001"}],
                "unmatched": [],
            },
            {
                "record_id": "scan:old-2",
                "record_type": "operator_snapshot",
                "entries": [{"operator_id": "char_002"}],
                "unmatched": [],
            },
        ]
        replacement = {
            "record_id": "scan:new-1",
            "record_type": "operator_snapshot",
            "entries": [{"operator_id": "char_001"}],
            "unmatched": [],
        }

        self.assertTrue(reader._upsert_exchange_record(records, replacement))
        self.assertEqual(records[0], replacement)
        self.assertEqual(records[1]["entries"][0]["operator_id"], "char_002")

        new_record = {
            "record_id": "scan:new-3",
            "record_type": "operator_snapshot",
            "entries": [{"operator_id": "char_003"}],
            "unmatched": [],
        }
        self.assertFalse(reader._upsert_exchange_record(records, new_record))
        self.assertEqual(records[-1], new_record)

    def test_upsert_replaces_same_operator_after_disc_identity_correction(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        ordinary = {
            "record_type": "operator_snapshot",
            "entries": [
                {
                    "operator_id": "char_013_chendeng",
                    "diagnostics": {"collection_debug": {"name_raw": "陈登觉醒"}},
                }
            ],
            "unmatched": [],
        }
        wrong_sp = {
            "record_type": "operator_snapshot",
            "entries": [
                {
                    "operator_id": "char_013_chendeng",
                    "diagnostics": {"collection_debug": {"name_raw": "陈登泰王"}},
                }
            ],
            "unmatched": [],
        }
        records = [ordinary, wrong_sp]
        corrected = {
            "record_type": "operator_snapshot",
            "entries": [
                {
                    "operator_id": "char_084_chendengsp",
                    "diagnostics": {
                        "collection_debug": {
                            "operator_match": "disc",
                            "name_raw": "陈登泰王",
                            "name_match_operator_id": "char_013_chendeng",
                        }
                    },
                }
            ],
            "unmatched": [],
        }

        self.assertTrue(reader._upsert_exchange_record(records, corrected))
        self.assertEqual(records, [ordinary, corrected])

    def test_operator_name_removes_awakened_badge(self):
        self.assertEqual(_clean_operator_name("王粲觉醒"), "王粲")
        self.assertEqual(_clean_operator_name("王粲已覺醒"), "王粲")
        self.assertEqual(_clean_operator_name("贾翊觉醒"), "贾翊")
        self.assertTrue(_has_awakened_badge("王粲覺醒"))
        self.assertFalse(_has_awakened_badge("王粲"))
        self.assertTrue(_is_pending_awaken_layout("五星 待覺醒"))
        self.assertTrue(_is_max_huaji_layout("已進階至最高等級"))
        self.assertFalse(_is_max_huaji_layout("化極"))
        self.assertEqual(_operator_name_key("史子眇·赴烛觉醒"), "史子眇赴烛")

    def test_sp_name_without_middle_dot_prefers_full_operator_name(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.operators = {
            _operator_name_key("史子眇"): {
                "id": "char_023_shizimiao",
                "name": "史子眇",
            },
            _operator_name_key("史子眇·赴烛"): {
                "id": "char_085_shizimiaosp",
                "name": "史子眇·赴烛",
            },
            _operator_name_key("陈登"): {
                "id": "char_013_chendeng",
                "name": "陈登",
            },
            _operator_name_key("陈登·黍王"): {
                "id": "char_084_chendengsp",
                "name": "陈登·黍王",
            },
        }

        self.assertEqual(
            reader._operator_for_name("史子眇赴烛")["id"],
            "char_085_shizimiaosp",
        )
        self.assertEqual(
            reader._operator_for_name("陈登黍王觉醒")["id"],
            "char_084_chendengsp",
        )

    def test_unique_sp_discs_override_truncated_base_operator_name(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.operators = {
            _operator_name_key("陈登"): {
                "id": "char_013_chendeng",
                "name": "陈登",
                "discs": [{"ot_name": "初始能量+1"}],
            },
            _operator_name_key("陈登·黍王"): {
                "id": "char_084_chendengsp",
                "name": "陈登·黍王",
                "discs": [
                    {"ot_name": "初始能量+1"},
                    {"ot_name": "时和岁丰"},
                    {"ot_name": "雨顺物康"},
                    {"ot_name": "积善馀庆"},
                ],
            },
        }
        configs = [
            {
                "available": True,
                "slots": [
                    {"state": "active", "name": "初始能量+1"},
                    {"state": "active", "name": "时和岁丰"},
                    {"state": "active", "name": "积善馀庆"},
                ],
            },
            {
                "available": True,
                "slots": [{"state": "active", "name": "雨顺物康"}],
            },
        ]
        huaji_operator_ids = []
        reader._collect_details = lambda main: {}
        reader._collect_discs = lambda main: configs
        reader._resolve_locked_disc_names = lambda current, operator: None
        reader._collect_huaji = lambda record: (
            huaji_operator_ids.append(record.get("operator_id"))
            or {"layout": "sp", "stars": 3, "nodes": [], "awakened": False}
        )

        record = reader.collect_current_from_main(
            {
                "operator_id": "char_013_chendeng",
                "name": "陈登",
                "name_raw": "陈登泰王",
                "name_cleaned": "陈登泰王",
                "operator_lookup": True,
                "_operator_match": "name",
                "_operator_candidates": [],
                "stats": {},
            }
        )

        self.assertEqual(record["operator_id"], "char_084_chendengsp")
        self.assertEqual(record["name"], "陈登·黍王")
        self.assertEqual(record["collection_debug"]["operator_match"], "disc")
        self.assertEqual(
            record["collection_debug"]["name_match_operator_id"],
            "char_013_chendeng",
        )
        self.assertEqual(huaji_operator_ids, ["char_084_chendengsp"])

    def test_fuzzy_name_waits_for_unique_disc_confirmation(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.operators = {
            "贾诩": {
                "id": "char_jiaxu",
                "name": "贾诩",
                "discs": [
                    {"ot_name": "冷酷毒计", "desp": "我方全体的技能增伤提升"},
                    {"ot_name": "技能伤害大幅增加", "desp": "技能系数增加20%"},
                ],
            },
            "贾南": {
                "id": "char_jianan",
                "name": "贾南",
                "discs": [
                    {"ot_name": "南风", "desp": "受到攻击时回复能量"},
                    {"ot_name": "技能伤害大幅增加", "desp": "技能系数增加40%"},
                ],
            },
        }

        self.assertIsNone(reader._operator_for_name("贾翊觉醒"))
        self.assertIsNone(reader._operator_for_name("贾"))
        candidates = reader._fuzzy_operator_candidates("贾翊觉醒")
        self.assertEqual(
            {candidate["operator_id"] for candidate in candidates},
            {"char_jiaxu", "char_jianan"},
        )
        main = {
            "name_raw": "贾翊觉醒",
            "_operator_candidates": candidates,
        }
        confirmed = reader._confirm_operator_from_discs(
            main,
            [
                {
                    "slots": [
                        {
                            "state": "active",
                            "name": "冷酷毒计",
                        }
                    ]
                }
            ],
        )
        self.assertEqual(confirmed["id"], "char_jiaxu")

    def test_shared_disc_name_cannot_confirm_fuzzy_operator(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.operators = {
            "贾诩": {
                "id": "char_jiaxu",
                "name": "贾诩",
                "discs": [
                    {"ot_name": "技能伤害大幅增加", "desp": "技能系数增加20%"}
                ],
            },
            "贾南": {
                "id": "char_jianan",
                "name": "贾南",
                "discs": [
                    {"ot_name": "技能伤害大幅增加", "desp": "技能系数增加40%"}
                ],
            },
        }
        main = {
            "name_raw": "贾翊",
            "_operator_candidates": reader._fuzzy_operator_candidates("贾翊"),
        }

        self.assertIsNone(
            reader._confirm_operator_from_discs(
                main,
                [
                    {
                        "slots": [
                            {"state": "active", "name": "技能伤害大幅增加"},
                            {
                                "state": "locked",
                                "unlock_description": "技能系数增加20%",
                            },
                        ]
                    }
                ],
            )
        )

    def test_two_unique_inactive_discs_confirm_without_name_candidate(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.operators = {
            "甘宁": {
                "id": "char_ganning",
                "name": "甘宁",
                "discs": [
                    {"ot_name": "坐享其成", "desp": "description one"},
                    {"ot_name": "同流合污", "desp": "description two"},
                ],
            },
            "张辽": {
                "id": "char_zhangliao",
                "name": "张辽",
                "discs": [
                    {"ot_name": "无路可退", "desp": "description three"}
                ],
            },
        }
        main = {"name_raw": "北", "_operator_candidates": []}
        configs = [
            {
                "slots": [
                    {"state": "inactive", "name": "坐享其成"},
                    {"state": "inactive", "name": "同流合污"},
                ]
            }
        ]

        confirmed = reader._confirm_operator_from_discs(main, configs)

        self.assertEqual(confirmed["id"], "char_ganning")

    def test_one_unique_disc_cannot_override_unmatched_name(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.operators = {
            "甘宁": {
                "id": "char_ganning",
                "name": "甘宁",
                "discs": [{"ot_name": "坐享其成", "desp": "description"}],
            }
        }

        self.assertIsNone(
            reader._confirm_operator_from_discs(
                {"name_raw": "觉", "_operator_candidates": []},
                [{"slots": [{"state": "inactive", "name": "坐享其成"}]}],
            )
        )

    def test_disc_scan_keeps_known_inactive_cells_without_clicking(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.operators = {
            "甘宁": {
                "id": "char_ganning",
                "name": "甘宁",
                "discs": [
                    {"ot_name": "坐享其成", "desp": "description one"},
                    {"ot_name": "同流合污", "desp": "description two"},
                ],
            }
        }
        image = object()
        reader.screenshot = lambda: image
        cell_text = {
            DISC_CELLS[0][1]: "坐享其成",
            DISC_CELLS[1][1]: "同流合污",
        }
        reader.ocr_text = lambda current, roi: (
            "命盘一" if roi == (180, 142, 101, 42) else cell_text.get(roi, "")
        )
        reader.click = lambda *args, **kwargs: self.fail(
            "inactive disc cells should not be clicked"
        )

        result = reader._scan_disc_config(None)

        self.assertEqual(
            [(slot["position"], slot["state"], slot["name"]) for slot in result["slots"]],
            [
                (DISC_CELLS[0][0], "inactive", "坐享其成"),
                (DISC_CELLS[1][0], "inactive", "同流合污"),
            ],
        )

    def test_locked_disc_is_collected_as_active_without_star_scan(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        operator = {
            "id": "char_ganning",
            "name": "甘宁",
            "discs": [{"ot_name": "坐享其成", "desp": "description one"}],
        }
        reader.operators = {"甘宁": operator}
        image = object()
        reader.screenshot = lambda: image
        calls = []

        def ocr_text(current, roi):
            if roi == (180, 142, 101, 42):
                return "命盘一"
            if roi == DISC_CELLS[0][1]:
                return "可解锁"
            if roi == (100, 900, 600, 180):
                return "解锁即获得：description one"
            return ""

        reader.ocr_text = ocr_text
        reader.click = lambda point, current, settle_ms=0: calls.append(point)
        reader._read_star_stones = lambda current: self.fail(
            "locked disc must not scan star stones"
        )

        result = reader._scan_disc_config(operator)

        self.assertEqual(len(result["slots"]), 1)
        self.assertEqual(
            result["slots"][0],
            {
                "position": DISC_CELLS[0][0],
                "state": "active",
                "locked": True,
                "name": "坐享其成",
                "unlock_description": "descriptionone",
            },
        )
        self.assertEqual(len(calls), 1)

    def test_huaji_readiness_accepts_normal_and_awakened_pages(self):
        import json

        resource_root = Path(__file__).resolve().parents[2] / "assets/resource"
        for resource in ("base", "zh_tw"):
            path = resource_root / resource / "pipeline/agent_info_collector.json"
            with path.open(encoding="utf-8") as handle:
                node = json.load(handle)["密探信息采集-化极界面就绪"]
                recognition = node["recognition"]
                branches = recognition["param"]["any_of"]
            self.assertEqual(recognition["type"], "Or")
            expected = [branch["expected"] for branch in branches]
            self.assertEqual(_normalise(expected[0]), "化极")
            self.assertEqual(_normalise(expected[1]), "已觉醒")
            self.assertEqual(_normalise(expected[2]), "待觉醒")
            self.assertEqual(_normalise(expected[3]), "最高等级")
            self.assertEqual(branches[0]["roi"], [430, 1070, 260, 100])
            self.assertEqual(branches[1]["roi"], [0, 800, 720, 480])
            self.assertEqual(branches[2]["roi"], [150, 800, 420, 110])
            self.assertEqual(branches[3]["roi"], [180, 900, 420, 180])
            self.assertEqual(node["pre_delay"], 500)

    def test_awakened_name_skips_huaji_page(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader._require_page = lambda page: self.fail(
            f"awakened operator should not open {page} page"
        )
        reader.click = lambda *args, **kwargs: self.fail(
            "awakened operator should not click the huaji entry"
        )

        result = reader._collect_huaji({"name_raw": "王粲已覺醒"})

        self.assertEqual(result["layout"], "awakened")
        self.assertEqual(result["stars"], 5)
        self.assertEqual(result["nodes"], [])
        self.assertTrue(result["awakened"])

    def test_max_huaji_text_is_reported_as_awakened(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        pages = []
        image = object()
        reader._require_page = lambda page: pages.append(page) or image
        reader.click = lambda *args, **kwargs: None
        reader.ocr = lambda *args, **kwargs: None
        reader.ocr_text = lambda current, roi: (
            "已进阶至最高等级" if roi == HUAJI_MAX_ROI else ""
        )

        result = reader._collect_huaji({"name_raw": "毛玠"})

        self.assertEqual(pages, ["main", "huaji", "main"])
        self.assertEqual(
            result,
            {
                "layout": "awakened",
                "stars": 5,
                "nodes": [],
                "awakened": True,
            },
        )

    def test_page_transition_uses_pipeline_readiness_node_once(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        reader.current_page = None
        calls = []
        screenshot = object()
        reader._run_task = lambda name: calls.append(name) or True
        reader.screenshot = lambda: screenshot

        self.assertIs(reader._wait_for_page("detail"), screenshot)
        self.assertIs(reader._wait_for_page("detail"), screenshot)
        self.assertEqual(calls, [PAGE_NODES["detail"]])

    def test_ratio_accepts_percent_before_slash(self):
        self.assertEqual(_ratio("治疗加成15%/15%"), (15, 15))
        self.assertEqual(_ratio("攻击力0/500"), (0, 500))

    def test_oddity_rows_use_fixed_primary_labels_and_whitelisted_third_label(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        values = {
            ODDITY_ROWS[0]["value_roi"]: "0/500",
            ODDITY_ROWS[1]["value_roi"]: "0/2600",
            ODDITY_ROWS[2]["value_roi"]: "15%/15%",
            ODDITY_ROWS[2]["label_roi"]: "治療加成",
        }
        reader.ocr_text = lambda image, roi: values.get(roi, "ignored")

        self.assertEqual(
            reader._read_oddities(object()),
            {
                "攻击力": {"current": 0, "max": 500},
                "生命值": {"current": 0, "max": 2600},
                "治疗加成": {"current": 15, "max": 15},
            },
        )

    def test_unknown_third_oddity_label_does_not_become_garbage_key(self):
        reader = _AgentInfoReader.__new__(_AgentInfoReader)
        values = {
            ODDITY_ROWS[0]["value_roi"]: "0/500",
            ODDITY_ROWS[1]["value_roi"]: "0/2600",
            ODDITY_ROWS[2]["value_roi"]: "6/15",
            ODDITY_ROWS[2]["label_roi"]: ".",
        }
        reader.ocr_text = lambda image, roi: values.get(roi, "")

        result = reader._read_oddities(object())

        self.assertIn("field_3", result)
        self.assertNotIn(".", result)

    def test_normalise_converts_traditional_unlock_text(self):
        self.assertEqual(
            _normalise("解鎖即獲得：自身提供的治療效果提升10％"),
            "解锁即获得:自身提供的治疗效果提升10%",
        )

    def test_unlock_description_drops_material_prompt(self):
        from custom.action.agent_info_collector import _unlock_description

        self.assertEqual(
            _unlock_description("自身提供的治疗效果提升10%消耗材料可以解锁"),
            "自身提供的治疗效果提升10%",
        )

    def test_star_stone_fields_are_parsed_independently(self):
        self.assertEqual(
            _star_stone_from_parts("天機", "60级"),
            {"level": 60, "name": "天机"},
        )
        self.assertEqual(
            _star_stone_from_parts("地劫", ""),
            {"level": None, "name": "地劫"},
        )
        self.assertIsNone(_star_stone_from_parts("", ""))
        self.assertEqual(
            _star_stone_from_parts("属性攻击+200", "1级", ("天机",)),
            None,
        )
        self.assertEqual(
            _star_stone_from_parts("天机", "60级", ("天机",)),
            {"level": 60, "name": "天机"},
        )

    def test_huaji_nodes_only_contain_index_and_active(self):
        import numpy as np

        nodes = _huaji_nodes(np.zeros((1280, 720, 3), dtype=np.uint8))
        self.assertEqual(len(nodes), 5)
        self.assertTrue(all(set(node) == {"index", "active"} for node in nodes))

    def test_regular_huaji_advance_preview_is_read_as_three_five(self):
        import cv2

        image = cv2.imread(
            str(
                Path(__file__).resolve().parents[2]
                / "debug/dhy/base/agentinfo-huaji-3-5.png"
            )
        )
        state = _regular_huaji_advance_state(image)
        self.assertIsNotNone(state)
        self.assertEqual(state["stars"], 3)
        self.assertEqual(
            state["nodes"],
            [{"index": index, "active": True} for index in range(1, 6)],
        )

    def test_sp_huaji_layout_is_detected_from_two_level_labels(self):
        self.assertTrue(_is_sp_huaji_layout("王巫遮天1级王巫遮天2级生命447>982"))
        self.assertFalse(_is_sp_huaji_layout("攻击力+22生命值+118"))

    def test_sp_huaji_layout_falls_back_to_left_star_group(self):
        import cv2

        image = cv2.imread(
            str(
                Path(__file__).resolve().parents[2]
                / "debug/dhy/base/agentinfo-huaji-sp.png"
            )
        )
        self.assertTrue(_is_sp_huaji_layout("", image))


if __name__ == "__main__":
    unittest.main()
