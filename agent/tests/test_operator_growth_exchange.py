import copy
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from custom.action.operator_growth_exchange import (  # noqa: E402
    build_v3_document,
    commit_v3_document,
    discover_v3_schema,
    preview_v3_document,
    _redact,
    validate_v3_document,
    write_v3_document,
)


class OperatorGrowthExchangeTests(unittest.TestCase):
    def sample_record(self, **changes):
        record = {
            "operator_id": "char_095_zhangyan",
            "name": "张燕",
            "name_raw": "张燕",
            "stats": {"level": 100, "cultivation": 17, "attack": 8000, "life": 30000},
            "oddities": {
                "攻击力": {"current": 500, "max": 500},
                "生命值": {"current": 2600, "max": 2600},
                "治疗加成": {"current": 15, "max": 15},
            },
            "huaji": {
                "layout": "regular",
                "stars": 3,
                "nodes": [
                    {"index": 1, "active": True},
                    {"index": 2, "active": True},
                    {"index": 3, "active": False},
                    {"index": 4, "active": False},
                    {"index": 5, "active": False},
                ],
                "awakened": False,
            },
            "disc_configs": [
                {
                    "available": True,
                    "slots": [
                        {"state": "active", "name": "攻击力大幅提升", "star_stones": {"main": {"name": "天机", "level": 60}, "support": {"name": "地劫", "level": 46}}},
                        {"state": "active", "name": "噢", "star_stones": {"main": None, "support": None}},
                        {"state": "active", "name": "初始能量+1", "star_stones": {"main": None, "support": None}},
                    ],
                },
                {
                    "available": True,
                    "slots": [
                        {"state": "active", "name": "攻击力大幅提升", "star_stones": {"main": {"name": "天机", "level": 60}, "support": {"name": "地劫", "level": 46}}},
                        {"state": "active", "name": "噢", "star_stones": {"main": None, "support": None}},
                        {"state": "active", "name": "初始能量+1", "star_stones": {"main": None, "support": None}},
                    ],
                },
            ],
            "collection_debug": {"operator_match": "name", "operator_candidates": []},
        }
        record.update(changes)
        return record

    def test_document_shape_and_one_source_account(self):
        document = build_v3_document([self.sample_record()], "stable-1", "代号鸢")
        validate_v3_document(document)
        self.assertEqual(document["format"], "myshare-operator-exchange")
        self.assertEqual(document["version"], 3)
        self.assertEqual(len(document["accounts"]), 1)
        self.assertEqual(document["records"][0]["record_id"], "scan:stable-1")
        self.assertEqual(document["records"][0]["game"], "代号鸢")
        self.assertNotIn("game", document["records"][0]["entries"][0])
        self.assertIn("observed_at", document["records"][0]["entries"][0])

    def test_current_agent_report_can_be_converted(self):
        report_path = Path(__file__).resolve().parents[2] / "AgentInfoReport.json"
        if not report_path.exists():
            self.skipTest("AgentInfoReport.json is not present")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        document = build_v3_document(
            report.get("records", []),
            "sample-report",
            "代号鸢",
            exported_at=report.get("updated_at"),
            effective_at=report.get("updated_at"),
        )
        validate_v3_document(document)
        self.assertEqual(document["records"][0]["record_type"], "operator_snapshot")

    def test_same_scan_id_is_stable_and_new_scan_id_changes(self):
        first = build_v3_document([self.sample_record()], "stable-1", "代号鸢", exported_at="2026-01-01T00:00:00+08:00", effective_at="2026-01-01T00:00:00+08:00")
        second = build_v3_document([self.sample_record()], "stable-1", "代号鸢", exported_at="2026-01-01T00:00:00+08:00", effective_at="2026-01-01T00:00:00+08:00")
        third = build_v3_document([self.sample_record()], "stable-2", "代号鸢", exported_at="2026-01-01T00:00:00+08:00", effective_at="2026-01-01T00:00:00+08:00")
        self.assertEqual(first, second)
        self.assertNotEqual(first["records"][0]["record_id"], third["records"][0]["record_id"])

    def test_multiple_single_operator_records_form_valid_import_document(self):
        first = build_v3_document([self.sample_record()], "checkpoint-1", "代号鸢")
        second_record = self.sample_record(
            operator_id="char_084_chendengsp",
            name="陈登·黍王",
            name_raw="陈登黍王",
            disc_configs=None,
        )
        second = build_v3_document([second_record], "checkpoint-2", "代号鸢")
        combined = copy.deepcopy(second)
        combined["records"] = [first["records"][0], second["records"][0]]

        validate_v3_document(combined, discover_v3_schema())

        self.assertEqual(
            [record["record_id"] for record in combined["records"]],
            ["scan:checkpoint-1", "scan:checkpoint-2"],
        )

    def test_unmatched_identity_is_not_emitted_as_entry(self):
        document = build_v3_document([self.sample_record(operator_id=None)], "stable-1")
        self.assertEqual(document["records"][0]["entries"], [])
        self.assertEqual(len(document["records"][0]["unmatched"]), 1)

    def test_star_level_mappings_and_node_review(self):
        pending = self.sample_record(huaji={"layout": "pending_awaken", "stars": 5, "nodes": [], "awakened": False})
        awakened = self.sample_record(huaji={"layout": "awakened", "stars": 5, "nodes": [], "awakened": True})
        broken = self.sample_record(huaji={"layout": "regular", "stars": 2, "nodes": [{"index": 1, "active": False}]})
        for record, expected in ((pending, 30), (awakened, 31)):
            entry = build_v3_document([record], "x")["records"][0]["entries"][0]
            self.assertEqual(entry["star_level"], expected)
        entry = build_v3_document([broken], "x")["records"][0]["entries"][0]
        self.assertNotIn("star_level", entry)
        self.assertEqual(entry["section_status"]["huaji"], "review")

    def test_sp_star_level_is_direct(self):
        record = self.sample_record(huaji={"layout": "sp", "stars": 4, "nodes": [], "awakened": False})
        entry = build_v3_document([record], "x")["records"][0]["entries"][0]
        self.assertEqual(entry["star_level"], 4)

    def test_four_star_oddity_uses_305_limit_and_stable_keys(self):
        record = self.sample_record(
            huaji={
                "layout": "regular",
                "stars": 4,
                "nodes": [{"index": i, "active": i <= 2} for i in range(1, 6)],
                "awakened": False,
            },
            oddities={
                "field_1": {"current": 305},
                "field_2": {"current": 1820},
                "field_3": {"current": 11},
            },
        )
        entry = build_v3_document([record], "x")["records"][0]["entries"][0]
        self.assertEqual(
            entry["combat_stats"]["oddities"],
            {
                "attack": {"current": 305},
                "hp": {"current": 1820},
                "special": {"current": 11},
            },
        )
        self.assertEqual(entry["section_status"]["oddities"], "ready")

    def test_inactive_discs_are_not_exported_and_support_maps_to_assist(self):
        record = self.sample_record()
        record["disc_configs"][0]["slots"].append({"state": "inactive", "name": "专属四"})
        document = build_v3_document([record], "x")
        entry = document["records"][0]["entries"][0]
        self.assertEqual(
            entry["disc_loadouts"][0]["discs"],
            [{"ot_name": "攻击力大幅提升"}, {"ot_name": "噢"}, {"ot_name": "初始能量+1"}],
        )
        self.assertEqual(
            entry["equipped_star_stones"],
            [
                {"type": "main1", "name": "天机", "level": 60},
                {"type": "assist1", "name": "地劫", "level": 46},
            ],
        )

    def test_locked_disc_is_exported_with_active_discs(self):
        record = self.sample_record()
        record["disc_configs"] = [
            {
                "available": True,
                "label": "命盘一",
                "slots": [
                    {"state": "locked", "name": "防御时恢复生命"},
                    {"state": "active", "name": "噢", "star_stones": {"main": None, "support": None}},
                    {"state": "active", "name": "啥？", "star_stones": {"main": None, "support": None}},
                ],
            },
            {"available": False, "slots": []},
        ]

        entry = build_v3_document([record], "partial-discs")["records"][0]["entries"][0]

        self.assertEqual(
            entry["disc_loadouts"],
            [
                {
                    "id": "disc_1",
                    "name": "命盘一",
                    "discs": [
                        {"ot_name": "防御时恢复生命"},
                        {"ot_name": "噢"},
                        {"ot_name": "啥？"},
                    ],
                }
            ],
        )
        self.assertEqual(entry["section_status"]["disc_loadouts"], "partial")

    def test_disc_names_normalise_ui_punctuation_and_variant_character(self):
        cases = (
            (
                "char_095_zhangyan",
                ["攻击力大幅提升", "噢", "初始能量+1"],
                ["噢", "啥?", "初始能量+1"],
                ["噢", "啥？", "初始能量+1"],
            ),
            (
                "char_084_chendengsp",
                ["雨顺物康", "初始能量+3", "普攻伤害小幅增加"],
                ["初始能量+1", "时和岁丰", "积善余庆"],
                ["初始能量+1", "时和岁丰", "积善馀庆"],
            ),
        )
        for operator_id, first_names, raw_names, expected_names in cases:
            with self.subTest(operator_id=operator_id):
                record = self.sample_record(operator_id=operator_id)
                record["disc_configs"] = [
                    {
                        "available": True,
                        "label": "命盘一",
                        "slots": [
                            {"state": "active", "name": name}
                            for name in first_names
                        ],
                    },
                    {
                        "available": True,
                        "label": "命盘二",
                        "slots": [
                            {"state": "active", "name": name}
                            for name in raw_names
                        ],
                    },
                ]

                entry = build_v3_document([record], "disc-normalisation")["records"][0]["entries"][0]

                self.assertEqual(
                    [item["ot_name"] for item in entry["disc_loadouts"][1]["discs"]],
                    expected_names,
                )
                self.assertEqual(entry["section_status"]["disc_loadouts"], "ready")

    def test_one_unmatched_disc_name_does_not_drop_the_whole_loadout(self):
        record = self.sample_record(operator_id="char_095_zhangyan")
        record["disc_configs"] = [
            {
                "available": True,
                "label": "命盘一",
                "slots": [
                    {"state": "active", "name": "噢"},
                    {"state": "active", "name": "无法确认"},
                    {"state": "active", "name": "初始能量+1"},
                ],
            },
            {"available": False, "slots": []},
        ]

        entry = build_v3_document([record], "partial-disc")["records"][0]["entries"][0]

        self.assertEqual(
            entry["disc_loadouts"][0]["discs"],
            [{"ot_name": "噢"}, {"ot_name": "初始能量+1"}],
        )
        self.assertEqual(entry["section_status"]["disc_loadouts"], "partial")
        self.assertEqual(entry["diagnostics"]["disc_name_review"], ["无法确认"])

    def test_unknown_catalog_operator_never_passes_raw_disc_names(self):
        record = self.sample_record(operator_id="char_not_in_catalog")

        entry = build_v3_document([record], "missing-catalog")["records"][0]["entries"][0]

        self.assertNotIn("disc_loadouts", entry)
        self.assertEqual(entry["section_status"]["disc_loadouts"], "review")
        self.assertEqual(
            entry["diagnostics"]["disc_catalog_review"]["operator_id"],
            "char_not_in_catalog",
        )

    def test_final_catalog_guard_rejects_non_catalog_canonical_name(self):
        record = self.sample_record()
        with mock.patch(
            "custom.action.operator_growth_exchange._canonical_disc_name",
            return_value="不存在的命盘",
        ):
            entry = build_v3_document([record], "final-disc-guard")["records"][0]["entries"][0]

        self.assertNotIn("disc_loadouts", entry)
        self.assertEqual(entry["section_status"]["disc_loadouts"], "review")

    def test_similar_catalog_name_is_not_forced_by_fuzzy_matching(self):
        record = self.sample_record()
        record["disc_configs"] = [
            {
                "available": True,
                "label": "命盘一",
                "slots": [
                    {"state": "active", "name": "初始能量+2"},
                    {"state": "active", "name": "噢"},
                ],
            },
            {"available": False, "slots": []},
        ]

        entry = build_v3_document([record], "no-fuzzy-disc")["records"][0]["entries"][0]

        self.assertEqual(
            entry["disc_loadouts"][0]["discs"],
            [{"ot_name": "噢"}],
        )
        self.assertEqual(entry["section_status"]["disc_loadouts"], "partial")
        self.assertEqual(entry["diagnostics"]["disc_name_review"], ["初始能量+2"])

    def test_single_letter_ocr_suffix_resolves_to_catalog_name(self):
        record = self.sample_record(operator_id="char_109_chenqun")
        record["disc_configs"] = [
            {
                "available": True,
                "label": "命盘一",
                "slots": [
                    {"state": "active", "name": "行药C"},
                    {"state": "active", "name": "伐谋"},
                ],
            },
            {"available": False, "slots": []},
        ]

        entry = build_v3_document([record], "disc-letter-suffix")["records"][0]["entries"][0]

        self.assertEqual(
            entry["disc_loadouts"][0]["discs"],
            [{"ot_name": "行药"}, {"ot_name": "伐谋"}],
        )
        self.assertEqual(entry["section_status"]["disc_loadouts"], "partial")
        self.assertNotIn("disc_name_review", entry["diagnostics"])

    def test_disc_description_alias_preserves_meaningful_trailing_number(self):
        record = self.sample_record(operator_id="char_084_chendengsp")
        record["disc_configs"] = [
            {
                "available": True,
                "label": "命盘一",
                "slots": [
                    {"state": "active", "name": "雨顺物康"},
                    {"state": "active", "name": "初始能量+3"},
                    {"state": "active", "name": "普攻伤害小幅增加"},
                ],
            },
            {"available": False, "slots": []},
        ]

        entry = build_v3_document([record], "numeric-disc-name")["records"][0]["entries"][0]

        self.assertEqual(
            entry["disc_loadouts"][0]["discs"],
            [
                {"ot_name": "雨顺物康"},
                {"ot_name": "初始能量+3"},
                {"ot_name": "普攻伤害小幅增加"},
            ],
        )

    def test_unavailable_sections_do_not_emit_empty_values(self):
        record = self.sample_record(disc_configs=None, oddities={})
        entry = build_v3_document([record], "x")["records"][0]["entries"][0]
        self.assertNotIn("disc_loadouts", entry)
        self.assertNotIn("equipped_star_stones", entry)
        self.assertNotIn("oddities", entry["combat_stats"])
        self.assertEqual(entry["section_status"]["disc_loadouts"], "unavailable")

    def test_write_does_not_contain_token(self):
        document = build_v3_document([self.sample_record()], "x")
        with tempfile.TemporaryDirectory() as directory:
            path = write_v3_document(document, Path(directory) / "exchange.json")
            self.assertNotIn("secret-token", path.read_text(encoding="utf-8"))

    def test_error_redaction_does_not_expose_token(self):
        self.assertNotIn("secret-token", _redact("server echoed secret-token", "secret-token"))

    def test_preview_and_commit_send_identical_frozen_document(self):
        document = build_v3_document([self.sample_record()], "same-body")
        bodies = []

        class Response:
            status = 200

            def read(self):
                return b'{"data":{"accepted":1}}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def open_request(request, timeout):
            bodies.append(request.data)
            return Response()

        with mock.patch("custom.action.operator_growth_exchange.urllib_request.urlopen", side_effect=open_request):
            preview_v3_document(document, "http://example.test", "secret-token")
            commit_v3_document(document, "http://example.test", "secret-token")

        self.assertEqual(bodies[0], bodies[1])
        self.assertNotIn("secret-token", bodies[0].decode("utf-8"))

    def test_star_stones_differing_between_loadouts_are_review(self):
        record = self.sample_record()
        record["disc_configs"][1]["slots"][0]["star_stones"]["main"]["level"] = 59
        entry = build_v3_document([record], "x")["records"][0]["entries"][0]
        self.assertEqual(entry["section_status"]["equipment"], "review")
        self.assertNotIn("equipped_star_stones", entry)


if __name__ == "__main__":
    unittest.main()
