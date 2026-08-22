import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
sys.path.insert(0, str(AGENT_ROOT))

from custom.action.fixed_reward_reporting import (  # noqa: E402
    ITEMS_PATH,
    FixedRewardReporting,
    build_fixed_reward_results,
    parse_fixed_reward_plan,
)


class FixedRewardPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interface = json.loads(
            (REPO_ROOT / "assets" / "interface.json").read_text(encoding="utf-8")
        )

    def _case_attach(self, option_name, case_name):
        option = self.interface["option"][option_name]
        case = next(case for case in option["cases"] if case["name"] == case_name)
        return case["pipeline_override"]["历练奖励配置"]["attach"]

    def test_builds_fixed_reward_totals_and_ignores_zero_tiers(self):
        plan = parse_fixed_reward_plan(
            {
                "acquisition_channel": "历练-风火",
                "sweep_count": 6,
                "reward_item_ids": ["juanshan", "cuishan", "jinsishan"],
                "reward_counts": [0, 40, 30],
            }
        )

        results = build_fixed_reward_results(
            plan,
            {"cuishan": "翠扇", "jinsishan": "金丝扇"},
        )

        self.assertEqual(
            [(result["item_id"], result["count"]) for result in results],
            [("cuishan", 240), ("jinsishan", 180)],
        )

    def test_rejects_invalid_sweep_count(self):
        with self.assertRaisesRegex(ValueError, "sweep_count"):
            parse_fixed_reward_plan(
                {
                    "acquisition_channel": "历练-经验",
                    "sweep_count": 0,
                    "reward_item_ids": ["bingshucanjuan"],
                    "reward_counts": [25],
                }
            )

    def test_rejects_unknown_catalog_item(self):
        plan = parse_fixed_reward_plan(
            {
                "acquisition_channel": "历练-经验",
                "sweep_count": 1,
                "reward_item_ids": ["missing"],
                "reward_counts": [25],
            }
        )
        with self.assertRaisesRegex(ValueError, "未知道具"):
            build_fixed_reward_results(plan, {})

    def test_interface_experience_reward_table_is_complete(self):
        expected = {
            "新手历练": [25, 0],
            "初级历练": [25, 2],
            "中级历练": [30, 3],
            "高级历练": [40, 4],
            "终极历练": [50, 5],
            "实战历练": [65, 6],
            "鏖战历练": [85, 8],
            "绝境历练-最高": [100, 9],
        }
        actual = {
            case["name"]: case["pipeline_override"]["历练奖励配置"]["attach"][
                "reward_counts"
            ]
            for case in self.interface["option"]["经验历练级别"]["cases"]
        }
        self.assertEqual(actual, expected)

    def test_interface_element_options_form_valid_reward_plans(self):
        categories = self.interface["option"]["修为历练类别"]["cases"]
        levels = self.interface["option"]["修为历练级别"]["cases"]
        item_names = {
            item["id"]: item["name"]
            for item in json.loads(ITEMS_PATH.read_text(encoding="utf-8"))["items"]
        }
        for category in categories:
            category_attach = category["pipeline_override"]["历练奖励配置"][
                "attach"
            ]
            for level in levels:
                level_attach = level["pipeline_override"]["历练奖励配置"][
                    "attach"
                ]
                with self.subTest(category=category["name"], level=level["name"]):
                    plan = parse_fixed_reward_plan(
                        {
                            **category_attach,
                            **level_attach,
                            "sweep_count": 6,
                        }
                    )
                    results = build_fixed_reward_results(plan, item_names)
                    self.assertIn(len(results), (1, 2))

        wind_ten = parse_fixed_reward_plan(
            {
                **self._case_attach("修为历练类别", "风火历练"),
                **self._case_attach("修为历练级别", "十"),
                "sweep_count": 6,
            }
        )
        self.assertEqual(
            [
                (item_id, count * wind_ten.sweep_count)
                for item_id, count in wind_ten.rewards
            ],
            [("yushan", 300), ("xianmenshan", 240)],
        )

    def test_action_writes_local_reward_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "DailyRewards-test.txt"
            nodes = {
                "在线上传认证": {
                    "attach": {
                        "mode": "仅保存到本地",
                        "base_url": "https://example.test",
                    }
                },
                "历练奖励配置": {
                    "attach": {
                        "acquisition_channel": "历练-经验",
                        "sweep_count": 4,
                        "reward_item_ids": [
                            "bingshucanjuan",
                            "bingshuquanjuan",
                        ],
                        "reward_counts": [100, 9],
                        "inventory_report_path": str(report_path),
                    }
                },
            }
            context = SimpleNamespace(get_node_data=nodes.get)
            argv = SimpleNamespace(
                custom_action_param={"config_node": "历练奖励配置"}
            )

            result = FixedRewardReporting().run(context, argv)

            self.assertTrue(result.success)
            report = report_path.read_text(encoding="utf-8-sig")
            self.assertIn("渠道：历练-经验", report)
            self.assertIn("兵书残卷 × 400", report)
            self.assertIn("兵书全卷 × 36", report)


if __name__ == "__main__":
    unittest.main()
