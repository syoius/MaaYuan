import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from custom.action.cvpls_config import (
    CVPLSConfigure,
    parse_cvpls_department,
    parse_cvpls_project_day,
)


def _ocr(text: str, score: float = 0.99):
    return {
        "detail": {
            "filtered": [
                {"box": [10, 10, 100, 30], "score": score, "text": text}
            ]
        }
    }


class CVPLSConfigParsingTests(unittest.TestCase):
    def test_parses_project_day(self):
        self.assertEqual(parse_cvpls_project_day(_ocr("潜伏暗桩-第一天")), 1)
        self.assertEqual(parse_cvpls_project_day(_ocr("四部统筹—第3天")), 3)

    def test_rejects_ambiguous_project_day(self):
        detail = {
            "filtered": [
                {"box": [0, 0, 10, 10], "text": "第一天"},
                {"box": [20, 0, 10, 10], "text": "第二天"},
            ]
        }
        with self.assertRaises(ValueError):
            parse_cvpls_project_day(detail)

    def test_matches_exact_and_single_character_ocr_error_department(self):
        self.assertEqual(parse_cvpls_department(_ocr("项目名称：潜伏暗桩")), "潜伏暗桩")
        self.assertEqual(parse_cvpls_department(_ocr("四部綂筹")), "四部统筹")

    def test_matches_traditional_department_and_label(self):
        self.assertEqual(
            parse_cvpls_department(_ocr("項目名稱：潛伏暗樁")), "潜伏暗桩"
        )


class _Controller:
    def post_screencap(self):
        return self

    def wait(self):
        return self

    def get(self):
        return object()


class _Context:
    def __init__(self, name="潜伏暗桩", date="项目进度-第二天"):
        self.tasker = SimpleNamespace(controller=_Controller())
        self.name = name
        self.date = date
        self.recognition_calls = []
        self.overrides = []

    def run_recognition(self, name, image):
        del image
        self.recognition_calls.append(name)
        if name == "提取项目日期":
            return _ocr(self.date)
        if name == "提取项目名称":
            return _ocr(self.name)
        raise AssertionError(f"unexpected recognition: {name}")

    def override_pipeline(self, override):
        self.overrides.append(override)
        return True


class CVPLSConfigureTests(unittest.TestCase):
    def test_construction_office_uses_fixed_day_when_date_has_no_day(self):
        context = _Context(name="搬砖办人才招募", date="搬砖办人才招募")
        action = CVPLSConfigure()
        argv = SimpleNamespace(custom_action_param="{}")

        result = action.run(context, argv)

        self.assertTrue(result.success)
        final_params = context.overrides[0]["自动审简历"]["action"]["param"][
            "custom_action_param"
        ]
        self.assertEqual(final_params["department"], "搬砖办")
        self.assertEqual(final_params["day"], 1)

    def test_recognizes_and_overrides_screen_action_parameters(self):
        context = _Context()
        action = CVPLSConfigure()
        argv = SimpleNamespace(
            custom_action_param=json.dumps(
                {
                    "target_node": "自动审简历",
                    "screen_params": {
                        "portrait_debug": True,
                        "portrait_similarity_threshold": 0.65,
                    },
                },
                ensure_ascii=False,
            )
        )

        result = action.run(context, argv)

        self.assertTrue(result.success)
        self.assertEqual(
            context.recognition_calls, ["提取项目日期", "提取项目名称"]
        )
        final_params = context.overrides[0]["自动审简历"]["action"]["param"][
            "custom_action_param"
        ]
        self.assertEqual(
            final_params,
            {
                "portrait_debug": True,
                "portrait_similarity_threshold": 0.65,
                "department": "潜伏暗桩",
                "day": 2,
            },
        )

    def test_does_not_override_when_project_name_is_unknown(self):
        context = _Context(name="完全未知项目")
        action = CVPLSConfigure()
        argv = SimpleNamespace(custom_action_param="{}")

        result = action.run(context, argv)

        self.assertFalse(result.success)
        self.assertEqual(context.overrides, [])


if __name__ == "__main__":
    unittest.main()
