import unittest
import sys
import json
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from custom.action.cvpls import (
    CVPLSScreen,
    build_requirements,
    collect_comment_options,
    extract_ocr_items,
    parse_certificate_info,
    parse_resume_info,
    _find_comment_option,
    _portrait_similarity,
    _write_portrait_debug_sample,
    _wait_task_detail,
)


COMMENT_OCR = {
    "detail": {
        "filtered": [
            {
                "box": [82, 1203, 241, 26],
                "score": 0.999545,
                "text": "简历与毕业证信息不一致",
            },
            {"box": [135, 1144, 137, 30], "score": 0.999634, "text": "无肖像者勿投"},
            {"box": [153, 1084, 98, 35], "score": 0.999196, "text": "简历无误"},
            {
                "box": [359, 1109, 309, 99],
                "score": 0.879447,
                "text": "非广陵学校毕业需持证应明",
            },
            {"box": [439, 1086, 137, 30], "score": 0.999911, "text": "优质单位专场"},
            {"box": [457, 1197, 102, 38], "score": 0.999842, "text": "学府作假"},
            {"box": [612, 1073, 53, 37], "score": 0.999879, "text": "新"},
        ]
    }
}

RESUME_OCR = {
    "detail": {
        "filtered": [
            {"box": [297, 701, 140, 30], "text": "工作/实习经验"},
            {"box": [300, 738, 191, 26], "text": "青丘戏坊-百戏艺人"},
            {"box": [301, 652, 159, 30], "text": "状态：在职员工"},
            {"box": [302, 547, 156, 27], "text": "姓名：师马光临"},
            {"box": [304, 601, 242, 27], "text": "毕业学校:雒阳鸿都门学"},
            {"box": [508, 701, 60, 35], "text": "大又"},
            {"box": [541, 734, 49, 36], "text": "1年"},
        ]
    }
}

CERTIFICATE_OCR = {
    "detail": {
        "filtered": [
            {"box": [400, 600, 160, 30], "text": "姓名：师马光临"},
            {"box": [400, 650, 240, 30], "text": "毕业学校：雒阳鸿都门学"},
            {"box": [400, 700, 180, 30], "text": "毕业时间：招聘10年"},
        ]
    }
}


class RequirementsTests(unittest.TestCase):
    def test_builds_and_orders_department_requirements(self):
        requirements = build_requirements("雀部管培", 3)
        self.assertEqual(
            [item["type"] for item in requirements],
            [
                "fresh_graduate_only",
                "minimum_work_or_internship_segments",
                "portrait_required",
                "certificate_required_for_fresh_graduate",
            ],
        )

    def test_accepts_department_key(self):
        requirements = build_requirements("construction_office", 1)
        self.assertEqual(
            [item["text"] for item in requirements],
            ["无肖像者勿投", "应届生需持证应聘"],
        )

    def test_accepts_traditional_department_name(self):
        requirements = build_requirements("搬磚辦", 1)
        self.assertEqual(
            [item["text"] for item in requirements],
            ["无肖像者勿投", "应届生需持证应聘"],
        )

    def test_rejects_day_outside_range(self):
        with self.assertRaises(ValueError):
            build_requirements("搬砖办", 4)


class OcrCollectionTests(unittest.TestCase):
    def test_normalizes_traditional_ocr_and_preserves_raw_text(self):
        items = extract_ocr_items(
            {
                "filtered": [
                    {
                        "box": [10, 20, 30, 40],
                        "score": 0.99,
                        "text": "簡歷無誤",
                    }
                ]
            }
        )
        self.assertEqual(items[0]["text"], "简历无误")
        self.assertEqual(items[0]["raw_text"], "簡歷無誤")

    def test_finds_simplified_requirement_in_traditional_comment_ocr(self):
        options = collect_comment_options(
            {
                "filtered": [
                    {
                        "box": [10, 20, 100, 30],
                        "score": 0.99,
                        "text": "所有人需持證應聘",
                    }
                ]
            }
        )

        option = _find_comment_option(options, "所有人需持证应聘")

        self.assertIsNotNone(option)
        self.assertEqual(option["box"], [10, 20, 100, 30])

    def test_collects_six_comments_and_filters_badge(self):
        options = collect_comment_options(COMMENT_OCR)
        self.assertEqual(len(options), 6)
        self.assertNotIn("新", [item["text"] for item in options])
        self.assertEqual(
            next(item for item in options if item["text"] == "简历无误")["box"],
            [153, 1084, 98, 35],
        )

    def test_comment_collection_prefers_six_highest_scores(self):
        detail = {
            "filtered": [
                {
                    "box": [index * 10, 100, 10, 10],
                    "score": score,
                    "text": f"评语{index}",
                }
                for index, score in enumerate(
                    [0.10, 0.90, 0.80, 0.70, 0.60, 0.50, 0.99]
                )
            ]
            + [
                {
                    "box": [100, 100, 10, 10],
                    "score": 1.0,
                    "text": "新",
                }
            ]
        }

        options = collect_comment_options(detail)

        self.assertEqual(
            [option["text"] for option in options],
            ["评语6", "评语1", "评语2", "评语3", "评语4", "评语5"],
        )

    def test_parses_resume_and_joins_experience_row(self):
        resume = parse_resume_info(RESUME_OCR)
        self.assertEqual(resume["name"], "师马光临")
        self.assertEqual(resume["school"], "雒阳鸿都门学")
        self.assertEqual(resume["status"], "在职员工")
        self.assertEqual(resume["experience_count"], 1)
        self.assertEqual(
            resume["experiences"][0],
            {
                "company": "青丘戏坊",
                "role": "百戏艺人",
                "duration": "1年",
                "duration_years": 1.0,
                "text": "青丘戏坊-百戏艺人 1年",
            },
        )

    def test_parses_multiple_experience_rows(self):
        detail = {
            "filtered": [
                {"box": [10, 100, 100, 20], "text": "工作/实习经验"},
                {"box": [10, 140, 150, 20], "text": "里八华-死士"},
                {"box": [200, 139, 40, 20], "text": "2年"},
                {"box": [10, 180, 150, 20], "text": "绣衣楼-密探"},
                {"box": [200, 182, 40, 20], "text": "3年"},
            ]
        }
        resume = parse_resume_info(detail)
        self.assertEqual(resume["experience_count"], 2)
        self.assertEqual(
            [item["duration_years"] for item in resume["experiences"]], [2.0, 3.0]
        )

    def test_parses_certificate_information(self):
        certificate = parse_certificate_info(CERTIFICATE_OCR)
        self.assertEqual(certificate["name"], "师马光临")
        self.assertEqual(certificate["school"], "雒阳鸿都门学")
        self.assertEqual(certificate["graduation_time"], "招聘10年")
        self.assertEqual(certificate["graduation_year"], 10)

    def test_parses_traditional_resume_and_experience(self):
        detail = {
            "filtered": [
                {"box": [10, 100, 100, 20], "text": "姓名：師馬光臨"},
                {"box": [10, 130, 180, 20], "text": "畢業學校：雒陽鴻都門學"},
                {"box": [10, 160, 150, 20], "text": "狀態：在職員工"},
                {"box": [10, 200, 150, 20], "text": "工作/實習經驗"},
                {"box": [10, 240, 170, 20], "text": "繡衣樓-親衛"},
                {"box": [200, 240, 40, 20], "text": "2年"},
            ]
        }

        resume = parse_resume_info(detail)

        self.assertEqual(resume["name"], "师马光临")
        self.assertEqual(resume["school"], "雒阳鸿都门学")
        self.assertEqual(resume["status"], "在职员工")
        self.assertEqual(resume["experience_count"], 1)
        self.assertEqual(resume["experiences"][0]["company"], "绣衣楼")
        self.assertEqual(resume["experiences"][0]["role"], "亲卫")

    def test_parses_traditional_certificate_information(self):
        detail = {
            "filtered": [
                {"box": [10, 100, 100, 20], "text": "姓名：師馬光臨"},
                {"box": [10, 130, 180, 20], "text": "畢業學校：雒陽鴻都門學"},
                {"box": [10, 160, 160, 20], "text": "畢業時間：招聘10年"},
            ]
        }

        certificate = parse_certificate_info(detail)

        self.assertEqual(certificate["name"], "师马光临")
        self.assertEqual(certificate["school"], "雒阳鸿都门学")
        self.assertEqual(certificate["graduation_time"], "招聘10年")
        self.assertEqual(certificate["graduation_year"], 10)

    def test_portrait_similarity_compares_resized_images(self):
        random = np.random.default_rng(1)
        portrait = random.integers(0, 256, (185, 133, 3), dtype=np.uint8)
        resized = cv2.resize(portrait, (101, 140), interpolation=cv2.INTER_AREA)
        self.assertGreater(_portrait_similarity(portrait, portrait), 0.99)
        self.assertGreater(_portrait_similarity(portrait, resized), 0.7)

    def test_writes_portrait_debug_images_and_csv_record(self):
        portrait = np.full((40, 30, 3), 127, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory) / "中文目录"
            files = _write_portrait_debug_sample(
                directory,
                2,
                {"name": "甲", "school": "广陵书院"},
                {"name": "甲", "school": "广陵书院"},
                portrait,
                portrait,
                0.93456789,
                0.72,
            )

            self.assertTrue(Path(files["简历头像"]).is_file())
            self.assertTrue(Path(files["证书头像"]).is_file())
            with Path(files["对比记录"]).open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["序号"], "2")
            self.assertEqual(rows[0]["相似度"], "0.934568")
            self.assertEqual(rows[0]["自动判定头像一致"], "是")
            self.assertEqual(rows[0]["人工标注同一人"], "")


class _FakeController:
    def __init__(self):
        self.screencap_count = 0

    def post_screencap(self):
        self.screencap_count += 1
        return self

    def wait(self):
        return self

    def get(self):
        return object()


class _FakeContext:
    def __init__(self):
        self.tasker = SimpleNamespace(controller=_FakeController())
        self.task_calls = []
        self.recognition_calls = []
        self.overrides = []

    def run_task(self, name, pipeline_override=None):
        self.task_calls.append((name, pipeline_override))
        failed = name == "检查简历-查看下一位"
        return SimpleNamespace(status=SimpleNamespace(failed=failed))

    def run_recognition(self, name, image):
        del image
        self.recognition_calls.append(name)
        if name == "检查评语位置":
            return COMMENT_OCR
        if name == "获取简历信息":
            return RESUME_OCR
        if name == "完成审批check":
            return SimpleNamespace(hit=True)
        return SimpleNamespace(hit=False)

    def override_pipeline(self, override):
        self.overrides.append(override)
        return True


class _AllPassedScreen(CVPLSScreen):
    def _evaluate_requirements(self, context, image, requirements, resume):
        del context, image, resume
        return {
            "complete": True,
            "unsupported": [],
            "results": [
                {"type": item["type"], "text": item["text"], "passed": True}
                for item in requirements
            ],
        }


class ScreeningFlowTests(unittest.TestCase):
    def test_dragging_second_page_comment_switches_there_and_back(self):
        context = _FakeContext()
        action = CVPLSScreen()
        action._comment_page_delay = 0

        dragged = action._drag_comment(
            context,
            [{"text": "学府作假", "box": [400, 1100, 120, 30], "page": 2}],
            "学府作假",
        )

        self.assertTrue(dragged)
        self.assertEqual(action._comment_page, 1)
        self.assertEqual(
            [name for name, _ in context.task_calls],
            ["评语翻页-去第二页", "拖评语-action", "评语翻页-回第一页"],
        )
        self.assertEqual(
            context.overrides[-1]["拖评语-action"]["action"]["param"]["target"],
            [400, 1100, 120, 30],
        )

    def test_certificate_second_round_checks_are_short_circuited(self):
        action = CVPLSScreen()
        base_resume = {
            "name": "师马光临",
            "school": "雒阳鸿都门学",
            "status": "应届毕业生",
            "experiences": [],
            "experience_count": 0,
        }
        valid_certificate = {
            "name": "师马光临",
            "school": "雒阳鸿都门学",
            "graduation_time": "招聘10年",
            "graduation_year": 10,
        }

        mismatch = action._evaluate_certificate_data(
            base_resume,
            {**valid_certificate, "name": "张三"},
            0.95,
            0.72,
            10,
        )
        forged = action._evaluate_certificate_data(
            base_resume,
            {**valid_certificate, "school": "雒阳鸿都门字"},
            0.95,
            0.72,
            10,
        )
        old_graduation = action._evaluate_certificate_data(
            base_resume,
            {
                **valid_certificate,
                "graduation_time": "招聘9年",
                "graduation_year": 9,
            },
            0.95,
            0.72,
            10,
        )
        employed_without_experience = action._evaluate_resume_second_round_data(
            {**base_resume, "status": "在职员工"}
        )

        self.assertEqual(mismatch["results"][0]["text"], "简历与毕业证信息不一致")
        self.assertEqual(
            mismatch["results"][0]["evidence"],
            {
                "不一致项目": {
                    "姓名": {"简历": "师马光临", "毕业证": "张三"}
                }
            },
        )
        self.assertEqual(forged["results"][0]["text"], "学府作假")
        self.assertEqual(
            old_graduation["results"][0]["text"], "应届毕业年早于当年"
        )
        self.assertEqual(
            employed_without_experience["results"][0]["text"],
            "在职却无工作经历",
        )

    def test_resume_certificate_mismatch_lists_only_actual_differences(self):
        action = CVPLSScreen()
        resume = {
            "name": "师马光临",
            "school": "雒阳鸿都门学",
            "status": "应届毕业生",
            "experiences": [],
            "experience_count": 0,
        }
        certificate = {
            "name": "张三",
            "school": "长安太学",
            "graduation_time": "招聘10年",
            "graduation_year": 10,
        }

        evaluation = action._evaluate_certificate_data(
            resume, certificate, 0.5, 0.72, 10
        )

        self.assertEqual(
            evaluation["results"][0]["evidence"]["不一致项目"],
            {
                "姓名": {"简历": "师马光临", "毕业证": "张三"},
                "毕业学校": {"简历": "雒阳鸿都门学", "毕业证": "长安太学"},
                "头像": {"相似度": 0.5, "最低阈值": 0.72},
            },
        )

    def test_certificate_second_round_rejects_unknown_company(self):
        action = CVPLSScreen()
        resume = {
            "name": "师马光临",
            "school": "雒阳鸿都门学",
            "status": "应届毕业生",
            "experiences": [{"company": "不存在商号", "role": "厨师"}],
            "experience_count": 1,
        }
        certificate = {
            "name": "师马光临",
            "school": "雒阳鸿都门学",
            "graduation_time": "招聘10年",
            "graduation_year": 10,
        }

        evaluation = action._evaluate_resume_second_round_data(resume)

        self.assertEqual(evaluation["results"][0]["text"], "公司作假")
        self.assertEqual(
            evaluation["results"][0]["evidence"]["异常工作/实习经历"],
            [{"单位": "不存在商号", "职位": "厨师", "原因": "公司不存在"}],
        )

    def test_traditional_valid_company_and_role_are_not_marked_forged(self):
        resume = parse_resume_info(
            {
                "filtered": [
                    {"box": [10, 100, 150, 20], "text": "狀態：在職員工"},
                    {"box": [10, 140, 150, 20], "text": "工作/實習經驗"},
                    {"box": [10, 180, 170, 20], "text": "繡衣樓-親衛"},
                    {"box": [200, 180, 40, 20], "text": "2年"},
                ]
            }
        )

        evaluation = CVPLSScreen()._evaluate_resume_second_round_data(resume)

        company_check = next(
            result
            for result in evaluation["results"]
            if result["type"] == "forged_company"
        )
        self.assertTrue(company_check["passed"])
        self.assertEqual(company_check["evidence"]["异常工作/实习经历"], [])

    def test_certificate_second_round_rejects_role_from_another_company(self):
        action = CVPLSScreen()
        resume = {
            "name": "师马光临",
            "school": "雒阳鸿都门学",
            "status": "应届毕业生",
            "experiences": [{"company": "里八华", "role": "密探"}],
            "experience_count": 1,
        }
        certificate = {
            "name": "师马光临",
            "school": "雒阳鸿都门学",
            "graduation_time": "招聘10年",
            "graduation_year": 10,
        }

        evaluation = action._evaluate_resume_second_round_data(resume)

        self.assertEqual(evaluation["results"][0]["text"], "公司作假")
        self.assertEqual(
            evaluation["results"][0]["evidence"]["异常工作/实习经历"],
            [
                {
                    "单位": "里八华",
                    "职位": "密探",
                    "原因": "岗位不属于该公司",
                }
            ],
        )

    def test_certificate_second_round_accepts_known_company_role_pair(self):
        action = CVPLSScreen()
        resume = {
            "name": "师马光临",
            "school": "雒阳鸿都门学",
            "status": "应届毕业生",
            "experiences": [{"company": "里八华", "role": "厨师"}],
            "experience_count": 1,
        }
        certificate = {
            "name": "师马光临",
            "school": "雒阳鸿都门学",
            "graduation_time": "招聘10年",
            "graduation_year": 10,
        }

        evaluation = action._evaluate_resume_second_round_data(resume)

        company_check = next(
            result
            for result in evaluation["results"]
            if result["text"] == "公司作假"
        )
        self.assertTrue(company_check["passed"])
        self.assertEqual(
            company_check["evidence"]["异常工作/实习经历"], []
        )

    def test_failed_resume_precheck_runs_before_certificate_detection(self):
        context = _FakeContext()
        action = _AllPassedScreen()
        action._evaluate_resume_second_round_data = lambda resume: {
            "complete": True,
            "results": [
                {
                    "type": "forged_company",
                    "text": "公司作假",
                    "passed": False,
                    "evidence": {"异常工作/实习经历": []},
                }
            ],
        }
        action._drag_comment = lambda context, options, text: True
        argv = SimpleNamespace(
            custom_action_param=json.dumps(
                {"department": "搬砖办", "day": 1}, ensure_ascii=False
            )
        )

        result = action.run(context, argv)

        self.assertTrue(result.success)
        self.assertNotIn("持证check", context.recognition_calls)
        self.assertNotIn("获取证书信息", context.recognition_calls)
        self.assertEqual(action.last_session["resumes"][0]["comments"], ["公司作假"])
        self.assertIsNone(action.last_session["resumes"][0]["evaluation"])

    def test_data_requirement_failure_skips_portrait_and_certificate_checks(self):
        context = _FakeContext()
        action = CVPLSScreen()
        action._drag_comment = lambda context, options, text: True
        argv = SimpleNamespace(
            custom_action_param=json.dumps(
                {"department": "四部统筹", "day": 3}, ensure_ascii=False
            )
        )

        result = action.run(context, argv)

        self.assertTrue(result.success)
        self.assertNotIn("无肖像check", context.recognition_calls)
        self.assertNotIn("持证check", context.recognition_calls)
        self.assertEqual(
            action.last_session["resumes"][0]["comments"], ["选拔应届毕业生"]
        )

    def test_external_stop_interrupts_before_resume_is_processed(self):
        context = _FakeContext()
        original_recognition = context.run_recognition

        def recognize_and_stop(name, image):
            result = original_recognition(name, image)
            if name == "获取简历信息":
                context.tasker.stopping = True
            return result

        context.run_recognition = recognize_and_stop
        action = _AllPassedScreen()
        argv = SimpleNamespace(
            custom_action_param=json.dumps(
                {"department": "搬砖办", "day": 1}, ensure_ascii=False
            )
        )

        result = action.run(context, argv)

        self.assertFalse(result.success)
        self.assertEqual(action.last_session["stop_reason"], "external_stop")
        self.assertEqual(
            [name for name, _ in context.task_calls],
            ["检查简历-新简历", "评语翻页-去第二页", "评语翻页-回第一页"],
        )
        self.assertEqual(context.overrides, [])

    def test_fresh_graduate_check_reuses_collected_resume_status(self):
        context = _FakeContext()
        action = CVPLSScreen()
        evaluation = action._evaluate_requirements(
            context,
            object(),
            [{"type": "fresh_graduate_only", "text": "选拔应届毕业生"}],
            {"status": "应届毕业生"},
        )

        self.assertTrue(evaluation["complete"])
        self.assertTrue(evaluation["results"][0]["passed"])
        self.assertEqual(context.recognition_calls, [])

    def test_past_graduate_check_reuses_collected_resume_status(self):
        action = CVPLSScreen()
        requirement = [
            {"type": "past_graduate_only", "text": "往届毕业生限定"}
        ]

        employed = action._evaluate_requirements(
            _FakeContext(), object(), requirement, {"status": "在职员工"}
        )
        fresh = action._evaluate_requirements(
            _FakeContext(), object(), requirement, {"status": "应届毕业生"}
        )

        self.assertTrue(employed["results"][0]["passed"])
        self.assertFalse(fresh["results"][0]["passed"])

    def test_everyone_must_have_certificate(self):
        requirement = [
            {
                "type": "certificate_required_for_everyone",
                "text": "所有人需持证应聘",
            }
        ]

        certified_context = _FakeContext()

        def recognize_certificate(name, image):
            del image
            certified_context.recognition_calls.append(name)
            return SimpleNamespace(hit=name == "持证check")

        certified_context.run_recognition = recognize_certificate
        certified = CVPLSScreen()._evaluate_requirements(
            certified_context, object(), requirement, {"status": "在职员工"}
        )
        uncertified_context = _FakeContext()
        uncertified = CVPLSScreen()._evaluate_requirements(
            uncertified_context, object(), requirement, {"status": "往届毕业生"}
        )

        self.assertTrue(certified["results"][0]["passed"])
        self.assertEqual(certified_context.recognition_calls, ["持证check"])
        self.assertFalse(uncertified["results"][0]["passed"])

    def test_guangling_graduate_does_not_need_certificate_check(self):
        context = _FakeContext()
        action = CVPLSScreen()
        evaluation = action._evaluate_requirements(
            context,
            object(),
            [
                {
                    "type": "certificate_required_for_non_guangling_school_graduate",
                    "text": "非广陵学校毕业需持证应聘",
                }
            ],
            {"school": "广陵书院", "experiences": []},
        )

        self.assertTrue(evaluation["results"][0]["passed"])
        self.assertEqual(context.recognition_calls, [])

    def test_non_guangling_graduate_must_have_certificate(self):
        context = _FakeContext()

        def recognize(name, image):
            del image
            context.recognition_calls.append(name)
            return SimpleNamespace(hit=name == "持证check")

        context.run_recognition = recognize
        action = CVPLSScreen()
        evaluation = action._evaluate_requirements(
            context,
            object(),
            [
                {
                    "type": "certificate_required_for_non_guangling_school_graduate",
                    "text": "非广陵学校毕业需持证应聘",
                }
            ],
            {"school": "雒阳鸿都门学", "experiences": []},
        )

        self.assertTrue(evaluation["results"][0]["passed"])
        self.assertEqual(context.recognition_calls, ["持证check"])

        no_certificate = CVPLSScreen()._evaluate_requirements(
            _FakeContext(),
            object(),
            [
                {
                    "type": "certificate_required_for_non_guangling_school_graduate",
                    "text": "非广陵学校毕业需持证应聘",
                }
            ],
            {"school": "长安太学", "experiences": []},
        )
        self.assertFalse(no_certificate["results"][0]["passed"])

    def test_premium_company_requirement_uses_collected_experiences(self):
        action = CVPLSScreen()
        requirement = [
            {
                "type": "premium_company_experience_required",
                "text": "优质单位专场",
            }
        ]

        premium = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {"experiences": [{"company": "绣衣楼", "role": "密探"}]},
        )
        ordinary = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {"experiences": [{"company": "青丘戏坊", "role": "百戏艺人"}]},
        )

        self.assertTrue(premium["results"][0]["passed"])
        self.assertFalse(ordinary["results"][0]["passed"])

    def test_first_failed_requirement_stops_later_checks(self):
        context = _FakeContext()
        action = CVPLSScreen()
        evaluation = action._evaluate_requirements(
            context,
            object(),
            [
                {
                    "type": "certificate_required_for_non_guangling_school_graduate",
                    "text": "非广陵学校毕业需持证应聘",
                },
                {
                    "type": "premium_company_experience_required",
                    "text": "优质单位专场",
                },
            ],
            {
                "school": "雒阳鸿都门学",
                "experiences": [{"company": "绣衣楼", "role": "密探"}],
            },
        )

        self.assertTrue(evaluation["complete"])
        self.assertEqual(len(evaluation["results"]), 1)
        self.assertEqual(
            evaluation["results"][0]["type"],
            "certificate_required_for_non_guangling_school_graduate",
        )
        self.assertFalse(evaluation["results"][0]["passed"])

    def test_martial_role_requirement_uses_company_and_role(self):
        action = CVPLSScreen()
        requirement = [
            {
                "type": "matching_martial_role_experience_required",
                "text": "武力类经历对口专招",
            }
        ]

        martial = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {
                "experiences": [
                    {"company": "绣衣楼", "role": "亲卫"},
                    {"company": "青丘戏坊", "role": "乐师"},
                ]
            },
        )
        non_martial = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {"experiences": [{"company": "绣衣楼", "role": "谋士"}]},
        )

        self.assertTrue(martial["results"][0]["passed"])
        self.assertEqual(
            martial["results"][0]["evidence"]["命中的武力类经历"],
            [{"单位": "绣衣楼", "职位": "亲卫"}],
        )
        self.assertFalse(non_martial["results"][0]["passed"])

    def test_fresh_graduate_certificate_requirement_and_cache(self):
        context = _FakeContext()

        def recognize(name, image):
            del image
            context.recognition_calls.append(name)
            return SimpleNamespace(hit=name == "持证check")

        context.run_recognition = recognize
        action = CVPLSScreen()
        evaluation = action._evaluate_requirements(
            context,
            object(),
            [
                {
                    "type": "certificate_required_for_fresh_graduate",
                    "text": "应届生需持证应聘",
                },
                {
                    "type": "certificate_required_for_everyone",
                    "text": "所有人需持证应聘",
                },
            ],
            {"status": "应届毕业生"},
        )

        self.assertTrue(evaluation["complete"])
        self.assertTrue(all(result["passed"] for result in evaluation["results"]))
        self.assertEqual(context.recognition_calls, ["持证check"])

        non_fresh_context = _FakeContext()
        non_fresh = CVPLSScreen()._evaluate_requirements(
            non_fresh_context,
            object(),
            [
                {
                    "type": "certificate_required_for_fresh_graduate",
                    "text": "应届生需持证应聘",
                }
            ],
            {"status": "在职员工"},
        )
        self.assertTrue(non_fresh["results"][0]["passed"])
        self.assertEqual(non_fresh_context.recognition_calls, [])

        uncertified_fresh = CVPLSScreen()._evaluate_requirements(
            _FakeContext(),
            object(),
            [
                {
                    "type": "certificate_required_for_fresh_graduate",
                    "text": "应届生需持证应聘",
                }
            ],
            {"status": "应届毕业生"},
        )
        self.assertFalse(uncertified_fresh["results"][0]["passed"])

    def test_minimum_experience_segments_and_inconsistent_count(self):
        requirement = [
            {
                "type": "minimum_work_or_internship_segments",
                "params": {"minimum": 2},
                "text": "两段工作/实习经验起投",
            }
        ]
        experiences = [
            {"company": "里八华", "role": "厨师"},
            {"company": "绣衣楼", "role": "谋士"},
        ]
        action = CVPLSScreen()

        passed = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {"experiences": experiences, "experience_count": 2},
        )
        below_minimum = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {"experiences": experiences[:1], "experience_count": 1},
        )
        inconsistent = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {"experiences": experiences, "experience_count": 1},
        )

        self.assertTrue(passed["results"][0]["passed"])
        self.assertFalse(below_minimum["results"][0]["passed"])
        self.assertIn("计数与解析出的经历数量不一致", inconsistent["data_error"])

    def test_matching_role_requires_valid_company_role_pair(self):
        requirement = [
            {
                "type": "matching_role_experience_required",
                "params": {"role": "死士"},
                "text": "死士经历对口专招",
            }
        ]
        action = CVPLSScreen()
        valid = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {
                "experiences": [{"company": "里八华", "role": "死士"}],
                "experience_count": 1,
            },
        )
        invalid_pair = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {
                "experiences": [{"company": "青丘戏坊", "role": "死士"}],
                "experience_count": 1,
            },
        )

        self.assertTrue(valid["results"][0]["passed"])
        self.assertFalse(invalid_pair["results"][0]["passed"])

    def test_required_and_forbidden_company_experience(self):
        action = CVPLSScreen()
        resume = {
            "experiences": [{"company": "绣衣楼", "role": "谋士"}],
            "experience_count": 1,
        }
        required = action._evaluate_requirements(
            _FakeContext(),
            object(),
            [
                {
                    "type": "former_company_member_only",
                    "params": {"company_id": "embroidered_uniform_tower"},
                    "text": "绣衣楼旧部专场",
                }
            ],
            resume,
        )
        forbidden = action._evaluate_requirements(
            _FakeContext(),
            object(),
            [
                {
                    "type": "company_experience_forbidden",
                    "params": {"company_id": "embroidered_uniform_tower"},
                    "text": "禁止有绣衣楼经历者入内",
                }
            ],
            resume,
        )
        other_company_resume = {
            "experiences": [{"company": "青丘戏坊", "role": "乐师"}],
            "experience_count": 1,
        }
        required_missing = action._evaluate_requirements(
            _FakeContext(),
            object(),
            [
                {
                    "type": "former_company_member_only",
                    "params": {"company_id": "embroidered_uniform_tower"},
                    "text": "绣衣楼旧部专场",
                }
            ],
            other_company_resume,
        )
        forbidden_absent = action._evaluate_requirements(
            _FakeContext(),
            object(),
            [
                {
                    "type": "company_experience_forbidden",
                    "params": {"company_id": "embroidered_uniform_tower"},
                    "text": "禁止有绣衣楼经历者入内",
                }
            ],
            other_company_resume,
        )

        self.assertTrue(required["results"][0]["passed"])
        self.assertFalse(forbidden["results"][0]["passed"])
        self.assertFalse(required_missing["results"][0]["passed"])
        self.assertTrue(forbidden_absent["results"][0]["passed"])

    def test_top_tier_school_uses_conservative_resume_ocr_matching(self):
        requirement = [
            {
                "type": "top_tier_school_alumni_only",
                "text": "顶尖学府校友专场",
            }
        ]
        action = CVPLSScreen()
        fuzzy_top_tier = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {"school": "长安太字"},
        )
        fuzzy_non_top_tier = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {"school": "雒阳鸿都门字"},
        )

        self.assertTrue(fuzzy_top_tier["results"][0]["passed"])
        self.assertEqual(
            fuzzy_top_tier["results"][0]["evidence"]["匹配学校"], "长安太学"
        )
        self.assertFalse(fuzzy_non_top_tier["results"][0]["passed"])

    def test_minimum_experience_years_sums_segments_and_rejects_missing_duration(self):
        requirement = [
            {
                "type": "minimum_work_or_internship_years",
                "params": {"minimum": 10},
                "text": "10年实习/工作经验起投",
            }
        ]
        action = CVPLSScreen()
        passed = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {
                "experiences": [
                    {"company": "里八华", "role": "厨师", "duration_years": 4.0},
                    {"company": "绣衣楼", "role": "谋士", "duration_years": 6.0},
                ],
                "experience_count": 2,
            },
        )
        below_minimum = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {
                "experiences": [
                    {"company": "里八华", "role": "厨师", "duration_years": 4.0},
                    {"company": "绣衣楼", "role": "谋士", "duration_years": 5.0},
                ],
                "experience_count": 2,
            },
        )
        missing_duration = action._evaluate_requirements(
            _FakeContext(),
            object(),
            requirement,
            {
                "experiences": [{"company": "里八华", "role": "厨师"}],
                "experience_count": 1,
            },
        )

        self.assertTrue(passed["results"][0]["passed"])
        self.assertEqual(
            passed["results"][0]["evidence"]["累计工作/实习年限"], 10.0
        )
        self.assertFalse(below_minimum["results"][0]["passed"])
        self.assertIn("缺少时长", missing_duration["data_error"])

    def test_one_call_controls_a_complete_round(self):
        context = _FakeContext()
        action = _AllPassedScreen()
        argv = SimpleNamespace(
            custom_action_param=json.dumps(
                {"department": "搬砖办", "day": 1}, ensure_ascii=False
            )
        )

        result = action.run(context, argv)

        self.assertTrue(result.success)
        self.assertEqual(context.tasker.controller.screencap_count, 4)
        self.assertEqual(context.recognition_calls.count("检查评语位置"), 2)
        self.assertEqual(context.recognition_calls.count("获取简历信息"), 1)
        self.assertEqual(context.recognition_calls.count("完成审批check"), 1)
        self.assertEqual(
            [name for name, _ in context.task_calls],
            [
                "检查简历-新简历",
                "评语翻页-去第二页",
                "评语翻页-回第一页",
                "拖评语-action",
                "处置简历-合格",
            ],
        )
        self.assertEqual(
            context.task_calls[4][1],
            {"处置简历-合格": {"post_delay": 1000}},
        )
        self.assertTrue(action.last_session["completed"])
        self.assertEqual(action.last_session["stop_reason"], "approval_completed")
        self.assertEqual(len(action.last_session["resumes"]), 1)

    def test_next_resume_failure_is_not_completion_without_approval_hit(self):
        context = _FakeContext()
        original_recognition = context.run_recognition

        def recognition_without_approval(name, image):
            if name == "完成审批check":
                context.recognition_calls.append(name)
                return SimpleNamespace(hit=False)
            return original_recognition(name, image)

        context.run_recognition = recognition_without_approval
        action = _AllPassedScreen()
        argv = SimpleNamespace(
            custom_action_param=json.dumps(
                {"department": "搬砖办", "day": 1}, ensure_ascii=False
            )
        )

        result = action.run(context, argv)

        self.assertFalse(result.success)
        self.assertFalse(action.last_session["completed"])
        self.assertEqual(action.last_session["stop_reason"], "next_resume_failed")

    def test_unknown_requirement_is_still_reported_as_unsupported(self):
        context = _FakeContext()
        action = CVPLSScreen()
        evaluation = action._evaluate_requirements(
            context,
            object(),
            [{"type": "unknown_future_requirement", "text": "未来要求"}],
            {},
        )

        self.assertFalse(evaluation["complete"])
        self.assertEqual(evaluation["unsupported"], ["unknown_future_requirement"])

    def test_waits_for_a_nested_task_that_is_not_done_yet(self):
        completed_detail = SimpleNamespace(
            task_id=7,
            status=SimpleNamespace(done=True, succeeded=True, failed=False),
            nodes=[],
        )

        class Job:
            def __init__(self):
                self.waited = False

            def wait(self):
                self.waited = True
                return self

            def get(self):
                return completed_detail

        job = Job()
        context = _FakeContext()
        context.tasker._gen_task_job = lambda task_id: job if task_id == 7 else None
        running_detail = SimpleNamespace(
            task_id=7,
            status=SimpleNamespace(done=False, running=True, failed=False),
            nodes=[],
        )

        result = _wait_task_detail(context, running_detail)

        self.assertTrue(job.waited)
        self.assertIs(result, completed_detail)


if __name__ == "__main__":
    unittest.main()
