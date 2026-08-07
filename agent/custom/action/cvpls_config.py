import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger

from .cvpls import _should_stop_context, extract_ocr_items, load_cvpls_data


_DAY_PATTERN = re.compile(r"第\s*([一二三1-3])\s*天")
_DAY_VALUES = {"一": 1, "二": 2, "三": 3, "1": 1, "2": 2, "3": 3}
_DEPARTMENT_LABELS = ("项目名称", "招聘项目", "招募项目")


def _ocr_texts(detail: Any) -> list[str]:
    return [item["text"].strip() for item in extract_ocr_items(detail)]


def parse_cvpls_project_day(detail: Any) -> int:
    """从类似“XXXXXX-第一天”的 OCR 结果中提取 1 到 3。"""
    days = set()
    for text in _ocr_texts(detail):
        for match in _DAY_PATTERN.finditer(text):
            days.add(_DAY_VALUES[match.group(1)])
    if len(days) != 1:
        raise ValueError(
            "无法唯一确定项目日期，OCR文本："
            + json.dumps(_ocr_texts(detail), ensure_ascii=False)
        )
    return days.pop()


def _compact_department_text(text: str) -> str:
    compact = re.sub(r"[\s:：\-—_]+", "", text)
    for label in _DEPARTMENT_LABELS:
        compact = compact.replace(label, "")
    return compact


def parse_cvpls_department(
    detail: Any, data: Optional[Dict[str, Any]] = None
) -> str:
    """将项目名称 OCR 结果保守匹配为 cvpls.json 中的正式部门名称。"""
    rules = data if data is not None else load_cvpls_data()
    department_names = [
        str(department.get("name", "")).strip()
        for department in rules.get("departments", {}).values()
        if department.get("name")
    ]
    texts = _ocr_texts(detail)
    candidates = [_compact_department_text(text) for text in texts]

    exact_matches = {
        department
        for candidate in candidates
        for department in department_names
        if department in candidate
    }
    if len(exact_matches) == 1:
        return exact_matches.pop()
    if len(exact_matches) > 1:
        raise ValueError(
            "项目名称包含多个部门："
            + json.dumps(sorted(exact_matches), ensure_ascii=False)
        )

    department_scores = (
        sorted(
            (
                max(
                    SequenceMatcher(None, candidate, department).ratio()
                    for candidate in candidates
                    if candidate
                ),
                department,
            )
            for department in department_names
        )
        if any(candidates)
        else []
    )
    if not department_scores:
        raise ValueError("项目名称 OCR 未返回可用文本")
    best = department_scores[-1]
    second_score = department_scores[-2][0] if len(department_scores) > 1 else 0.0
    if best[0] < 0.75 or best[0] - second_score < 0.1:
        raise ValueError(
            "无法可靠匹配项目名称，OCR文本："
            + json.dumps(texts, ensure_ascii=False)
        )
    return best[1]


def _parse_config_params(raw: Any) -> Dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("custom_action_param must be a JSON object")


@AgentServer.custom_action("CVPLSConfigure")
class CVPLSConfigure(CustomAction):
    """识别当前招聘项目，将部门和日期写入“自动审简历”的调用参数。"""

    def __init__(self):
        super().__init__()
        self.last_result: Optional[Dict[str, Any]] = None

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        self.last_result = None
        if _should_stop_context(context):
            logger.info("[简历筛选配置] 检测到外部结束任务，停止识别项目参数")
            return CustomAction.RunResult(success=False)
        try:
            params = _parse_config_params(argv.custom_action_param)
            target_node = params.get("target_node", "自动审简历")
            screen_params = params.get("screen_params", {})
            if not isinstance(target_node, str) or not target_node.strip():
                raise ValueError("target_node must be a non-empty string")
            if not isinstance(screen_params, dict):
                raise ValueError("screen_params must be a JSON object")

            image = context.tasker.controller.post_screencap().wait().get()
            date_detail = context.run_recognition("提取项目日期", image)
            if _should_stop_context(context):
                logger.info("[简历筛选配置] 识别项目日期时收到外部结束任务")
                return CustomAction.RunResult(success=False)
            name_detail = context.run_recognition("提取项目名称", image)
            if _should_stop_context(context):
                logger.info("[简历筛选配置] 识别项目名称时收到外部结束任务")
                return CustomAction.RunResult(success=False)

            day = parse_cvpls_project_day(date_detail)
            department = parse_cvpls_department(name_detail)
            date_texts = _ocr_texts(date_detail)
            name_texts = _ocr_texts(name_detail)
            final_params = dict(screen_params)
            final_params.update({"department": department, "day": day})
            override = {
                target_node.strip(): {
                    "action": {
                        "type": "Custom",
                        "param": {
                            "custom_action": "CVPLSScreen",
                            "custom_action_param": final_params,
                        },
                    }
                }
            }
            success = bool(context.override_pipeline(override))
            self.last_result = {
                "department": department,
                "day": day,
                "target_node": target_node.strip(),
                "screen_params": final_params,
                "date_ocr_texts": date_texts,
                "name_ocr_texts": name_texts,
                "override": override,
                "success": success,
            }
            if not success:
                logger.error("[简历筛选配置] 覆盖“自动审简历”参数失败")
                return CustomAction.RunResult(success=False)
            logger.info(
                "[简历筛选配置] 已识别并写入筛选参数："
                + json.dumps(
                    {
                        "目标节点": target_node.strip(),
                        "部门": department,
                        "天数": day,
                        "项目名称OCR": name_texts,
                        "项目日期OCR": date_texts,
                        "完整参数": final_params,
                    },
                    ensure_ascii=False,
                )
            )
            return CustomAction.RunResult(success=True)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            logger.error(f"[简历筛选配置] 识别或写入项目参数失败：{error}")
            return CustomAction.RunResult(success=False)
