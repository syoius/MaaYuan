import csv
import json
import re
import time
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
from zhconv import convert

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger

_CVPLS_DATA_PATH = Path(__file__).resolve().parents[2] / "resources" / "cvpls.json"
_REQUIREMENT_PRIORITY = {
    # 已采集字段即可判断。
    "fresh_graduate_only": 10,
    "past_graduate_only": 11,
    "minimum_work_or_internship_segments": 20,
    "minimum_work_or_internship_years": 21,
    "top_tier_school_alumni_only": 30,
    "company_experience_forbidden": 31,
    "former_company_member_only": 32,
    "premium_company_experience_required": 33,
    "matching_role_experience_required": 34,
    "matching_martial_role_experience_required": 35,
    # 需要额外图像识别，统一放到纯数据检查之后。
    "portrait_required": 90,
    "certificate_required_for_fresh_graduate": 100,
    "certificate_required_for_non_guangling_school_graduate": 101,
    "certificate_required_for_everyone": 102,
}
_IGNORED_COMMENT_TEXTS = {"新", "NEW"}
_IGNORED_RESUME_TEXTS = {"大又"}
_FIELD_PATTERN = re.compile(r"^(姓名|毕业学校|状态)\s*[:：]\s*(.*)$")
_DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?\s*年)")
_ROLE_SEPARATOR_PATTERN = re.compile(r"[-－—–]")
_CERTIFICATE_FIELD_PATTERN = re.compile(r"^(姓名|毕业学校|毕业时间)\s*[:：]\s*(.*)$")
_RESUME_PORTRAIT_ROI = (146, 541, 133, 185)
_CERTIFICATE_PORTRAIT_ROI = (475, 717, 101, 140)
_DEFAULT_PORTRAIT_DEBUG_ROOT = (
    Path(__file__).resolve().parents[3] / "debug" / "cvpls_portraits"
)
_PORTRAIT_DEBUG_FIELDS = (
    "序号",
    "简历姓名",
    "证书姓名",
    "简历学校",
    "证书学校",
    "算法",
    "简历特征点数",
    "证书特征点数",
    "候选匹配数",
    "几何内点数",
    "特征覆盖率",
    "几何一致率",
    "相似度",
    "当前阈值",
    "自动判定头像一致",
    "人工标注同一人",
    "简历头像文件",
    "证书头像文件",
)


def normalize_cvpls_text(value: Any) -> str:
    """将 OCR/传入文字统一为简体，供 CVPLS 规则解析与比较使用。"""
    return convert(str(value or ""), "zh-cn")


def load_cvpls_data(path: Optional[Path] = None) -> Dict[str, Any]:
    """读取简历筛选规则数据。"""
    data_path = Path(path) if path is not None else _CVPLS_DATA_PATH
    with data_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data.get("departments"), dict):
        raise ValueError(f"cvpls data has no valid departments: {data_path}")
    return data


def build_requirements(
    department: str, day: int, data: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    生成指定部门在指定天数的有效招聘要求。

    ``day`` 是 1 到 3 的整数。纯简历数据检查优先，需要额外图像识别的
    肖像与持证检查置后；相同优先级保持 cvpls.json 中的原始顺序。
    """
    if not isinstance(department, str) or not department.strip():
        raise ValueError("department must be a non-empty string")
    if isinstance(day, bool) or not isinstance(day, int) or day not in (1, 2, 3):
        raise ValueError("day must be an integer from 1 to 3")

    rules = data if data is not None else load_cvpls_data()
    department_value = normalize_cvpls_text(department).strip()
    department_entry = None
    for key, candidate in rules["departments"].items():
        if department_value in (key, candidate.get("name")):
            department_entry = candidate
            break
    if department_entry is None:
        raise ValueError(f"unknown department: {department_value}")

    base = department_entry.get("base_requirements", [])
    additions = department_entry.get("day_additions", {}).get(str(day), [])
    if not isinstance(base, list) or not isinstance(additions, list):
        raise ValueError(f"invalid requirements for {department_value}, day {day}")

    requirements = deepcopy(base + additions)
    return sorted(
        requirements,
        key=lambda item: _REQUIREMENT_PRIORITY.get(item.get("type", ""), 70),
    )


def _get_value(value: Any, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_box(box: Any) -> Optional[List[int]]:
    if box is None:
        return None
    try:
        values = [int(value) for value in box]
    except (TypeError, ValueError):
        return None
    return values if len(values) == 4 else None


def _result_to_item(result: Any) -> Optional[Dict[str, Any]]:
    raw_text = str(_get_value(result, "text", "") or "").strip()
    text = normalize_cvpls_text(raw_text).strip()
    box = _as_box(_get_value(result, "box"))
    if not text or box is None:
        return None
    # text 用于后续业务判断；raw_text 保留 OCR 原文，方便繁中环境排错。
    item = {"text": text, "raw_text": raw_text, "box": box}
    score = _get_value(result, "score")
    if score is not None:
        try:
            item["score"] = float(score)
        except (TypeError, ValueError):
            pass
    return item


def _first_result_collection(detail: Any) -> Optional[Iterable[Any]]:
    if detail is None:
        return None

    nested_detail = _get_value(detail, "detail")
    if nested_detail is not None:
        nested = _first_result_collection(nested_detail)
        if nested is not None:
            return nested

    for name in ("filtered_results", "filterd_results", "filtered"):
        results = _get_value(detail, name)
        if results:
            return results
    for name in ("all_results", "all"):
        results = _get_value(detail, name)
        if results:
            return results

    raw_detail = _get_value(detail, "raw_detail")
    if raw_detail is not None and raw_detail is not detail:
        nested = _first_result_collection(raw_detail)
        if nested is not None:
            return nested

    best = _get_value(detail, "best_result") or _get_value(detail, "best")
    return [best] if best is not None else None


def extract_ocr_items(detail: Any) -> List[Dict[str, Any]]:
    """兼容 MAA 对象和 OCR 日志字典，提取去重后的 text/box/score。"""
    results = _first_result_collection(detail) or []
    items = []
    seen = set()
    for result in results:
        item = _result_to_item(result)
        if item is None:
            continue
        identity = (item["text"], tuple(item["box"]))
        if identity in seen:
            continue
        seen.add(identity)
        items.append(item)
    return items


def collect_comment_options(
    detail: Any, limit: int = 6, page: int = 1
) -> List[Dict[str, Any]]:
    """过滤 badge 后按 OCR 置信度降序采集一页评语，并记录其页码。"""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if page not in (1, 2):
        raise ValueError("page must be 1 or 2")
    options = []
    for item in extract_ocr_items(detail):
        compact_text = re.sub(r"\s+", "", item["text"])
        if compact_text.upper() in _IGNORED_COMMENT_TEXTS:
            continue
        item["page"] = page
        options.append(item)
    options.sort(
        key=lambda item: (
            item.get("score") is not None,
            item.get("score", float("-inf")),
        ),
        reverse=True,
    )
    return options[:limit]


def _merge_comment_options(*pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并分页评语；同名评语重复出现时优先保留较早页面。"""
    merged = []
    seen_texts = set()
    for options in pages:
        for option in options:
            compact_text = re.sub(r"\s+", "", option["text"])
            if compact_text in seen_texts:
                continue
            seen_texts.add(compact_text)
            merged.append(option)
    return merged


def _box_center_y(item: Dict[str, Any]) -> float:
    return item["box"][1] + item["box"][3] / 2


def _group_items_by_row(
    items: List[Dict[str, Any]], tolerance: int = 20
) -> List[List[Dict[str, Any]]]:
    rows: List[List[Dict[str, Any]]] = []
    row_centers: List[float] = []
    for item in sorted(
        items, key=lambda value: (_box_center_y(value), value["box"][0])
    ):
        center_y = _box_center_y(item)
        if rows and abs(center_y - row_centers[-1]) <= tolerance:
            rows[-1].append(item)
            row_centers[-1] = sum(_box_center_y(value) for value in rows[-1]) / len(
                rows[-1]
            )
        else:
            rows.append([item])
            row_centers.append(center_y)
    for row in rows:
        row.sort(key=lambda value: value["box"][0])
    return rows


def parse_resume_info(detail: Any) -> Dict[str, Any]:
    """从“获取简历信息”的 OCR 结果中提取基本字段与逐行经历。"""
    items = []
    for item in extract_ocr_items(detail):
        if re.sub(r"\s+", "", item["text"]) in _IGNORED_RESUME_TEXTS:
            continue
        items.append(item)

    resume = {
        "name": "",
        "school": "",
        "status": "",
        "experiences": [],
        "experience_count": 0,
    }
    field_names = {"姓名": "name", "毕业学校": "school", "状态": "status"}
    heading_bottom = None
    experience_items = []

    for item in items:
        compact_text = item["text"].strip()
        if re.sub(r"\s+", "", compact_text) == "工作/实习经验":
            heading_bottom = item["box"][1] + item["box"][3]
            continue
        match = _FIELD_PATTERN.match(compact_text)
        if match:
            resume[field_names[match.group(1)]] = match.group(2).strip()
            continue
        experience_items.append(item)

    if heading_bottom is not None:
        experience_items = [
            item for item in experience_items if _box_center_y(item) >= heading_bottom
        ]

    for row in _group_items_by_row(experience_items):
        texts = [item["text"].strip() for item in row]
        row_text = " ".join(text for text in texts if text)
        duration_match = _DURATION_PATTERN.search(row_text)
        duration = re.sub(r"\s+", "", duration_match.group(1)) if duration_match else ""
        role_text = _DURATION_PATTERN.sub("", row_text).strip()
        role_parts = _ROLE_SEPARATOR_PATTERN.split(role_text, maxsplit=1)
        if len(role_parts) != 2 or not all(part.strip() for part in role_parts):
            continue

        company, role = (part.strip() for part in role_parts)
        experience = {
            "company": company,
            "role": role,
            "duration": duration,
            "text": f"{company}-{role}" + (f" {duration}" if duration else ""),
        }
        if duration:
            experience["duration_years"] = float(duration[:-1])
        resume["experiences"].append(experience)

    resume["experience_count"] = len(resume["experiences"])
    return resume


def parse_certificate_info(detail: Any) -> Dict[str, Any]:
    """从“获取证书信息”的 OCR 结果中提取姓名、毕业学校和毕业时间。"""
    certificate = {
        "name": "",
        "school": "",
        "graduation_time": "",
        "graduation_year": None,
    }
    field_names = {
        "姓名": "name",
        "毕业学校": "school",
        "毕业时间": "graduation_time",
    }
    for item in extract_ocr_items(detail):
        match = _CERTIFICATE_FIELD_PATTERN.match(item["text"].strip())
        if match:
            certificate[field_names[match.group(1)]] = match.group(2).strip()

    year_match = re.search(r"(\d+)\s*年", certificate["graduation_time"])
    if year_match:
        certificate["graduation_year"] = int(year_match.group(1))
    return certificate


def _crop_image(image: Any, roi: Iterable[int]) -> Optional[np.ndarray]:
    if image is None or not hasattr(image, "shape"):
        return None
    x, y, width, height = (int(value) for value in roi)
    image_height, image_width = image.shape[:2]
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        return None
    return image[y : y + height, x : x + width].copy()


def _portrait_similarity_detail(
    first: np.ndarray, second: np.ndarray
) -> Dict[str, Any]:
    """使用 ORB 特征与 RANSAC 几何校验比较两张头像。"""
    empty_result = {
        "algorithm": "ORB_RANSAC_V1",
        "first_keypoints": 0,
        "second_keypoints": 0,
        "candidate_matches": 0,
        "inlier_matches": 0,
        "feature_coverage": 0.0,
        "geometric_consistency": 0.0,
        "score": 0.0,
    }
    if first is None or second is None or first.size == 0 or second.size == 0:
        return empty_result

    def to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 小头像需要较低 FAST 阈值和较小边缘留白。ORB 会寻找发饰、五官、
    # 衣纹等局部特征，不会让大面积相同的浅色背景主导结果。
    detector = cv2.ORB_create(
        nfeatures=500,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=8,
        patchSize=15,
        fastThreshold=5,
    )
    first_keypoints, first_descriptors = detector.detectAndCompute(
        to_gray(first), None
    )
    second_keypoints, second_descriptors = detector.detectAndCompute(
        to_gray(second), None
    )
    result = dict(empty_result)
    result["first_keypoints"] = len(first_keypoints)
    result["second_keypoints"] = len(second_keypoints)
    if (
        first_descriptors is None
        or second_descriptors is None
        or min(len(first_keypoints), len(second_keypoints)) < 4
    ):
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    nearby_matches = matcher.knnMatch(
        first_descriptors, second_descriptors, k=2
    )
    candidate_matches = [
        best
        for match_group in nearby_matches
        if len(match_group) >= 2
        for best, second_best in [match_group[:2]]
        if best.distance < 0.75 * second_best.distance
    ]
    result["candidate_matches"] = len(candidate_matches)
    if len(candidate_matches) < 4:
        return result

    first_points = np.float32(
        [first_keypoints[match.queryIdx].pt for match in candidate_matches]
    ).reshape(-1, 1, 2)
    second_points = np.float32(
        [second_keypoints[match.trainIdx].pt for match in candidate_matches]
    ).reshape(-1, 1, 2)
    try:
        _, inlier_mask = cv2.findHomography(
            first_points,
            second_points,
            cv2.RANSAC,
            4.0,
        )
    except cv2.error:
        inlier_mask = None
    inlier_matches = int(inlier_mask.sum()) if inlier_mask is not None else 0
    keypoint_base = min(len(first_keypoints), len(second_keypoints))
    feature_coverage = inlier_matches / keypoint_base
    geometric_consistency = inlier_matches / len(candidate_matches)

    # 现有同一人样本的覆盖率约为 0.40～0.48。先将 0.40 映射为满分，
    # 再用几何一致率抑制背景纹样等偶然匹配；最终分数仍为 0～1，继续兼容
    # portrait_similarity_threshold 的现有含义与默认值 0.65。
    coverage_score = min(1.0, feature_coverage / 0.40)
    geometry_score = max(
        0.0, min(1.0, (geometric_consistency - 0.35) / 0.65)
    )
    score = coverage_score * geometry_score
    result.update(
        {
            "inlier_matches": inlier_matches,
            "feature_coverage": feature_coverage,
            "geometric_consistency": geometric_consistency,
            "score": max(0.0, min(1.0, float(score))),
        }
    )
    return result


def _portrait_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """返回 ORB/RANSAC 头像相似度，范围为 0 到 1。"""
    return float(_portrait_similarity_detail(first, second)["score"])


def _write_png(path: Path, image: np.ndarray) -> None:
    """通过内存编码保存 PNG，兼容包含中文的 Windows 路径。"""
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"无法编码头像图片：{path}")
    path.write_bytes(buffer.tobytes())


def _write_portrait_debug_sample(
    directory: Path,
    resume_index: int,
    resume: Dict[str, Any],
    certificate: Dict[str, Any],
    resume_portrait: np.ndarray,
    certificate_portrait: np.ndarray,
    similarity: float,
    threshold: float,
    similarity_detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """保存一组头像样本，并向可人工标注的 CSV 追加一行。"""
    directory.mkdir(parents=True, exist_ok=True)
    sample_name = f"{resume_index:03d}"
    resume_path = directory / f"{sample_name}_简历头像.png"
    certificate_path = directory / f"{sample_name}_证书头像.png"
    record_path = directory / "头像对比记录.csv"
    _write_png(resume_path, resume_portrait)
    _write_png(certificate_path, certificate_portrait)

    detail = similarity_detail or {}
    should_write_header = not record_path.exists() or record_path.stat().st_size == 0
    with record_path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_PORTRAIT_DEBUG_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(
            {
                "序号": resume_index,
                "简历姓名": resume.get("name", ""),
                "证书姓名": certificate.get("name", ""),
                "简历学校": resume.get("school", ""),
                "证书学校": certificate.get("school", ""),
                "算法": detail.get("algorithm", ""),
                "简历特征点数": detail.get("first_keypoints", ""),
                "证书特征点数": detail.get("second_keypoints", ""),
                "候选匹配数": detail.get("candidate_matches", ""),
                "几何内点数": detail.get("inlier_matches", ""),
                "特征覆盖率": (
                    f'{detail["feature_coverage"]:.6f}'
                    if "feature_coverage" in detail
                    else ""
                ),
                "几何一致率": (
                    f'{detail["geometric_consistency"]:.6f}'
                    if "geometric_consistency" in detail
                    else ""
                ),
                "相似度": f"{similarity:.6f}",
                "当前阈值": f"{threshold:.6f}",
                "自动判定头像一致": "是" if similarity >= threshold else "否",
                "人工标注同一人": "",
                "简历头像文件": resume_path.name,
                "证书头像文件": certificate_path.name,
            }
        )
    return {
        "简历头像": str(resume_path),
        "证书头像": str(certificate_path),
        "对比记录": str(record_path),
    }


def _recognition_box(detail: Any) -> Optional[List[int]]:
    items = extract_ocr_items(detail)
    if items:
        return items[0]["box"]
    return _as_box(_get_value(detail, "box"))


def override_drag_target(
    context: Context, box: List[int], node_name: str = "拖评语-action"
) -> bool:
    """供后续核验逻辑按评语 box 替换拖拽动作起点。"""
    target = _as_box(box)
    if target is None:
        raise ValueError("comment box must contain four integers")
    return bool(
        context.override_pipeline(
            {node_name: {"action": {"type": "TouchDown", "param": {"target": target}}}}
        )
    )


def _parse_action_params(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("custom_action_param must be a JSON object")


def _task_failed(detail: Any) -> bool:
    """run_task 未返回结果或明确失败时均视为失败。"""
    if detail is None:
        return True
    status = getattr(detail, "status", None)
    if status is None:
        return True
    done = getattr(status, "done", None)
    if done is False:
        return True
    if done is True:
        return not bool(getattr(status, "succeeded", False))
    return bool(getattr(status, "failed", True))


def _wait_task_detail(context: Context, detail: Any) -> Any:
    """确保 Context.run_task 返回的任务已经真正结束。"""
    if detail is None or _should_stop_context(context):
        return detail
    status = getattr(detail, "status", None)
    if status is not None and bool(getattr(status, "done", False)):
        return detail

    generate_job = getattr(getattr(context, "tasker", None), "_gen_task_job", None)
    if callable(generate_job):
        try:
            return generate_job(detail.task_id).wait().get()
        except Exception:
            logger.exception("[简历筛选] 等待流水线任务完成时发生异常")
    return detail


def _run_task_wait(
    context: Context, name: str, pipeline_override: Optional[Dict[str, Any]] = None
) -> Any:
    """运行流水线任务，并等待动作、延迟和后续节点全部完成。"""
    if _should_stop_context(context):
        return None
    detail = context.run_task(name, pipeline_override or {})
    return _wait_task_detail(context, detail)


def _task_debug_summary(detail: Any) -> Dict[str, Any]:
    if detail is None:
        return {"任务状态": "未返回任务结果", "已执行节点": []}
    status = getattr(detail, "status", None)
    if status is not None and bool(getattr(status, "succeeded", False)):
        status_text = "成功"
    elif status is not None and bool(getattr(status, "failed", False)):
        status_text = "失败"
    elif status is not None and bool(getattr(status, "running", False)):
        status_text = "仍在运行"
    else:
        status_text = "状态未知"
    return {
        "任务状态": status_text,
        "已执行节点": [
            {
                "节点": getattr(node, "name", ""),
                "已完成": "是" if getattr(node, "completed", False) else "否",
            }
            for node in (getattr(detail, "nodes", None) or [])
        ],
    }


def _should_stop_context(context: Context) -> bool:
    """协作式检测外部 StopTask/Tasker.stop()，与 AutoFormation 保持一致。"""
    try:
        if bool(getattr(context, "stop", False)):
            return True
        tasker = getattr(context, "tasker", None)
        if tasker is not None:
            if bool(getattr(tasker, "stopping", False)):
                return True
            if getattr(tasker, "running", None) is False:
                return True
    except Exception:
        return False
    return False


def _recognition_hit(detail: Any) -> bool:
    return bool(detail and getattr(detail, "hit", False))


def _find_comment_option(
    options: List[Dict[str, Any]], expected_text: str
) -> Optional[Dict[str, Any]]:
    compact_expected = re.sub(r"\s+", "", expected_text)
    for option in options:
        if re.sub(r"\s+", "", option["text"]) == compact_expected:
            return option

    # OCR 偶尔会把“聘”识别成“明”等近形字；仅在候选足够接近时回退。
    matches = [
        (
            SequenceMatcher(
                None, compact_expected, re.sub(r"\s+", "", option["text"])
            ).ratio(),
            option,
        )
        for option in options
    ]
    best = max(matches, key=lambda match: match[0]) if matches else None
    return best[1] if best and best[0] >= 0.72 else None


@AgentServer.custom_action("CVPLSScreen")
class CVPLSScreen(CustomAction):
    """
    在一次调用中控制当天、当前部门的整轮简历筛选。

    参数示例：``{"department": "搬砖办", "day": 1}``。
    评语位置只在开始时采集一次；简历信息则在每位候选人处重新采集。

    cvpls.json 中的招聘要求均已实装。若以后新增未知 requirement，会在拖评语
    和处置简历前安全停止，避免误判整批简历。
    """

    def __init__(self):
        super().__init__()
        self.last_session: Optional[Dict[str, Any]] = None
        self._cvpls_data: Optional[Dict[str, Any]] = None
        self._comment_page = 1
        self._comment_page_delay = 500
        self._portrait_debug_dir: Optional[Path] = None
        self._certificate_checked = False
        self._certificate_detail: Any = None

    def _get_certificate_detail(self, context: Context, image: Any) -> Any:
        """同一份简历中的持证识别只运行一次。"""
        if not self._certificate_checked:
            self._certificate_detail = context.run_recognition("持证check", image)
            self._certificate_checked = True
        return self._certificate_detail

    def _company_roles(self) -> Dict[str, set]:
        rules = self._cvpls_data or load_cvpls_data()
        return {
            re.sub(r"\s+", "", str(company.get("name", ""))): {
                re.sub(r"\s+", "", str(role.get("name", "")))
                for role in company.get("roles", [])
                if role.get("name")
            }
            for company in rules.get("companies", {}).values()
            if company.get("name")
        }

    def _matching_role_experiences(
        self, resume: Dict[str, Any], required_role: str
    ) -> List[Dict[str, str]]:
        company_roles = self._company_roles()
        expected_role = re.sub(r"\s+", "", required_role)
        matches = []
        for experience in resume.get("experiences", []):
            company_name = re.sub(r"\s+", "", str(experience.get("company", "")))
            role_name = re.sub(r"\s+", "", str(experience.get("role", "")))
            if role_name == expected_role and role_name in company_roles.get(
                company_name, set()
            ):
                matches.append(
                    {
                        "单位": experience.get("company", ""),
                        "职位": experience.get("role", ""),
                    }
                )
        return matches

    def _requirement_company_name(self, requirement: Dict[str, Any]) -> str:
        params = requirement.get("params", {})
        company_id = str(params.get("company_id", "") or "").strip()
        rules = self._cvpls_data or load_cvpls_data()
        company = rules.get("companies", {}).get(company_id)
        if not isinstance(company, dict) or not company.get("name"):
            raise ValueError(f"unknown requirement company_id: {company_id}")
        return str(company["name"])

    def _match_known_school(self, school_text: str) -> Optional[Dict[str, Any]]:
        """对简历学校做保守模糊匹配；证书学校不使用此逻辑。"""
        compact_school = re.sub(r"\s+", "", str(school_text or ""))
        if not compact_school:
            return None
        rules = self._cvpls_data or load_cvpls_data()
        candidates = []
        for school_id, school in rules.get("schools", {}).items():
            school_name = re.sub(r"\s+", "", str(school.get("name", "")))
            if not school_name:
                continue
            score = SequenceMatcher(None, compact_school, school_name).ratio()
            candidates.append((score, school_id, school, school_name))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return None
        best = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        if best[0] < 0.75 or (best[0] < 1.0 and best[0] - second_score < 0.1):
            return None
        return {
            "id": best[1],
            "name": best[2].get("name", best[3]),
            "is_top_tier": bool(best[2].get("is_top_tier")),
            "score": best[0],
        }

    def _has_premium_company_experience(self, resume: Dict[str, Any]) -> bool:
        rules = self._cvpls_data or load_cvpls_data()
        premium_companies = {
            re.sub(r"\s+", "", str(company.get("name", "")))
            for company in rules.get("companies", {}).values()
            if company.get("is_premium") and company.get("name")
        }
        return any(
            re.sub(r"\s+", "", str(experience.get("company", ""))) in premium_companies
            for experience in resume.get("experiences", [])
        )

    def _matching_martial_experiences(
        self, resume: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        rules = self._cvpls_data or load_cvpls_data()
        martial_roles = set()
        for company in rules.get("companies", {}).values():
            company_name = re.sub(r"\s+", "", str(company.get("name", "")))
            if not company_name:
                continue
            for role in company.get("roles", []):
                role_name = re.sub(r"\s+", "", str(role.get("name", "")))
                if role.get("is_martial") and role_name:
                    martial_roles.add((company_name, role_name))

        matches = []
        for experience in resume.get("experiences", []):
            company_name = re.sub(r"\s+", "", str(experience.get("company", "")))
            role_name = re.sub(r"\s+", "", str(experience.get("role", "")))
            if (company_name, role_name) in martial_roles:
                matches.append(
                    {
                        "单位": experience.get("company", ""),
                        "职位": experience.get("role", ""),
                    }
                )
        return matches

    def _invalid_company_experiences(
        self, resume: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """找出公司不存在，或岗位不属于对应公司的工作/实习经历。"""
        company_roles = self._company_roles()

        invalid = []
        for experience in resume.get("experiences", []):
            company_name = re.sub(r"\s+", "", str(experience.get("company", "")))
            role_name = re.sub(r"\s+", "", str(experience.get("role", "")))
            if company_name not in company_roles:
                reason = "公司不存在"
            elif role_name not in company_roles[company_name]:
                reason = "岗位不属于该公司"
            else:
                continue
            invalid.append(
                {
                    "单位": experience.get("company", ""),
                    "职位": experience.get("role", ""),
                    "原因": reason,
                }
            )
        return invalid

    def _wait_animation(self, context: Context, delay_ms: int) -> bool:
        deadline = time.monotonic() + delay_ms / 1000.0
        while time.monotonic() < deadline:
            if _should_stop_context(context):
                return False
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return not _should_stop_context(context)

    def _evaluate_resume_second_round_data(
        self, resume: Dict[str, Any]
    ) -> Dict[str, Any]:
        """在展开证书前，先执行只依赖已采集简历信息的二轮检查。"""
        invalid_company_experiences = self._invalid_company_experiences(resume)
        checks = [
            {
                "type": "forged_company",
                "text": "公司作假",
                "passed": not invalid_company_experiences,
                "evidence": {
                    "异常工作/实习经历": invalid_company_experiences,
                },
            },
            {
                "type": "employed_without_experience",
                "text": "在职却无工作经历",
                "passed": not (
                    re.sub(r"\s+", "", str(resume.get("status", ""))) == "在职员工"
                    and int(resume.get("experience_count", 0)) == 0
                ),
                "evidence": {
                    "简历状态": resume.get("status", ""),
                    "经历数量": resume.get("experience_count", 0),
                },
            },
        ]
        for check in checks:
            if not check["passed"]:
                return {"complete": True, "results": [check]}
        return {"complete": True, "results": checks}

    def _evaluate_certificate_data(
        self,
        resume: Dict[str, Any],
        certificate: Dict[str, Any],
        portrait_similarity: float,
        portrait_threshold: float,
        recruitment_year: int,
    ) -> Dict[str, Any]:
        rules = self._cvpls_data or load_cvpls_data()
        valid_schools = {
            re.sub(r"\s+", "", str(school.get("name", "")))
            for school in rules.get("schools", {}).values()
            if school.get("name")
        }
        resume_name = re.sub(r"\s+", "", str(resume.get("name", "")))
        certificate_name = re.sub(r"\s+", "", str(certificate.get("name", "")))
        resume_school = re.sub(r"\s+", "", str(resume.get("school", "")))
        certificate_school = re.sub(r"\s+", "", str(certificate.get("school", "")))
        certificate_school_is_valid = certificate_school in valid_schools
        mismatch_items = {}
        if resume_name != certificate_name:
            mismatch_items["姓名"] = {
                "简历": resume.get("name", ""),
                "毕业证": certificate.get("name", ""),
            }
        if resume_school != certificate_school:
            mismatch_items["毕业学校"] = {
                "简历": resume.get("school", ""),
                "毕业证": certificate.get("school", ""),
            }
        if portrait_similarity < portrait_threshold:
            mismatch_items["头像"] = {
                "相似度": round(portrait_similarity, 4),
                "最低阈值": portrait_threshold,
            }

        checks = [
            {
                "type": "forged_school",
                "text": "学府作假",
                "passed": certificate_school_is_valid,
                "evidence": {
                    "证书学校": certificate.get("school", ""),
                    "是否为已知学校": "是" if certificate_school_is_valid else "否",
                },
            },
            {
                "type": "fresh_graduate_year_too_early",
                "text": "应届毕业年早于当年",
                "passed": not (
                    "应届" in str(resume.get("status", ""))
                    and certificate.get("graduation_year") != recruitment_year
                ),
                "evidence": {
                    "简历状态": resume.get("status", ""),
                    "证书毕业时间": certificate.get("graduation_time", ""),
                    "当前招聘年份": f"招聘{recruitment_year}年",
                },
            },
            {
                "type": "resume_certificate_mismatch",
                "text": "简历与毕业证信息不一致",
                "passed": not mismatch_items,
                "evidence": {"不一致项目": mismatch_items},
            },
        ]
        for check in checks:
            if not check["passed"]:
                return {"complete": True, "results": [check]}
        return {"complete": True, "results": checks}

    def _screen_certificate(
        self,
        context: Context,
        resume_image: Any,
        resume: Dict[str, Any],
        certificate_detail: Any,
        expand_delay_ms: int,
        portrait_threshold: float,
        recruitment_year: int,
        resume_index: int = 0,
    ) -> Dict[str, Any]:
        resume_portrait = _crop_image(resume_image, _RESUME_PORTRAIT_ROI)
        view_box = _recognition_box(certificate_detail)
        if resume_portrait is None or view_box is None:
            return {
                "complete": False,
                "stopped": False,
                "error": "无法截取简历头像或获取证书查看按钮位置",
                "results": [],
            }

        center_x = view_box[0] + view_box[2] // 2
        center_y = view_box[1] + view_box[3] // 2
        context.tasker.controller.post_click(center_x, center_y).wait()
        if not self._wait_animation(context, expand_delay_ms):
            return {
                "complete": False,
                "stopped": True,
                "error": "等待证书展开时收到外部停止",
                "results": [],
            }

        expanded_image = context.tasker.controller.post_screencap().wait().get()
        certificate_ocr = context.run_recognition("获取证书信息", expanded_image)
        if _should_stop_context(context):
            return {
                "complete": False,
                "stopped": True,
                "error": "采集证书信息时收到外部停止",
                "results": [],
            }
        certificate = parse_certificate_info(certificate_ocr)
        missing = [
            field
            for field in ("name", "school", "graduation_time")
            if not certificate.get(field)
        ]
        certificate_portrait = _crop_image(expanded_image, _CERTIFICATE_PORTRAIT_ROI)
        if missing or certificate_portrait is None:
            return {
                "complete": False,
                "stopped": False,
                "error": "证书信息缺失：" + "、".join(missing),
                "certificate": certificate,
                "results": [],
            }

        similarity_detail = _portrait_similarity_detail(
            resume_portrait, certificate_portrait
        )
        similarity = float(similarity_detail["score"])
        debug_files = None
        if self._portrait_debug_dir is not None:
            try:
                debug_files = _write_portrait_debug_sample(
                    self._portrait_debug_dir,
                    resume_index,
                    resume,
                    certificate,
                    resume_portrait,
                    certificate_portrait,
                    similarity,
                    portrait_threshold,
                    similarity_detail,
                )
                logger.info(
                    "[简历筛选] 已保存第 "
                    f"{resume_index} 份头像对比样本："
                    + json.dumps(debug_files, ensure_ascii=False)
                )
            except Exception:
                # 调试落盘不应影响正常筛选与处置。
                logger.exception(f"[简历筛选] 保存第 {resume_index} 份头像对比样本失败")
        evaluation = self._evaluate_certificate_data(
            resume,
            certificate,
            similarity,
            portrait_threshold,
            recruitment_year,
        )
        evaluation.update(
            {
                "stopped": False,
                "certificate": certificate,
                "portrait_similarity": similarity,
                "portrait_similarity_detail": similarity_detail,
                "portrait_debug_files": debug_files,
            }
        )
        return evaluation

    def _evaluate_requirements(
        self,
        context: Context,
        image: Any,
        requirements: List[Dict[str, Any]],
        resume: Dict[str, Any],
    ) -> Dict[str, Any]:
        results = []
        unsupported = []
        for requirement in requirements:
            if _should_stop_context(context):
                return {
                    "complete": False,
                    "stopped": True,
                    "results": results,
                    "unsupported": unsupported,
                }
            requirement_type = requirement.get("type", "")
            evidence: Dict[str, Any] = {}
            if requirement_type == "portrait_required":
                no_portrait = context.run_recognition("无肖像check", image)
                no_portrait_hit = _recognition_hit(no_portrait)
                passed = not no_portrait_hit
                evidence = {
                    "识别节点": "无肖像check",
                    "识别到无肖像": "是" if no_portrait_hit else "否",
                }
            elif requirement_type == "fresh_graduate_only":
                status = re.sub(r"\s+", "", str(resume.get("status", "") or ""))
                passed = "应届" in status
                evidence = {"简历状态": status}
            elif requirement_type == "certificate_required_for_fresh_graduate":
                status = re.sub(r"\s+", "", str(resume.get("status", "") or ""))
                if "应届" not in status:
                    passed = True
                    evidence = {
                        "简历状态": status,
                        "证书检查": "非应届毕业生，无需检查",
                    }
                else:
                    certificate = self._get_certificate_detail(context, image)
                    certificate_hit = _recognition_hit(certificate)
                    passed = certificate_hit
                    evidence = {
                        "简历状态": status,
                        "识别节点": "持证check",
                        "识别到证书": "是" if certificate_hit else "否",
                    }
            elif requirement_type == "past_graduate_only":
                status = re.sub(r"\s+", "", str(resume.get("status", "") or ""))
                passed = status == "在职员工"
                evidence = {"简历状态": status}
            elif requirement_type == "certificate_required_for_everyone":
                certificate = self._get_certificate_detail(context, image)
                certificate_hit = _recognition_hit(certificate)
                passed = certificate_hit
                evidence = {
                    "识别节点": "持证check",
                    "识别到证书": "是" if certificate_hit else "否",
                }
            elif (
                requirement_type
                == "certificate_required_for_non_guangling_school_graduate"
            ):
                school = re.sub(r"\s+", "", str(resume.get("school", "") or ""))
                if "广陵" in school:
                    passed = True
                    evidence = {
                        "毕业学校": school,
                        "证书检查": "广陵学校毕业，无需检查",
                    }
                else:
                    certificate = self._get_certificate_detail(context, image)
                    certificate_hit = _recognition_hit(certificate)
                    passed = certificate_hit
                    evidence = {
                        "毕业学校": school,
                        "识别节点": "持证check",
                        "识别到证书": "是" if certificate_hit else "否",
                    }
            elif requirement_type == "premium_company_experience_required":
                passed = self._has_premium_company_experience(resume)
                evidence = {
                    "工作/实习单位": [
                        experience.get("company", "")
                        for experience in resume.get("experiences", [])
                    ]
                }
            elif requirement_type == "matching_martial_role_experience_required":
                martial_matches = self._matching_martial_experiences(resume)
                passed = bool(martial_matches)
                evidence = {
                    "工作/实习经历": [
                        {
                            "单位": experience.get("company", ""),
                            "职位": experience.get("role", ""),
                        }
                        for experience in resume.get("experiences", [])
                    ],
                    "命中的武力类经历": martial_matches,
                }
            elif requirement_type == "minimum_work_or_internship_segments":
                experiences = resume.get("experiences", [])
                experience_count = resume.get("experience_count", 0)
                if experience_count != len(experiences):
                    return {
                        "complete": False,
                        "stopped": False,
                        "results": results,
                        "unsupported": [],
                        "data_error": (
                            "工作/实习经历计数与解析出的经历数量不一致："
                            f"{experience_count} != {len(experiences)}"
                        ),
                    }
                minimum = requirement.get("params", {}).get("minimum")
                if (
                    isinstance(minimum, bool)
                    or not isinstance(minimum, int)
                    or minimum < 0
                ):
                    raise ValueError("segment minimum must be a non-negative integer")
                passed = experience_count >= minimum
                evidence = {
                    "经历数量": experience_count,
                    "最低要求": minimum,
                }
            elif requirement_type == "matching_role_experience_required":
                required_role = str(
                    requirement.get("params", {}).get("role", "") or ""
                ).strip()
                if not required_role:
                    raise ValueError("matching role requirement has no role")
                role_matches = self._matching_role_experiences(resume, required_role)
                passed = bool(role_matches)
                evidence = {
                    "要求岗位": required_role,
                    "工作/实习经历": [
                        {
                            "单位": experience.get("company", ""),
                            "职位": experience.get("role", ""),
                        }
                        for experience in resume.get("experiences", [])
                    ],
                    "命中的合法岗位经历": role_matches,
                }
            elif requirement_type == "former_company_member_only":
                target_company = self._requirement_company_name(requirement)
                companies = [
                    str(experience.get("company", ""))
                    for experience in resume.get("experiences", [])
                ]
                compact_target = re.sub(r"\s+", "", target_company)
                passed = any(
                    re.sub(r"\s+", "", company) == compact_target
                    for company in companies
                )
                evidence = {
                    "要求单位": target_company,
                    "工作/实习单位": companies,
                }
            elif requirement_type == "company_experience_forbidden":
                target_company = self._requirement_company_name(requirement)
                companies = [
                    str(experience.get("company", ""))
                    for experience in resume.get("experiences", [])
                ]
                compact_target = re.sub(r"\s+", "", target_company)
                forbidden_matches = [
                    company
                    for company in companies
                    if re.sub(r"\s+", "", company) == compact_target
                ]
                passed = not forbidden_matches
                evidence = {
                    "禁止单位": target_company,
                    "工作/实习单位": companies,
                    "命中的禁止单位": forbidden_matches,
                }
            elif requirement_type == "top_tier_school_alumni_only":
                school = str(resume.get("school", "") or "")
                school_match = self._match_known_school(school)
                passed = bool(school_match and school_match["is_top_tier"])
                evidence = {
                    "简历毕业学校": school,
                    "匹配学校": school_match["name"] if school_match else "未匹配",
                    "匹配相似度": (
                        round(school_match["score"], 4) if school_match else None
                    ),
                    "是否顶尖学府": (
                        "是" if school_match and school_match["is_top_tier"] else "否"
                    ),
                }
            elif requirement_type == "minimum_work_or_internship_years":
                experiences = resume.get("experiences", [])
                experience_count = resume.get("experience_count", 0)
                if experience_count != len(experiences):
                    return {
                        "complete": False,
                        "stopped": False,
                        "results": results,
                        "unsupported": [],
                        "data_error": (
                            "工作/实习经历计数与解析出的经历数量不一致："
                            f"{experience_count} != {len(experiences)}"
                        ),
                    }
                missing_durations = [
                    {
                        "单位": experience.get("company", ""),
                        "职位": experience.get("role", ""),
                    }
                    for experience in experiences
                    if experience.get("duration_years") is None
                ]
                if missing_durations:
                    return {
                        "complete": False,
                        "stopped": False,
                        "results": results,
                        "unsupported": [],
                        "data_error": (
                            "以下工作/实习经历缺少时长："
                            + json.dumps(missing_durations, ensure_ascii=False)
                        ),
                    }
                minimum = requirement.get("params", {}).get("minimum")
                if (
                    isinstance(minimum, bool)
                    or not isinstance(minimum, (int, float))
                    or minimum < 0
                ):
                    raise ValueError("year minimum must be a non-negative number")
                total_years = sum(
                    float(experience["duration_years"]) for experience in experiences
                )
                passed = total_years >= float(minimum)
                evidence = {
                    "累计工作/实习年限": total_years,
                    "最低要求": minimum,
                    "逐段年限": [
                        {
                            "单位": experience.get("company", ""),
                            "职位": experience.get("role", ""),
                            "年限": experience.get("duration_years"),
                        }
                        for experience in experiences
                    ],
                }
            else:
                unsupported.append(requirement_type)
                continue
            if _should_stop_context(context):
                return {
                    "complete": False,
                    "stopped": True,
                    "results": results,
                    "unsupported": unsupported,
                }
            results.append(
                {
                    "type": requirement_type,
                    "text": requirement.get("text", ""),
                    "passed": passed,
                    "evidence": evidence,
                }
            )
            # 任一招聘要求不通过即可判定整份简历不合格；后续项目不再检查，
            # 也只会拖入这一条失败评语。
            if not passed:
                return {
                    "complete": True,
                    "stopped": False,
                    "results": results,
                    "unsupported": [],
                }
        return {
            "complete": not unsupported,
            "stopped": False,
            "results": results,
            "unsupported": unsupported,
        }

    def _drag_comment(
        self,
        context: Context,
        options: List[Dict[str, Any]],
        text: str,
    ) -> bool:
        if _should_stop_context(context):
            return False
        option = _find_comment_option(options, text)
        if option is None:
            logger.error(f"[简历筛选] 未找到评语：{text}")
            return False
        target_page = int(option.get("page", 1))
        if not self._switch_comment_page(context, target_page):
            return False
        # logger.info(
        #     "[简历筛选] 正在拖入评语："
        #     + json.dumps(
        #         {
        #             "评语": option["text"],
        #             "页码": target_page,
        #             "位置": option["box"],
        #         },
        #         ensure_ascii=False,
        #     )
        # )
        if not override_drag_target(context, option["box"]):
            logger.error(f"[简历筛选] 替换拖拽起点失败：{text}")
            return False
        detail = _run_task_wait(context, "拖评语-action")
        if _should_stop_context(context):
            return False
        if _task_failed(detail):
            logger.error(f"[简历筛选] 拖入评语失败：{text}")
            return False
        if self._comment_page != 1 and not self._switch_comment_page(context, 1):
            return False
        return True

    def _switch_comment_page(self, context: Context, target_page: int) -> bool:
        if target_page not in (1, 2):
            logger.error(f"[简历筛选] 无效的评语页码：{target_page}")
            return False
        if self._comment_page == target_page:
            return True
        task_name = "评语翻页-去第二页" if target_page == 2 else "评语翻页-回第一页"
        # logger.info(f"[简历筛选] 评语翻页：第 {self._comment_page} 页 → 第 {target_page} 页")
        detail = _run_task_wait(
            context,
            task_name,
            {task_name: {"post_delay": self._comment_page_delay}},
        )
        if _should_stop_context(context):
            return False
        if _task_failed(detail):
            logger.error(f"[简历筛选] 评语翻页失败：{task_name}")
            return False
        self._comment_page = target_page
        return True

    def _fail_session(self, reason: str) -> CustomAction.RunResult:
        if self.last_session is not None:
            self.last_session["stop_reason"] = reason
            self.last_session["completed"] = False
        return CustomAction.RunResult(success=False)

    def _external_stop_result(self, stage: str) -> CustomAction.RunResult:
        logger.info(f"[简历筛选] 检测到外部结束任务，停止筛选；当前阶段：{stage}")
        return self._fail_session("external_stop")

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        self.last_session = None
        self._cvpls_data = None
        self._comment_page = 1
        self._comment_page_delay = 500
        self._portrait_debug_dir = None
        self._certificate_checked = False
        self._certificate_detail = None
        if _should_stop_context(context):
            return self._external_stop_result("启动")
        try:
            params = _parse_action_params(argv.custom_action_param)
            department = params.get("department")
            day = params.get("day")
            self._cvpls_data = load_cvpls_data()
            requirements = build_requirements(department, day, self._cvpls_data)
            max_resumes = params.get("max_resumes", 100)
            if (
                isinstance(max_resumes, bool)
                or not isinstance(max_resumes, int)
                or not 1 <= max_resumes <= 1000
            ):
                raise ValueError("max_resumes must be an integer from 1 to 1000")
            disposition_delay = params.get("disposition_post_delay", 1000)
            if (
                isinstance(disposition_delay, bool)
                or not isinstance(disposition_delay, int)
                or not 0 <= disposition_delay <= 10000
            ):
                raise ValueError(
                    "disposition_post_delay must be an integer from 0 to 10000"
                )
            certificate_expand_delay = params.get("certificate_expand_delay", 1200)
            if (
                isinstance(certificate_expand_delay, bool)
                or not isinstance(certificate_expand_delay, int)
                or not 0 <= certificate_expand_delay <= 10000
            ):
                raise ValueError(
                    "certificate_expand_delay must be an integer from 0 to 10000"
                )
            portrait_threshold = params.get("portrait_similarity_threshold", 0.65)
            if (
                isinstance(portrait_threshold, bool)
                or not isinstance(portrait_threshold, (int, float))
                or not 0 <= float(portrait_threshold) <= 1
            ):
                raise ValueError(
                    "portrait_similarity_threshold must be a number from 0 to 1"
                )
            portrait_threshold = float(portrait_threshold)
            portrait_debug = params.get("portrait_debug", False)
            if not isinstance(portrait_debug, bool):
                raise ValueError("portrait_debug must be a boolean")
            portrait_debug_root = params.get("portrait_debug_dir")
            if portrait_debug_root is not None and (
                not isinstance(portrait_debug_root, str)
                or not portrait_debug_root.strip()
            ):
                raise ValueError("portrait_debug_dir must be a non-empty string")
            if portrait_debug:
                debug_root = (
                    Path(portrait_debug_root).expanduser()
                    if portrait_debug_root
                    else _DEFAULT_PORTRAIT_DEBUG_ROOT
                )
                if not debug_root.is_absolute():
                    debug_root = Path.cwd() / debug_root
                run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                self._portrait_debug_dir = debug_root / run_name
                try:
                    self._portrait_debug_dir.mkdir(parents=True, exist_ok=False)
                except Exception:
                    logger.exception("[简历筛选] 创建头像对比调试目录失败，已关闭保存")
                    self._portrait_debug_dir = None
            recruitment_year = params.get("recruitment_year", 10)
            if (
                isinstance(recruitment_year, bool)
                or not isinstance(recruitment_year, int)
                or recruitment_year < 0
            ):
                raise ValueError("recruitment_year must be a non-negative integer")
            comment_page_delay = params.get("comment_page_delay", 500)
            if (
                isinstance(comment_page_delay, bool)
                or not isinstance(comment_page_delay, int)
                or not 0 <= comment_page_delay <= 10000
            ):
                raise ValueError(
                    "comment_page_delay must be an integer from 0 to 10000"
                )
            self._comment_page_delay = comment_page_delay

            self.last_session = {
                "department": department.strip(),
                "day": day,
                "requirements": requirements,
                "comment_options": [],
                "comment_boxes": {},
                "resumes": [],
                "completed": False,
                "stop_reason": "",
                "portrait_debug": {
                    "enabled": self._portrait_debug_dir is not None,
                    "directory": (
                        str(self._portrait_debug_dir)
                        if self._portrait_debug_dir is not None
                        else ""
                    ),
                    "threshold": portrait_threshold,
                },
            }
            logger.info(
                "[简历筛选] 开始本轮筛选："
                + json.dumps(
                    {
                        "部门": department.strip(),
                        "天数": day,
                        "招聘要求": [
                            requirement.get("text", "") for requirement in requirements
                        ],
                        "头像对比阈值": portrait_threshold,
                        # "保存头像对比样本": (
                        #     str(self._portrait_debug_dir)
                        #     if self._portrait_debug_dir is not None
                        #     else "否"
                        # ),
                    },
                    ensure_ascii=False,
                )
            )

            if _should_stop_context(context):
                return self._external_stop_result("确认简历筛选界面前")
            initial_detail = _run_task_wait(context, "检查简历-新简历")
            if _should_stop_context(context):
                return self._external_stop_result("确认简历筛选界面")
            if _task_failed(initial_detail):
                logger.error("[简历筛选] 未确认当前处于简历筛选界面")
                return self._fail_session("initial_page_check_failed")

            if _should_stop_context(context):
                return self._external_stop_result("采集评语位置前")
            image = context.tasker.controller.post_screencap().wait().get()
            comment_detail = context.run_recognition("检查评语位置", image)
            if _should_stop_context(context):
                return self._external_stop_result("采集评语位置")
            first_page_options = collect_comment_options(comment_detail, page=1)
            if not self._switch_comment_page(context, 2):
                if _should_stop_context(context):
                    return self._external_stop_result("采集第二页评语前")
                return self._fail_session("comment_page_switch_failed")
            image = context.tasker.controller.post_screencap().wait().get()
            second_page_detail = context.run_recognition("检查评语位置", image)
            if _should_stop_context(context):
                return self._external_stop_result("采集第二页评语")
            second_page_options = collect_comment_options(second_page_detail, page=2)
            if not self._switch_comment_page(context, 1):
                if _should_stop_context(context):
                    return self._external_stop_result("返回第一页评语")
                return self._fail_session("comment_page_switch_failed")
            comment_options = _merge_comment_options(
                first_page_options, second_page_options
            )
            self.last_session["comment_options"] = comment_options
            self.last_session["comment_boxes"] = {
                option["text"]: {
                    "page": option["page"],
                    "box": option["box"],
                }
                for option in comment_options
            }
            # logger.info(
            #     "[简历筛选] 已采集评语位置："
            #     + json.dumps(
            #         [
            #             {
            #                 "评语": option["text"],
            #                 "页码": option["page"],
            #                 "位置": option["box"],
            #                 "置信度": option.get("score"),
            #             }
            #             for option in comment_options
            #         ],
            #         ensure_ascii=False,
            #     )
            # )
            if _find_comment_option(comment_options, "简历无误") is None:
                logger.error("[简历筛选] 未找到“简历无误”评语")
                return self._fail_session("comment_collection_failed")

            for index in range(max_resumes):
                if _should_stop_context(context):
                    return self._external_stop_result(f"采集第 {index + 1} 份简历前")
                image = context.tasker.controller.post_screencap().wait().get()
                self._certificate_checked = False
                self._certificate_detail = None
                resume_detail = context.run_recognition("获取简历信息", image)
                if _should_stop_context(context):
                    return self._external_stop_result(f"采集第 {index + 1} 份简历")
                resume = parse_resume_info(resume_detail)
                record = {
                    "index": index + 1,
                    "resume": resume,
                    "evaluation": None,
                    "resume_second_round_evaluation": None,
                    "certificate_evaluation": None,
                    "comments": [],
                    "disposition": None,
                }
                self.last_session["resumes"].append(record)
                # logger.info(
                #     f"[简历筛选] 第 {index + 1} 份简历信息："
                #     + json.dumps(
                #         {
                #             "姓名": resume["name"],
                #             "毕业学校": resume["school"],
                #             "状态": resume["status"],
                #             "工作/实习经历": [
                #                 {
                #                     "单位": experience["company"],
                #                     "职位": experience["role"],
                #                     "时长": experience["duration"],
                #                 }
                #                 for experience in resume["experiences"]
                #             ],
                #             "经历数量": resume["experience_count"],
                #         },
                #         ensure_ascii=False,
                #     )
                # )

                missing = [
                    name
                    for name in ("name", "school", "status")
                    if not resume.get(name)
                ]
                if missing:
                    missing_names = {
                        "name": "姓名",
                        "school": "毕业学校",
                        "status": "状态",
                    }
                    logger.error(
                        f"[简历筛选] 第 {index + 1} 份简历缺少字段："
                        + "、".join(missing_names[name] for name in missing)
                    )
                    return self._fail_session("resume_collection_failed")

                resume_second_round_evaluation = (
                    self._evaluate_resume_second_round_data(resume)
                )
                record["resume_second_round_evaluation"] = (
                    resume_second_round_evaluation
                )
                for result in resume_second_round_evaluation["results"]:
                    if result["passed"]:
                        continue
                    logger.info(
                        f"[简历筛选] 第 {index + 1} 份简历通用检查结果："
                        + json.dumps(
                            {
                                "检查项目": result["text"],
                                "结果": ("通过" if result["passed"] else "不通过"),
                                "判断依据": result.get("evidence", {}),
                            },
                            ensure_ascii=False,
                        )
                    )
                failed_requirements = [
                    result
                    for result in resume_second_round_evaluation["results"]
                    if not result["passed"]
                ]
                if not failed_requirements:
                    evaluation = self._evaluate_requirements(
                        context, image, requirements, resume
                    )
                    record["evaluation"] = evaluation
                    if evaluation.get("stopped") or _should_stop_context(context):
                        return self._external_stop_result(f"检查第 {index + 1} 份简历")
                    for result in evaluation["results"]:
                        if result["passed"]:
                            continue
                        logger.info(
                            f"[简历筛选] 第 {index + 1} 份简历检查结果："
                            + json.dumps(
                                {
                                    "检查项目": result["text"],
                                    "结果": ("通过" if result["passed"] else "不通过"),
                                    "判断依据": result.get("evidence", {}),
                                },
                                ensure_ascii=False,
                            )
                        )
                    if evaluation.get("data_error"):
                        logger.error(
                            f"[简历筛选] 第 {index + 1} 份简历数据不足："
                            + evaluation["data_error"]
                        )
                        return self._fail_session("resume_collection_failed")
                    if not evaluation["complete"]:
                        unsupported_texts = [
                            requirement.get("text", requirement.get("type", ""))
                            for requirement in requirements
                            if requirement.get("type") in evaluation["unsupported"]
                        ]
                        logger.warning(
                            "[简历筛选] 尚未实现的检查项目："
                            + "、".join(unsupported_texts)
                        )
                        return self._fail_session("unsupported_requirements")
                    failed_requirements = [
                        result
                        for result in evaluation["results"]
                        if not result["passed"]
                    ]

                if not failed_requirements:
                    if _should_stop_context(context):
                        return self._external_stop_result(
                            f"检查第 {index + 1} 份简历是否持证前"
                        )
                    certificate_detail = self._get_certificate_detail(context, image)
                    if _should_stop_context(context):
                        return self._external_stop_result(
                            f"检查第 {index + 1} 份简历是否持证"
                        )
                    has_certificate = _recognition_hit(certificate_detail)
                    # logger.info(
                    #     f"[简历筛选] 第 {index + 1} 份简历第二轮检查："
                    #     + (
                    #         "检测到证书，开始核对"
                    #         if has_certificate
                    #         else "未持证，跳过"
                    #     )
                    # )
                    if has_certificate:
                        certificate_evaluation = self._screen_certificate(
                            context,
                            image,
                            resume,
                            certificate_detail,
                            certificate_expand_delay,
                            portrait_threshold,
                            recruitment_year,
                            index + 1,
                        )
                        record["certificate_evaluation"] = certificate_evaluation
                        if certificate_evaluation.get("stopped"):
                            return self._external_stop_result(
                                f"核对第 {index + 1} 份简历与证书"
                            )
                        if not certificate_evaluation.get("complete"):
                            logger.error(
                                f"[简历筛选] 第 {index + 1} 份证书信息采集失败："
                                + certificate_evaluation.get("error", "未知原因")
                            )
                            return self._fail_session("certificate_collection_failed")

                        certificate = certificate_evaluation["certificate"]
                        # logger.info(
                        #     f"[简历筛选] 第 {index + 1} 份证书信息："
                        #     + json.dumps(
                        #         {
                        #             "姓名": certificate["name"],
                        #             "毕业学校": certificate["school"],
                        #             "毕业时间": certificate["graduation_time"],
                        #             "头像相似度": round(
                        #                 certificate_evaluation["portrait_similarity"],
                        #                 4,
                        #             ),
                        #         },
                        #         ensure_ascii=False,
                        #     )
                        # )
                        for result in certificate_evaluation["results"]:
                            if result["passed"]:
                                continue
                            logger.info(
                                f"[简历筛选] 第 {index + 1} 份证书核对结果："
                                + json.dumps(
                                    {
                                        "检查项目": result["text"],
                                        "结果": (
                                            "通过" if result["passed"] else "不通过"
                                        ),
                                        "判断依据": result.get("evidence", {}),
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        failed_requirements = [
                            result
                            for result in certificate_evaluation["results"]
                            if not result["passed"]
                        ]
                comments = (
                    [failed_requirements[0]["text"]]
                    if failed_requirements
                    else ["简历无误"]
                )
                for comment in comments:
                    if _should_stop_context(context):
                        return self._external_stop_result(
                            f"拖入第 {index + 1} 份简历评语前"
                        )
                    if not self._drag_comment(context, comment_options, comment):
                        if _should_stop_context(context):
                            return self._external_stop_result(
                                f"拖入第 {index + 1} 份简历评语"
                            )
                        return self._fail_session("drag_comment_failed")
                    record["comments"].append(comment)

                disposition = (
                    "处置简历-不合格" if failed_requirements else "处置简历-合格"
                )
                if _should_stop_context(context):
                    return self._external_stop_result(f"处置第 {index + 1} 份简历前")
                logger.info(f"[简历筛选] 开始处置第 {index + 1} 份简历：{disposition}")
                disposition_detail = _run_task_wait(
                    context,
                    disposition,
                    {
                        disposition: {
                            "post_delay": disposition_delay,
                        }
                    },
                )
                # logger.info(
                #     f"[简历筛选] 第 {index + 1} 份简历处置任务状态："
                #     + json.dumps(
                #         _task_debug_summary(disposition_detail), ensure_ascii=False
                #     )
                # )
                if _should_stop_context(context):
                    return self._external_stop_result(f"处置第 {index + 1} 份简历")
                if _task_failed(disposition_detail):
                    logger.error(
                        f"[简历筛选] 第 {index + 1} 份简历处置失败：{disposition}"
                    )
                    return self._fail_session("disposition_failed")
                record["disposition"] = disposition
                # logger.info(f"[简历筛选] 第 {index + 1} 份简历已处置：{disposition}")

                if _should_stop_context(context):
                    return self._external_stop_result("检查是否完成审批前")
                image = context.tasker.controller.post_screencap().wait().get()
                approval_detail = context.run_recognition("完成审批check", image)
                if _should_stop_context(context):
                    return self._external_stop_result("检查是否完成审批")
                approval_completed = _recognition_hit(approval_detail)
                # logger.info(
                #     "[简历筛选] 完成审批检查结果："
                #     + ("已完成" if approval_completed else "尚未完成")
                # )
                if approval_completed:
                    self.last_session["completed"] = True
                    self.last_session["stop_reason"] = "approval_completed"
                    # logger.info(
                    #     "[简历筛选] 本轮筛选完成："
                    #     + json.dumps(
                    #         {
                    #             "部门": department.strip(),
                    #             "天数": day,
                    #             "已处理简历数": len(self.last_session["resumes"]),
                    #             "结束原因": "完成审批检查已命中",
                    #         },
                    #         ensure_ascii=False,
                    #     )
                    # )
                    return CustomAction.RunResult(success=True)

                if _should_stop_context(context):
                    return self._external_stop_result("查看下一份简历前")
                # logger.info("[简历筛选] 开始查找并打开下一份简历")
                next_detail = _run_task_wait(context, "检查简历-查看下一位")
                # logger.info(
                #     "[简历筛选] 查看下一份简历任务状态："
                #     + json.dumps(_task_debug_summary(next_detail), ensure_ascii=False)
                # )
                if _should_stop_context(context):
                    return self._external_stop_result("查看下一份简历")
                if _task_failed(next_detail):
                    logger.error("[简历筛选] 完成审批检查未命中，但无法打开下一份简历")
                    return self._fail_session("next_resume_failed")
                # logger.info(f"[简历筛选] 已切换到第 {index + 2} 份简历")

            logger.error(f"[简历筛选] 已达到简历处理上限：{max_resumes}")
            return self._fail_session("max_resumes_reached")
        except Exception:
            logger.exception("[简历筛选] 筛选过程发生异常")
            return self._fail_session("exception")
