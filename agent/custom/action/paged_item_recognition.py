from __future__ import annotations

import json
import math
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from zhconv import convert

from custom.reco.agent_item import (
    AgentIndex,
    DEFAULT_INDEX_PATH,
    REPO_ROOT,
    LayoutCell,
    _clip_rect,
    _crop,
    _load_index,
    _ocr_results,
    _parse_auto_grid_hint,
    _parse_rect,
    _record_results,
    _resolve_path,
    _resolve_record_path,
    _run_count_ocr,
    auto_recognition_params,
    detect_auto_layout,
    recognize_agent_in_cell,
    recognize_bag_count,
    recognize_item_grid,
)
from custom.action.inventory_reporting import (
    AUTO_UPLOAD_MODE,
    DAILY_TOKEN_REQUIRED_HINT,
    LOCAL_ONLY_MODE,
    TOKEN_REQUIRED_HINT,
    append_inventory_report,
    bound_account_report_filename,
    build_exchange_document,
    get_bound_account,
    inventory_recovery_hint,
    is_dispatch_reward,
    parse_record_options,
    read_upload_settings,
    resolve_inventory_report_destination,
    update_upload_status,
    upload_inventory_document,
    validate_stamina_cost,
)
from custom.action.toolbox_diagnostics import call as diagnostic_call, start_bag_diagnostics
from custom.reco.xinzhi_online_index import load_xinzhi_catalog, supplement_xinzhi_index
from utils import logger

OPERATORS_PATH = REPO_ROOT / "agent" / "operators.json"
DEFAULT_STAMINA_COST_ROI = (510, 375, 50, 48)
SP_OPERATOR_IDS = frozenset({"char_084_chendengsp", "char_085_shizimiaosp"})
TAB1_ITEM_IDS = ("fuchuan", "tianjifuchuan", "jizhi", "mazi", "sherou", "zhuyu")
TAB3_1_ITEM_IDS = (
    "zhuangjinboli",
    "baimozhijiu",
    "baoshijing",
    "bawanglei",
    "beihuifengshan",
    "cuishan",
    "jinsishan",
    "juanshan",
    "lingshanquan",
    "liubojing",
    "liujinjing",
    "mulanzhuilu",
    "qingjiu",
    "shuijing",
    "tongjing",
    "xianmenshan",
    "xinghanjing",
    "yushan",
    "zhuojiu",
    "huaiyinjinsuo",
    "huoyuanjinsuo",
    "shuixinjinsuo",
    "tianfengjinsuo",
    "yangmingjinsuo",
    "zaidijinsuo",
    "gongguoge",
    "gusuanchou",
    "jinsuanchou",
    "shanebu",
    "bingshucanjuan",
    "bingshuquanjuan",
    "caiwendun",
    "diguanghe",
    "fujunhaitang",
    "jianjia",
    "jincuodao",
    "liutaobingshu",
    "menghunlan",
    "panlonggu",
    "qingtingyan",
    "qingtongdao",
    "tietaigong",
    "xijiaogong",
    "yingqiongyao",
    "yinwendao",
    "yuanyu",
    "yuguidun",
    "zhentiangu",
    "ziyunying",
    "zuigucao",
    "jiezheping",
    "jieyangping",
    "jiezhuping",
)


@dataclass
class PageScan:
    results: list[dict]
    rejected: list[dict]
    row_features: list[dict[int, np.ndarray]]
    column_count: int
    layout: dict
    row_entities: list[dict[int, str]] = field(default_factory=list)
    row_counts: list[dict[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class SnapshotList:
    name: str
    entity_type: str
    ids: tuple[str, ...]


def _parse_params(raw: Any) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        params = raw
    elif isinstance(raw, str):
        params = json.loads(raw)
    else:
        raise ValueError("custom_action_param 必须是 JSON 对象")
    if not isinstance(params, dict):
        raise ValueError("custom_action_param 必须是 JSON 对象")
    return params


def _parse_entity_type_filter(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("entity_type_filter 必须是 agent 或 item")
    entity_type = value.strip().lower()
    if entity_type not in {"agent", "item"}:
        raise ValueError("entity_type_filter 必须是 agent 或 item")
    return entity_type


def _filter_results(results: list[dict], entity_type: str | None) -> list[dict]:
    if entity_type is None:
        return results
    return [result for result in results if result.get("entity_type") == entity_type]


def _load_tab3_2_ids(path: Path = OPERATORS_PATH) -> tuple[str, ...]:
    try:
        data = load_xinzhi_catalog(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 tab3-2 密探目录 {path}: {exc}") from exc
    operators = data.get("OPERATORS") if isinstance(data, dict) else None
    if not isinstance(operators, list) or not operators:
        raise ValueError(f"tab3-2 密探目录格式无效: {path}")

    operator_ids: list[str] = []
    seen: set[str] = set()
    for operator in operators:
        operator_id = (
            str(operator.get("id", "")).strip() if isinstance(operator, dict) else ""
        )
        if not operator_id:
            raise ValueError(f"tab3-2 密探目录包含无效 ID: {path}")
        if operator_id in seen:
            raise ValueError(f"tab3-2 密探目录包含重复 ID: {operator_id}")
        seen.add(operator_id)
        if not (operator.get("name") or "").strip():
            continue
        if operator_id not in SP_OPERATOR_IDS:
            operator_ids.append(operator_id)
    if not operator_ids:
        raise ValueError("tab3-2 密探目录排除 SP 和未命名占位记录后为空")
    return tuple(operator_ids)


def _parse_snapshot_list(value: Any) -> SnapshotList | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("snapshot_list 必须是 tab1、tab3-1 或 tab3-2")
    name = value.strip().lower()
    if name == "tab1":
        return SnapshotList(name, "item", TAB1_ITEM_IDS)
    if name == "tab3-1":
        return SnapshotList(name, "item", TAB3_1_ITEM_IDS)
    if name == "tab3-2":
        return SnapshotList(name, "agent", _load_tab3_2_ids())
    raise ValueError("snapshot_list 必须是 tab1、tab3-1 或 tab3-2")


def _snapshot_record_options(
    params: dict, snapshot_list: SnapshotList | None
) -> tuple[str, str | None]:
    if snapshot_list is None:
        return parse_record_options(params)
    record_type = params.get("record_type")
    if record_type not in (None, "", "stock_snapshot"):
        raise ValueError("snapshot_list 只能用于 record_type: stock_snapshot")
    snapshot_scope = params.get("snapshot_scope")
    if snapshot_scope not in (None, "", "listed"):
        raise ValueError("snapshot_list 只能用于 snapshot_scope: listed")
    return "stock_snapshot", "listed"


def _index_entries(index: AgentIndex) -> dict[tuple[str, str], dict]:
    entries: dict[tuple[str, str], dict] = {}
    for position, raw_entity_type in enumerate(index.entity_types):
        entity_type = str(raw_entity_type)
        if entity_type == "agent":
            entity_id = str(index.operator_ids[position])
            entry = {
                "entity_type": "agent",
                "operator_id": entity_id,
                "operator_name": str(index.operator_names[position]),
                "count": 0,
            }
        else:
            entity_id = str(index.agent_ids[position])
            entry = {
                "entity_type": "item",
                "item_id": entity_id,
                "item_name": str(index.operator_names[position]),
                "count": 0,
            }
        key = (entity_type, entity_id)
        if key in entries:
            raise ValueError(f"索引中包含重复稳定 ID: {entity_type}/{entity_id}")
        entries[key] = entry
    return entries


def _result_stable_id(result: dict, entity_type: str) -> str:
    key = "operator_id" if entity_type == "agent" else "item_id"
    return str(result.get(key, "")).strip()


def _snapshot_target_rows(page: PageScan, snapshot_list: SnapshotList) -> set[int]:
    target_ids = set(snapshot_list.ids)
    rows: set[int] = set()
    for result in page.results:
        if _result_stable_id(result, snapshot_list.entity_type) in target_ids:
            rows.add(int(result["row"]))
    for rejected in page.rejected:
        if rejected.get("reason") not in {
            "count-not-recognized",
            "count-too-close-to-roi-bottom",
        }:
            continue
        if _result_stable_id(rejected, snapshot_list.entity_type) in target_ids:
            rows.add(int(rejected["row"]))
    return rows


def _has_snapshot_target_boundary(
    page: PageScan,
    snapshot_list: SnapshotList,
    previously_seen_target: bool,
) -> tuple[bool, bool]:
    target_rows = _snapshot_target_rows(page, snapshot_list)
    seen_target = previously_seen_target or bool(target_rows)
    if not seen_target:
        return False, False

    first_candidate_row = max(target_rows) + 1 if target_rows else 0
    for row in range(first_candidate_row, len(page.row_features)):
        if len(page.row_features[row]) == page.column_count:
            return True, True
    return False, True


def _apply_snapshot_list(
    results: list[dict],
    snapshot_list: SnapshotList,
    index: AgentIndex,
    uncertain_ids: set[str] | None = None,
    count_skipped_ids: set[str] | None = None,
) -> tuple[list[dict], list[str], int]:
    index_entries = _index_entries(index)
    missing = [
        entity_id
        for entity_id in snapshot_list.ids
        if (snapshot_list.entity_type, entity_id) not in index_entries
    ]
    if (
        missing
        and snapshot_list.name == "tab3-2"
        and snapshot_list.entity_type == "agent"
    ):
        operators = load_xinzhi_catalog(OPERATORS_PATH)["OPERATORS"]
        names = {operator["id"]: operator["name"] for operator in operators}
        logger.warning(
            f"以下新密探心纸缺图或在线补全失败，本次跳过：{', '.join(names.get(entity_id, entity_id) for entity_id in missing)}"
        )
    elif missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            f"snapshot_list {snapshot_list.name} 有 {len(missing)} 个 ID 不在当前索引中: "
            f"{preview}{suffix}；请先更新对应 NPZ"
        )

    target_ids = set(snapshot_list.ids) - set(missing)
    recognized: dict[str, dict] = {}
    completed: list[dict] = []
    ignored: list[str] = []
    for result in results:
        entity_id = _result_stable_id(result, snapshot_list.entity_type)
        if entity_id not in target_ids:
            ignored.append(entity_id or "<missing-id>")
            continue
        if entity_id in recognized:
            first = recognized[entity_id]
            raise ValueError(
                f"snapshot_list {snapshot_list.name} 中重复识别到: {entity_id}; "
                f"first={_result_origin(first)}; second={_result_origin(result)}"
            )
        recognized[entity_id] = result
        output = dict(index_entries[(snapshot_list.entity_type, entity_id)])
        output.update(result)
        completed.append(output)

    count_skipped_ids = count_skipped_ids or set()
    uncertain_ids = (((uncertain_ids or set()) | count_skipped_ids) & target_ids) - recognized.keys()
    unconfirmed_ids = uncertain_ids - count_skipped_ids
    if unconfirmed_ids:
        logger.warning(
            f"【广陵库房】以下候选未确认，本次不覆盖其库存：{', '.join(sorted(unconfirmed_ids))}"
        )
    for entity_id in snapshot_list.ids:
        if entity_id in target_ids and entity_id not in uncertain_ids and entity_id not in recognized:
            completed.append(dict(index_entries[(snapshot_list.entity_type, entity_id)]))
    logger.bind(console=False).info(
        f"【广陵库房】未识别对象补 0：{len(completed) - len(recognized)} 项"
    )
    return completed, ignored, len(recognized)


def _uncertain_snapshot_ids(
    rejected: dict, index: AgentIndex, params: dict
) -> set[str]:
    """Protect plausible rejected candidates, not unrelated low-score icons."""
    direct_id = (
        rejected.get("operator_id")
        if rejected.get("entity_type") == "agent"
        else rejected.get("item_id")
    )
    if direct_id:
        return {str(direct_id)}
    floor = float(params.get("match_low_threshold", params.get("match_threshold", 0.9)))
    if float(rejected.get("match_score", 0)) < floor:
        return set()
    candidate_ids = {
        rejected.get("best_agent_id"),
        rejected.get("match_runner_up_agent_id"),
    }
    if rejected.get("refined"):
        group = next(
            group
            for group in index.refine_groups
            if group.group_id == rejected["refine_group"]
        )
        margin = float(params.get("refine_min_margin", group.min_margin))
        best_score = float(rejected.get("refine_score", 0))
        candidate_ids = {
            candidate.get("item_id")
            for candidate in rejected.get("refine_candidates", [])
            if best_score - float(candidate["score"]) <= margin
        } or {rejected.get("best_agent_id")}
    return {
        str(
            index.operator_ids[position]
            if str(index.entity_types[position]) == "agent"
            else raw_id
        )
        for position, raw_id in enumerate(index.agent_ids)
        if str(raw_id) in candidate_ids
    }


def _load_debug_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"调试图片不存在: {path}")
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取调试图片: {path}")
    return image


def _result_origin(result: dict) -> str:
    return (
        f"page={result.get('_source_page', '?')},"
        f"page_row={result.get('_source_page_row', '?')},"
        f"global_row={result.get('row', '?')},"
        f"column={result.get('column', '?')},"
        f"count={result.get('count', '?')},"
        f"match_score={float(result.get('match_score', 0.0)):.4f}"
    )


def _prepare_page_debug_directory(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("debug_page_dir 必须是非空路径字符串")
    base = _resolve_path(value.strip(), REPO_ROOT)
    run_name = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _debug_json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法写入调试 JSON 的类型: {type(value).__name__}")


def _write_debug_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_debug_json_default,
        ),
        encoding="utf-8",
    )


def _write_debug_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"无法编码分页调试截图: {path}")
    path.write_bytes(encoded.tobytes())


def _should_stop(context: Context) -> bool:
    try:
        if bool(getattr(context, "stop", False)):
            return True
        tasker = getattr(context, "tasker", None)
        if tasker is not None:
            if bool(getattr(tasker, "stopping", False)):
                return True
            if not tasker.running:
                return True
    except Exception:
        return False
    return False


def _circle_feature(image: np.ndarray, cell: LayoutCell) -> np.ndarray:
    x, y, width, height = cell.box
    patch = image[y : y + height, x : x + width]
    resized = cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA)
    vector = resized.astype(np.float32).reshape(-1)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        raise ValueError("物品圆形区域没有有效图像特征")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def scan_page(
    image: np.ndarray,
    index,
    roi: tuple[int, int, int, int],
    grid_hint: tuple[int | None, int | None],
    params: dict,
    context: Context | None,
) -> PageScan:
    cells, layout = detect_auto_layout(image, index, roi, grid_hint, params)
    if not cells:
        raise RuntimeError(f"分页物品识别未检测到完整圆形物品: {layout}")

    effective_params = auto_recognition_params(params)
    results, rejected = recognize_item_grid(
        image,
        index,
        roi,
        (1, 1),
        effective_params,
        context,
        layout_cells=cells,
    )
    row_count = max(cell.row for cell in cells) + 1
    row_features: list[dict[int, np.ndarray]] = [dict() for _ in range(row_count)]
    for cell in cells:
        row_features[cell.row][cell.column] = _circle_feature(image, cell)
    row_entities: list[dict[int, str]] = [dict() for _ in range(row_count)]
    row_counts: list[dict[int, int]] = [dict() for _ in range(row_count)]
    overlap_results = results
    if params.get("count_mode") == "bag_consensus":
        # Keep icon identities at clipped count badges; only template counts anchor row quantities.
        overlap_results = results + [entry for entry in rejected if entry.get("reason") == "count-not-recognized"]
    for result in overlap_results:
        entity_type = str(result.get("entity_type", ""))
        id_key = "operator_id" if entity_type == "agent" else "item_id"
        entity_id = str(result.get(id_key, "")).strip()
        count = result.get("count")
        if params.get("count_mode") == "bag_consensus":
            count = next((reading["_template_count"] for reading in result.get("count_readings", [])
                          if "_template_count" in reading), None)
            if count is not None and count < max(0, int(params.get("count_min_value", 1))):
                count = None
        if entity_id and (count is not None or params.get("count_mode") == "bag_consensus"):
            row = int(result["row"])
            column = int(result["column"])
            row_entities[row][column] = f"{entity_type}:{entity_id}"
            if count is not None:
                row_counts[row][column] = int(count)
    column_count = max(cell.column for cell in cells) + 1
    return PageScan(
        results,
        rejected,
        row_features,
        column_count,
        layout,
        row_entities,
        row_counts,
    )


def _row_similarity(
    previous: dict[int, np.ndarray], current: dict[int, np.ndarray]
) -> float | None:
    common_columns = sorted(set(previous).intersection(current))
    required = min(2, len(previous), len(current))
    if len(common_columns) < required:
        return None
    scores = [float(previous[column] @ current[column]) for column in common_columns]
    return float(np.mean(scores))


def _row_identity_matches(previous: dict[int, str], current: dict[int, str]) -> int:
    common_columns = set(previous).intersection(current)
    return sum(previous[column] == current[column] for column in common_columns)


def _row_identity_count_matches(
    previous_entities: dict[int, str],
    current_entities: dict[int, str],
    previous_counts: dict[int, int],
    current_counts: dict[int, int],
) -> int:
    common_columns = (
        set(previous_entities)
        .intersection(current_entities)
        .intersection(previous_counts)
        .intersection(current_counts)
    )
    return sum(
        previous_entities[column] == current_entities[column]
        and previous_counts[column] == current_counts[column]
        for column in common_columns
    )


def find_row_overlap(
    previous: list[dict[int, np.ndarray]],
    current: list[dict[int, np.ndarray]],
    threshold: float,
    previous_entities: list[dict[int, str]] | None = None,
    current_entities: list[dict[int, str]] | None = None,
    previous_counts: list[dict[int, int]] | None = None,
    current_counts: list[dict[int, int]] | None = None,
) -> tuple[int, list[float]]:
    for count in range(min(len(previous), len(current)), 0, -1):
        scores: list[float] = []
        all_image_rows_match = True
        has_identity_anchor = False
        previous_identity_rows = (
            previous_entities[-count:] if previous_entities else [{}] * count
        )
        current_identity_rows = (
            current_entities[:count] if current_entities else [{}] * count
        )
        previous_count_rows = (
            previous_counts[-count:] if previous_counts else [{}] * count
        )
        current_count_rows = current_counts[:count] if current_counts else [{}] * count
        rows = zip(
            previous[-count:],
            current[:count],
            previous_identity_rows,
            current_identity_rows,
            previous_count_rows,
            current_count_rows,
        )
        for (
            previous_row,
            current_row,
            previous_ids,
            current_ids,
            previous_row_counts,
            current_row_counts,
        ) in rows:
            score = _row_similarity(previous_row, current_row)
            identity_matches = _row_identity_matches(previous_ids, current_ids)
            identity_count_matches = _row_identity_count_matches(
                previous_ids,
                current_ids,
                previous_row_counts,
                current_row_counts,
            )
            image_matches = score is not None and score >= threshold
            all_image_rows_match = all_image_rows_match and image_matches
            if (
                identity_count_matches >= 1
                or identity_matches >= 2
                or (
                    identity_matches == 1
                    and score is not None
                    and score >= max(0.0, threshold - 0.05)
                )
            ):
                has_identity_anchor = True
            scores.append(1.0 if score is None else score)
        if has_identity_anchor or all_image_rows_match:
            return count, scores
    return 0, []


def overlap_candidate_scores(
    previous: list[dict[int, np.ndarray]],
    current: list[dict[int, np.ndarray]],
    previous_entities: list[dict[int, str]] | None = None,
    current_entities: list[dict[int, str]] | None = None,
    previous_counts: list[dict[int, int]] | None = None,
    current_counts: list[dict[int, int]] | None = None,
) -> list[dict]:
    candidates: list[dict] = []
    for count in range(min(len(previous), len(current)), 0, -1):
        scores = [
            _row_similarity(previous_row, current_row)
            for previous_row, current_row in zip(previous[-count:], current[:count])
        ]
        previous_identity_rows = (
            previous_entities[-count:] if previous_entities else [{}] * count
        )
        current_identity_rows = (
            current_entities[:count] if current_entities else [{}] * count
        )
        previous_count_rows = (
            previous_counts[-count:] if previous_counts else [{}] * count
        )
        current_count_rows = current_counts[:count] if current_counts else [{}] * count
        candidates.append(
            {
                "count": count,
                "scores": [
                    None if score is None else round(score, 6) for score in scores
                ],
                "identity_matches": [
                    _row_identity_matches(previous_ids, current_ids)
                    for previous_ids, current_ids in zip(
                        previous_identity_rows, current_identity_rows
                    )
                ],
                "identity_count_matches": [
                    _row_identity_count_matches(
                        previous_ids,
                        current_ids,
                        previous_row_counts,
                        current_row_counts,
                    )
                    for (
                        previous_ids,
                        current_ids,
                        previous_row_counts,
                        current_row_counts,
                    ) in zip(
                        previous_identity_rows,
                        current_identity_rows,
                        previous_count_rows,
                        current_count_rows,
                    )
                ],
            }
        )
    return candidates


def _parse_swipe(
    params: dict,
) -> tuple[tuple[int, int, int, int, int] | None, int]:
    default_duration = int(params.get("swipe_duration", 500))
    if default_duration <= 0:
        raise ValueError("swipe_duration 必须大于 0")
    configured = params.get("swipe")
    if configured is not None:
        if not isinstance(configured, (list, tuple)) or len(configured) not in {4, 5}:
            raise ValueError("swipe 必须为 [x1, y1, x2, y2] 或再追加 duration")
        values = [int(value) for value in configured]
        duration = values[4] if len(values) == 5 else default_duration
        if duration <= 0:
            raise ValueError("swipe duration 必须大于 0")
        return (values[0], values[1], values[2], values[3], duration), duration
    return None, default_duration


def _parse_swipe_rows(params: dict) -> float | None:
    value = params.get("swipe_rows")
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("swipe_rows 必须是大于 0 的数字")
    try:
        rows = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("swipe_rows 必须是大于 0 的数字") from exc
    if not math.isfinite(rows) or rows <= 0:
        raise ValueError("swipe_rows 必须是大于 0 的数字")
    return rows


def automatic_swipe(
    layout: dict,
    roi: tuple[int, int, int, int],
    duration: int,
    swipe_rows: float | None = None,
) -> tuple[int, int, int, int, int]:
    roi_x, roi_y, roi_width, roi_height = roi
    columns = sorted(float(value) for value in layout.get("column_centers", []))
    rows = sorted(float(value) for value in layout.get("row_centers", []))
    if not columns or not rows:
        raise ValueError("自动滑动缺少行列中心")

    roi_center_x = roi_x + roi_width / 2
    if len(columns) >= 2:
        gaps = [(left + right) / 2 for left, right in zip(columns, columns[1:])]
        swipe_x = min(gaps, key=lambda value: abs(value - roi_center_x))
    else:
        radius_range = layout.get("circle_radius", [0, 0])
        radius = sum(float(value) for value in radius_range) / 2
        left_edge = columns[0] - radius
        right_edge = columns[0] + radius
        left_width = max(0.0, left_edge - roi_x)
        right_width = max(0.0, roi_x + roi_width - right_edge)
        if left_width >= right_width:
            swipe_x = roi_x + left_width / 2
        else:
            swipe_x = right_edge + right_width / 2

    if len(rows) >= 2:
        start_y = rows[-1]
        if swipe_rows is None:
            end_y = rows[0]
        else:
            row_spacing = float(np.median(np.diff(rows)))
            row_count = min(swipe_rows, len(rows) - 1)
            end_y = start_y - row_spacing * row_count
    else:
        start_y = roi_y + roi_height * 0.82
        end_y = roi_y + roi_height * 0.37
    return (
        int(round(swipe_x)),
        int(round(start_y)),
        int(round(swipe_x)),
        int(round(end_y)),
        duration,
    )


def _prefer_result(candidate: dict, current: dict) -> bool:
    candidate_score = (
        float(candidate.get("match_score", 0.0)),
        float(candidate.get("count_score", 0.0)),
    )
    current_score = (
        float(current.get("match_score", 0.0)),
        float(current.get("count_score", 0.0)),
    )
    return candidate_score > current_score


def _resolve_bag_consensus(
    observations: list[dict], roi: tuple[int, int, int, int], params: dict,
) -> tuple[dict | None, dict | None]:
    identity = max(observations, key=lambda entry: float(entry.get("match_score", 0)))
    entity_type = identity["entity_type"]
    entity_id = _result_stable_id(identity, entity_type)
    observations = [entry for entry in observations if entry["entity_type"] == entity_type
                    and _result_stable_id(entry, entity_type) == entity_id]
    details = [entry for entry in observations if entry.get("_count_source") == "detail"]
    confirmed = [entry for entry in observations if entry.get("count") is not None]
    values = {entry["count"] for entry in confirmed}
    chosen = None
    if not any(entry.get("_count_force_skip") for entry in observations):
        if details and len({entry["count"] for entry in details}) == 1:
            chosen = dict(details[-1])
        elif not details:
            frames: dict[int, set[int | None]] = {}
            for entry in observations:
                frames.setdefault(entry["_image_index"], set()).add(entry.get("count"))
            votes: dict[int, int] = {}
            for frame_values in frames.values():
                if len(frame_values) == 1 and None not in frame_values:
                    count = next(iter(frame_values))
                    votes[count] = votes.get(count, 0) + 1
            needs_majority = len(values) > 1 or any(entry.get("_needs_count_vote") for entry in observations)
            for count, support in votes.items():
                if (not needs_majority or (support >= 2 and support * 2 > len(frames))):
                    chosen = dict(next(entry for entry in confirmed if entry["count"] == count))
                    break

    reasons = []
    if chosen is None:
        reasons.append("count_unresolved")
    if len(values) > 1:
        reasons.append("page_conflict")
    if any(len({reading["count"] for reading in entry.get("count_readings", [])
                if reading["count"] is not None}) > 1 for entry in observations):
        reasons.append("threshold_disagree")
    minimum = float(params.get("count_min_roi_bottom_distance", 0))
    if all(minimum <= roi[1] + roi[3] - entry["item_center"][1] < 150 for entry in observations):
        reasons.append("edge_zone")
    if not reasons:
        return chosen, None

    readings = [{
        "image_index": reading.get("_image_index", entry["_image_index"]),
        "count_box": list(reading["count_box"]), "count": reading["count"],
        "raw": reading["raw"], "score": round(max(0.0, min(1.0, reading["score"])), 3),
        "source": reading["source"],
        **({"threshold": reading["threshold"]} if "threshold" in reading else {}),
    } for entry in observations for reading in entry.get("count_readings", [])]
    useful = []
    seen = set()
    for reading in readings:
        key = reading["image_index"], reading["count"]
        if key not in seen:
            useful.append(reading)
            seen.add(key)
    evidence = [reading for reading in useful if chosen is not None and reading["count"] == chosen["count"]][:2]
    seen_values = {reading["count"] for reading in evidence}
    for reading in useful:
        if reading["count"] not in seen_values:
            evidence.append(reading)
            seen_values.add(reading["count"])
    readings = (evidence + [reading for reading in useful if reading not in evidence]
                + [reading for reading in readings if reading not in useful])[:8]
    return chosen, dict(entity_type=entity_type, entity_id=entity_id,
                        chosen=chosen["count"] if chosen else None, reasons=reasons, readings=readings)


def _capture_bag_frame(context: Context) -> np.ndarray:
    if _should_stop(context):
        raise RuntimeError("背包扫描已停止，已保留原数量")
    capture = context.tasker.controller.post_screencap().wait()
    if not capture.succeeded:
        raise RuntimeError("背包扫描截图操作失败，拒绝使用缓存截图，已保留原数量")
    frame = capture.get()
    if frame is None:
        raise RuntimeError("背包扫描截图失败，已保留原数量")
    return frame


def _record_bag_image(frame: np.ndarray, roi: tuple[int, int, int, int], diagnostics, state: dict) -> int:
    image_index = state["image_index"]
    state["image_index"] += 1
    diagnostic_call(diagnostics, "observe_bag_page", frame, roi)
    return image_index


def _bag_cell_match(frame: np.ndarray, candidate: dict, index: AgentIndex, params: dict):
    matched, _ = recognize_agent_in_cell(frame, tuple(candidate["item_center"]), index, auto_recognition_params(params))
    if matched is None or matched.entity_type != candidate["entity_type"]:
        return None
    identity = matched.operator_id if matched.entity_type == "agent" else matched.agent_id
    if identity != _result_stable_id(candidate, matched.entity_type):
        return None
    if any(abs(a - b) > 3 for a, b in zip(matched.item_center, candidate["item_center"])):
        return None
    return matched


def _bag_page_unchanged(
    before: np.ndarray, after: np.ndarray, roi: tuple[int, int, int, int], threshold: float = 0.98,
) -> bool:
    if before.shape != after.shape or _clip_rect(roi, after) != roi:
        return False
    features = []
    for frame in (before, after):
        gray = cv2.cvtColor(_crop(frame, roi), cv2.COLOR_BGR2GRAY)
        feature = cv2.resize(gray, (84, 118), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
        feature -= float(feature.mean())
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-6:
            return False
        features.append(feature / norm)
    return float(features[0] @ features[1]) >= threshold


def _bag_name_key(value: str) -> str:
    text = convert(unicodedata.normalize("NFKC", value), "zh-cn").replace("・", "·").replace("•", "·")
    return re.sub(r"[^\w·]", "", text).lower()


def _bag_detail_names(candidate: dict, state: dict) -> set[str]:
    if "aliases" not in state:
        aliases = {}
        for kind, path, field in (("item", REPO_ROOT / "agent/items.json", "items"),
                                  ("agent", OPERATORS_PATH, "OPERATORS")):
            try:
                catalogue = json.loads(path.read_text(encoding="utf-8"))
                for entry in catalogue[field]:
                    raw = entry.get("aliases", entry.get("alias", []))
                    raw = raw.split() if isinstance(raw, str) else raw
                    names = [entry.get("name", ""), *raw]
                    aliases[(kind, entry["id"])] = {_bag_name_key(str(name)) for name in names if name}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning(f"【广陵库房】详情名称目录读取失败，改用当前索引名称核对：{exc}")
        state["aliases"] = aliases
    kind = candidate["entity_type"]
    entity_id = _result_stable_id(candidate, kind)
    names = set(state["aliases"].get((kind, entity_id), set()))
    name = candidate.get("operator_name") if kind == "agent" else candidate.get("item_name")
    if name:
        names.add(_bag_name_key(name))
    if kind == "agent":
        names.update(name.removesuffix("心纸") for name in tuple(names))
        names.update("心纸" + name for name in tuple(names) if name)
    return names - {""}


def _bag_popup_present(context: Context, frame: np.ndarray) -> bool:
    detail = context.run_recognition("背包数量-详情已出现", frame)
    if detail is None:
        raise RuntimeError("背包详情识别节点不可用，已保留原数量")
    return bool(detail.hit)


def _wait_bag_popup(context: Context, expected: bool) -> tuple[np.ndarray, bool]:
    stable = 0
    for _ in range(8):
        time.sleep(0.2)
        frame = _capture_bag_frame(context)
        present = _bag_popup_present(context, frame)
        stable = stable + 1 if present == expected else 0
        if stable >= 2:
            return frame, present
    if present == expected:
        raise RuntimeError("背包详情状态未连续稳定两帧，停止兜底，已保留原数量")
    return frame, present


def _read_bag_detail(
    context: Context, candidate: dict, index: AgentIndex, roi: tuple[int, int, int, int],
    params: dict, diagnostics, state: dict,
) -> tuple[dict | None, str]:
    before = _capture_bag_frame(context)
    if before.shape[:2] != (1280, 720):
        return None, "截图不是720×1280，未点开详情"
    if (not _bag_page_unchanged(state["page_frame"], before, roi)
            or _bag_popup_present(context, before)
            or _bag_cell_match(before, candidate, index, params) is None):
        raise RuntimeError("背包页面状态或格子位置已变化，停止兜底，已保留原数量")
    name = candidate.get("item_name") or candidate.get("operator_name") or candidate.get("item_id")
    logger.info(f"【广陵库房】{name} 点开详情核对（{state['detail_attempts']}/60），投票读数={candidate.get('count')}")
    x, y = (int(round(value)) for value in candidate["item_center"])
    if _should_stop(context):
        raise RuntimeError("背包扫描已停止，已保留原数量")
    action = context.run_action("背包数量-打开详情", box=(x, y, 1, 1))
    popup, present = _wait_bag_popup(context, True)
    if not present:
        if (_bag_page_unchanged(before, popup, roi)
                and _bag_cell_match(popup, candidate, index, params) is not None):
            return None, "详情弹窗未出现" if action and action.success else "详情点击未成功"
        raise RuntimeError(f"{name} 详情状态不明，停止兜底，已保留原数量")

    resolved = None
    reason = "详情未读到合法的拥有数量"
    try:
        node = context.get_node_data("背包数量-详情数量")
        count_box = _parse_rect(node["recognition"]["param"]["roi"], "详情数量ROI")
        image_index = _record_bag_image(popup, roi, diagnostics, state)
        if state.get("debug_dir") is not None:
            _write_debug_image(state["debug_dir"] / f"detail-{state['detail_attempts']:02d}.png", popup)
        popup_reading = dict(count=None, raw="", score=0.0, source="fallback",
                             count_box=count_box, _image_index=image_index)
        candidate["count_readings"] = [*candidate.get("count_readings", []), popup_reading]
        names = _bag_detail_names(candidate, state)
        name_detail = context.run_recognition("背包数量-详情名称", popup)
        name_ok = False
        for result in _ocr_results(name_detail):
            title = _bag_name_key(str(getattr(result, "text", "")))
            if (float(getattr(result, "score", 0)) >= 0.6 and any(
                title == alias or (candidate["entity_type"] == "agent" and title.endswith("·" + alias))
                for alias in names
            )):
                name_ok = True
                break
        if not name_ok:
            reason = "详情名称与格子身份不一致或未识别"
        else:
            detail = context.run_recognition("背包数量-详情数量", popup)
            counts = []
            for result in _ocr_results(detail):
                text = unicodedata.normalize("NFKC", str(getattr(result, "text", "")))
                match = re.fullmatch(r"[拥擁]有[:：]([0-9]{1,8})", re.sub(r"\s+", "", text))
                score = float(getattr(result, "score", 0))
                if match and math.isfinite(score) and score >= 0.6:
                    counts.append((int(match.group(1)), score))
            if counts and len({count for count, _ in counts}) == 1:
                count, score = max(counts, key=lambda value: value[1])
                popup_reading.update(count=count, raw=str(count), score=score)
                resolved = dict(candidate)
                resolved.update(count=count, count_score=score, count_raw=str(count), _count_source="detail")
    except Exception as exc:
        reason = f"详情识别失败：{exc}"
    finally:
        if _should_stop(context):
            raise RuntimeError("背包扫描已停止，已保留原数量")
        closing_frame = _capture_bag_frame(context)
        if closing_frame.shape[:2] != (1280, 720):
            raise RuntimeError(f"{name} 详情画面尺寸变化，停止兜底，已保留原数量")
        if _bag_popup_present(context, closing_frame):
            close = context.run_action("背包数量-关闭详情")
            if close is None or not close.success:
                raise RuntimeError(f"{name} 详情关闭失败，停止兜底，已保留原数量")
        after, present = _wait_bag_popup(context, False)
        if (present or not _bag_page_unchanged(before, after, roi)
                or _bag_cell_match(after, candidate, index, params) is None):
            raise RuntimeError(f"{name} 详情未关闭或背包页面已变化，停止兜底，已保留原数量")
    return resolved, reason


def _fallback_bag_counts(
    context: Context, candidates: list[tuple[tuple[int, int], dict]],
    observations: dict[tuple[int, int], list[dict]], index: AgentIndex,
    roi: tuple[int, int, int, int], params: dict, diagnostics, state: dict,
) -> None:
    pending = []
    for key, candidate in candidates:
        values = {entry["count"] for entry in observations[key] if entry.get("count") is not None}
        probe = state["probe_remaining"] > 0
        if candidate.get("count") is not None and len(values) <= 1 and not probe:
            continue
        name = candidate.get("item_name") or candidate.get("operator_name") or candidate.get("item_id")
        kind = candidate["entity_type"]
        detail_counts = {entry["count"] for entry in observations[key]
                         if entry.get("_count_source") == "detail" and entry["entity_type"] == kind
                         and _result_stable_id(entry, kind) == _result_stable_id(candidate, kind)}
        if (not probe and len(detail_counts) == 1
                and (candidate.get("count") is None or candidate["count"] in detail_counts)):
            logger.info(f"【广陵库房】{name} 复用本次扫描同一格已核对的详情数量={next(iter(detail_counts))}")
            continue
        if state["detail_attempts"] >= 60:
            candidate["_count_force_skip"] = True
            candidate["_count_failure"] = "已达到60次详情兜底上限"
            logger.warning(f"【广陵库房】{name} 已达到60次详情兜底上限，本次跳过，已保留原数量")
            continue
        state["detail_attempts"] += 1
        if probe:
            state["probe_remaining"] -= 1
        vote_count = candidate.get("count")
        candidate["_needs_count_vote"] = True
        resolved, reason = _read_bag_detail(context, candidate, index, roi, params, diagnostics, state)
        if resolved is not None:
            candidate.update(resolved)
            state["detail_successes"] += 1
            logger.info(f"【广陵库房】{name} 详情核对通过，数量={candidate['count']}，投票读数={vote_count}，一致={candidate['count'] == vote_count}")
        else:
            candidate["_count_failure"] = reason
            logger.info(f"【广陵库房】{name} {reason}，退回原地补帧投票")
            pending.append((key, candidate))
    for _ in range(2):
        if not pending:
            break
        time.sleep(0.2)
        frame = _capture_bag_frame(context)
        state["extra_frames"] += 1
        if (_clip_rect(roi, frame) != roi or _bag_popup_present(context, frame)
                or not _bag_page_unchanged(state["page_frame"], frame, roi)):
            raise RuntimeError("背包补帧页面状态异常，已保留原数量")
        image_index = _record_bag_image(frame, roi, diagnostics, state)
        remaining = []
        for key, candidate in pending:
            matched = _bag_cell_match(frame, candidate, index, params)
            if matched is None:
                raise RuntimeError("背包补帧格子身份或位置变化，已保留原数量")
            readings: list[dict] = []
            count, score, raw, box = recognize_bag_count(context, frame, matched, params, readings)
            observation = dict(candidate)
            observation.update(count=count, count_score=score, count_raw=raw, count_box=list(box),
                               count_readings=readings, _image_index=image_index)
            observations[key].append(observation)
            chosen, _ = _resolve_bag_consensus(observations[key], roi, params)
            if chosen is None:
                remaining.append((key, candidate))
        pending = remaining


def _resolve_bag_count(
    observations: list[dict], roi: tuple[int, int, int, int], params: dict
) -> tuple[dict | None, dict | None]:
    if params.get("count_mode") == "bag_consensus":
        return _resolve_bag_consensus(observations, roi, params)
    identity = observations[0]
    for observation in observations[1:]:
        if _prefer_result(observation, identity):
            identity = observation
    entity_type = identity["entity_type"]
    entity_id = _result_stable_id(identity, entity_type)
    observations = [
        observation for observation in observations
        if observation["entity_type"] == entity_type
        and _result_stable_id(observation, entity_type) == entity_id
    ]
    minimum = float(params.get("count_min_roi_bottom_distance", 0))
    non_edge = [
        observation for observation in observations
        if not minimum <= roi[1] + roi[3] - observation["item_center"][1] < 150
    ]
    attempts = []
    for observation in observations:
        page_attempts = observation.get("count_readings") or [{
            "count": observation["count"], "raw": observation["count_raw"],
            "score": observation["count_score"], "count_box": observation["count_box"],
            "source": "primary",
        }]
        attempts.extend((observation, attempt) for attempt in page_attempts)

    qualified = [(observation, attempt) for observation, attempt in attempts
                 if attempt["count"] is not None and attempt["score"] >= 0.70]
    primary_values = {str(attempt["count"]) for _, attempt in qualified if attempt["source"] == "primary"}
    values = {str(attempt["count"]) for _, attempt in qualified}
    grounded = {value for value in values if not primary_values or any(
        value.startswith(primary) or value.endswith(primary) for primary in primary_values
    )}
    # Admit replacements as well as completions, but do not promote unrelated extra digits.
    candidates = grounded | {value for value in values if any(len(value) == len(other) for other in grounded)}
    candidates = {value for value in candidates if not any(
        len(value) < len(other) and (other.startswith(value) or other.endswith(value))
        for other in grounded
    )}
    votes = {value: set() for value in candidates}
    non_edge_pages = {observation["_source_page"] for observation in non_edge}
    page_values = {}
    page_anchors = {}
    for observation in observations:
        page = observation["_source_page"]
        page_attempts = [attempt for other, attempt in qualified if other is observation]
        page_values[page] = {str(attempt["count"]) for attempt in page_attempts}
        primary = {str(attempt["count"]) for attempt in page_attempts if attempt["source"] == "primary"}
        # A page contributes once; its primary anchors truncation, not a global veto.
        anchors = primary or page_values[page]
        page_anchors[page] = anchors
        for value in candidates:
            if any(value.startswith(anchor) or value.endswith(anchor) for anchor in anchors):
                votes[value].add(page)

    winners = {value for value in candidates if votes[value] and len(votes[value]) == max(map(len, votes.values()))}
    replacement_conflict = len(winners) > 1 and any(len(values & winners) > 1 for values in page_values.values())
    chosen = None
    if winners and not replacement_conflict:
        # Only clean page evidence can use geometry/quality to break an equal-support tie.
        ranked = [(len(votes[str(attempt["count"])] & non_edge_pages), attempt["score"], observation, attempt)
                  for observation, attempt in qualified if str(attempt["count"]) in winners]
        best_rank = max(rank[:2] for rank in ranked)
        best = [rank for rank in ranked if rank[:2] == best_rank]
        if len({rank[3]["count"] for rank in best}) == 1:
            _, _, observation, attempt = best[0]
            chosen = dict(observation)
            chosen.update(
                count=attempt["count"], count_raw=attempt["raw"],
                count_score=attempt["score"], count_box=attempt["count_box"],
            )

    reasons = []
    if chosen is None:
        reasons.append("count_unresolved")
    if replacement_conflict:
        reasons.append("threshold_disagree")
    if len(page_values) > 1 and (chosen is None or any(
        not any(str(chosen["count"]).startswith(value) or str(chosen["count"]).endswith(value)
                for value in values) for values in page_anchors.values() if values
    )):
        reasons.append("page_conflict")
    if not non_edge:
        reasons.append("edge_zone")
    if not reasons:
        return chosen, None

    readings = [{
            "image_index": observation["_source_page"] - 1,
            "count_box": list(attempt["count_box"]),
            "count": attempt["count"], "raw": attempt["raw"],
            "score": round(max(0.0, min(1.0, attempt["score"])), 3),
            "source": attempt["source"],
            **({"threshold": attempt["threshold"]} if "threshold" in attempt else {}),
        } for observation, attempt in attempts]
    return chosen, {
        "entity_type": entity_type, "entity_id": entity_id,
        "chosen": chosen["count"] if chosen is not None else None,
        "reasons": reasons, "readings": readings,
    }


def _resolve_inventory_report_context(
    params: dict,
    snapshot_list: SnapshotList | None,
    upload_settings,
) -> tuple[Any | None, Path]:
    bound_account = None
    account_error: Exception | None = None
    if upload_settings.mode == AUTO_UPLOAD_MODE:
        try:
            bound_account = get_bound_account(upload_settings)
        except Exception as exc:
            account_error = exc

    filename_part = (
        bound_account_report_filename(bound_account)
        if bound_account is not None
        else upload_settings.report_filename
    )
    filename_prefix = (
        "StockReport"
        if snapshot_list is not None
        and upload_settings.mode == LOCAL_ONLY_MODE
        and filename_part not in (None, "")
        else "DailyRewards"
    )
    report_path = resolve_inventory_report_destination(
        params.get("inventory_report_path"),
        filename_part,
        REPO_ROOT,
        filename_prefix=filename_prefix,
    )
    if account_error is not None:
        logger.warning(
            "【广陵库房】 查询 Token 绑定账号失败，继续扫描并保存"
            f"本地报告至 {report_path}；扫描完成后仍会尝试自动上报："
            f"{account_error}"
        )
    return bound_account, report_path


def recognize_baijinbi_count(context: Context, image: np.ndarray) -> int:
    detail = context.run_recognition("背包-白金币识别", image)
    for result in _ocr_results(detail):
        raw = str(result.text).strip()
        if raw.isascii() and raw.isdecimal():
            return int(raw)
    raise ValueError("无法通过背包-白金币识别节点读取顶部白金币数量")


def recognize_stamina_cost(context: Context, image: np.ndarray, params: dict) -> int:
    roi = _parse_rect(
        params.get("stamina_cost_roi", list(DEFAULT_STAMINA_COST_ROI)),
        "stamina_cost_roi",
    )
    if _clip_rect(roi, image) != roi:
        raise ValueError(f"stamina_cost_roi 超出截图范围: {roi}, image={image.shape}")

    crop = _crop(image, roi)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(1, int(params.get("stamina_cost_ocr_scale", 4)))
    prepared = cv2.resize(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    threshold = float(params.get("stamina_cost_ocr_threshold", 0.45))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("stamina_cost_ocr_threshold 必须位于 [0, 1]")
    result = _run_count_ocr(
        context,
        prepared,
        str(params.get("stamina_cost_ocr_model", "")),
        threshold,
    )
    if result is None:
        raise ValueError(f"无法从固定区域 {roi} 识别本次派遣消耗体力")
    digits, _, raw = result
    stamina_cost = int(digits)
    try:
        return validate_stamina_cost(stamina_cost)
    except ValueError as exc:
        raise ValueError(f"派遣消耗体力识别结果无效: {raw!r}; {exc}") from exc


@AgentServer.custom_action("PagedItemRecognition")
class PagedItemRecognition(CustomAction):
    """Scan a scrollable item list, or one local image in debug mode.

    Set ``debug_image_path`` to recognize that image once without taking a
    screenshot, swiping, or requiring bottom-of-list confirmation.
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        try:
            params = _parse_params(argv.custom_action_param)
            if str(params.get("layout_mode", "auto")).lower() != "auto":
                raise ValueError("PagedItemRecognition 只支持 layout_mode: auto")
            entity_type_filter = _parse_entity_type_filter(
                params.get("entity_type_filter")
            )
            snapshot_list_value = params.get("snapshot_list")
            if isinstance(snapshot_list_value, str) and snapshot_list_value.strip().lower() == "tab3-2":
                try:
                    from custom.action.operator_catalog_sync import refresh_operator_catalog

                    refresh_operator_catalog(OPERATORS_PATH)
                except Exception as exc:
                    logger.bind(console=False).warning(f"背包心纸目录刷新失败，使用现有名单：{exc}")
            snapshot_list = _parse_snapshot_list(params.get("snapshot_list"))
            if snapshot_list is not None:
                if (
                    entity_type_filter is not None
                    and entity_type_filter != snapshot_list.entity_type
                ):
                    raise ValueError(
                        "entity_type_filter 与 snapshot_list 的对象类型冲突"
                    )
                entity_type_filter = snapshot_list.entity_type
            stop_on_target_boundary = params.get("stop_on_target_boundary", False)
            if not isinstance(stop_on_target_boundary, bool):
                raise ValueError("stop_on_target_boundary 必须是布尔值")
            if stop_on_target_boundary and snapshot_list is None:
                raise ValueError(
                    "stop_on_target_boundary 必须与 snapshot_list 一起使用"
                )
            acquisition_channel_value = params.get("acquisition_channel", "")
            if not isinstance(acquisition_channel_value, str):
                raise ValueError("acquisition_channel 必须是字符串")
            acquisition_channel = acquisition_channel_value.strip()
            if len(acquisition_channel) > 64:
                raise ValueError("acquisition_channel 不能超过 64 个字符")

            roi = _parse_rect(params.get("roi"), "roi")
            grid_hint = _parse_auto_grid_hint(params.get("grid"))
            index_path = _resolve_path(
                params.get(
                    "index_path", params.get("npz_path", params.get("npx_path"))
                ),
                DEFAULT_INDEX_PATH,
            )
            record_path = _resolve_record_path(
                params.get("record_path", params.get("output_path"))
            )
            index = _load_index(index_path)
            if snapshot_list is not None and snapshot_list.name == "tab3-2":
                index = supplement_xinzhi_index(index, OPERATORS_PATH, snapshot_list.ids)
            debug_image_value = params.get("debug_image_path")
            if debug_image_value not in (None, ""):
                if snapshot_list is not None:
                    raise ValueError("本地图片调试模式不支持 snapshot_list")
                if not isinstance(debug_image_value, str):
                    raise ValueError("debug_image_path 必须是字符串")
                debug_image_path = _resolve_path(debug_image_value, REPO_ROOT)
                image = _load_debug_image(debug_image_path)
                if _clip_rect(roi, image) != roi:
                    raise ValueError(
                        f"roi 超出调试图片范围: {roi}, image={image.shape}"
                    )
                page = scan_page(image, index, roi, grid_hint, params, context)
                page.results = _filter_results(page.results, entity_type_filter)
                if not page.results:
                    logger.error(
                        "【火眼金睛麻圆酱】调试图片未识别到有效条目，"
                        f"image={debug_image_path}, rejected={page.rejected}"
                    )
                    return CustomAction.RunResult(success=False)

                results: list[dict] = []
                for result in page.results:
                    output = dict(result)
                    output["acquisition_channel"] = acquisition_channel
                    results.append(output)
                timestamp = (
                    datetime.now().astimezone().isoformat(timespec="milliseconds")
                )
                invocation_id = uuid.uuid4().hex
                _record_results(
                    record_path,
                    timestamp,
                    invocation_id,
                    "debug_image",
                    index_path,
                    results,
                )
                logger.info(
                    "【火眼金睛麻圆酱】本地图片调试完成，"
                    f"image={debug_image_path}, circles={page.layout['detected_count']}, "
                    f"recognized={len(results)}, 已记录至 {record_path}"
                )
                return CustomAction.RunResult(success=True)

            if not acquisition_channel:
                raise ValueError("正式分页记录必须设置 acquisition_channel")
            if not bool(params.get("recognize_count", True)):
                raise ValueError("库存记录必须启用 recognize_count")
            if not bool(params.get("count_required", True)):
                raise ValueError("库存记录必须启用 count_required")
            upload_settings = read_upload_settings(
                context, TOKEN_REQUIRED_HINT if snapshot_list is not None else DAILY_TOKEN_REQUIRED_HINT,
            )
            bound_account, inventory_report_path = _resolve_inventory_report_context(
                params, snapshot_list, upload_settings
            )
            bag_tab = snapshot_list.name if snapshot_list is not None else ""
            bag_diagnostics = (
                start_bag_diagnostics(context, upload_settings)
                if bag_tab
                else None
            )
            record_type, snapshot_scope = _snapshot_record_options(
                params, snapshot_list
            )
            dispatch_reward = is_dispatch_reward(record_type, acquisition_channel)
            stamina_cost: int | None = None
            scan_started_at = (
                datetime.now().astimezone().isoformat(timespec="milliseconds")
            )

            max_pages = int(params.get("max_pages", 10))
            if max_pages < 2:
                raise ValueError("max_pages 必须至少为 2")
            overlap_threshold = float(params.get("overlap_threshold", 0.90))
            if not math.isfinite(overlap_threshold) or not 0 < overlap_threshold <= 1:
                raise ValueError("overlap_threshold 必须位于 (0, 1]")
            settle_seconds = max(0.0, float(params.get("swipe_wait_ms", 700)) / 1000)
            configured_swipe, swipe_duration = _parse_swipe(params)
            swipe_rows = _parse_swipe_rows(params)
            debug_page_value = params.get("debug_page_dir")
            if "debug_page_dir" not in params and snapshot_list is not None:
                try:
                    node = context.get_node_data("背包调试截图配置")
                    base_dir = node.get("attach", {}).get("debug_page_dir", "")
                    if isinstance(base_dir, str) and base_dir.strip():
                        debug_page_value = str(Path(base_dir.strip()) / snapshot_list.name)
                except Exception:
                    debug_page_value = None
            page_debug_dir = _prepare_page_debug_directory(debug_page_value)
            if page_debug_dir is not None:
                logger.info("【火眼金睛麻圆酱】分页调试输出目录=" f"{page_debug_dir}")

            previous: PageScan | None = None
            previous_row_ids: list[int] = []
            next_row_id = 0
            max_columns = 1
            collected: dict[tuple[int, int], dict] = {}
            count_observations: dict[tuple[int, int], list[dict]] = {}
            uncertain_cells: dict[tuple[int, int], set[str]] = {}
            reached_bottom = False
            reached_target_boundary = False
            seen_snapshot_target = False
            # Optional runtime probe: force the first N candidates through detail, at most 60 per scan.
            detail_probe = params.get("bag_detail_probe", 0)
            if type(detail_probe) is not int or not 0 <= detail_probe <= 60:
                raise ValueError("bag_detail_probe 必须是 0～60 的整数")
            bag_state = {"image_index": 0, "detail_attempts": 0, "detail_successes": 0,
                         "extra_frames": 0, "probe_remaining": detail_probe, "debug_dir": page_debug_dir}

            for page_number in range(max_pages):
                page_started = time.monotonic()
                attempts_before = bag_state["detail_attempts"]
                successes_before = bag_state["detail_successes"]
                extra_before = bag_state["extra_frames"]
                if _should_stop(context):
                    logger.info("【火眼金睛麻圆酱】 任务已停止")
                    return CustomAction.RunResult(success=False)
                image = (_capture_bag_frame(context) if params.get("count_mode") == "bag_consensus"
                         else context.tasker.controller.post_screencap().wait().get())
                if image is None:
                    raise RuntimeError("分页物品识别截图失败")
                if params.get("count_mode") == "bag_consensus":
                    # A fixed swipe delay can still capture the list while it is coasting.
                    stable = 0
                    for _ in range(8):
                        time.sleep(0.2)
                        settled = _capture_bag_frame(context)
                        stable = stable + 1 if _bag_page_unchanged(image, settled, roi, 0.999) else 0
                        image = settled
                        if stable >= 2:
                            break
                    else:
                        raise RuntimeError("背包页面滚动未稳定，停止扫描，已保留原数量")
                if page_debug_dir is not None:
                    _write_debug_image(
                        page_debug_dir / f"page-{page_number + 1:02d}.png",
                        image,
                    )
                if _clip_rect(roi, image) != roi:
                    raise ValueError(f"roi 超出截图范围: {roi}, image={image.shape}")
                page_image_index = _record_bag_image(image, roi, bag_diagnostics, bag_state)
                if page_number == 0 and dispatch_reward:
                    try:
                        stamina_cost = recognize_stamina_cost(context, image, params)
                    except Exception as exc:
                        logger.error(
                            "【广陵库房】派遣奖励停止上报："
                            f"acquisition_channel={acquisition_channel}, "
                            f"stamina_cost=未取得, error={exc}"
                        )
                        return CustomAction.RunResult(success=False)

                page = scan_page(image, index, roi, grid_hint, params, context)
                page.results = _filter_results(page.results, entity_type_filter)
                max_columns = max(max_columns, page.column_count)
                if previous is None:
                    overlap = 0
                    overlap_scores: list[float] = []
                    overlap_candidates: list[dict] = []
                    row_ids = list(range(len(page.row_features)))
                    next_row_id = len(row_ids)
                else:
                    overlap_candidates = (
                        overlap_candidate_scores(
                            previous.row_features,
                            page.row_features,
                            previous.row_entities,
                            page.row_entities,
                            previous.row_counts,
                            page.row_counts,
                        )
                        if page_debug_dir is not None
                        else []
                    )
                    overlap, overlap_scores = find_row_overlap(
                        previous.row_features,
                        page.row_features,
                        overlap_threshold,
                        previous.row_entities,
                        page.row_entities,
                        previous.row_counts,
                        page.row_counts,
                    )
                    row_ids = previous_row_ids[-overlap:] if overlap else []
                    new_row_count = len(page.row_features) - overlap
                    row_ids.extend(range(next_row_id, next_row_id + new_row_count))
                    next_row_id += new_row_count

                if snapshot_list is not None:
                    for rejected in page.rejected:
                        key = (row_ids[int(rejected["row"])], int(rejected["column"]))
                        uncertain_cells.setdefault(key, set()).update(
                            _uncertain_snapshot_ids(rejected, index, params)
                        )
                page_count_candidates = []
                for result in page.results:
                    page_row = int(result["row"])
                    virtual_row = row_ids[page_row]
                    column = int(result["column"])
                    candidate = dict(result)
                    candidate["row"] = virtual_row
                    candidate["_source_page"] = page_number + 1
                    candidate["_source_page_row"] = page_row
                    candidate["_image_index"] = page_image_index
                    key = (virtual_row, column)
                    if snapshot_list is not None:
                        count_observations.setdefault(key, []).append(candidate)
                        page_count_candidates.append((key, candidate))
                    current = collected.get(key)
                    if current is None or _prefer_result(candidate, current):
                        collected[key] = candidate

                if snapshot_list is not None and params.get("count_mode") == "bag_consensus":
                    bag_state["page_frame"] = image
                    for rejected in page.rejected:
                        if (rejected.get("reason") != "count-not-recognized"
                                or rejected.get("entity_type") != entity_type_filter):
                            continue
                        page_row = int(rejected["row"])
                        key = (row_ids[page_row], int(rejected["column"]))
                        candidate = dict(rejected)
                        candidate.update(count=None, row=key[0], _source_page=page_number + 1,
                                         _source_page_row=page_row, _image_index=page_image_index)
                        count_observations.setdefault(key, []).append(candidate)
                        page_count_candidates.append((key, candidate))
                    _fallback_bag_counts(
                        context, page_count_candidates, count_observations, index,
                        roi, params, bag_diagnostics, bag_state,
                    )

                if params.get("count_mode") == "bag_consensus":
                    logger.info(
                        f"【广陵库房】页统计 tab={bag_tab} page={page_number + 1} "
                        f"seconds={time.monotonic() - page_started:.3f} cells={len(page_count_candidates)} "
                        f"vote_direct={len(page.results)} "
                        f"detail_attempts={bag_state['detail_attempts'] - attempts_before} "
                        f"detail_successes={bag_state['detail_successes'] - successes_before} "
                        f"extra_frames={bag_state['extra_frames'] - extra_before}"
                    )
                # logger.info(
                #     "【火眼金睛麻圆酱】"
                #     f"page={page_number + 1}, circles={page.layout['detected_count']}, "
                #     f"recognized={len(page.results)}, overlap={overlap}, "
                #     f"overlap_scores={[round(score, 3) for score in overlap_scores]}"
                # )
                if page_debug_dir is not None:
                    _write_debug_json(
                        page_debug_dir / f"page-{page_number + 1:02d}.json",
                        {
                            "page": page_number + 1,
                            "layout": page.layout,
                            "virtual_row_ids": row_ids,
                            "overlap": overlap,
                            "overlap_scores": overlap_scores,
                            "overlap_candidates": overlap_candidates,
                            "results": page.results,
                            "rejected": page.rejected,
                        },
                    )
                    # if previous is not None and overlap == 0:
                    #     # logger.info(
                    #     #     "【火眼金睛麻圆酱】overlap=0 调试候选="
                    #     #     f"{overlap_candidates}"
                    #     # )

                if stop_on_target_boundary and snapshot_list is not None:
                    reached_target_boundary, seen_snapshot_target = (
                        _has_snapshot_target_boundary(
                            page, snapshot_list, seen_snapshot_target
                        )
                    )
                    if reached_target_boundary:
                        # logger.info(
                        #     "【火眼金睛麻圆酱】"
                        #     f"snapshot_list={snapshot_list.name} 已发现完整的下一行"
                        #     "非目标格子，在当前位置结束扫描"
                        # )
                        break

                if previous is not None and overlap == len(page.row_features):
                    reached_bottom = True
                    break
                if page_number + 1 >= max_pages:
                    break

                previous = page
                previous_row_ids = row_ids
                swipe = configured_swipe or automatic_swipe(
                    page.layout, roi, swipe_duration, swipe_rows
                )
                # logger.info(f"【火眼金睛麻圆酱】swipe={swipe}")
                swipe_job = context.tasker.controller.post_swipe(*swipe).wait()
                if params.get("count_mode") == "bag_consensus" and not swipe_job.succeeded:
                    raise RuntimeError("背包翻页操作失败，停止扫描，已保留原数量")
                if settle_seconds:
                    time.sleep(settle_seconds)

            if not reached_bottom and not reached_target_boundary:
                logger.error(
                    f"【火眼金睛麻圆酱】已经翻了 {max_pages} 页，但仍未确认列表底部，"
                    "本次不写入报告"
                )
                return CustomAction.RunResult(success=False)

            bag_cells: list[dict] = []
            skipped_names: list[str] = []
            for key, observations in count_observations.items():
                chosen, suspect = _resolve_bag_count(observations, roi, params)
                if chosen is None:
                    current = collected.pop(key, observations[0])
                    entity_id = _result_stable_id(current, current["entity_type"])
                    name = current.get("item_name") or current.get("operator_name") or entity_id
                    skipped_names.append(name)
                    uncertain_cells.setdefault(key, set()).add(entity_id)
                    reason = current.get("_count_failure", "数量读数冲突或未确认")
                    logger.warning(f"【广陵库房】{name} {reason}，补充证据仍不足，本次跳过，已保留原数量")
                else:
                    collected[key] = chosen
                if suspect is not None:
                    bag_cells.append(suspect)
                    logger.bind(console=False).info(
                        f"【广陵库房】数量可疑 id={suspect['entity_id']} "
                        f"readings={suspect['readings']} reasons={suspect['reasons']} chosen={suspect['chosen']}"
                    )
            if bag_cells or skipped_names:
                skipped = "、".join(dict.fromkeys(skipped_names)) or "无"
                logger.info(f"【广陵库房】数量可疑诊断 {len(bag_cells)} 格，本次跳过：{skipped}；已保留原数量")

            results: list[dict] = []
            for (row, column), result in sorted(collected.items()):
                output = dict(result)
                output["slot"] = row * max_columns + column
                output["acquisition_channel"] = acquisition_channel
                results.append(output)
            if page_debug_dir is not None:
                _write_debug_json(
                    page_debug_dir / "collected.json",
                    {"results": results, "bag_cells": bag_cells},
                )
            recognized_count = len(results)
            if snapshot_list is not None:
                if not any(_result_stable_id(result, snapshot_list.entity_type) in snapshot_list.ids
                           for result in results):
                    raise RuntimeError("背包未确认任何目标格子，拒绝将整份库存补零，已保留原数量")
                results, ignored_ids, recognized_count = _apply_snapshot_list(
                    results,
                    snapshot_list,
                    index,
                    set().union(
                        *(
                            ids
                            for key, ids in uncertain_cells.items()
                            if key not in collected
                        )
                    ),
                    count_skipped_ids={
                        cell["entity_id"] for cell in bag_cells if cell["chosen"] is None
                    },
                )
                # if ignored_ids:
                #     logger.warning(
                #         f"【火眼金睛麻圆酱】snapshot_list={snapshot_list.name} "
                #         f"忽略 {len(ignored_ids)} 个范围外识别结果: "
                #         f"{', '.join(ignored_ids)}"
                #     )
                if snapshot_list.name == "tab1":
                    try:
                        baijinbi_count = recognize_baijinbi_count(context, image)
                    except ValueError as exc:
                        logger.warning(f"【广陵库房】{exc}；本次只记录符传和鸟食库存")
                    else:
                        results.append(
                            {
                                "entity_type": "item",
                                "item_id": "baijinbi",
                                "item_name": "白金币",
                                "count": baijinbi_count,
                                "acquisition_channel": acquisition_channel,
                            }
                        )
            if not results:
                logger.info(
                    "【火眼金睛麻圆酱】已完成扫描，但没有可上报条目，"
                    "本次不写入库存报告或上传库存"
                )
                if not _should_stop(context):
                    diagnostic_call(bag_diagnostics, "finish_bag", results, index_path, bag_tab, bag_cells,
                                    scan_complete=reached_bottom)
                return CustomAction.RunResult(success=True)

            exported_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
            invocation_id = uuid.uuid4().hex
            document = build_exchange_document(
                results,
                invocation_id,
                scan_started_at,
                exported_at,
                acquisition_channel,
                record_type,
                snapshot_scope,
                bound_account.id if bound_account is not None else None,
                stamina_cost,
            )
            initial_status = (
                "等待自动上报"
                if upload_settings.mode == AUTO_UPLOAD_MODE
                else "仅保存到本地"
            )
            append_inventory_report(inventory_report_path, document, initial_status)
            record_ids = [record["record_id"] for record in document["records"]]

            if upload_settings.mode == LOCAL_ONLY_MODE:
                logger.info(
                    f"【火眼金睛麻圆酱】已完成，共 {next_row_id} 个物理行、"
                    f"{len(results)} 个目标，仅保存至 {inventory_report_path}"
                )
                return CustomAction.RunResult(success=True)

            upload_result = upload_inventory_document(document, upload_settings)
            final_status = (
                f"自动上报成功（{upload_result.message}）"
                if upload_result.success
                else f"自动上报失败（{upload_result.message}）"
            )
            try:
                update_upload_status(inventory_report_path, record_ids, final_status)
            except Exception as exc:
                logger.warning(
                    "【广陵库房】无法更新库存 TXT 上报状态，"
                    f"原始识别记录仍然有效: {exc}"
                )

            if upload_result.success:
                logger.info(
                    "【广陵库房】库存记录已自动上报，"
                    f"并保存至 {inventory_report_path}"
                )
            else:
                record_context = "; ".join(
                    "record_id={record_id}, acquisition_channel={channel}, "
                    "stamina_cost={stamina}".format(
                        record_id=record.get("record_id", ""),
                        channel=record.get("acquisition_channel", ""),
                        stamina=record.get("stamina_cost", "<omitted>"),
                    )
                    for record in document["records"]
                )
                logger.warning(
                    "【广陵库房】自动上报失败，"
                    f"{record_context}, response={upload_result.message}。"
                    "识别结果已保存至 "
                    f"{inventory_report_path}。{inventory_recovery_hint(upload_result)}"
                )
            if not _should_stop(context):
                diagnostic_call(bag_diagnostics, "finish_bag", results, index_path, bag_tab, bag_cells,
                                scan_complete=reached_bottom)
            return CustomAction.RunResult(success=True)
        except Exception as exc:
            logger.exception(f"PagedItemRecognition 失败: {exc}")
            return CustomAction.RunResult(success=False)
