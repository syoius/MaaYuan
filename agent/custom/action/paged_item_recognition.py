from __future__ import annotations

import json
import math
import time
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

from custom.reco.agent_item import (
    AgentIndex,
    DEFAULT_INDEX_PATH,
    REPO_ROOT,
    LayoutCell,
    _clip_rect,
    _crop,
    _load_index,
    _parse_auto_grid_hint,
    _parse_rect,
    _record_results,
    _resolve_path,
    _resolve_record_path,
    _run_count_ocr,
    auto_recognition_params,
    detect_auto_layout,
    recognize_count,
    recognize_item_grid,
)
from custom.action.inventory_reporting import (
    AUTO_UPLOAD_MODE,
    LOCAL_ONLY_MODE,
    append_inventory_report,
    bound_account_report_filename,
    build_exchange_document,
    get_bound_account,
    is_dispatch_reward,
    parse_record_options,
    read_upload_settings,
    resolve_inventory_report_destination,
    update_upload_status,
    upload_inventory_document,
    validate_stamina_cost,
)
from utils import logger

OPERATORS_PATH = REPO_ROOT / "agent" / "operators.json"
DEFAULT_STAMINA_COST_ROI = (510, 375, 50, 48)
BAIJINBI_COUNT_ROI = (414, 55, 92, 40)
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
        data = json.loads(path.read_text(encoding="utf-8"))
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
        if operator_id not in SP_OPERATOR_IDS:
            operator_ids.append(operator_id)
    if not operator_ids:
        raise ValueError("tab3-2 密探目录排除 SP 后为空")
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
    results: list[dict], snapshot_list: SnapshotList, index: AgentIndex
) -> tuple[list[dict], list[str], int]:
    index_entries = _index_entries(index)
    missing = [
        entity_id
        for entity_id in snapshot_list.ids
        if (snapshot_list.entity_type, entity_id) not in index_entries
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            f"snapshot_list {snapshot_list.name} 有 {len(missing)} 个 ID 不在当前索引中: "
            f"{preview}{suffix}；请先更新对应 NPZ"
        )

    target_ids = set(snapshot_list.ids)
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

    for entity_id in snapshot_list.ids:
        if entity_id not in recognized:
            completed.append(
                dict(index_entries[(snapshot_list.entity_type, entity_id)])
            )
    return completed, ignored, len(recognized)


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
    for result in results:
        entity_type = str(result.get("entity_type", ""))
        id_key = "operator_id" if entity_type == "agent" else "item_id"
        entity_id = str(result.get(id_key, "")).strip()
        if entity_id:
            row = int(result["row"])
            column = int(result["column"])
            row_entities[row][column] = f"{entity_type}:{entity_id}"
            row_counts[row][column] = int(result["count"])
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


def recognize_baijinbi_count(image: np.ndarray) -> int:
    # The fixed top-bar crop excludes the currency icon and the purchase button.
    count, score, raw, box = recognize_count(
        None,
        image,
        (0.0, 0.0),
        {
            "count_box": list(BAIJINBI_COUNT_ROI),
            "count_binary_threshold": 165,
        },
    )
    if count is None:
        raise ValueError(
            f"无法识别顶部白金币数量: raw={raw!r}, score={score:.4f}, box={box}"
        )
    return count


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
            upload_settings = read_upload_settings(context)
            bound_account, inventory_report_path = _resolve_inventory_report_context(
                params, snapshot_list, upload_settings
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
            page_debug_dir = _prepare_page_debug_directory(params.get("debug_page_dir"))
            if page_debug_dir is not None:
                logger.info("【火眼金睛麻圆酱】分页调试输出目录=" f"{page_debug_dir}")

            previous: PageScan | None = None
            previous_row_ids: list[int] = []
            next_row_id = 0
            max_columns = 1
            collected: dict[tuple[int, int], dict] = {}
            reached_bottom = False
            reached_target_boundary = False
            seen_snapshot_target = False

            for page_number in range(max_pages):
                if _should_stop(context):
                    logger.info("【火眼金睛麻圆酱】 任务已停止")
                    return CustomAction.RunResult(success=False)
                image = context.tasker.controller.post_screencap().wait().get()
                if image is None:
                    raise RuntimeError("分页物品识别截图失败")
                if page_debug_dir is not None:
                    _write_debug_image(
                        page_debug_dir / f"page-{page_number + 1:02d}.png",
                        image,
                    )
                if _clip_rect(roi, image) != roi:
                    raise ValueError(f"roi 超出截图范围: {roi}, image={image.shape}")
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

                for result in page.results:
                    page_row = int(result["row"])
                    virtual_row = row_ids[page_row]
                    column = int(result["column"])
                    candidate = dict(result)
                    candidate["row"] = virtual_row
                    candidate["_source_page"] = page_number + 1
                    candidate["_source_page_row"] = page_row
                    key = (virtual_row, column)
                    current = collected.get(key)
                    if current is None or _prefer_result(candidate, current):
                        collected[key] = candidate

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
                context.tasker.controller.post_swipe(*swipe).wait()
                if settle_seconds:
                    time.sleep(settle_seconds)

            if not reached_bottom and not reached_target_boundary:
                logger.error(
                    f"【火眼金睛麻圆酱】已经翻了 {max_pages} 页，但仍未确认列表底部，"
                    "本次不写入报告"
                )
                return CustomAction.RunResult(success=False)

            results: list[dict] = []
            for (row, column), result in sorted(collected.items()):
                output = dict(result)
                output["slot"] = row * max_columns + column
                output["acquisition_channel"] = acquisition_channel
                results.append(output)
            if page_debug_dir is not None:
                _write_debug_json(
                    page_debug_dir / "collected.json",
                    {"results": results},
                )
            recognized_count = len(results)
            if snapshot_list is not None:
                results, ignored_ids, recognized_count = _apply_snapshot_list(
                    results, snapshot_list, index
                )
                # if ignored_ids:
                #     logger.warning(
                #         f"【火眼金睛麻圆酱】snapshot_list={snapshot_list.name} "
                #         f"忽略 {len(ignored_ids)} 个范围外识别结果: "
                #         f"{', '.join(ignored_ids)}"
                #     )
                missing_count = len(results) - recognized_count
                # logger.info(
                #     f"【火眼金睛麻圆酱】snapshot_list={snapshot_list.name}, "
                #     f"预设={len(results)}, 识别={recognized_count}, 补零={missing_count}"
                # )
                if snapshot_list.name == "tab1":
                    try:
                        baijinbi_count = recognize_baijinbi_count(image)
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
                    "【火眼金睛麻圆酱】已完成扫描，但没有发现有价值的道具，"
                    "本次不写入报告"
                )
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
                    f"{inventory_report_path}，可稍后手动补传"
                )
            return CustomAction.RunResult(success=True)
        except Exception as exc:
            logger.exception(f"PagedItemRecognition 失败: {exc}")
            return CustomAction.RunResult(success=False)
