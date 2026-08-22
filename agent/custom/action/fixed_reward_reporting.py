from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from custom.action.inventory_reporting import (
    AUTO_UPLOAD_MODE,
    LOCAL_ONLY_MODE,
    append_inventory_report,
    build_exchange_document,
    read_upload_settings,
    update_upload_status,
    upload_inventory_document,
)
from custom.action.paged_item_recognition import (
    _parse_params,
    _resolve_inventory_report_context,
)
from custom.reco.agent_item import REPO_ROOT
from utils import logger


DEFAULT_CONFIG_NODE = "历练奖励配置"
ITEMS_PATH = REPO_ROOT / "agent" / "items.json"


@dataclass(frozen=True)
class FixedRewardPlan:
    acquisition_channel: str
    sweep_count: int
    rewards: tuple[tuple[str, int], ...]


@lru_cache(maxsize=1)
def _item_names(path: Path = ITEMS_PATH) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"道具目录格式无效: {path}")
    names = {
        str(item.get("id", "")).strip(): str(item.get("name", "")).strip()
        for item in items
        if isinstance(item, dict)
    }
    if "" in names:
        del names[""]
    return names


def parse_fixed_reward_plan(attach: Any) -> FixedRewardPlan:
    if not isinstance(attach, dict):
        raise ValueError("历练奖励配置缺少 attach 对象")

    acquisition_channel = str(attach.get("acquisition_channel", "")).strip()
    if not acquisition_channel:
        raise ValueError("历练奖励配置缺少 acquisition_channel")
    if len(acquisition_channel) > 64:
        raise ValueError("acquisition_channel 不能超过 64 个字符")

    raw_sweep_count = attach.get("sweep_count")
    if isinstance(raw_sweep_count, bool):
        raise ValueError("sweep_count 必须是 1 至 6 的整数")
    try:
        sweep_count = int(raw_sweep_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("sweep_count 必须是 1 至 6 的整数") from exc
    if sweep_count != raw_sweep_count or sweep_count not in range(1, 7):
        raise ValueError("sweep_count 必须是 1 至 6 的整数")

    item_ids = attach.get("reward_item_ids")
    counts = attach.get("reward_counts")
    if not isinstance(item_ids, list) or not item_ids:
        raise ValueError("reward_item_ids 必须是非空数组")
    if not isinstance(counts, list) or len(counts) != len(item_ids):
        raise ValueError("reward_counts 必须与 reward_item_ids 等长")

    rewards: list[tuple[str, int]] = []
    seen: set[str] = set()
    for position, (raw_item_id, raw_count) in enumerate(zip(item_ids, counts)):
        item_id = str(raw_item_id).strip()
        if not item_id or item_id in seen:
            raise ValueError(f"reward_item_ids[{position}] 为空或重复")
        seen.add(item_id)
        if isinstance(raw_count, bool):
            raise ValueError(f"reward_counts[{position}] 必须是非负整数")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"reward_counts[{position}] 必须是非负整数"
            ) from exc
        if count != raw_count or count < 0:
            raise ValueError(f"reward_counts[{position}] 必须是非负整数")
        if count > 0:
            rewards.append((item_id, count))
    if not rewards:
        raise ValueError("reward_counts 至少需要一个正数")
    return FixedRewardPlan(acquisition_channel, sweep_count, tuple(rewards))


def build_fixed_reward_results(
    plan: FixedRewardPlan,
    item_names: dict[str, str],
) -> list[dict]:
    unknown = [item_id for item_id, _ in plan.rewards if item_id not in item_names]
    if unknown:
        raise ValueError("历练奖励包含未知道具: " + ", ".join(unknown))
    return [
        {
            "entity_type": "item",
            "item_id": item_id,
            "item_name": item_names[item_id],
            "count": count_per_sweep * plan.sweep_count,
        }
        for item_id, count_per_sweep in plan.rewards
    ]


def _save_and_upload(
    context: Context,
    config: dict,
    plan: FixedRewardPlan,
    results: list[dict],
) -> Path:
    upload_settings = read_upload_settings(context)
    bound_account, report_path = _resolve_inventory_report_context(
        config,
        None,
        upload_settings,
    )
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    document = build_exchange_document(
        results,
        uuid.uuid4().hex,
        timestamp,
        timestamp,
        plan.acquisition_channel,
        "reward_delta",
        None,
        bound_account.id if bound_account is not None else None,
        None,
    )
    initial_status = (
        "等待自动上报"
        if upload_settings.mode == AUTO_UPLOAD_MODE
        else "仅保存到本地"
    )
    append_inventory_report(report_path, document, initial_status)
    if upload_settings.mode == LOCAL_ONLY_MODE:
        return report_path

    upload_result = upload_inventory_document(document, upload_settings)
    record_ids = [record["record_id"] for record in document["records"]]
    final_status = (
        f"自动上报成功（{upload_result.message}）"
        if upload_result.success
        else f"自动上报失败（{upload_result.message}）"
    )
    try:
        update_upload_status(report_path, record_ids, final_status)
    except Exception as exc:
        logger.warning(
            "【广陵库房】无法更新历练奖励 TXT 上报状态，"
            f"原始记录仍然有效: {exc}"
        )
    if not upload_result.success:
        logger.warning(
            "【广陵库房】历练奖励自动上报失败，"
            f"response={upload_result.message}；记录已保存至 {report_path}"
        )
    return report_path


@AgentServer.custom_action("FixedRewardReporting")
class FixedRewardReporting(CustomAction):
    """Report rewards whose item IDs and counts are fixed by the task chain."""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        try:
            params = _parse_params(argv.custom_action_param)
            config_node = str(
                params.get("config_node", DEFAULT_CONFIG_NODE)
            ).strip()
            if not config_node:
                raise ValueError("config_node 不能为空")
            config = context.get_node_data(config_node)
            if not isinstance(config, dict):
                raise ValueError(f"未找到固定奖励配置节点: {config_node}")
            attach = config.get("attach")
            plan = parse_fixed_reward_plan(attach)
            results = build_fixed_reward_results(plan, _item_names())
            report_path = _save_and_upload(context, attach, plan, results)
            logger.info(
                "【广陵库房】历练扫荡奖励已记录："
                f"channel={plan.acquisition_channel}, sweeps={plan.sweep_count}, "
                f"items={len(results)}, report={report_path}"
            )
            return CustomAction.RunResult(success=True)
        except Exception as exc:
            logger.exception(f"FixedRewardReporting 失败: {exc}")
            return CustomAction.RunResult(success=False)
