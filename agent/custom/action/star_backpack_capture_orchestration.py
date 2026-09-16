"""Production three-section star-backpack capture orchestration.

This module deliberately owns only sequence, tab switching, and CaptureBatch
assembly.  Pixel motion and semantic-overlap decisions remain in the already
verified ``StarBackpackCaptureProbe`` continuous-capture primitive.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from custom.action.star_backpack_capture_probe import (
    StarBackpackCaptureProbe,
    _parse_int,
    _parse_params,
    _prepare_run_directory,
    _write_json,
    _write_png,
    parse_capture_probe_params,
)
from custom.action.star_capture_transport import upload_full_capture_batch
from utils import logger


def parse_star_backpack_capture_orchestration_params(raw: Any) -> dict[str, Any]:
    """Parse the B5 production entry without enabling B4 transport."""
    raw_params = _parse_params(raw)
    if raw_params.get("mode") != "continuous_capture":
        raise ValueError("正式星石背包采集必须使用 continuous_capture")
    parsed = parse_capture_probe_params(raw_params)
    for field in ("support_tab_task", "experience_tab_task"):
        value = raw_params.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} 必须是非空 pipeline 节点名")
        parsed[field] = value.strip()
    parsed["tab_settle_ms"] = _parse_int(
        raw_params.get("tab_settle_ms"), "tab_settle_ms", minimum=0
    )
    return parsed


def _task_succeeded(detail: Any) -> bool:
    if isinstance(detail, bool):
        return detail
    status = getattr(detail, "status", None)
    return bool(status and getattr(status, "succeeded", False))


def _switch_tab(context: Context, task_name: str, settle_ms: int) -> None:
    if not _task_succeeded(context.run_task(task_name)):
        raise RuntimeError(f"星石背包切换 Tab 失败: {task_name}")
    if settle_ms:
        time.sleep(settle_ms / 1000)


def _section_payload(
    section: str,
    session: dict[str, Any],
    capture_id: str,
    source_order: int,
) -> tuple[dict[str, Any], int]:
    image_names = session["retained_images"]
    source_id_by_name = {
        image_name: f"{capture_id}:{section}:{index:03d}"
        for index, image_name in enumerate(image_names)
    }
    images = []
    for index, image_name in enumerate(image_names):
        images.append(
            {
                "sourceImageId": source_id_by_name[image_name],
                "sourceOrder": source_order + index,
                # MaaYuan retains PNGs locally.  The later transport/import
                # boundary supplies File objects and is intentionally out of B5.
                "fileName": f"{section}/{image_name}",
            }
        )
    relations = [
        {
            "previousSourceImageId": source_id_by_name[relation["previous_image"]],
            "currentSourceImageId": source_id_by_name[relation["current_image"]],
            "relation": "overlap",
        }
        for relation in session["adjacent_relations"]
    ]
    return (
        {
            "images": images,
            "adjacentRelations": relations,
            "complete": True,
            "stopReason": "single_capture" if section == "experience" else "bottom_no_move",
        },
        source_order + len(images),
    )


def assemble_capture_batch(
    run_dir: Path,
    game_version: str,
    main: dict[str, Any],
    support: dict[str, Any],
    experience: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the local, file-name based representation of CaptureBatchV1."""
    if not main["success"] or not support["success"]:
        raise ValueError("未完成的主星或辅星采集不能组装完整 CaptureBatchV1")
    capture_id = "star-" + uuid.uuid5(uuid.NAMESPACE_URL, run_dir.resolve().as_uri()).hex
    next_order = 1
    sections: dict[str, dict[str, Any]] = {}
    for section, session in (("main", main), ("support", support), ("experience", experience)):
        sections[section], next_order = _section_payload(
            section, session, capture_id, next_order
        )
    return {
        "schemaVersion": 1,
        "captureId": capture_id,
        "source": "maayuan",
        "gameVersion": game_version,
        "sections": sections,
    }


@AgentServer.custom_action("StarBackpackCaptureOrchestration")
class StarBackpackCaptureOrchestration(CustomAction):
    """Run main → support → experience capture with no OCR or transport."""

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        try:
            params = parse_star_backpack_capture_orchestration_params(
                argv.custom_action_param
            )
            run_dir = _prepare_run_directory(params["debug_dir"])
            probe = StarBackpackCaptureProbe()

            (run_dir / "main").mkdir(parents=True, exist_ok=False)
            main = probe._run_continuous_capture(
                context, params, run_dir / "main", probe._capture(context)
            )
            if not main["success"]:
                logger.warning("星石背包主星未完整采集，停止后续 Tab 切换。")
                return CustomAction.RunResult(success=False)

            _switch_tab(context, params["support_tab_task"], params["tab_settle_ms"])
            (run_dir / "support").mkdir(parents=True, exist_ok=False)
            support = probe._run_continuous_capture(
                context, params, run_dir / "support", probe._capture(context)
            )
            if not support["success"]:
                logger.warning("星石背包辅星未完整采集，停止经验星曜截图。")
                return CustomAction.RunResult(success=False)

            _switch_tab(context, params["experience_tab_task"], params["tab_settle_ms"])
            experience_dir = run_dir / "experience"
            experience_dir.mkdir(parents=True, exist_ok=False)
            experience_image = probe._capture(context)
            experience_name = "capture-00.png"
            _write_png(experience_dir / experience_name, experience_image)
            experience = {
                "success": True,
                "stop_reason": "single_capture",
                "retained_images": [experience_name],
                "retained_image_count": 1,
                "adjacent_relations": [],
                "transition_count": 0,
                "failed_transition": None,
            }
            _write_json(experience_dir / "session.json", experience)

            batch = assemble_capture_batch(
                run_dir, params["game_version"], main, support, experience
            )
            _write_json(run_dir / "capture-batch.json", batch)
            upload_result = upload_full_capture_batch(context, run_dir)
            logger.info(
                "星石背包正式三段采集完成: "
                f"main={main['retained_image_count']}, "
                f"support={support['retained_image_count']}, experience=1, dir={run_dir}, "
                f"transport={'local-only' if upload_result is None else upload_result.message}"
            )
            return CustomAction.RunResult(success=True)
        except Exception as exc:
            logger.exception(f"星石背包正式三段采集失败: {exc}")
            return CustomAction.RunResult(success=False)
