"""Only dispatches to caller-provided action nodes after stable recognition."""
import json
import re
from pathlib import Path
import time
import unicodedata

import cv2

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from utils import logger

from .dragon_smart_vision import DragonVision

VERSION = "2.4.1"


def describe_state(result):
    """User-facing state; internal branch names are not executed actions."""
    if result["vein"] == "outside":
        energy = {True: "能量足够", False: "能量不足", None: "能量未确认"}[result["skill_ready"]]
        return f"龙脉外，{energy}"
    if result["vein"] == "inside":
        qi = result["qi"]
        return f"龙脉内，{qi}龙气" if qi is not None else "龙脉内，龙气未确认"
    return "龙脉状态未确认"


def parse_round_text(value):
    """Read current round, optionally followed by a round limit or separator."""
    text = unicodedata.normalize("NFKC", str(value))
    match = re.fullmatch(r"\s*([0-9]{1,3})\s*(?:/\s*([0-9]{1,3})|[/|\\])?\s*", text)
    if not match or int(match[1]) < 1:
        return None
    if match[2] is not None and int(match[2]) < int(match[1]):
        return None
    return str(int(match[1]))


def decode_params(value):
    """Accept SDK JSON and the extra string encoding used by older clients."""
    for _ in range(3):
        if not isinstance(value, str):
            break
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("智能圈参数必须是对象")
    return value


@AgentServer.custom_action("DragonSmartAction")
class DragonSmartAction(CustomAction):
    def run(self, context, argv):
        phase = "读取作业参数"
        try:
            params = decode_params(argv.custom_action_param)
            node = params["node"]
            if not isinstance(node, str) or not node:
                raise ValueError("智能圈入口名称无效")
            match = re.match(r"^回合([0-9]+)行动", node)
            expected_round = str(int(match[1])) if match else None
            position = params.get("position", 1)
            if type(position) is not int or position not in range(1, 6):
                raise ValueError("智能圈位置必须是1至5的整数")
            branches = params["branches"]
            if not isinstance(branches, dict) or set(branches) != {"normal", "ultimate", "defense"}:
                raise ValueError("智能圈必须提供普攻、技能、下拉三个分支")
            if any(not isinstance(n, str) or not n or not context.get_node_data(n) for n in branches.values()):
                raise ValueError("智能圈分支指向不存在的作业步骤")
            if not context.get_node_data(node):
                raise ValueError("智能圈入口步骤不存在")
            logger.info(f"诸葛亮智能圈 v{VERSION}：{node}，检查{position}号位状态")
            # Resolve assets in the deployed bundle, with a source-tree fallback
            # for offline development. No installed files are changed here.
            root = Path(__file__).resolve().parents[3]
            resource = root / "resource/base"
            if not (resource / "dragon_smart.json").is_file():
                resource = root / "assets/resource/base"
            phase = "加载识别资源"
            vision = DragonVision(resource)
            phase = "识别战斗状态"
            previous = None
            previous_image = None
            last_reason = "尚无有效采样"
            for attempt in range(6):
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)
                image = context.tasker.controller.post_screencap().wait().get()
                result = vision.inspect(image, position)
                # Confirm this is still a battle before interpreting absence of
                # the dragon background. Menus and victory screens must not cast.
                battle = context.run_recognition("战斗回合数识别", image)
                best = getattr(battle, "best_result", None)
                round_raw = str(getattr(best, "text", "")).strip()
                round_text = parse_round_text(round_raw)
                round_valid = bool(getattr(battle, "hit", False)) and round_text is not None
                round_valid = round_valid and (expected_round is None or round_text == expected_round)
                # Some tables explicitly map ALL inside states to the same
                # node. In that case qi is irrelevant, just like energy inside.
                same_inside_target = result["vein"] == "inside" and branches["normal"] == branches["defense"]
                target = branches["normal"] if same_inside_target else branches.get(result["action"])
                relevant_state = (min(result["qi"], 2) if result["qi"] is not None else None) if result["vein"] == "inside" else result["skill_ready"]
                signature = (round_text, result["vein"], None if same_inside_target else relevant_state, target)
                normalized = cv2.resize(image, (720, 1280))
                # Compare exposed bottom background only, not the animated boss
                # or sleeve in the left-side recognition region.
                stable = previous_image is not None and vision.stable_background(normalized, previous_image)
                if not getattr(battle, "hit", False) or round_text is None:
                    last_reason = f"回合OCR未确认，原文={round_raw!r}"
                elif expected_round is not None and round_text != expected_round:
                    last_reason = f"回合不符，识别={round_text}，节点预期={expected_round}"
                elif result["vein"] == "unknown":
                    last_reason = "龙脉状态未确认"
                elif not target:
                    last_reason = "龙气数量未确认" if result["vein"] == "inside" else "技能能量未确认"
                elif signature != previous:
                    last_reason = "尚未获得连续两次相同的状态判断"
                elif not stable:
                    last_reason = "底部背景仍在变化"
                else:
                    last_reason = "状态已确认"
                logger.debug(f"DragonSmartAction position={position} observation={result} round_raw={round_raw!r} round={round_text} stable={stable} check={last_reason}")
                # A stop may arrive during screenshot capture or OCR.
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)
                if round_valid and target and signature == previous and stable:
                    # Replace next afresh on EVERY invocation. No mutation cache
                    # and no gestures during recognition or retries.
                    if not context.override_next(node, [target]):
                        raise RuntimeError("无法切换智能圈动作分支")
                    # Report the mapped node, not the default policy's label:
                    # a table may map 'defense' to normal or a refinement node.
                    logger.info(f"诸葛亮智能圈：{describe_state(result)}；转入「{target}」")
                    return CustomAction.RunResult(success=True)
                if attempt == 2:
                    logger.info(f"诸葛亮智能圈：{last_reason}，正在重试")
                # A missed OCR hit must never count as one of two valid frames,
                # even if its stale best_result happens to contain the right text.
                previous = signature if round_valid and target else None
                previous_image = normalized
                time.sleep(0.2)
            logger.error(f"诸葛亮智能圈：{last_reason}；已停止，未执行圈动作")
        except Exception:
            logger.error(f"诸葛亮智能圈：{phase}时发生异常，已停止；详细原因见调试日志")
            if hasattr(logger, "opt"):
                logger.opt(exception=True).debug(f"DragonSmartAction execution error phase={phase}")
            else:
                logger.debug(f"DragonSmartAction execution error phase={phase}", exc_info=True)
        return CustomAction.RunResult(success=False)
