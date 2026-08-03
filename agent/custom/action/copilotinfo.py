import json
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.library import *
from utils import logger

# 缓存每个节点的原始 next（首次进入时从未被 override 的干净状态读取）
_original_next_cache = {}

# 重开目标节点，全部为此即代表 next 已被 override 污染
_RESTART_TARGETS = {"抄作业点左上角重开"}


def _get_original_next(context: Context, node_name: str) -> list:
    """
    获取节点原始的 next 列表，首次读取后缓存，后续不受 override 污染。

    必须在 override 之前调用（如 run() 开头），否则首次读到的可能是被污染的 next。
    若读到的 next 含重开目标（已被污染），则不缓存，返回 [] 等下次再试。
    """
    if node_name in _original_next_cache:
        return list(_original_next_cache[node_name])

    node_data = context.get_node_data(node_name)
    raw_next = node_data.get("next", []) if node_data else []
    # get_node_data 返回的 next 是 [{"name": "xxx", ...}, ...]，提取纯 name
    names = [item["name"] if isinstance(item, dict) else item for item in raw_next]

    # 仅当 next 全部是重开目标时才判定为被 override 污染（override_next 整体替换后只剩重开目标）。
    # 原始 next 里允许包含重开目标（如检测节点后接"重开:全灭"，全灭重开是合法分支），不缓存返回空。
    if names and all(n in _RESTART_TARGETS for n in names):
        return []

    _original_next_cache[node_name] = names
    return list(names)


def _recognition_hit(reco_detail) -> bool:
    """兼容识别未执行（返回 None）的情况。"""
    return bool(reco_detail and getattr(reco_detail, "hit", False))


def _get_ocr_text(reco_detail) -> str:
    """从不同版本的 OCR 识别结果中汇总命中文本。"""
    if not reco_detail:
        return ""

    results = []
    best_result = getattr(reco_detail, "best_result", None)
    if best_result:
        results.append(best_result)

    for attr in ("filtered_results", "filterd_results", "all_results"):
        attr_results = getattr(reco_detail, attr, None)
        if attr_results:
            results.extend(attr_results)

    texts = []
    for result in results:
        text = str(getattr(result, "text", "") or "")
        if text and text not in texts:
            texts.append(text)
    return "".join(texts)


def _get_restart_reasons(down_detail, retreat_detail) -> list:
    """返回所有命中的重开原因；两种识别可同时贡献原因。"""
    reasons = []
    if _recognition_hit(down_detail):
        reasons.append("已阵亡")

    if _recognition_hit(retreat_detail):
        retreat_text = _get_ocr_text(retreat_detail)
        if "退" in retreat_text:
            reasons.append("已退场")
        if "放" in retreat_text:
            reasons.append("已放逐")
        if "退" not in retreat_text and "放" not in retreat_text:
            reasons.append("已退场或放逐")

    return reasons


@AgentServer.custom_action("CopilotInfo")
class CopilotInfo(CustomAction):
    """
    读取并打印作业文件中的"作业信息"
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        context.run_task("作业信息")
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("DownRestart")
class DownRestart(CustomAction):
    """
    同时检测指定位置密探是否已阵亡、退场或放逐，任一命中则改写 next 为左上角重开

    Args:
        - "node": "当前节点名称"
        - "position": [1,5]
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        COLORMATCH_ROIS = [
            [],
            [21, 811, 124, 378],
            [156, 810, 128, 375],
            [298, 811, 129, 378],
            [439, 809, 127, 376],
            [579, 808, 126, 378],
        ]
        params = json.loads(argv.custom_action_param)
        current_node_name = params["node"]
        position = params["position"]
        # 在 override 前预读原始 next，缓存干净值
        original_next = _get_original_next(context, current_node_name)
        cmroi = COLORMATCH_ROIS[position]
        img = context.tasker.controller.post_screencap().wait().get()
        down_detail = context.run_recognition(
            "downTest", img, {"downTest": {"roi": cmroi}}
        )
        retreat_detail = context.run_recognition(
            "RetreatCheck", img, {"RetreatCheck": {"roi": cmroi}}
        )
        restart_reasons = _get_restart_reasons(down_detail, retreat_detail)
        if restart_reasons:
            context.override_next(current_node_name, ["抄作业点左上角重开"])
            logger.info(
                f"检测到{position}号位{'、'.join(restart_reasons)}，正在尝试点左上角重开"
            )
            return CustomAction.RunResult(success=True)
        else:
            if original_next:
                context.override_next(current_node_name, original_next)
            logger.info(f"检测到{position}号位存活，正常执行后续动作")
            return CustomAction.RunResult(success=True)


@AgentServer.custom_action("RetreatRestart")
class RetreatRestart(CustomAction):
    """
    同时检测指定位置密探是否已阵亡、退场或放逐，任一命中则改写 next 为左上角重开

    Args:
        - "node": "当前节点名称"
        - "position": [1,5]
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        RETREAT_ROIS = [
            [],
            [21, 811, 124, 378],
            [156, 810, 128, 375],
            [298, 811, 129, 378],
            [439, 809, 127, 376],
            [579, 808, 126, 378],
        ]
        params = json.loads(argv.custom_action_param)
        current_node_name = params["node"]
        position = params["position"]
        # 在 override 前预读原始 next，缓存干净值
        original_next = _get_original_next(context, current_node_name)
        retreatroi = RETREAT_ROIS[position]
        img = context.tasker.controller.post_screencap().wait().get()
        down_detail = context.run_recognition(
            "downTest", img, {"downTest": {"roi": retreatroi}}
        )
        retreat_detail = context.run_recognition(
            "RetreatCheck", img, {"RetreatCheck": {"roi": retreatroi}}
        )
        restart_reasons = _get_restart_reasons(down_detail, retreat_detail)
        if restart_reasons:
            context.override_next(current_node_name, ["抄作业点左上角重开"])
            logger.info(
                f"检测到{position}号位{'、'.join(restart_reasons)}，正在尝试点左上角重开"
            )
            return CustomAction.RunResult(success=True)
        else:
            if original_next:
                context.override_next(current_node_name, original_next)
            logger.info(f"检测到{position}号位存活，正常执行后续动作")
            return CustomAction.RunResult(success=True)


@AgentServer.custom_action("BirdRestart")
class BirdRestart(CustomAction):
    """
    TemplateMatch 检测指定位置密探是否有鹦鹉图片，如有则正常执行后续动作，无则改写 next 为左上角重开

    Args:
        - "node": "当前节点名称"
        - "position": [1,5]
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        BIRD_ROIS = [
            [],
            [21, 811, 124, 378],
            [156, 810, 128, 375],
            [298, 811, 129, 378],
            [439, 809, 127, 376],
            [579, 808, 126, 378],
        ]
        params = json.loads(argv.custom_action_param)
        current_node_name = params["node"]
        position = params["position"]
        # 在 override 前预读原始 next，缓存干净值
        original_next = _get_original_next(context, current_node_name)
        birdroi = BIRD_ROIS[position]
        img = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition(
            "BirdCheck", img, {"BirdCheck": {"roi": birdroi}}
        )
        if reco_detail.hit:
            if original_next:
                context.override_next(current_node_name, original_next)
            logger.info(f"检测到{position}号位有鹦鹉，正常执行后续动作")
            return CustomAction.RunResult(success=True)
        else:
            context.override_next(current_node_name, ["抄作业点左上角重开"])
            logger.info(f"检测到{position}号位无鹦鹉，正在尝试点左上角重开")
            return CustomAction.RunResult(success=True)


@AgentServer.custom_action("DragonRestart")
class DragonRestart(CustomAction):
    """
    TemplateMatch 检测指定位置密探是否有龙气，如有则正常执行后续动作，无则改写 next 为左上角重开

    Args:
        - "node": "当前节点名称"
        - "position": [1,5]
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        DRAGON_ROIS = [
            [],
            [5, 1000, 50, 160],
            [145, 1000, 50, 160],
            [285, 1000, 50, 160],
            [425, 1000, 50, 160],
            [565, 1000, 50, 160],
        ]
        params = json.loads(argv.custom_action_param)
        current_node_name = params["node"]
        position = params["position"]
        # 在 override 前预读原始 next，缓存干净值
        original_next = _get_original_next(context, current_node_name)
        dragonroi = DRAGON_ROIS[position]
        img = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition(
            "DragonCheck", img, {"DragonCheck": {"roi": dragonroi}}
        )
        if reco_detail.hit:
            if original_next:
                context.override_next(current_node_name, original_next)
            logger.info(f"检测到{position}号位有2龙气，正常执行后续动作")
            return CustomAction.RunResult(success=True)
        else:
            context.override_next(current_node_name, ["抄作业点左上角重开"])
            logger.info(f"检测到{position}号位无2龙气，正在尝试点左上角重开")
            return CustomAction.RunResult(success=True)
