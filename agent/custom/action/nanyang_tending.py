import json

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


_VALID_TASKS = {"营养过剩", "营养不足"}
_DISABLED_TASKS: set[str] = set()

_FOCUS_PREFIX = "【躬耕南阳】正在进行自动护理"
_FOCUS_ITEMS = [
    ("营养不足", True),
    ("营养过剩", True),
    ("除虫", False),
]


def _parse_params(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"NanyangTendingAbandon: 参数解析失败: {raw}")
            return {}
        if isinstance(data, dict):
            return data
        logger.warning(f"NanyangTendingAbandon: 参数类型错误: {type(data)}")
        return {}
    logger.warning(f"NanyangTendingAbandon: 参数类型错误: {type(raw)}")
    return {}


def _build_focus(disabled_tasks: set[str]) -> str:
    parts = []
    for label, default_enabled in _FOCUS_ITEMS:
        enabled = default_enabled and label not in disabled_tasks
        parts.append(("✅" if enabled else "❌") + label)
    return f"{_FOCUS_PREFIX} " + " ".join(parts)


@AgentServer.custom_action("NanyangTendingAbandon")
class NanyangTendingAbandon(CustomAction):
    """
    根据参数禁用对应的南阳自动护理任务。

    Args:
        - task: "营养过剩" | "营养不足"
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        params = _parse_params(argv.custom_action_param)
        task = params.get("task")
        if isinstance(task, str):
            task = task.strip()
        if task not in _VALID_TASKS:
            logger.warning(f"NanyangTendingAbandon: 无效 task 参数: {task}")
            return CustomAction.RunResult(success=False)

        _DISABLED_TASKS.add(task)
        focus = _build_focus(_DISABLED_TASKS)
        override = {name: {"enabled": False} for name in _DISABLED_TASKS}
        override["南阳-自动护理"] = {"focus": focus}
        ok = context.override_pipeline(override)
        if ok:
            logger.info(f"NanyangTendingAbandon: 已禁用任务 {task}")
        else:
            logger.warning(f"NanyangTendingAbandon: override_pipeline 失败: {task}")
        return CustomAction.RunResult(success=ok)
