import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from zhconv import convert

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


def _clamp_roi(roi: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


@AgentServer.custom_action("AutoFormation")
class AutoFormation(CustomAction):
    """自动编队：仅负责根据 copilot 方案完成选人。命盘校验由独立的自定义识别完成。"""

    _last_plan: Dict = {}
    _last_lang: str = "zh-cn"

    ALREADY_DEPLOYED_OFFSET = (-22, -102, 43, 41)
    EFFECTIVE_DISC_OFFSET = (-47, -83, 92, 39)

    REMOVE_FIRST_SLOT_ROI = (69, 425, 42, 51)
    TARGET_OFFSET = (0, -50, 0, 0)

    MAX_SCROLL = 10
    SCROLL_TASK = "下滑-密探编队-一行"
    FIRST_SLOT_EMPTY_NODE = "自动编队-第一位为空"
    CLICK_TARGET_NODE = "自动编队-点击目标密探"
    EFFECTIVE_DISC_NODE = "自动编队-读取生效中命盘"

    def __init__(self):
        super().__init__()
        self._operators: Dict[str, dict] = self._load_operators()
        self._lang: str = "zh-cn"

    def _convert(self, text: str) -> str:
        return convert(text or "", self._lang)

    @staticmethod
    def get_last_plan() -> Dict:
        """供后续节点读取最近一次编队生成的方案信息。"""
        return AutoFormation._last_plan

    # ---------------- 数据读取 ----------------
    def _load_operators(self) -> Dict[str, dict]:
        path = Path("agent") / "operators.json"
        if not path.exists():
            logger.error(f"未找到 operators.json: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {op.get("name"): op for op in data.get("OPERATORS", [])}
        except Exception:
            logger.exception("加载 operators.json 失败")
            return {}

    def _parse_action_param(self, argv: CustomAction.RunArg) -> Dict:
        if not argv.custom_action_param:
            return {}
        try:
            return json.loads(argv.custom_action_param)
        except json.JSONDecodeError:
            logger.warning(f"AutoFormation 参数解析失败: {argv.custom_action_param}")
            return {}

    def _locate_resource_root(self, params: Dict) -> Optional[Path]:
        preferred = params.get("resource")
        candidates: List[Path] = []
        cwd = Path(".").resolve()
        for base in [cwd, cwd / "assets"]:
            res_dir = base / "resource"
            if res_dir.exists():
                candidates.append(res_dir)

        if preferred:
            for root in candidates:
                target = root / preferred
                if (target / "pipeline" / "copilot" / "auto_formation.json").exists():
                    return target

        for root in candidates:
            for child in root.iterdir():
                if child.is_dir() and (child / "pipeline" / "copilot" / "auto_formation.json").exists():
                    return child
        return None

    def _pick_copilot_filename(self, pipeline_dir: Path) -> Optional[str]:
        if not pipeline_dir.exists():
            return None
        json_files = sorted([p.name for p in pipeline_dir.glob("*.json")])
        filtered = [n for n in json_files if n not in {"auto_formation.json", "copilot_config.json"}]
        if not filtered:
            logger.error("未找到可用的编队方案文件")
            return None
        if len(filtered) > 1:
            logger.error("存在多个编队方案文件，无法确定使用哪个，终止自动编队")
            return None
        return filtered[0]

    def _load_plan(self, resource_root: Path) -> Optional[Dict]:
        pipeline_dir = resource_root / "pipeline" / "copilot"
        filename = self._pick_copilot_filename(pipeline_dir)
        if not filename:
            logger.error("无法确定 copilot 方案文件")
            return None

        cache_file = resource_root.parent / "copilot-cache" / filename
        if not cache_file.exists():
            logger.error(f"未找到 copilot 缓存: {cache_file}")
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            opers = data.get("opers", [])
            for oper in opers:
                oper["discs_ot_names"] = self._map_discs(oper)
            data["opers_num"] = len(opers)
            data["filename"] = filename
            return data
        except Exception:
            logger.exception(f"加载 copilot 缓存失败: {cache_file}")
            return None

    def _map_discs(self, oper: Dict) -> List[str]:
        indexes = oper.get("discs_selected") or []
        if not indexes:
            return []
        operator = self._operators.get(oper.get("name"))
        if not operator:
            logger.warning(f"operators.json 中未找到密探: {oper.get('name')}")
            return []
        discs = operator.get("discs", [])
        result = []
        for idx in indexes:
            try:
                disc = discs[int(idx)]
                name = disc.get("ot_name") or ""
                if name:
                    result.append(self._convert(name))
            except (ValueError, IndexError, TypeError):
                logger.warning(f"命盘索引无效 {idx} 对应 {oper.get('name')}")
        return result

    # ---------------- OCR/动作辅助 ----------------
    def _screenshot(self, context: Context):
        return context.tasker.controller.post_screencap().wait().get()

    def _is_first_slot_empty(self, context: Context) -> bool:
        img = self._screenshot(context)
        result = context.run_recognition(self.FIRST_SLOT_EMPTY_NODE, img)
        hit = bool(result and getattr(result, "hit", False))
        logger.info(f"第一位是否为空: {hit}")
        return hit

    def _recognize_target(self, context: Context, img, name: str):
        expected = self._convert(name)
        override = {
            self.CLICK_TARGET_NODE: {
                "custom_recognition_param": {"expected": expected, "lang": self._lang},
            }
        }
        return context.run_recognition(self.CLICK_TARGET_NODE, img, override)

    def _scroll_once(self, context: Context):
        context.run_task(self.SCROLL_TASK)

    def _click_box(self, context: Context, box: Tuple[int, int, int, int]):
        ox, oy, ow, oh = self.TARGET_OFFSET
        cx = box[0] + ox + (box[2] + ow) // 2
        cy = box[1] + oy + (box[3] + oh) // 2
        context.tasker.controller.post_click(cx, cy).wait()

    def _remove_old_first(self, context: Context):
        x, y, w, h = self.REMOVE_FIRST_SLOT_ROI
        cx, cy = x + w // 2, y + h // 2
        logger.info("移除原一号位")
        context.tasker.controller.post_click(cx, cy).wait()

    def _exists_already_deployed(self, context: Context, img, hit_box: Tuple[int, int, int, int]) -> bool:
        ox, oy, ow, oh = self.ALREADY_DEPLOYED_OFFSET
        roi = (
            hit_box[0] + ox,
            hit_box[1] + oy,
            hit_box[2] + ow,
            hit_box[3] + oh,
        )
        roi = _clamp_roi(roi, img.shape[1], img.shape[0])
        override = {
            self.EFFECTIVE_DISC_NODE: {
                "recognition": {
                    "type": "Custom",
                    "param": {
                        "custom_recognition": "ActiveDiscText",
                        "roi": list(roi),
                        "expected": self._convert("上"),
                        "lang": self._lang,
                    },
                }
            }
        }
        result = context.run_recognition(self.EFFECTIVE_DISC_NODE, img, override)
        return bool(result and getattr(result, "hit", False))

    def _find_and_click(self, context: Context, name: str, need_check_first: bool) -> bool:
        attempts = 0
        target_name = self._convert(name)
        while attempts < self.MAX_SCROLL:
            img = self._screenshot(context)
            reco = self._recognize_target(context, img, target_name)
            if reco and getattr(reco, "hit", False) and reco.best_result and reco.best_result.box:
                box = tuple(int(v) for v in reco.best_result.box)
                if need_check_first:
                    if self._exists_already_deployed(context, img, box):
                        logger.info(f"{target_name} 已在一号位")
                        return True
                    self._click_box(context, box)
                    self._remove_old_first(context)
                    return True
                self._click_box(context, box)
                return True
            self._scroll_once(context)
            attempts += 1
        logger.error(f"未找到目标密探: {target_name}")
        return False

    # ---------------- 主流程 ----------------
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        params = self._parse_action_param(argv)
        resource_root = self._locate_resource_root(params)
        if not resource_root:
            logger.error("未找到资源目录，停止自动编队")
            return CustomAction.RunResult(success=False)

        name_lower = resource_root.name.lower()
        self._lang = "zh-tw" if "zh_tw" in name_lower or "zh-tw" in name_lower else "zh-cn"
        AutoFormation._last_lang = self._lang

        plan = self._load_plan(resource_root)
        if not plan:
            return CustomAction.RunResult(success=False)
        AutoFormation._last_plan = plan

        opers: List[Dict] = plan.get("opers", [])
        if not opers:
            logger.error("编队信息为空")
            return CustomAction.RunResult(success=False)

        first_empty = self._is_first_slot_empty(context)
        for idx, oper in enumerate(opers):
            name = oper.get("name")
            if not name:
                continue
            ok = self._find_and_click(context, name, need_check_first=(idx == 0 and not first_empty))
            if not ok:
                return CustomAction.RunResult(success=False)

        return CustomAction.RunResult(success=True)
