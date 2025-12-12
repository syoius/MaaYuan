import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").strip(), (b or "").strip()).ratio()


def _clamp_roi(roi: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


@AgentServer.custom_action("AutoFormation")
class AutoFormation(CustomAction):
    """自动编队：读取 copilot 配置，选人并校验命盘。"""

    # 由示例计算出的偏移量
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

    # --------- 数据读取 ---------
    def _load_operators(self) -> Dict[str, dict]:
        path = Path("agent") / "operators.json"
        if not path.exists():
            logger.error(f"未找到密探数据文件: {path}")
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

        # 优先资源名匹配
        if preferred:
            for root in candidates:
                target = root / preferred
                if (target / "pipeline" / "copilot" / "auto_formation.json").exists():
                    return target

        # 其次尝试 base/zh_tw 等已有目录
        for root in candidates:
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                if (child / "pipeline" / "copilot" / "auto_formation.json").exists():
                    return child
        return None

    def _pick_copilot_filename(self, pipeline_dir: Path) -> Optional[str]:
        if not pipeline_dir.exists():
            return None
        json_files = sorted([p.name for p in pipeline_dir.glob("*.json")])
        filtered = [n for n in json_files if n not in {"auto_formation.json", "copilot_config.json"}]
        if len(filtered) < 3:
            logger.warning("copilot 文件不足 3 个，默认取可用的第一个")
            return filtered[0] if filtered else None
        return filtered[2]  # 第3个

    def _load_plan(self, resource_root: Path) -> Optional[Dict]:
        pipeline_dir = resource_root / "pipeline" / "copilot"
        filename = self._pick_copilot_filename(pipeline_dir)
        if not filename:
            logger.error("未找到可用的编队文件")
            return None

        cache_file = resource_root.parent / "copilot-cache" / filename
        if not cache_file.exists():
            logger.error(f"未找到编队缓存: {cache_file}")
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
            logger.exception(f"加载编队缓存失败: {cache_file}")
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
                    result.append(name)
            except (ValueError, IndexError, TypeError):
                logger.warning(f"命盘索引无效: {oper.get('name')} -> {idx}")
        return result

    # --------- OCR & 识别辅助 ---------
    def _screenshot(self, context: Context):
        return context.tasker.controller.post_screencap().wait().get()

    def _is_first_slot_empty(self, context: Context) -> bool:
        img = self._screenshot(context)
        result = context.run_recognition(self.FIRST_SLOT_EMPTY_NODE, img)
        hit = bool(result and getattr(result, "hit", False))
        logger.info(f"第一位是否为空: {hit}")
        return hit

    def _recognize_target(self, context: Context, img, name: str):
        override = {
            self.CLICK_TARGET_NODE: {
                "custom_recognition_param": {"expected": name},
            }
        }
        return context.run_recognition(self.CLICK_TARGET_NODE, img, override)

    def _scroll_once(self, context: Context):
        context.run_task(self.SCROLL_TASK)

    def _click_box(self, context: Context, box: Tuple[int, int, int, int]):
        # 模拟节点内 action 的 target_offset 行为
        ox, oy, ow, oh = self.TARGET_OFFSET
        cx = box[0] + ox + (box[2] + ow) // 2
        cy = box[1] + oy + (box[3] + oh) // 2
        context.tasker.controller.post_click(cx, cy).wait()

    def _remove_old_first(self, context: Context):
        x, y, w, h = self.REMOVE_FIRST_SLOT_ROI
        cx, cy = x + w // 2, y + h // 2
        logger.info("尝试移除原一号位")
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
                "roi": list(roi),
                "expected": "上",
            }
        }
        result = context.run_recognition(self.EFFECTIVE_DISC_NODE, img, override)
        return bool(result and getattr(result, "hit", False))

    def _find_and_click(self, context: Context, name: str, need_check_first: bool) -> bool:
        attempts = 0
        while attempts < self.MAX_SCROLL:
            img = self._screenshot(context)
            reco = self._recognize_target(context, img, name)
            if reco and getattr(reco, "hit", False) and reco.best_result and reco.best_result.box:
                box = tuple(int(v) for v in reco.best_result.box)
                if need_check_first:
                    if self._exists_already_deployed(context, img, box):
                        logger.info(f"{name} 已在一号位，无需点击")
                        return True
                    self._click_box(context, box)
                    self._remove_old_first(context)
                    return True
                self._click_box(context, box)
                return True
            self._scroll_once(context)
            attempts += 1
        logger.error(f"未找到目标密探: {name}")
        return False

    # --------- 命盘校验 ---------
    def _collect_active_disc_texts(self, context: Context) -> List[str]:
        img = self._screenshot(context)
        reco = context.run_recognition(self.EFFECTIVE_DISC_NODE, img)
        results = []
        if not reco:
            return results

        candidates = getattr(reco, "filterd_results", None) or getattr(reco, "filtered_results", None) or reco.all_results
        if not candidates:
            return results

        for res in candidates:
            if not getattr(res, "box", None):
                continue
            box = tuple(int(v) for v in res.box)
            ox, oy, ow, oh = self.EFFECTIVE_DISC_OFFSET
            roi = (
                box[0] + ox,
                box[1] + oy,
                box[2] + ow,
                box[3] + oh,
            )
            roi = _clamp_roi(roi, img.shape[1], img.shape[0])
            override = {self.EFFECTIVE_DISC_NODE: {"roi": list(roi), "expected": ""}}
            detail = context.run_recognition(self.EFFECTIVE_DISC_NODE, img, override)
            if detail and getattr(detail, "best_result", None):
                text = detail.best_result.text.strip()
                if text:
                    results.append(text)
        return results

    def _verify_discs(self, context: Context, opers: List[Dict]) -> bool:
        required = []
        for oper in opers:
            required.extend(oper.get("discs_ot_names") or [])
        required = [r for r in required if r]
        if not required:
            logger.info("无命盘要求，跳过校验")
            return True

        active_texts = self._collect_active_disc_texts(context)
        if not active_texts:
            logger.error("未识别到任何生效命盘")
            return False

        for need in required:
            if not any(_similar(need, txt) >= 0.55 for txt in active_texts):
                logger.error(f"命盘缺失: {need}")
                return False
        return True

    # --------- 主流程 ---------
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        params = self._parse_action_param(argv)
        resource_root = self._locate_resource_root(params)
        if not resource_root:
            logger.error("无法确定资源目录，自动编队终止")
            return CustomAction.RunResult(success=False)

        plan = self._load_plan(resource_root)
        if not plan:
            return CustomAction.RunResult(success=False)

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

        if not self._verify_discs(context, opers):
            return CustomAction.RunResult(success=False)

        return CustomAction.RunResult(success=True)
