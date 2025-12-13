import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from zhconv import convert

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


def _safe_parse_json(raw, name: str) -> Dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"{name} 参数解析失败: {raw}")
        except Exception:
            logger.exception(f"{name} 参数解析异常")
    return {}


def _lang_from_resource(
    resource_hint: str = "", resource_root: Optional[Path] = None
) -> str:
    hint = (resource_hint or "").strip().lower()
    if "zh_tw" in hint or "zh-tw" in hint:
        return "zh-tw"

    if resource_root:
        name_lower = resource_root.name.lower()
        if "zh_tw" in name_lower or "zh-tw" in name_lower:
            return "zh-tw"

    return "zh-cn"


def _clamp_roi(
    roi: Tuple[int, int, int, int], width: int, height: int
) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def _extract_recognition(detail):
    if detail is None:
        return None
    if hasattr(detail, "nodes"):
        nodes = getattr(detail, "nodes", None) or []
        if nodes:
            return getattr(nodes[0], "recognition", None)
        return None
    return detail


def _get_results(recognition) -> list:
    recognition = _extract_recognition(recognition)
    if recognition is None:
        return []
    for attr in ("filterd_results", "filtered_results", "all_results", "all"):
        results = getattr(recognition, attr, None)
        if results:
            return results
    detail = getattr(recognition, "detail", None)
    if isinstance(detail, dict):
        for key in ("filtered", "filterd", "all"):
            if key in detail and detail[key]:
                return detail[key]
    return []


@AgentServer.custom_action("AutoFormation")
class AutoFormation(CustomAction):
    """自动编队：仅负责根据 copilot 方案完成选人。命盘校验由独立的自定义识别完成。"""

    _last_plan: Dict = {}
    _last_resource: str = ""
    _last_resource_lang: str = "zh-cn"

    ALREADY_DEPLOYED_OFFSET = (-22, -102, 43, 41)
    EFFECTIVE_DISC_OFFSET = (-47, -83, 92, 39)

    REMOVE_FIRST_SLOT_ROI = (69, 425, 42, 51)
    TARGET_OFFSET = (0, -50, 0, 0)

    ATTR_FILTER_ROIS: Dict[str, Tuple[int, int, int, int]] = {
        "地": (141, 393, 17, 11),
        "水": (284, 393, 14, 13),
        "火": (424, 389, 17, 14),
        "风": (568, 389, 20, 18),
        "阳": (142, 462, 13, 11),
        "阴": (284, 467, 13, 9),
        "混沌": (425, 466, 20, 11),
    }
    SUBPROF_FILTER_ROIS: Dict[str, Tuple[int, int, int, int]] = {
        "pojun": (143, 605, 12, 11),
        "longdun": (281, 607, 14, 9),
        "qihuang": (430, 606, 13, 9),
        "shenji": (579, 606, 10, 10),
        "guidao": (145, 682, 19, 12),
    }
    NAME_CONFUSION_MAP: Dict[str, List[str]] = {
        "士": ["土"],
        "郃": ["部"],
    }
    TARGET_RECO_ROIS: List[Tuple[int, int, int, int]] = [
        (27, 1013, 136, 61),
        (166, 1013, 123, 52),
        (295, 1012, 124, 52),
        (426, 1012, 126, 55),
        (557, 1012, 121, 55),
    ]

    MAX_SCROLL = 5
    FILTER_ENTRY_NODE = "自动编队-打开筛选面板"
    RESET_SCROLL_NODE = "自动编队-重置列表滚动位置"
    FILTER_ATTR_NODE = "自动编队-筛选-属性"
    FILTER_PROF_NODE = "自动编队-筛选-职业"
    SCROLL_TASK = "上滑-密探编队-一行"
    FIRST_SLOT_EMPTY_NODE = "自动编队-第一位为空"
    CLICK_TARGET_NODE = "自动编队-点击目标密探"
    EXISTED_NODE = "自动编队-识别目标密探"

    def __init__(self):
        super().__init__()
        self._operators: Dict[str, dict] = self._load_operators()
        self._resource_hint: str = ""
        self._resource_lang: str = "zh-cn"

    def _convert(self, text: str) -> str:
        return convert(text or "", self._resource_lang)

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
        params = _safe_parse_json(
            getattr(argv, "custom_action_param", None), "AutoFormation"
        )
        resource = (
            str(params.get("resource", "") or "").strip()
            if isinstance(params, dict)
            else ""
        )
        if resource:
            self._resource_hint = resource
        return params if isinstance(params, dict) else {}

    def _locate_resource_root(self, params: Optional[Dict]) -> Optional[Path]:
        params = params or {}
        preferred = params.get("resource")
        candidates: List[Path] = []
        cwd = Path(".").resolve()
        for base in [cwd, cwd / "assets"]:
            res_dir = base / "resource"
            if res_dir.exists():
                candidates.append(res_dir)

        if preferred:
            pref_path = Path(str(preferred))
            if pref_path.exists():
                target = pref_path
                if (
                    target / "pipeline" / "autoformation" / "auto_formation.json"
                ).exists():
                    return target

            for root in candidates:
                target = root / preferred
                if (
                    target / "pipeline" / "autoformation" / "auto_formation.json"
                ).exists():
                    return target

        for root in candidates:
            for child in root.iterdir():
                if (
                    child.is_dir()
                    and (
                        child / "pipeline" / "autoformation" / "auto_formation.json"
                    ).exists()
                ):
                    return child
        return None

    def _pick_copilot_filename(self, pipeline_dir: Path) -> Optional[str]:
        if not pipeline_dir.exists():
            return None
        json_files = sorted([p.name for p in pipeline_dir.glob("*.json")])
        filtered = [n for n in json_files if n not in {"copilot_config.json"}]
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
                operator = self._operators.get(oper.get("name"))
                oper["discs_ot_names"] = self._map_discs(oper, operator)
                oper["prof"] = oper.get("prof") or (
                    operator.get("prof") if operator else None
                )
                oper["subProf"] = oper.get("subProf") or (
                    operator.get("subProf") if operator else None
                )
            data["opers_num"] = len(opers)
            data["filename"] = filename
            return data
        except Exception:
            logger.exception(f"加载 copilot 缓存失败: {cache_file}")
            return None

    def _map_discs(self, oper: Dict, operator: Optional[dict] = None) -> List[str]:
        indexes = oper.get("discs_selected") or []
        if not indexes:
            return []
        if operator is None:
            operator = self._operators.get(oper.get("name"))
        if not operator:
            logger.warning(f"operators.json 中未找到密探: {oper.get('name')}")
            return []
        discs = operator.get("discs", [])
        result = []
        for idx in indexes:
            try:
                idx_int = int(idx)
            except (ValueError, TypeError):
                logger.warning(f"命盘索引格式无效 {idx} 对应 {oper.get('name')}")
                continue

            if idx_int <= 0:
                # 0 或负数代表空槽，跳过
                continue

            real_idx = idx_int - 1  # copilot 序号从 1 开始
            if real_idx < 0 or real_idx >= len(discs):
                logger.warning(f"命盘索引越界 {idx_int} 对应 {oper.get('name')}")
                continue

            disc = discs[real_idx]
            name = disc.get("ot_name") or ""
            if name:
                result.append(self._convert(name))
        return result

    # ---------------- OCR/动作辅助 ----------------
    def _screenshot(self, context: Context):
        return context.tasker.controller.post_screencap().wait().get()

    def _is_first_slot_empty(self, context: Context) -> bool:
        img = self._screenshot(context)
        result = context.run_recognition(self.FIRST_SLOT_EMPTY_NODE, img)
        hit = bool(result and getattr(result, "hit", False))
        logger.info(f"第一位为空: {hit}")
        return hit

    def _recognize_target(
        self, context: Context, img, name: str, roi: Tuple[int, int, int, int]
    ):
        expected = self._convert(name or "")
        rec_param = {
            "expected": expected,
            "roi": list(roi),
        }
        if self._resource_hint:
            rec_param["resource"] = self._resource_hint

        override = {
            self.CLICK_TARGET_NODE: {
                "custom_recognition_param": rec_param,
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
        logger.info("移除原一号位")
        result = context.run_task("自动编队-移除一号位")
        if (
            not result
            or not getattr(result, "status", None)
            or not result.status.succeeded
        ):
            logger.warning("自动编队-移除一号位 执行失败")

    def _exists_already_deployed(
        self, context: Context, img, hit_box: Tuple[int, int, int, int]
    ) -> bool:
        ox, oy, ow, oh = self.ALREADY_DEPLOYED_OFFSET
        roi = (
            hit_box[0] + ox,
            hit_box[1] + oy,
            hit_box[2] + ow,
            hit_box[3] + oh,
        )
        roi = _clamp_roi(roi, img.shape[1], img.shape[0])
        override = {
            self.EXISTED_NODE: {
                "recognition": {
                    "param": {
                        "roi": list(roi),
                        "expected": self._convert("上"),
                    },
                }
            }
        }
        result = context.run_recognition(self.EXISTED_NODE, img, override)
        return bool(result and getattr(result, "hit", False))

    def _reset_scroll_position(self, context: Context) -> bool:
        result = context.run_task(self.RESET_SCROLL_NODE)
        ok = bool(
            result and getattr(result, "status", None) and result.status.succeeded
        )
        if not ok:
            logger.warning("重置列表滚动位置失败")
        return ok

    def _build_filter_overrides(
        self, attr_roi: Tuple[int, int, int, int], sub_roi: Tuple[int, int, int, int]
    ) -> Dict:
        return {
            self.FILTER_ATTR_NODE: {
                "action": {"type": "Click", "param": {"target": list(attr_roi)}},
            },
            self.FILTER_PROF_NODE: {
                "action": {"type": "Click", "param": {"target": list(sub_roi)}},
            },
        }

    def _apply_filters(self, context: Context, oper: Dict) -> bool:
        prof = oper.get("prof")
        sub_prof = oper.get("subProf")
        attr_roi = self.ATTR_FILTER_ROIS.get(prof or "")
        sub_roi = self.SUBPROF_FILTER_ROIS.get(sub_prof or "")
        if not attr_roi or not sub_roi:
            logger.warning(f"缺少筛选坐标，跳过筛选: prof={prof}, subProf={sub_prof}")
            return False

        overrides = self._build_filter_overrides(attr_roi, sub_roi)
        result = context.run_task(self.FILTER_ENTRY_NODE, overrides)
        ok = bool(
            result and getattr(result, "status", None) and result.status.succeeded
        )
        if not ok:
            logger.error("筛选流程执行失败")
        return ok

    def _generate_fallback_names(self, name: str) -> List[str]:
        """根据易错表为名字生成替代候选。"""
        fallbacks: set = set()
        for wrong, replaces in self.NAME_CONFUSION_MAP.items():
            if wrong in name:
                for rep in replaces:
                    fallbacks.add(name.replace(wrong, rep))
        fallbacks.discard(name)
        return list(fallbacks)

    def _search_and_click(
        self, context: Context, name: str, need_check_first: bool
    ) -> bool:
        attempts = 0
        target_name = self._convert(name)
        fallback_names = self._generate_fallback_names(name)
        while attempts < self.MAX_SCROLL:
            img = self._screenshot(context)
            for candidate_name in [name] + fallback_names:
                for roi in self.TARGET_RECO_ROIS:
                    reco = self._recognize_target(context, img, candidate_name, roi)
                    if (
                        reco
                        and getattr(reco, "hit", False)
                        and reco.best_result
                        and reco.best_result.box
                    ):
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

    def _find_and_click(
        self, context: Context, oper: Dict, need_check_first: bool
    ) -> bool:
        name = oper.get("name")
        if not name:
            logger.error("缺少密探名称，无法选人")
            return False

        filtered = self._apply_filters(context, oper)
        if not filtered:
            self._reset_scroll_position(context)

        ok = self._search_and_click(context, name, need_check_first)
        if ok:
            self._reset_scroll_position(context)
        return ok

    # ---------------- 主流程 ----------------
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        params = self._parse_action_param(argv) or {}
        resource_root = self._locate_resource_root(params)
        if not resource_root:
            logger.error("未找到资源目录，停止自动编队")
            return CustomAction.RunResult(success=False)

        name_lower = resource_root.name.lower()
        resource_hint_lower = str(params.get("resource", "") or "").strip().lower()
        AutoFormation._last_resource = resource_hint_lower or name_lower
        if not self._resource_hint:
            self._resource_hint = AutoFormation._last_resource

        self._resource_lang = _lang_from_resource(self._resource_hint, resource_root)
        AutoFormation._last_resource_lang = self._resource_lang

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
            ok = self._find_and_click(
                context, oper, need_check_first=(idx == 0 and not first_empty)
            )
            if not ok:
                return CustomAction.RunResult(success=False)

        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("DiscChecker")
class DiscChecker(CustomAction):
    """命盘生效检测，必要时尝试切换命盘。"""

    DEFAULT_THRESHOLD = 0.55
    RETRY_TASK = "自动编队-尝试切换命盘"
    NEXT_TASK = "自动编队-命盘检测-向右切换"
    CLOSE_PROMPT_TASK = "自动编队-关闭命盘提示"
    RETRY_WAIT_MS = 2000
    CLOSE_WAIT_MS = 1000

    def __init__(self):
        super().__init__()
        self._resource_lang: str = "zh-cn"

    def _convert(self, text: str) -> str:
        return convert(text or "", self._resource_lang)

    def _screenshot(self, context: Context):
        return context.tasker.controller.post_screencap().wait().get()

    def _run_task(
        self,
        context: Context,
        name: str,
        wait_ms: Optional[int] = None,
        log_warning: bool = True,
    ) -> bool:
        result = context.run_task(name)
        ok = bool(
            result and getattr(result, "status", None) and result.status.succeeded
        )
        if (not ok) and log_warning:
            logger.warning(f"{name} 执行失败")
        if wait_ms and wait_ms > 0:
            time.sleep(wait_ms / 1000.0)
        return ok

    def _read_active_effects(self, context: Context) -> List[str]:
        img = self._screenshot(context)
        reco = context.run_recognition("自动编队-读取生效中命盘", img)
        candidates = _get_results(reco)
        if not candidates:
            logger.info("未识别到生效中的命盘效果")
            return []

        offset = list(getattr(AutoFormation, "EFFECTIVE_DISC_OFFSET", [0, 0, 0, 0]))
        dx, dy, dw, dh = [int(v) for v in offset]
        texts: List[str] = []

        for res in candidates:
            box = getattr(res, "box", None)
            if not box:
                continue
            roi = (
                int(box[0]) + dx,
                int(box[1]) + dy,
                int(box[2]) + dw,
                int(box[3]) + dh,
            )
            roi = _clamp_roi(roi, img.shape[1], img.shape[0])

            override = {
                "自动编队-读取生效中命盘": {
                    "recognition": {
                        "type": "OCR",
                        "param": {
                            "roi": list(roi),
                            "expected": "",
                        },
                    }
                }
            }
            detail = context.run_recognition("自动编队-读取生效中命盘", img, override)
            text_raw = ""
            if detail and getattr(detail, "best_result", None):
                text_raw = getattr(detail.best_result, "text", "") or ""
            else:
                recog = _extract_recognition(detail)
                if recog and getattr(recog, "best_result", None):
                    text_raw = getattr(recog.best_result, "text", "") or ""

            text = self._convert(str(text_raw).strip())
            if text:
                texts.append(text)

        logger.info(f"当前生效的命盘: {texts}")
        return texts

    def _find_missing(self, required: List[str], effects: List[str]) -> List[str]:
        missing: List[str] = []
        for need in required:
            best_score = 0.0
            best_effect = ""
            for effect in effects:
                score = SequenceMatcher(None, need, effect).ratio()
                if score > best_score:
                    best_score = score
                    best_effect = effect
            logger.info(
                f"命盘匹配: 需要={need}, 当前已有={best_effect}, 相似度={best_score:.3f}"
            )
            if best_score < self.DEFAULT_THRESHOLD:
                missing.append(need)
        return missing

    def _ensure_plan(self, argv: CustomAction.RunArg) -> Dict:
        """若未缓存方案则尝试按 AutoFormation 逻辑生成。"""
        params = _safe_parse_json(
            getattr(argv, "custom_action_param", None), "DiscChecker"
        )
        resource_hint_lower = str(params.get("resource", "") or "").strip().lower()

        plan = (
            AutoFormation.get_last_plan()
            if hasattr(AutoFormation, "get_last_plan")
            else {}
        )
        if plan:
            last_res_lower = getattr(AutoFormation, "_last_resource", "").lower()
            if (
                resource_hint_lower
                and last_res_lower
                and resource_hint_lower != last_res_lower
            ):
                AutoFormation._last_plan = {}
                AutoFormation._last_resource = ""
                plan = {}
            else:
                self._resource_lang = getattr(
                    AutoFormation, "_last_resource_lang", "zh-cn"
                )
                return plan

        try:
            af = AutoFormation()
            resource_root = (
                af._locate_resource_root(params)
                if hasattr(af, "_locate_resource_root")
                else None
            )
            if not resource_root:
                logger.error("未找到资源目录，无法生成编队方案")
                return {}

            name_lower = resource_root.name.lower()
            resource_lang = _lang_from_resource(resource_hint_lower, resource_root)

            af._resource_lang = resource_lang  # type: ignore[attr-defined]
            self._resource_lang = resource_lang
            AutoFormation._last_resource_lang = resource_lang
            AutoFormation._last_resource = resource_hint_lower or name_lower

            if hasattr(af, "_load_plan"):
                plan = af._load_plan(resource_root) or {}
                AutoFormation._last_plan = plan
                return plan
        except Exception:
            logger.exception("生成编队方案时异常")

        return {}

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        if AutoFormation is None:
            logger.error("未找到 AutoFormation，无法读取命盘方案")
            return CustomAction.RunResult(success=False)

        plan = self._ensure_plan(argv)
        if not plan:
            logger.error("尚未生成编队方案，无法进行命盘检测")
            return CustomAction.RunResult(success=False)

        self._resource_lang = getattr(AutoFormation, "_last_resource_lang", "zh-cn")
        opers: List[Dict] = plan.get("opers", []) or []
        opers_num = plan.get("opers_num", len(opers))

        has_any_discs = any((oper.get("discs_ot_names") for oper in opers))
        if not has_any_discs:
            logger.info("方案未配置命盘要求，跳过命盘检测")
            return CustomAction.RunResult(success=True)

        overall_success = True

        for idx in range(opers_num):
            oper = opers[idx] if idx < len(opers) else None
            oper_name = oper.get("name") if oper else ""
            required = []
            if oper:
                required = [
                    self._convert(str(name))
                    for name in (oper.get("discs_ot_names") or [])
                    if name
                ]

            if not required:
                logger.info(f"{idx + 1}号位({oper_name})未配置命盘要求，跳过")
            else:
                effects_before = self._read_active_effects(context)
                missing_before = self._find_missing(required, effects_before)

                if missing_before:
                    logger.info(
                        f"{idx + 1}号位({oper_name})缺少命盘: {missing_before}，尝试切换命盘"
                    )
                    self._run_task(context, self.RETRY_TASK, wait_ms=self.RETRY_WAIT_MS)
                    close_hit = self._run_task(
                        context,
                        self.CLOSE_PROMPT_TASK,
                        wait_ms=self.CLOSE_WAIT_MS,
                        log_warning=False,
                    )

                    effects_after = effects_before
                    missing_after = missing_before
                    if close_hit:
                        logger.info("检测到仅一组命盘，保持当前侧效果")
                    else:
                        effects_after = self._read_active_effects(context)
                        missing_after = self._find_missing(required, effects_after)

                    if not missing_after:
                        logger.info(
                            f"{idx + 1}号位({oper_name})切换后命盘已符合作业要求"
                        )
                        effects_final = effects_after
                        missing_final: List[str] = []
                    else:
                        # 仅在切换成功且缺失更差时回退
                        if (not close_hit) and len(missing_after) > len(missing_before):
                            logger.info(
                                f"切换后缺失更多({len(missing_after)}>{len(missing_before)}), 再切回原侧"
                            )
                            self._run_task(
                                context, self.RETRY_TASK, wait_ms=self.RETRY_WAIT_MS
                            )
                            self._run_task(
                                context,
                                self.CLOSE_PROMPT_TASK,
                                wait_ms=self.CLOSE_WAIT_MS,
                                log_warning=False,
                            )
                            effects_final = effects_before
                            missing_final = missing_before
                        else:
                            effects_final = effects_after
                            missing_final = missing_after

                        if missing_final:
                            overall_success = False
                            logger.info(
                                f"{idx + 1}号位({oper_name})最终停留侧的效果: {effects_final}, 缺失: {missing_final}"
                            )
                            for name in missing_final:
                                logger.error(f"缺少命盘: {name}")
                else:
                    logger.info(f"{idx + 1}号位({oper_name})命盘已符合作业要求")

            if idx < opers_num - 1:
                self._run_task(context, self.NEXT_TASK)

        return CustomAction.RunResult(success=overall_success)
