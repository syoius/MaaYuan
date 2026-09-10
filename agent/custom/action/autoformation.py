import json
import hashlib
import time
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from zhconv import convert

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger
from utils.remote_file_sync import create_https_context


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


def _merge_line_segments(results: List) -> List[str]:
    """将同一行的碎片文本拼接，避免跨行拼接。"""
    segments: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for res in results:
        text = getattr(res, "text", "") or ""
        box = getattr(res, "box", None)
        if text and box:
            segments.append((str(text), tuple(int(v) for v in box)))

    if not segments:
        return []

    def _same_line(b1, b2) -> bool:
        y1, h1 = b1[1], b1[3]
        y2, h2 = b2[1], b2[3]
        return abs(y1 - y2) <= max(h1, h2) * 0.6

    segments_sorted = sorted(segments, key=lambda s: (s[1][1], s[1][0]))
    merged_texts: List[str] = []
    current_text, current_box = segments_sorted[0]
    for seg_text, seg_box in segments_sorted[1:]:
        if _same_line(current_box, seg_box):
            gap = seg_box[0] - (current_box[0] + current_box[2])
            if gap <= max(current_box[2], seg_box[2]) * 0.6:
                current_text += seg_text
                new_x1 = min(current_box[0], seg_box[0])
                new_y1 = min(current_box[1], seg_box[1])
                new_x2 = max(current_box[0] + current_box[2], seg_box[0] + seg_box[2])
                new_y2 = max(current_box[1] + current_box[3], seg_box[1] + seg_box[3])
                current_box = (new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)
                continue
        merged_texts.append(current_text)
        current_text, current_box = seg_text, seg_box
    merged_texts.append(current_text)
    return merged_texts


def _should_stop_ctx(context: Context) -> bool:
    """协作式停止检查，兼容 StopTask/Tasker.stop()."""
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


def _extract_numbers(text: str) -> List[str]:
    """提取字符串中的数字片段，用于区分如 +1 / +3 等决定性差异。"""
    return re.findall(r"\d+", text or "")


DEGREE_KEYWORDS = {
    "big": ["大幅"],
    "mid": ["中幅"],
    "small": ["小幅"],
}


def _extract_degree(text: str) -> Optional[str]:
    """提取描述程度的关键词，用于区分大/中/小幅差异。"""
    content = text or ""
    for degree, kws in DEGREE_KEYWORDS.items():
        for kw in kws:
            if kw in content:
                return degree
    return None


@AgentServer.custom_action("AutoFormation")
class AutoFormation(CustomAction):
    """自动编队：仅负责根据 copilot 方案完成选人。命盘校验由独立的自定义识别完成。"""

    _last_plan: Dict = {}
    _last_resource: str = ""
    _last_resource_lang: str = "zh-cn"
    OPERATORS_LOCAL_PATH = Path("agent") / "operators.json"
    OPERATORS_SYNC_META_PATH = Path("agent") / "operators.sync.meta"
    OPERATORS_REMOTE_URL = "https://maayuan.top/operators.json"
    # 远程变更频率低（通常月内少量变更）。
    OPERATORS_REMOTE_CHECK_INTERVAL_SEC = 15 * 24 * 60 * 60
    # 本地文件缺失/损坏时仍尽快恢复，但避免过于频繁请求远端。
    OPERATORS_REMOTE_RETRY_INTERVAL_SEC = 43200
    OPERATORS_REMOTE_TIMEOUT_SEC = 8

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
        "郃": ["部", "邻"],
        "诩": ["谢"],
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

    def _convert_name(self, text: str) -> str:
        """保留特定简体字形以匹配实际显示。"""
        converted = self._convert(text or "")
        if self._resource_lang == "zh-tw" and "群" in (text or ""):
            return converted.replace("羣", "群")
        return converted

    @staticmethod
    def get_last_plan() -> Dict:
        """供后续节点读取最近一次编队生成的方案信息。"""
        return AutoFormation._last_plan

    # ---------------- 数据读取 ----------------
    def _is_valid_operators_data(self, data: Dict) -> bool:
        return isinstance(data, dict) and isinstance(data.get("OPERATORS"), list)

    def _operators_to_map(self, data: Dict) -> Dict[str, dict]:
        operators: Dict[str, dict] = {}
        for oper in data.get("OPERATORS", []):
            if not isinstance(oper, dict):
                continue
            name = oper.get("name")
            if not name:
                continue
            operators[str(name)] = oper
        return operators

    @staticmethod
    def _to_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _operators_hash(data: Dict) -> str:
        normalized = json.dumps(
            data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _read_operators_file(self) -> Optional[Dict]:
        path = self.OPERATORS_LOCAL_PATH
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not self._is_valid_operators_data(data):
                logger.error("本地 operators.json 结构无效")
                return None
            return data
        except Exception:
            logger.exception("读取本地 operators.json 失败")
            return None

    def _write_operators_file(self, data: Dict) -> bool:
        path = self.OPERATORS_LOCAL_PATH
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            tmp_path.replace(path)
            return True
        except Exception:
            logger.exception("写入 operators.json 失败")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return False

    def _read_sync_meta(self) -> Dict:
        path = self.OPERATORS_SYNC_META_PATH
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.warning("读取 operators 同步元数据失败，将重建元数据")
            return {}

    def _write_sync_meta(self, meta: Dict):
        path = self.OPERATORS_SYNC_META_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except Exception:
            logger.warning("写入 operators 同步元数据失败")

    def _fetch_remote_operators(
        self, meta: Dict, use_conditional: bool = True
    ) -> Tuple[str, Optional[Dict], Dict]:
        """
        返回值:
            ("updated", data, meta_updates): 成功获取到远程数据（可能与本地一致）
            ("not_modified", None, meta_updates): 远程确认未变更（304）
            ("error", None, {}): 拉取失败
        """
        headers = {"User-Agent": "MaaY-AutoFormation/1.0"}
        if use_conditional:
            etag = str(meta.get("etag", "") or "").strip()
            last_modified = str(meta.get("last_modified", "") or "").strip()
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified

        request = urllib_request.Request(
            self.OPERATORS_REMOTE_URL, headers=headers, method="GET"
        )
        try:
            with urllib_request.urlopen(
                request,
                timeout=self.OPERATORS_REMOTE_TIMEOUT_SEC,
                context=create_https_context(),
            ) as response:
                status = int(getattr(response, "status", response.getcode()))
                if status != 200:
                    logger.warning(f"拉取远程 operators.json 失败，HTTP {status}")
                    return "error", None, {}

                payload_bytes = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
                payload_text = payload_bytes.decode(charset, errors="replace")
                payload = json.loads(payload_text)
                if not self._is_valid_operators_data(payload):
                    logger.error("远程 operators.json 结构无效，忽略本次更新")
                    return "error", None, {}

                updates: Dict[str, str] = {}
                etag = response.headers.get("ETag")
                last_modified = response.headers.get("Last-Modified")
                if etag:
                    updates["etag"] = str(etag)
                if last_modified:
                    updates["last_modified"] = str(last_modified)
                updates["remote_hash"] = self._operators_hash(payload)
                return "updated", payload, updates
        except urllib_error.HTTPError as e:
            if e.code == 304:
                updates: Dict[str, str] = {}
                etag = e.headers.get("ETag") if e.headers else None
                last_modified = e.headers.get("Last-Modified") if e.headers else None
                if etag:
                    updates["etag"] = str(etag)
                if last_modified:
                    updates["last_modified"] = str(last_modified)
                return "not_modified", None, updates
            logger.warning(f"拉取远程 operators.json 失败，HTTP {e.code}")
            return "error", None, {}
        except urllib_error.URLError as e:
            logger.warning(f"拉取远程 operators.json 失败: {e}")
            return "error", None, {}
        except json.JSONDecodeError:
            logger.exception("解析远程 operators.json 失败")
            return "error", None, {}
        except Exception:
            logger.exception("拉取远程 operators.json 时发生异常")
            return "error", None, {}

    def _sync_operators_data(self, local_data: Optional[Dict]) -> Optional[Dict]:
        meta = self._read_sync_meta()
        now = int(time.time())
        local_available = local_data is not None
        last_check_ts = self._to_int(meta.get("last_check_ts"), 0)

        if local_available:
            interval = self.OPERATORS_REMOTE_CHECK_INTERVAL_SEC
        else:
            interval = self.OPERATORS_REMOTE_RETRY_INTERVAL_SEC

        should_check_remote = now - last_check_ts >= max(1, interval)
        if not should_check_remote:
            return local_data

        if not local_available:
            logger.warning("本地 operators.json 不可用，尝试从远程下载")

        status, remote_data, meta_updates = self._fetch_remote_operators(
            meta, use_conditional=local_available
        )

        meta["last_check_ts"] = now
        if status == "not_modified":
            meta["last_success_ts"] = now
            meta.update(meta_updates)
            self._write_sync_meta(meta)
            return local_data

        if status == "updated" and remote_data is not None:
            remote_hash = str(meta_updates.get("remote_hash", "") or "").strip()
            if not remote_hash:
                remote_hash = self._operators_hash(remote_data)

            local_hash = self._operators_hash(local_data) if local_data else ""
            meta.update(meta_updates)
            meta["last_success_ts"] = now
            meta["remote_hash"] = remote_hash

            if (not local_available) or remote_hash != local_hash:
                if self._write_operators_file(remote_data):
                    logger.info("operators.json 已更新为远程版本")
                else:
                    logger.warning(
                        "写入本地 operators.json 失败，当前运行将直接使用远程数据"
                    )
                local_data = remote_data
            self._write_sync_meta(meta)
            return local_data

        self._write_sync_meta(meta)
        if not local_available:
            logger.error("远程 operators.json 拉取失败，且本地无可用数据")
        else:
            logger.warning("远程 operators.json 检查失败，继续使用本地数据")
        return local_data

    def _load_operators(self) -> Dict[str, dict]:
        local_data = self._read_operators_file()
        data = self._sync_operators_data(local_data)
        if not data:
            logger.error(f"未找到可用 operators.json: {self.OPERATORS_LOCAL_PATH}")
            return {}
        return self._operators_to_map(data)

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

    def _resolve_resource_root(self, resource: str) -> Optional[Path]:
        """仅按传入 resource 参数解析资源目录，可为绝对路径或 base/zh_tw 等相对名。"""
        if not resource:
            return None

        # 绝对路径直接使用
        candidate = Path(resource)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        # 优先使用相对路径本身（install 后资源可能在 ./resource/...）
        candidate = Path(resource)
        if candidate.exists():
            return candidate

        # 再尝试 ./resource/<name>
        candidate = Path("resource") / resource
        if candidate.exists():
            return candidate

        # 最后回退到 assets/resource/<name>
        candidate = Path("assets") / "resource" / resource

        return candidate

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
                required, forbidden = self._map_discs(oper, operator)
                oper["discs_ot_names"] = required
                oper["discs_ot_forbidden"] = forbidden
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

    def _map_discs(
        self, oper: Dict, operator: Optional[dict] = None
    ) -> Tuple[List[str], List[str]]:
        indexes = oper.get("discs_selected") or []
        if not indexes:
            return [], []
        if operator is None:
            operator = self._operators.get(oper.get("name"))
        if not operator:
            logger.warning(f"operators.json 中未找到密探: {oper.get('name')}")
            return [], []
        discs = operator.get("discs", [])
        required: List[str] = []
        forbidden: List[str] = []
        for idx in indexes:
            try:
                idx_int = int(idx)
            except (ValueError, TypeError):
                logger.warning(f"命盘索引格式无效 {idx} 对应 {oper.get('name')}")
                continue

            if idx_int <= 0:
                if idx_int == 0:
                    # 0 代表空槽
                    continue
                real_idx = abs(idx_int) - 1
                if real_idx < 0 or real_idx >= len(discs):
                    logger.warning(f"命盘索引越界 {idx_int} 对应 {oper.get('name')}")
                    continue
                disc = discs[real_idx]
                name = disc.get("ot_name") or ""
                if name:
                    forbidden.append(self._convert(name))
                continue

            real_idx = idx_int - 1  # copilot 序号从 1 开始
            if real_idx < 0 or real_idx >= len(discs):
                logger.warning(f"命盘索引越界 {idx_int} 对应 {oper.get('name')}")
                continue

            disc = discs[real_idx]
            name = disc.get("ot_name") or ""
            if name:
                required.append(self._convert(name))

        # 去重保持顺序
        def _uniq(items: List[str]) -> List[str]:
            seen = set()
            out = []
            for item in items:
                if item in seen:
                    continue
                seen.add(item)
                out.append(item)
            return out

        return _uniq(required), _uniq(forbidden)

    # ---------------- OCR/动作辅助 ----------------
    def _screenshot(self, context: Context):
        if _should_stop_ctx(context):
            return None
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
        expected = self._convert_name(name or "")
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
        if _should_stop_ctx(context):
            return
        context.run_task(self.SCROLL_TASK)

    def _click_box(self, context: Context, box: Tuple[int, int, int, int]):
        ox, oy, ow, oh = self.TARGET_OFFSET
        cx = box[0] + ox + (box[2] + ow) // 2
        cy = box[1] + oy + (box[3] + oh) // 2
        context.tasker.controller.post_click(cx, cy).wait()

    def _stop_task(self, context: Context):
        """请求停止整个任务，避免后续节点继续运行。"""
        try:
            setattr(context, "stop", True)
            tasker = getattr(context, "tasker", None)
            if tasker is not None and not tasker.stopping:
                tasker.post_stop().wait()
            logger.info("已请求停止整个任务")
        except Exception:
            logger.exception("请求停止任务时异常")

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
        if _should_stop_ctx(context):
            return False
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
        if _should_stop_ctx(context):
            return False
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
        target_name = self._convert_name(name)
        fallback_names = self._generate_fallback_names(name)
        while attempts < self.MAX_SCROLL:
            if _should_stop_ctx(context):
                logger.info("检测到主动结束任务，终止搜索密探")
                return False
            img = self._screenshot(context)
            if img is None:
                return False
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
                            time.sleep(0.5)
                            self._remove_old_first(context)
                            return True
                        self._click_box(context, box)
                        return True
            self._scroll_once(context)
            attempts += 1
        logger.error(f"未找到目标密探: {target_name}")
        self._stop_task(context)
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
        if _should_stop_ctx(context):
            logger.info("检测到主动结束任务，终止自动编队")
            return CustomAction.RunResult(success=False)
        params = self._parse_action_param(argv) or {}
        resource_root = self._resolve_resource_root(
            str(params.get("resource", "") or "")
        )
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
            if _should_stop_ctx(context):
                logger.info("检测到主动结束任务，终止自动编队")
                return CustomAction.RunResult(success=False)
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
        if _should_stop_ctx(context):
            logger.info(f"检测到主动结束任务，跳过 {name}")
            return False
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
        if _should_stop_ctx(context):
            return []
        img = self._screenshot(context)
        if img is None:
            return []
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
            recog = _extract_recognition(detail)
            text_candidates: List[str] = []
            if recog:
                raw_list = [
                    getattr(res, "text", "") or "" for res in _get_results(recog)
                ]
                raw_list = [t for t in raw_list if t]
                if len(raw_list) > 1:
                    text_candidates.append("".join(raw_list))
                elif raw_list:
                    text_candidates.extend(raw_list)

            if not text_candidates and detail and getattr(detail, "best_result", None):
                text_candidates = [getattr(detail.best_result, "text", "") or ""]

            for text_raw in text_candidates:
                text = self._convert(str(text_raw).strip())
                if text and text not in texts:
                    texts.append(text)

        logger.info(f"当前生效的命盘: {texts}")
        return texts

    def _find_missing(self, required: List[str], effects: List[str]) -> List[str]:
        missing: List[str] = []
        for need in required:
            best_score = 0.0
            best_effect = ""
            for effect in effects:
                score = self._similarity(need, effect)
                if score > best_score:
                    best_score = score
                    best_effect = effect
            # logger.info(
            #     f"命盘匹配: 需要={need}, 当前已有={best_effect}, 相似度={best_score:.3f}"
            # )
            if best_score < self.DEFAULT_THRESHOLD:
                missing.append(need)
        return missing

    def _find_forbidden(self, forbidden: List[str], effects: List[str]) -> List[str]:
        hits: List[str] = []
        for ban in forbidden:
            best_score = 0.0
            best_effect = ""
            for effect in effects:
                score = self._similarity(ban, effect)
                if score > best_score:
                    best_score = score
                    best_effect = effect
            # logger.info(
            #     f"禁用命盘匹配: ban={ban}, best_effect={best_effect}, score={best_score:.3f}, effects={effects}"
            # )
            if best_score >= self.DEFAULT_THRESHOLD:
                hits.append(ban)
        return hits

    def _similarity(self, need: str, effect: str) -> float:
        """计算命盘描述相似度，若数字部分不同则视为强不匹配。"""
        need_nums = _extract_numbers(need)
        effect_nums = _extract_numbers(effect)
        if need_nums and effect_nums and need_nums != effect_nums:
            return 0.0

        need_degree = _extract_degree(need)
        effect_degree = _extract_degree(effect)
        if need_degree and effect_degree and need_degree != effect_degree:
            return 0.0

        return SequenceMatcher(None, need, effect).ratio()

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
            resource_root = af._resolve_resource_root(
                str(params.get("resource", "") or "")
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
        if _should_stop_ctx(context):
            logger.info("检测到主动结束任务，终止命盘检测")
            return CustomAction.RunResult(success=False)
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

        has_any_discs = any(
            (oper.get("discs_ot_names") or oper.get("discs_ot_forbidden"))
            for oper in opers
        )
        if not has_any_discs:
            logger.info("方案未配置命盘要求，跳过命盘检测")
            return CustomAction.RunResult(success=True)

        overall_success = True
        violations_summary: List[str] = []

        for idx in range(opers_num):
            if _should_stop_ctx(context):
                logger.info("检测到主动结束任务，终止命盘检测循环")
                return CustomAction.RunResult(success=False)
            oper = opers[idx] if idx < len(opers) else None
            oper_name = oper.get("name") if oper else ""
            required = []
            forbidden = []
            if oper:
                required = [
                    self._convert(str(name))
                    for name in (oper.get("discs_ot_names") or [])
                    if name
                ]
                forbidden = [
                    self._convert(str(name))
                    for name in (oper.get("discs_ot_forbidden") or [])
                    if name
                ]

            if not required and not forbidden:
                logger.info(f"{idx + 1}号位({oper_name})未配置命盘要求，跳过")
            else:
                effects_before = self._read_active_effects(context)
                missing_before = self._find_missing(required, effects_before)
                forbidden_before = self._find_forbidden(forbidden, effects_before)
                violations_before = len(missing_before) + len(forbidden_before)

                if missing_before or forbidden_before:
                    logger.info(
                        f"{idx + 1}号位({oper_name})缺少命盘: {missing_before}，"
                        f"禁止命盘: {forbidden_before}，尝试切换命盘"
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
                    forbidden_after = forbidden_before
                    violations_after = violations_before
                    if close_hit:
                        logger.info("检测到仅一组命盘，保持当前侧效果")
                    else:
                        effects_after = self._read_active_effects(context)
                        missing_after = self._find_missing(required, effects_after)
                        forbidden_after = self._find_forbidden(forbidden, effects_after)
                        violations_after = len(missing_after) + len(forbidden_after)

                    if not missing_after and not forbidden_after:
                        logger.info(
                            f"{idx + 1}号位({oper_name})切换后命盘已符合作业要求"
                        )
                        effects_final = effects_after
                        missing_final: List[str] = []
                        forbidden_final: List[str] = []
                    else:
                        # 仅在切换成功且缺失更差时回退
                        if (not close_hit) and violations_after > violations_before:
                            logger.info(
                                f"切换后违规更多({violations_after}>{violations_before}), 再切回原侧"
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
                            forbidden_final = forbidden_before
                        else:
                            effects_final = effects_after
                            missing_final = missing_after
                            forbidden_final = forbidden_after

                        if missing_final or forbidden_final:
                            overall_success = False
                            logger.info(
                                f"{idx + 1}号位({oper_name})最终停留侧的效果: {effects_final}, 缺失: {missing_final}"
                            )
                            if forbidden_final:
                                logger.info(
                                    f"{idx + 1}号位({oper_name})最终禁用命盘: {forbidden_final}"
                                )
                            for name in missing_final:
                                logger.error(f"{oper_name}缺少命盘: {name}")
                            for name in forbidden_final:
                                logger.error(f"{oper_name}出现禁用命盘: {name}")
                            parts: List[str] = []
                            if missing_final:
                                parts.append(f"缺少{','.join(missing_final)}命盘")
                            if forbidden_final:
                                parts.append(
                                    f"出现禁用命盘:{','.join(forbidden_final)}"
                                )
                            if parts:
                                violations_summary.append(
                                    f"{idx + 1}号位({oper_name}):" + "/".join(parts)
                                )
                else:
                    logger.info(f"{idx + 1}号位({oper_name})命盘已符合作业要求")

            if idx < opers_num - 1:
                self._run_task(context, self.NEXT_TASK, wait_ms=1000)

        if not overall_success and violations_summary:
            logger.error("存在不符合作业要求的命盘，汇总如下：")
            for line in violations_summary:
                logger.error(line)

        return CustomAction.RunResult(success=overall_success)
