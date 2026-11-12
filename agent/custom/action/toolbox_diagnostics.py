"""Explicit toolbox events, privacy allowlists and one bounded daemon worker."""
from __future__ import annotations

import base64
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
import unicodedata
from uuid import UUID, uuid4

from utils import logger
from custom.action.inventory_reporting import resolve_token
from custom.action.toolbox_diagnostics_log import collect_log, valid_log
from custom.action.toolbox_diagnostics_transport import MAX_IMAGE, MAX_MANIFEST, deliver, trusted_base

ROOT = Path(__file__).resolve().parents[3]
PENDING = ROOT / "debug/toolbox_diagnostics/pending"
CODES = frozenset("collector_exception account_lookup_failed annotations_failed catalog_unavailable navigation_failed screenshot_or_ocr_failed identity_unresolved data_review upload_network_failed upload_http_failed upload_rejected upload_response_unverified".split())
REASONS = frozenset("page_timeout resource_language_mismatch enter_operator_page_failed disc_plan_unknown disc_tip_close_failed disc_switch_unconfirmed disc_restore_failed identity_changed_on_reread page_arrow_failed list_boundary_unverified page_turn_unconfirmed return_first_limit current_operator_failed form_identity_unconfirmed form_switch_failed form_tip_close_failed name_and_disc_unresolved name_fuzzy_disc_unconfirmed unmatched_in_report section_needs_review server_marked_review screenshot_failed ocr_failed catalog_load_failed account_lookup_failed annotations_failed upload_network_error upload_http_error upload_rejected_by_server upload_response_invalid upload_auth_missing active_only_requires_sync unknown_scan_mode unexpected_exception task_failed_no_detail retry_recovered".split())
_REASON = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
UI_SCRIPTS = frozenset({"simplified", "traditional", "mixed", "unknown"})
_CANDIDATE_ID = re.compile(r"char_[A-Za-z0-9_]{1,80}\Z")
_NAME_SAMPLE = re.compile(r"[\u3400-\u9fff]{1,8}\Z")
STAGES = frozenset("initialize account annotations catalog enter main detail huaji disc screenshot ocr identity navigation checkpoint preview commit finalize".split())
SECTIONS = frozenset("identity basic combat_stats oddities huaji disc_loadouts".split())
STATUSES = frozenset("partial review unavailable unmatched".split())
COUNTERS = Counter()
_PRIVATE = re.compile(r"(?i)(bearer|acc_|https?[:/]|www\.|[a-z]:[\\/]|[/\\]|@|token|cookie|authorization|report.*\.json)")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_FIELD_PART = r"(?:records|entries|unmatched|identity|basic|combat_stats|oddities|huaji|disc_loadouts|discs|ot_name|operator_id|level|elite|star_level|values|observed_attack|observed_hp)"
_FIELD = re.compile(_FIELD_PART + r"(?:\[\d{1,3}\]|\." + _FIELD_PART + r")*\Z")
ERROR_TYPES = frozenset("ValueError RuntimeError TypeError OSError TimeoutError KeyError AttributeError ExchangeApiError Exception".split())
_TRACE_TEXT = re.compile(r"(?!.*[0-9]{9,})(?:[\u3400-\u9fff]{1,8}|[0-9+%.]{1,12})\Z")
_ALGORITHM = re.compile(r"[A-Za-z0-9_]{1,32}\Z")


def _node(value, secrets=()):
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        return ""
    if value != "(direct)" and any(c not in "_-·@ " and unicodedata.category(c)[0] not in "LN" for c in value):
        return ""
    return _text(value, re.compile(r".+"), secrets)


def _box(value, resolution):
    if not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value):
        return False
    if not re.fullmatch(r"[1-9][0-9]{0,4}x[1-9][0-9]{0,4}", resolution):
        return False
    width, height = map(int, resolution.split("x"))
    x, y, w, h = value
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height


def encode_manifest(manifest, secrets):
    """Recheck the complete wire schema, including packages restored from disk."""
    if set(manifest) - {"trace"} != {"schema_version", "event_id", "run_id", "feature", "client", "catalog", "occurred_at", "outcome", "issues", "images"}:
        raise ValueError("diagnostics_schema")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["feature"] != "operator_info_collection" or manifest["outcome"] not in {"failed", "upload_failed", "data_review"}:
        raise ValueError("diagnostics_schema")
    for key in ("event_id", "run_id"):
        if str(UUID(manifest[key])) != manifest[key]:
            raise ValueError("diagnostics_schema")
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT[0-9:.]+(?:Z|[+-]\d\d:\d\d)", manifest["occurred_at"]):
        raise ValueError("diagnostics_schema")
    client, catalog = manifest["client"], manifest["catalog"]
    if set(client) - {"ui_script"} != {"resource_version", "maafw_version", "resource", "resolution"} or set(catalog) != {"version", "source"}:
        raise ValueError("diagnostics_schema")
    if "ui_script" in client and (not isinstance(client["ui_script"], str) or client["ui_script"] not in UI_SCRIPTS):
        raise ValueError("diagnostics_schema")
    if client["resource"] not in {"base", "zh_tw"} or catalog["source"] not in {"remote", "cache", "builtin"}:
        raise ValueError("diagnostics_schema")
    for value in (client["resource_version"], client["maafw_version"], catalog["version"]):
        if not _text(value, re.compile(r"[A-Za-z0-9_.+-]{1,64}"), secrets):
            raise ValueError("diagnostics_schema")
    if not re.fullmatch(r"(?:unknown|[1-9][0-9]{0,4}x[1-9][0-9]{0,4})", client["resolution"]):
        raise ValueError("diagnostics_schema")
    trace = manifest.get("trace", [])
    if not isinstance(trace, list) or len(trace) > 64:
        raise ValueError("diagnostics_schema")
    previous_t = 0
    for step in trace:
        if (not isinstance(step, dict) or set(step) - {"algo", "score", "box", "text", "dt"} != {"kind", "node", "stage", "hit", "n", "t"}
                or step["kind"] not in {"reco", "action"} or not _node(step["node"], secrets)
                or step["stage"] not in STAGES or type(step["hit"]) is not bool
                or type(step["n"]) is not int or not 1 <= step["n"] <= 10000
                or type(step["t"]) is not int or not previous_t <= step["t"] <= 86400000):
            raise ValueError("diagnostics_schema")
        previous_t = step["t"]
        if "dt" in step and (type(step["dt"]) is not int or not 0 <= step["dt"] <= 86400000):
            raise ValueError("diagnostics_schema")
        if step["kind"] != "reco" and any(key in step for key in ("algo", "score", "text")):
            raise ValueError("diagnostics_schema")
        if "algo" in step and not _text(step["algo"], _ALGORITHM, secrets):
            raise ValueError("diagnostics_schema")
        if "score" in step and (type(step["score"]) not in {int, float} or not math.isfinite(step["score"])
                                or not -1 <= step["score"] <= 100000 or round(step["score"], 3) != step["score"]):
            raise ValueError("diagnostics_schema")
        if "box" in step and not _box(step["box"], client["resolution"]):
            raise ValueError("diagnostics_schema")
        if "text" in step and (step.get("algo") != "OCR" or not _text(step["text"], _TRACE_TEXT, secrets)):
            raise ValueError("diagnostics_schema")
    if not isinstance(manifest["issues"], list) or not 1 <= len(manifest["issues"]) <= 32:
        raise ValueError("diagnostics_schema")
    for item in manifest["issues"]:
        if (set(item) - {"code", "stage", "count", "operator_ids", "candidate_operator_ids", "reason", "error_type", "stack", "blocking_errors", "section_status", "ocr_samples", "node"}
                or item["code"] not in CODES or item["stage"] not in STAGES or type(item["count"]) is not int or item["count"] < 1):
            raise ValueError("diagnostics_schema")
        if "node" in item and (item["node"] == "(direct)" or not _node(item["node"], secrets)):
            raise ValueError("diagnostics_schema")
        if "reason" in item and (not isinstance(item["reason"], str) or not _REASON.fullmatch(item["reason"]) or item["reason"] not in REASONS):
            raise ValueError("diagnostics_schema")
        if not isinstance(item["operator_ids"], list) or len(item["operator_ids"]) > 64 or any(not _text(oid, re.compile(r"char_[a-z0-9_]{1,60}"), secrets) for oid in item["operator_ids"]):
            raise ValueError("diagnostics_schema")
        candidates = item.get("candidate_operator_ids", [])
        if (not isinstance(candidates, list) or len(candidates) > 5
                or any(not _text(oid, _CANDIDATE_ID, secrets) for oid in candidates)
                or len(candidates) != len(set(candidates))):
            raise ValueError("diagnostics_schema")
        if "error_type" in item and item["error_type"] not in ERROR_TYPES:
            raise ValueError("diagnostics_schema")
        stack = item.get("stack", [])
        if not isinstance(stack, list) or len(stack) > 8 or any(not re.fullmatch(r"agent/[A-Za-z0-9_/]+\.py:[A-Za-z_][A-Za-z0-9_]*:[0-9]{1,8}", frame) for frame in stack):
            raise ValueError("diagnostics_schema")
        errors = item.get("blocking_errors", [])
        if not isinstance(errors, list) or errors != blocking_errors({"blocking_errors": errors}):
            raise ValueError("diagnostics_schema")
        if any(k not in SECTIONS or v not in STATUSES for k, v in item.get("section_status", {}).items()):
            raise ValueError("diagnostics_schema")
        samples = item.get("ocr_samples", [])
        if not isinstance(samples, list) or len(samples) > 4 or any(not re.fullmatch(r"[\u3400-\u9fff0-9+%]{1,32}", sample) for sample in samples):
            raise ValueError("diagnostics_schema")
    if not isinstance(manifest["images"], list) or len(manifest["images"]) > 3:
        raise ValueError("diagnostics_schema")
    for index, meta in enumerate(manifest["images"]):
        if set(meta) - {"index", "role", "roi", "operator_id", "trace_index", "node_roi"} or meta["index"] != index or meta["role"] not in {"disc_cells", "main_panel", "name_area", "last_frame"}:
            raise ValueError("diagnostics_schema")
        if len(meta["roi"]) != 4 or any(type(value) is not int or not 0 <= value <= 99999 for value in meta["roi"]) or min(meta["roi"][2:]) <= 0:
            raise ValueError("diagnostics_schema")
        if "operator_id" in meta and not _text(meta["operator_id"], re.compile(r"char_[a-z0-9_]{1,60}"), secrets):
            raise ValueError("diagnostics_schema")
        if any(key in meta for key in ("trace_index", "node_roi")) and meta["role"] != "last_frame":
            raise ValueError("diagnostics_schema")
        if "trace_index" in meta and (type(meta["trace_index"]) is not int or not 0 <= meta["trace_index"] < len(trace)):
            raise ValueError("diagnostics_schema")
        if "node_roi" in meta and not _box(meta["node_roi"], client["resolution"]):
            raise ValueError("diagnostics_schema")
    raw = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
    decoded = raw.decode()
    if len(raw) > MAX_MANIFEST or any(secret and (secret in decoded or json.dumps(secret, ensure_ascii=False)[1:-1] in decoded) for secret in secrets):
        raise ValueError("diagnostics_privacy")
    return raw


def boundary(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            COUNTERS["diagnostics_internal_error"] += 1
            try:
                logger.bind(console=False).warning("toolbox_diagnostics_internal_error")
            except Exception:
                pass
            return None
    return guarded


def call(session, method, *args, **kwargs):
    if session is not None:
        # Also isolate replaced adapters and injected/failed worker implementations.
        try:
            return getattr(session, method)(*args, **kwargs)
        except Exception:
            COUNTERS["diagnostics_internal_error"] += 1


def _text(value, pattern, secrets=()):
    if not isinstance(value, str) or _PRIVATE.search(value) or any(s and s in value for s in secrets):
        return ""
    return value if pattern.fullmatch(value) else ""


def blocking_errors(payload):
    """Select code/field only; never retain a response object or message."""
    result = []
    if not isinstance(payload, dict):
        return result
    candidates = [payload]
    items = payload.get("items")
    if isinstance(items, list):
        candidates.extend(item for item in items[:300] if isinstance(item, dict))
    for item in candidates:
        errors = item.get("blocking_errors")
        if not isinstance(errors, list):
            continue
        for error in errors[:32]:
            if not isinstance(error, dict):
                continue
            code = _text(error.get("code"), _IDENTIFIER)
            field = _text(error.get("field"), _FIELD)
            if code:
                result.append({"code": code, **({"field": field} if field else {})})
    return result[:32]


class DiagnosticSession:
    def __init__(self, token, base_url, mode, resource):
        self.started = time.monotonic()
        self.started_at = datetime.now()
        self.log_directory = Path.cwd() / "debug/custom"
        self.token = token
        self.base_url = base_url
        self.mode = mode
        self.secrets = [token]
        self.run_id = str(uuid4())
        self.event_id = str(uuid4())
        self.occurred_at = datetime.now(timezone.utc).isoformat()
        self.client = {"resource_version": "unknown", "maafw_version": "unknown",
                       "resource": "zh_tw" if resource == "zh_tw" else "base", "resolution": "unknown"}
        self.catalog = {"version": "unknown", "source": "builtin"}
        self.issues = {}
        self.client_review_count = 0
        self.images = []
        self.last_frame = None
        self.image_cache = OrderedDict()
        self.problem_operators = {}
        self.known_seen = OrderedDict()
        self.samples = set()
        self.operator_id = ""
        self.stage = "initialize"
        self.epoch = _WORKER.configure(mode)
        for path in (ROOT / "interface.json", ROOT / "assets/interface.json"):
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8")).get("version", "unknown")
                self.client["resource_version"] = self.safe_version(value)
                break
        try:
            from maa.library import Library
            self.client["maafw_version"] = self.safe_version(Library.version())
        except Exception:
            pass
        self.trace = None
        try:
            from custom.action.toolbox_diagnostics_trace import DiagnosticTrace
            self.trace = DiagnosticTrace(self)
        except Exception:
            COUNTERS["diagnostics_internal_error"] += 1

    def safe_version(self, value):
        return _text(value, re.compile(r"[A-Za-z0-9_.+-]{1,64}"), self.secrets) or "unknown"

    @boundary
    def catalog_info(self, data, meta=None):
        meta = meta or {}
        version = meta.get("version", "unknown")
        if isinstance(version, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})", version):
            version = version[:10].replace("-", "") + version[10:].replace(":", "")
        self.catalog = {"version": self.safe_version(version),
                        "source": meta.get("source") if meta.get("source") in {"remote", "cache", "builtin"} else "builtin"}
        for operator in data.get("OPERATORS", []):
            for value in [operator.get("name"), *(disc.get("ot_name") for disc in operator.get("discs", []))]:
                if isinstance(value, str) and re.fullmatch(r"[\u3400-\u9fff0-9+%]{1,32}", value):
                    self.samples.add(value)

    @boundary
    def protect(self, *values):
        self.secrets.extend(value for value in values if isinstance(value, str) and value)

    @boundary
    def set_stage(self, stage):
        self.stage = stage if stage in STAGES else "initialize"

    @boundary
    def set_ui_script(self, script):
        if script in UI_SCRIPTS:
            self.client["ui_script"] = script

    @boundary
    def enable_trace_text(self, enabled):
        if self.trace is not None:
            self.trace.enable_text(enabled)

    @boundary
    def observe_main(self, frame, roi_config, operator_id):
        height, width = frame.shape[:2]
        self.client["resolution"] = f"{width}x{height}"
        self.operator_id = _text(operator_id, re.compile(r"char_[a-z0-9_]{1,60}"), self.secrets)
        # Unidentified pages cannot share evidence with the previous operator.
        if not self.operator_id:
            self.image_cache.pop("", None)
            self.known_seen.pop("", None)
        self.known_seen.setdefault(self.operator_id, [])
        self.known_seen.move_to_end(self.operator_id)
        if len(self.known_seen) > 200:
            self.known_seen.popitem(last=False)
        if not any(meta["role"] == "name_area" for meta, _ in self.image_cache.get(self.operator_id, [])):
            self.crop(frame, roi_config["main_name"], "name_area", operator_id)
        elif self.mode == "images":
            self.image_cache.move_to_end(self.operator_id)

    @boundary
    def resolve_operator(self, operator_id):
        oid = _text(operator_id, re.compile(r"char_[a-z0-9_]{1,60}"), self.secrets)
        if not oid or oid == self.operator_id:
            return
        images = self.image_cache.pop(self.operator_id, [])
        if images:
            self.image_cache[oid] = [({**meta, "operator_id": oid}, data) for meta, data in images]
            self.image_cache.move_to_end(oid)
        samples = self.known_seen.pop(self.operator_id, [])
        self.known_seen[oid] = samples
        self.known_seen.move_to_end(oid)
        if len(self.known_seen) > 200:
            self.known_seen.popitem(last=False)
        self.operator_id = oid

    @boundary
    def observe_disc(self, text):
        samples = self.known_seen.get(self.operator_id)
        if samples is not None and text in self.samples and len(samples) < 4 and text not in samples:
            samples.append(text)

    @boundary
    def observe_frame(self, frame):
        if self.trace is not None:
            self.trace.observe_frame(frame)
        else:
            height, width = frame.shape[:2]
            self.client["resolution"] = f"{width}x{height}"
            if self.mode == "images":
                self.last_frame = frame

    @boundary
    def _failure_image(self):
        failure = self.trace.failure_image() if self.trace is not None else None
        frame, annotations = failure if failure is not None else (self.last_frame, {})
        if frame is None:
            return None
        import cv2
        height, width = frame.shape[:2]
        self.client["resolution"] = f"{width}x{height}"
        for quality in (85, 70, 55, 40, 25, 10, 1):
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and encoded.nbytes <= MAX_IMAGE:
                meta = {"role": "last_frame", "roi": [0, 0, width, height], "index": 0, **annotations}
                if self.operator_id:
                    meta["operator_id"] = self.operator_id
                return meta, encoded.tobytes()
        return None

    @boundary
    def issue(self, code, stage=None, operator_id=None, exception=None, errors=(), statuses=None, sample=None, reason=None,
              name_samples=(), candidate_operator_ids=()):
        if code not in CODES:
            return
        if reason is None:
            reason = getattr(exception, "reason", None)
        if not isinstance(reason, str) or not _REASON.fullmatch(reason) or reason not in REASONS:
            reason = "unexpected_exception" if exception is not None else None
        stage = stage if stage in STAGES else self.stage if self.stage in STAGES else "initialize"
        node = self.trace.last_failed_node(stage) if self.trace is not None else ""
        oid = _text(self.operator_id if operator_id is None else operator_id, re.compile(r"char_[a-z0-9_]{1,60}"), self.secrets)
        clean_errors = blocking_errors({"blocking_errors": list(errors)})
        fields = sorted({error.get("field", "") for error in clean_errors}) or [""]
        for field in fields:
            key = (code, stage, field, reason, node)
            if code in {"data_review", "identity_unresolved"} and oid:
                key += (oid,)
            if key not in self.issues:
                if len(self.issues) >= 32:
                    COUNTERS["diagnostics_issues_dropped"] += 1
                    continue
                self.issues[key] = {"code": code, "stage": stage, "count": 0, "operator_ids": []}
                if node:
                    self.issues[key]["node"] = node
                if reason is not None:
                    self.issues[key]["reason"] = reason
            item = self.issues[key]
            item["count"] += 1
            if oid and oid not in item["operator_ids"] and len(item["operator_ids"]) < 64:
                item["operator_ids"].append(oid)
            if not oid or oid in item["operator_ids"]:
                self.problem_operators[oid] = (self.problem_operators.get(oid, False) or stage == "disc"
                                               or "disc_loadouts" in (statuses or {}) or (reason or "").startswith("disc_"))
            selected = [error for error in clean_errors if error.get("field", "") == field]
            if selected:
                existing = item.setdefault("blocking_errors", [])
                for error in selected:
                    if error not in existing and len(existing) < 32:
                        existing.append(error)
            if statuses:
                item.setdefault("section_status", {}).update({k: v for k, v in statuses.items() if k in SECTIONS and v in STATUSES})
            if code == "identity_unresolved":
                for value in (sample, *name_samples):
                    value = _text(value, _NAME_SAMPLE, self.secrets)
                    if value:
                        samples = item.setdefault("ocr_samples", [])
                        if value not in samples and len(samples) < 4:
                            samples.append(value)
                for candidate in candidate_operator_ids:
                    candidate = _text(candidate, _CANDIDATE_ID, self.secrets)
                    if candidate:
                        candidates = item.setdefault("candidate_operator_ids", [])
                        if candidate not in candidates and len(candidates) < 5:
                            candidates.append(candidate)
            else:
                if code == "data_review" and reason == "section_needs_review":
                    review_samples = [_text(value, _NAME_SAMPLE, self.secrets) for value in name_samples]
                    samples = list(dict.fromkeys(
                        value for value in (*review_samples, *item.get("ocr_samples", [])) if value
                    ))[:4]
                    if samples:
                        item["ocr_samples"] = samples
                if sample in self.samples and not _PRIVATE.search(sample) and not any(s in sample for s in self.secrets if s):
                    samples = item.setdefault("ocr_samples", [])
                    if sample not in samples and len(samples) < 4:
                        samples.append(sample[:32])
            if exception is not None:
                item["error_type"] = next((cls.__name__ for cls in type(exception).__mro__ if cls.__name__ in ERROR_TYPES), "Exception")
                stack = []
                tb = exception.__traceback__
                while tb is not None:
                    filename = Path(tb.tb_frame.f_code.co_filename)
                    try:
                        relative = filename.resolve().relative_to(ROOT).as_posix()
                        function = tb.tb_frame.f_code.co_name
                        if re.fullmatch(r"(?:agent/)[A-Za-z0-9_/.]+\.py", relative) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function):
                            stack.append(f"{relative}:{function}:{tb.tb_lineno}")
                    except ValueError:
                        pass
                    tb = tb.tb_next
                item["stack"] = stack[-8:]

    @boundary
    def checkpoint(self, document):
        self.client_review_count = 0
        for record in document.get("records", []):
            for entry in record.get("entries", []):
                if any(value != "ready" for value in entry.get("section_status", {}).values()):
                    self.client_review_count += 1
                statuses = {k: v for k, v in entry.get("section_status", {}).items() if v in STATUSES}
                if statuses:
                    diagnostics = entry.get("diagnostics")
                    disc_names = diagnostics.get("disc_name_review", []) if isinstance(diagnostics, dict) else []
                    self.issue("data_review", "checkpoint", entry.get("operator_id") or "", statuses=statuses,
                               sample=next(iter(self.known_seen.get(entry.get("operator_id"), [])), None), reason="section_needs_review",
                               name_samples=disc_names if isinstance(disc_names, list) else ())
            if record.get("unmatched"):
                for unmatched in record["unmatched"]:
                    self.issue("identity_unresolved", "checkpoint", unmatched.get("operator_id") or "", statuses={"identity": "unmatched"}, reason="unmatched_in_report",
                               name_samples=(unmatched.get("raw_name"),),
                               candidate_operator_ids=unmatched.get("candidates", []))

    @boundary
    def upload_response(self, payload, stage):
        from custom.action.operator_growth_exchange import upload_response_verified
        if not upload_response_verified(payload):
            self.issue("upload_response_unverified", stage, reason="upload_response_invalid")
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        if isinstance(data, dict) and sum(data[key] for key in ("partial", "review") if type(data.get(key)) is int and data[key] > 0) > self.client_review_count:
            self.issue("data_review", stage, reason="server_marked_review")

    @boundary
    def crop(self, frame, roi, role, operator_id=None):
        if self.mode != "images" or role not in {"disc_cells", "main_panel", "name_area"}:
            return
        oid = _text(self.operator_id if operator_id is None else operator_id, re.compile(r"char_[a-z0-9_]{1,60}"), self.secrets)
        images = self.image_cache.get(oid, [])
        if oid in self.image_cache:
            self.image_cache.move_to_end(oid)
        if len(images) >= 3:
            return
        import cv2
        height, width = frame.shape[:2]
        self.client["resolution"] = f"{width}x{height}"
        x, y, w, h = roi
        scaled = [round(x * width / 720), round(y * height / 1280), round(w * width / 720), round(h * height / 1280)]
        x, y, w, h = scaled
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            return
        ok, encoded = cv2.imencode(".jpg", frame[y:y+h, x:x+w].copy(), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok and encoded.nbytes <= MAX_IMAGE:
            meta = {"role": role, "roi": scaled}
            if oid:
                meta["operator_id"] = oid
            images.append((meta, encoded.tobytes()))
            self.image_cache[oid] = images
            if len(self.image_cache) > 200:
                self.image_cache.popitem(last=False)

    @boundary
    def finish(self, success, stopped=False):
        try:
            if self.trace is not None:
                self.trace.close()
            self._finish(success, stopped)
        finally:
            self.last_frame = None
            if self.trace is not None:
                self.trace.release_frames()

    def _finish(self, success, stopped):
        if stopped:
            return
        if not success and not self.issues:
            self.issue("collector_exception", "finalize", reason="task_failed_no_detail")
        if not self.issues:
            return
        codes = {item["code"] for item in self.issues.values()}
        outcome = "failed" if not success else "upload_failed" if any(code.startswith("upload_") for code in codes) else "data_review"
        self.images = []
        if self.mode == "images":
            if outcome == "failed":
                failure_image = self._failure_image()
                if failure_image is not None:
                    self.images.append(failure_image)
            candidates = []
            for oid, disc in self.problem_operators.items():
                if oid and self.image_cache.get(oid):
                    preferred = "disc_cells" if disc else "name_area"
                    candidates.append(sorted(self.image_cache[oid], key=lambda image: image[0]["role"] != preferred))
            if not candidates and self.image_cache.get(""):
                preferred = "disc_cells" if self.problem_operators.get("") else "name_area"
                candidates = [sorted(self.image_cache[""], key=lambda image: image[0]["role"] != preferred)]
            for batch in ([images[0] for images in candidates], [image for images in candidates for image in images[1:]]):
                for meta, data in batch:
                    if len(self.images) >= 3:
                        break
                    self.images.append(({**meta, "index": len(self.images)}, data))
        manifest = {"schema_version": 1, "event_id": self.event_id, "run_id": self.run_id,
                    "feature": "operator_info_collection", "client": self.client, "catalog": self.catalog,
                    "occurred_at": self.occurred_at, "outcome": outcome, "issues": list(self.issues.values()),
                    "images": [meta for meta, _ in self.images],
                    "trace": self.trace.snapshot() if self.trace is not None else []}
        try:
            encode_manifest(manifest, self.secrets)
        except (ValueError, TypeError, KeyError):
            COUNTERS["diagnostics_privacy_dropped"] += 1
            return
        log = collect_log(self.log_directory, self.started_at, self.secrets)
        _WORKER.submit(self, (manifest, [data for _, data in self.images], log))


class DiagnosticWorker:
    MAX_PACKAGES = 20
    MAX_BYTES = 20 * 1024 * 1024
    MAX_AGE = 7 * 86400

    def __init__(self, root=PENDING):
        self.root = root
        self.queue = queue.Queue(maxsize=4)
        self.lock = threading.Lock()
        self.thread = None
        self.mode = "off"
        self.epoch = 0
        self.purge = threading.Event()
        self.strip_images = threading.Event()

    def configure(self, mode):
        with self.lock:
            if self.mode != mode:
                self.epoch += 1
            self.mode = mode
            if mode == "off":
                self.purge.set()
            elif mode == "summary":
                self.strip_images.set()
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name="toolbox-diagnostics", daemon=True)
                self.thread.start()
            return self.epoch

    def submit(self, session, package=None):
        try:
            self.queue.put_nowait((session.epoch, session.token, session.base_url, package))
        except queue.Full:
            COUNTERS["diagnostics_queue_dropped"] += 1

    def _prune(self):
        paths = []
        if not self.root.exists():
            return paths
        for path in self.root.glob("*.json"):
            if path.is_symlink():
                continue
            stat = path.stat()
            if time.time() - stat.st_mtime > self.MAX_AGE or stat.st_size > self.MAX_BYTES:
                path.unlink()
                COUNTERS["diagnostics_pending_dropped"] += 1
            else:
                paths.append((stat.st_mtime, path, stat.st_size))
        paths.sort()
        size = sum(item[2] for item in paths)
        while len(paths) > self.MAX_PACKAGES or size > self.MAX_BYTES:
            _, path, length = paths.pop(0)
            path.unlink()
            size -= length
            COUNTERS["diagnostics_pending_dropped"] += 1
        return [path for _, path, _ in paths]

    def _write_package(self, path, envelope):
        previous = path.stat() if path.exists() else None
        raw = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(raw, encoding="utf-8")
            temporary.chmod(0o600)
            # Retrying/withdrawing images must not extend the seven-day retention.
            if previous is not None:
                os.utime(temporary, (previous.st_atime, previous.st_mtime))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _save(self, manifest, files, binding, retry_at, log=None):
        encode_manifest(manifest, ())
        if log is not None and not valid_log(log):
            log = None
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (str(uuid4()) + ".json")
        # The local envelope binds retries to the same credential without storing it.
        envelope = {"manifest": manifest, "files": [base64.b64encode(data).decode() for data in files],
                    "binding": binding, "retry_at": retry_at}
        if log is not None:
            envelope["log"] = base64.b64encode(log).decode()
        self._write_package(path, envelope)
        self._prune()

    def _send(self, manifest, files, token, base_url, log=None):
        if self.mode != "images":
            manifest = {**manifest, "images": []}
            files = []
        raw = encode_manifest(manifest, (token,))
        if len(manifest["images"]) != len(files):
            raise ValueError("diagnostics_images")
        return deliver(raw, files, base_url, token, direct=True, log=log)

    def _process(self, job):
        epoch, token, base_url, package = job
        if epoch != self.epoch or self.mode == "off":
            return
        binding = hashlib.sha256(token.encode()).hexdigest()
        # Persist first so an offline/terminated send remains recoverable with its event ID.
        if package is not None:
            manifest, files, log = package
            self._save(manifest, files, binding, 0, log)
        for path in self._prune():
            if epoch != self.epoch or self.mode == "off":
                return
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                encode_manifest(envelope["manifest"], (token,))
                if (set(envelope) - {"log"} != {"manifest", "files", "binding", "retry_at"}
                        or not isinstance(envelope["files"], list) or len(envelope["files"]) > 3
                        or not isinstance(envelope["retry_at"], (int, float))):
                    raise ValueError("diagnostics_pending")
                files = [base64.b64decode(data, validate=True) for data in envelope["files"]]
                if len(envelope["manifest"]["images"]) != len(files) or any(len(data) > MAX_IMAGE or not data.startswith(b"\xff\xd8") for data in files):
                    raise ValueError("diagnostics_images")
                log = None
                if "log" in envelope:
                    try:
                        log = base64.b64decode(envelope["log"], validate=True)
                        if not valid_log(log):
                            log = None
                    except (ValueError, TypeError):
                        pass
                    if log is None:
                        envelope.pop("log", None)
            except (ValueError, TypeError, KeyError):
                path.unlink(missing_ok=True)
                COUNTERS["diagnostics_pending_dropped"] += 1
                continue
            if envelope.get("binding") != binding or envelope.get("retry_at", 0) > time.time():
                continue
            manifest = envelope["manifest"]
            state, retry_at = self._send(manifest, files, token, base_url, log)
            if state == "accepted" and package is not None and manifest["event_id"] == package[0]["event_id"]:
                logger.info("已上报诊断信息，帮助开发者排查问题")
            if state != "pending":
                path.unlink(missing_ok=True)
            else:
                envelope["retry_at"] = retry_at
                if self.mode != "images":
                    envelope["files"] = []
                    envelope["manifest"]["images"] = []
                self._write_package(path, envelope)

    @boundary
    def _tick(self):
        if self.purge.is_set():
            self.purge.clear()
            if self.root.exists():
                for path in self.root.iterdir():
                    if path.suffix in {".json", ".tmp"} and not path.is_symlink():
                        path.unlink(missing_ok=True)
        if self.strip_images.is_set():
            self.strip_images.clear()
            for path in self._prune():
                try:
                    envelope = json.loads(path.read_text(encoding="utf-8"))
                    encode_manifest(envelope["manifest"], ())
                    envelope["files"] = []
                    envelope["manifest"]["images"] = []
                    self._write_package(path, envelope)
                except (ValueError, TypeError, KeyError):
                    path.unlink(missing_ok=True)
                    COUNTERS["diagnostics_pending_dropped"] += 1
        try:
            job = self.queue.get(timeout=0.25)
        except queue.Empty:
            return
        try:
            self._process(job)
        finally:
            self.queue.task_done()

    def _run(self):
        while True:
            self._tick()


_WORKER = DiagnosticWorker()


@boundary
def start_diagnostics(context, params):
    mode = params.get("toolbox_diagnostics", "summary")
    if mode not in {"summary", "images"}:
        _WORKER.configure("off")
        return None
    _WORKER.configure(mode)
    if not params.get("upload") or params.get("name_compare") or params.get("disc_compare"):
        return None
    if getattr(context, "stop", False) is True:
        return None
    try:
        node = context.get_node_data("在线上传认证")
    except Exception:
        return None
    attach = node.get("attach", {}) if isinstance(node, dict) else {}
    if not isinstance(attach, dict):
        return None
    token = attach.get("token")
    if token is None or (isinstance(token, str) and not token.strip()):
        token = resolve_token("")
    base_url = attach.get("base_url")
    if not isinstance(token, str) or not token.strip() or not isinstance(base_url, str) or not trusted_base(base_url):
        return None
    session = DiagnosticSession(token.strip(), base_url, mode, params.get("resource"))
    _WORKER.submit(session)
    return session


# Register independently of the legacy file-log sinks, before any task starts.
try:
    from custom.action import toolbox_diagnostics_trace
except Exception:
    COUNTERS["diagnostics_internal_error"] += 1
