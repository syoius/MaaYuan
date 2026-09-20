"""YuanHub operator growth exchange v3 conversion and transport.

The collector keeps its diagnostic report as the source model.  This module is
the only conversion boundary used by both the JSON exporter and OpenAPI
preview/commit calls.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from utils import logger

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "AgentInfoReport.json"
DEFAULT_SCHEMA_PATHS = (
    REPO_ROOT / "docs" / "schema" / "operator-growth-exchange-v3.schema.json",
    REPO_ROOT / "docs" / "schemas" / "operator-growth-exchange-v3.schema.json",
)
DEFAULT_GAME = "代号鸢"
DEFAULT_CATALOG_VERSION = None
PREVIEW_PATH = "/open-api/operator/scan-import/preview"
COMMIT_PATH = "/open-api/operator/scan-import/commit"
ANNOTATIONS_PATH = "/open-api/operator/annotations"
EXCHANGE_FORMAT = "myshare-operator-exchange"
EXCHANGE_VERSION = 3
VALID_SECTION_STATUS = {"ready", "partial", "review", "unavailable"}
ODDITY_KEYS = {"攻击力": "attack", "生命值": "hp", "治疗加成": "special", "增伤值": "special", "免伤值": "special", "attack": "attack", "hp": "hp", "special": "special"}
ODDITY_LIMITS = {
    3: {"attack": 300, "hp": 1560, "special": 9},
    4: {"attack": 305, "hp": 1820, "special": 11},
    5: {"attack": 500, "hp": 2600, "special": 15},
}


@lru_cache(maxsize=1)
def _operator_catalog_by_id() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(
            (REPO_ROOT / "agent" / "operators.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(item["id"]): item
        for item in payload.get("OPERATORS", [])
        if isinstance(item, dict) and item.get("id")
    }


def set_operator_catalog(payload: dict[str, Any]) -> None:
    """Use the scan's catalog even when its local file could not be updated."""
    catalog = _operator_catalog_by_id()
    catalog.clear()
    catalog.update({str(item["id"]): item for item in payload["OPERATORS"]})


@dataclass(frozen=True)
class FrozenExchange:
    document: dict[str, Any]
    scan_id: str
    output_path: Path


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def stable_scan_id(value: Any = None) -> str:
    text = str(value or "").strip()
    return text or uuid.uuid4().hex


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def scrub_v3_equipment(document: dict[str, Any]) -> None:
    """Remove retired agent-equipment scan data at the v3 document boundary."""
    def scrub_nested(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("star_stones", None)
            value.pop("equipped_star_stones", None)
            value.pop("star_stones_by_loadout", None)
            for nested in value.values():
                scrub_nested(nested)
        elif isinstance(value, list):
            for nested in value:
                scrub_nested(nested)

    records = document.get("records")
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict) or record.get("record_type") != "operator_snapshot":
            continue
        entries = record.get("entries")
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            statuses = entry.get("section_status")
            if isinstance(statuses, dict):
                statuses.pop("equipment", None)
            scrub_nested(entry)


def _status(value: str) -> str:
    return value if value in VALID_SECTION_STATUS else "review"


def _operator_match(record: dict[str, Any]) -> dict[str, Any]:
    operator_id = str(record.get("operator_id") or "").strip()
    debug = record.get("collection_debug")
    debug = debug if isinstance(debug, dict) else {}
    status = "ready" if operator_id else "review"
    method = "id" if operator_id else None
    if debug.get("operator_match") == "disc":
        method = "disc"
    elif debug.get("operator_match") == "name":
        method = "name"
    candidates = debug.get("operator_candidates")
    result: dict[str, Any] = {
        "status": status,
        "raw_name": record.get("name_raw") or record.get("name_cleaned") or record.get("name") or "",
    }
    if method:
        result["method"] = method
    if isinstance(candidates, list) and candidates:
        candidate_ids = []
        for candidate in candidates:
            candidate_id = candidate.get("operator_id") if isinstance(candidate, dict) else candidate
            if candidate_id and candidate_id not in candidate_ids:
                candidate_ids.append(str(candidate_id))
        if candidate_ids:
            result["candidates"] = candidate_ids
    if status == "ready":
        result["confidence"] = 1.0 if method in {"id", "name"} else 0.95
    return result


def _basic(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    required = ("level", "cultivation")
    if all(isinstance(stats.get(key), int) for key in required):
        return "ready", {"level": stats["level"], "elite": stats["cultivation"]}
    data = {key: stats[key] for key in required if isinstance(stats.get(key), int)}
    return ("partial" if data else "unavailable"), data


def _star_level(record: dict[str, Any]) -> tuple[str, int | None]:
    huaji = record.get("huaji") if isinstance(record.get("huaji"), dict) else {}
    if huaji.get("awakened") or huaji.get("layout") == "awakened":
        return "ready", 31
    stars = huaji.get("stars")
    nodes = huaji.get("nodes")
    if not isinstance(stars, int) or not 0 <= stars <= 5:
        return "unavailable", None
    if huaji.get("layout") == "pending_awaken":
        return ("ready", 30) if stars == 5 else ("review", None)
    if huaji.get("layout") == "sp":
        return ("ready", stars) if 1 <= stars <= 5 else ("review", None)
    if stars == 0:
        return "ready", 0
    if not isinstance(nodes, list) or len(nodes) != 5:
        return "review", None
    active = [node.get("active") is True for node in nodes if isinstance(node, dict)]
    if len(active) != 5 or any(active[index] and not all(active[:index]) for index in range(5)):
        return "review", None
    continuous = 0
    for value in active:
        if not value:
            break
        continuous += 1
    return "ready", 6 * (stars - 1) + continuous + 1


def _oddities(record: dict[str, Any], star_level: int | None) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    raw = record.get("oddities") if isinstance(record.get("oddities"), dict) else {}
    diagnostics: dict[str, Any] = {}
    formal: dict[str, Any] = {}
    labels: dict[str, str] = {}
    raw_items = list(raw.items())
    for index, (key, value) in enumerate(raw_items):
        canonical = ODDITY_KEYS.get(str(key))
        if canonical is None and str(key) in {"field_1", "field_2", "field_3"}:
            canonical = ("attack", "hp", "special")[int(str(key)[-1]) - 1]
        if canonical is None:
            continue
        labels[canonical] = str(key)
        if isinstance(value, dict) and isinstance(value.get("current"), int):
            formal[canonical] = {"current": value["current"]}
            rarity = _operator_rarity(record.get("operator_id"))
            if rarity in ODDITY_LIMITS:
                limit = ODDITY_LIMITS[rarity][canonical]
                if value["current"] < 0 or value["current"] > limit:
                    diagnostics.setdefault("oddity_warnings", []).append(
                        f"{canonical}={value['current']} exceeds {rarity}-star limit {limit}"
                    )
    if labels:
        diagnostics["raw_oddity_labels"] = labels
    if len(formal) == 3 and not diagnostics.get("oddity_warnings"):
        return "ready", {"values": formal, "source": "scan"}, diagnostics
    if formal:
        return ("review" if diagnostics.get("oddity_warnings") else "partial"), {"values": formal, "source": "scan"}, diagnostics
    return "unavailable", None, diagnostics


def _operator_rarity(operator_id: Any) -> int | None:
    if not operator_id:
        return None
    operator = _operator_catalog_by_id().get(str(operator_id))
    value = operator.get("rarity") if isinstance(operator, dict) else None
    return int(value) if isinstance(value, int) and value in ODDITY_LIMITS else None


def _disc_loadouts(record: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None, dict[str, Any]]:
    configs = record.get("disc_configs")
    diagnostics: dict[str, Any] = {}
    if not isinstance(configs, list):
        return "unavailable", None, diagnostics
    operator_id = str(record.get("operator_id") or "")
    operator = _operator_catalog_by_id().get(operator_id)
    if not isinstance(operator, dict):
        diagnostics["disc_catalog_review"] = {
            "operator_id": operator_id,
            "reason": "operator_id not found in operators.json",
        }
        return "review", None, diagnostics
    allowed_names = {
        str(item["ot_name"])
        for item in operator.get("discs", [])
        if isinstance(item, dict) and item.get("ot_name")
    }
    if not allowed_names:
        diagnostics["disc_catalog_review"] = {
            "operator_id": operator_id,
            "reason": "operator has no disc ot_name in operators.json",
        }
        return "review", None, diagnostics
    output: list[dict[str, Any]] = []
    for index, config in enumerate(configs[:2], 1):
        if not isinstance(config, dict) or not config.get("available"):
            continue
        selected = [
            slot
            for slot in config.get("slots", [])
            if isinstance(slot, dict) and slot.get("state") in {"active", "locked"}
        ]
        names: list[str] = []
        for slot in selected:
            raw_name = str(slot.get("name") or "").strip()
            canonical = _canonical_disc_name(operator_id, raw_name)
            if canonical is None and not raw_name:
                canonical = _canonical_disc_description(
                    operator_id,
                    str(slot.get("unlock_description") or "").strip(),
                )
            if canonical in allowed_names and canonical not in names:
                names.append(canonical)
            else:
                diagnostics.setdefault("disc_name_review", []).append(
                    raw_name or str(slot.get("unlock_description") or "")
                )
        if not 1 <= len(selected) <= 3:
            diagnostics.setdefault("disc_review", []).append({"index": index, "config": _json_safe(config)})
            continue
        if not names:
            diagnostics.setdefault("disc_review", []).append({"index": index, "config": _json_safe(config)})
            continue
        output.append(
            {
                "id": f"disc_{index}",
                "name": str(config.get("label") or f"命盘{index}"),
                "discs": [{"ot_name": name} for name in names],
            }
        )
    if not output:
        return ("review" if diagnostics else "unavailable"), None, diagnostics
    if diagnostics.get("disc_review") or diagnostics.get("disc_name_review") or len(output) < 2:
        return "partial", output, diagnostics
    return "ready", output, diagnostics


def _canonical_disc_name(operator_id: Any, raw_name: str) -> str | None:
    """Map OCR text to the stable ot_name from the local public catalog."""
    if not operator_id or not raw_name:
        return None
    operator = _operator_catalog_by_id().get(str(operator_id))
    if not isinstance(operator, dict):
        return None
    disc_items = [item for item in operator.get("discs", []) if isinstance(item, dict) and item.get("ot_name")]
    discs = [str(item["ot_name"]) for item in disc_items]
    if raw_name in discs:
        return raw_name
    normalised = _normalise_disc_name(raw_name)
    exact_matches = {
        str(item["ot_name"])
        for item in disc_items
        if any(
            _normalise_disc_name(alias) == normalised
            or _disc_ocr_name_matches(raw_name, alias)
            for alias in (item.get("ot_name"), item.get("abbreviation"), item.get("desp"))
            if alias
        )
    }
    if len(exact_matches) == 1:
        return next(iter(exact_matches))
    return None


def _canonical_disc_description(operator_id: Any, description: str) -> str | None:
    if not operator_id or not description:
        return None
    operator = _operator_catalog_by_id().get(str(operator_id))
    if not isinstance(operator, dict):
        return None
    needle = _normalise_disc_description(description)
    matches = {
        str(item["ot_name"])
        for item in operator.get("discs", [])
        if isinstance(item, dict)
        and item.get("ot_name")
        and _normalise_disc_description(item.get("desp"))
        and (
            _normalise_disc_description(item.get("desp")) == needle
            or _normalise_disc_description(item.get("desp")) in needle
            or needle in _normalise_disc_description(item.get("desp"))
        )
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _normalise_disc_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", text).replace("馀", "余").strip()


def _normalise_disc_description(value: Any) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", _normalise_disc_name(value))


def _disc_ocr_name_matches(raw_name: Any, catalog_name: Any) -> bool:
    raw = _normalise_disc_name(raw_name)
    catalog = _normalise_disc_name(catalog_name)
    return bool(raw and catalog and (raw == catalog or re.fullmatch(re.escape(catalog) + r"[A-Za-z]", raw)))


def _entry(record: dict[str, Any], game: str, observed_at: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    match = _operator_match(record)
    if match["status"] != "ready":
        debug = _json_safe(record.get("collection_debug", {}))
        debug["match"] = match
        unknown = {"raw_name": match["raw_name"], "diagnostics": debug}
        if match.get("candidates"):
            unknown["candidates"] = match["candidates"]
        return None, unknown
    basic_status, basic_data = _basic(record)
    star_status, star_level = _star_level(record)
    odd_status, odd_data, odd_diag = _oddities(record, star_level)
    combat_status = "ready"
    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    combat: dict[str, Any] = {"source": "scan", "observed_at": observed_at}
    if isinstance(stats.get("attack"), int): combat["observed_attack"] = stats["attack"]
    if isinstance(stats.get("life"), int): combat["observed_hp"] = stats["life"]
    if odd_data: combat["oddities"] = odd_data["values"]
    if "observed_attack" not in combat or "observed_hp" not in combat: combat_status = "partial"
    disc_status, disc_data, disc_diag = _disc_loadouts(record)
    statuses = {"basic": basic_status, "huaji": star_status, "oddities": odd_status, "combat_stats": combat_status, "disc_loadouts": disc_status}
    entry: dict[str, Any] = {
        "operator_id": record["operator_id"],
        "name": record.get("name") or record.get("name_cleaned") or record.get("name_raw"),
        "observed_at": observed_at,
        "match": match,
        "section_status": statuses,
    }
    entry.update(basic_data)
    if star_level is not None and star_status == "ready": entry["star_level"] = star_level
    if combat: entry["combat_stats"] = combat
    if disc_data is not None: entry["disc_loadouts"] = disc_data
    diagnostics = {"collection_debug": _json_safe(record.get("collection_debug", {})), **odd_diag, **disc_diag}
    entry["diagnostics"] = diagnostics
    return entry, None


def build_v3_document(records: Iterable[dict[str, Any]], scan_id: str, game: str = DEFAULT_GAME, *, exported_at: str | None = None, effective_at: str | None = None, catalog_version: str | None = DEFAULT_CATALOG_VERSION) -> dict[str, Any]:
    exported_at = exported_at or now_iso()
    effective_at = effective_at or exported_at
    entries: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for record in records:
        entry, unknown = _entry(record, game, effective_at)
        if entry is not None: entries.append(entry)
        if unknown is not None: unmatched.append(unknown)
    record: dict[str, Any] = {
        "account_id": "scan-source",
        "record_id": f"scan:{stable_scan_id(scan_id)}",
        "record_type": "operator_snapshot",
        "game": game,
        "source_kind": "scan",
        "snapshot_scope": "listed",
        "effective_at": effective_at,
        "entries": entries,
        "unmatched": unmatched,
    }
    document: dict[str, Any] = {
        "format": EXCHANGE_FORMAT,
        "version": EXCHANGE_VERSION,
        "exported_at": exported_at,
        "producer": {"platform": "maayuan", "version": "5"},
        "accounts": [{"id": "scan-source", "name": "MaaYuan 密探采集", "game_scope": "universal"}],
        "records": [record],
    }
    if catalog_version:
        document["catalog_version"] = catalog_version
    document = _json_safe(document)
    scrub_v3_equipment(document)
    return document


def validate_v3_document(document: dict[str, Any], schema_path: Path | None = None) -> None:
    if not isinstance(document, dict) or document.get("format") != EXCHANGE_FORMAT or document.get("version") != 3:
        raise ValueError("不是 myshare-operator-exchange v3 文档")
    if not isinstance(document.get("accounts"), list) or len(document["accounts"]) != 1:
        raise ValueError("v3 文档必须包含一个来源账号")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("v3 文档必须至少包含一个 operator_snapshot record")
    required = ("record_id", "record_type", "game", "source_kind", "snapshot_scope", "entries", "unmatched")
    for item in records:
        if not isinstance(item, dict) or any(key not in item for key in required) or item["record_type"] != "operator_snapshot":
            raise ValueError("operator_snapshot record 字段不完整")
        if not str(item["record_id"]).startswith("scan:"):
            raise ValueError("record_id 必须使用 scan:<stable-id>")
        if any(not isinstance(entry, dict) or not entry.get("operator_id") for entry in item["entries"]):
            raise ValueError("entries 中存在无效身份")
        for entry in item["entries"]:
            statuses = entry.get("section_status")
            if not isinstance(statuses, dict) or any(value not in VALID_SECTION_STATUS for value in statuses.values()):
                raise ValueError("entry.section_status 含非法状态")
    if schema_path is not None and schema_path.exists():
        try:
            import jsonschema  # type: ignore
        except ImportError as exc:
            raise RuntimeError("需要安装 jsonschema 才能执行权威 Schema 校验") from exc
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(document, schema)


def discover_v3_schema() -> Path | None:
    return next((path for path in DEFAULT_SCHEMA_PATHS if path.exists()), None)


def write_v3_document(document: dict[str, Any], path: Path) -> Path:
    scrub_v3_equipment(document)
    validate_v3_document(document)
    path = path if path.is_absolute() else REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _api_call(document: dict[str, Any], base_url: str, token: str, path: str, timeout: float = 15.0) -> dict[str, Any]:
    if not token:
        raise ValueError("密探自动上报需要 Token")
    url = base_url.rstrip("/") + path
    request = urllib_request.Request(url, data=json.dumps(document, ensure_ascii=False).encode("utf-8"), headers={"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {token}", "User-Agent": "MaaYuan-OperatorGrowth/3"}, method="POST")
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            raw_status = getattr(response, "status", None)
            status = int(raw_status if raw_status is not None else response.getcode())
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {_redact(body, token)[:500]}") from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"网络连接失败: {getattr(exc, 'reason', exc)}") from exc
    if not 200 <= status < 300:
        raise RuntimeError(f"HTTP {status}: {_redact(body, token)[:500]}")
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError:
        return {"raw": body}


def _redact(value: str, token: str) -> str:
    return value.replace(token, "<redacted>") if token else value


def read_growth_states(base_url: str, token: str, account_id: str, timeout: float = 15.0) -> dict[str, str]:
    """Read account-wide annotations; absent operators default to active in YuanHub."""
    if not token:
        raise ValueError("读取 YuanHub 养成状态需要连接码")
    request = urllib_request.Request(
        base_url.rstrip("/") + ANNOTATIONS_PATH,
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        raise RuntimeError(
            f"读取 YuanHub 养成状态失败（HTTP {exc.code}），请确认后端已更新且连接码具备 operator:read 权限"
        ) from exc
    except (urllib_error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError("读取 YuanHub 养成状态失败：网络或响应异常") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or data.get("account_id") != account_id or not isinstance(data.get("items"), list):
        raise ValueError("YuanHub 养成状态响应格式或绑定账号不匹配")
    states = {}
    for item in data["items"]:
        if (not isinstance(item, dict) or not isinstance(item.get("operator_id"), str)
                or not item["operator_id"] or item.get("growth_state") not in {"active", "graduated", "skip"}):
            raise ValueError("YuanHub 返回了无效的密探养成状态")
        states[item["operator_id"]] = item["growth_state"]
    return states


def preview_v3_document(document: dict[str, Any], base_url: str, token: str, timeout: float = 15.0) -> dict[str, Any]:
    scrub_v3_equipment(document)
    validate_v3_document(document)
    return _api_call(document, base_url, token, PREVIEW_PATH, timeout)


def commit_v3_document(document: dict[str, Any], base_url: str, token: str, timeout: float = 15.0) -> dict[str, Any]:
    scrub_v3_equipment(document)
    validate_v3_document(document)
    return _api_call(document, base_url, token, COMMIT_PATH, timeout)


def summarize_preview(payload: dict[str, Any]) -> str:
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    if not isinstance(data, dict): return "preview 已返回"
    parts = [f"{key}={data[key]}" for key in ("accepted", "partial", "review", "rejected", "unchanged") if key in data]
    return "preview: " + (", ".join(parts) if parts else "已返回")
