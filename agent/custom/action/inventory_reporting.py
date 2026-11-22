from __future__ import annotations

import codecs
import json
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from numbers import Integral
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

AUTH_NODE_NAME = "在线上传认证"
AUTO_UPLOAD_MODE = "自动上报"
LOCAL_ONLY_MODE = "仅保存到本地"
DEFAULT_BASE_URL = "https://api-hub.maayuan.com"
TOKEN_REQUIRED_HINT = "请在主界面『YuanHub 连接码』里填写"
DAILY_TOKEN_REQUIRED_HINT = TOKEN_REQUIRED_HINT
TOKEN_STORE_PATH = Path(__file__).resolve().parents[3] / "config" / "yuanhub_connection.json"
IMPORT_PATH = "/open-api/inventory/import"
ACCOUNT_PATH = "/open-api/inventory/account"
MACHINE_MARKER = "#@MaaYInventoryRefV2 "
MAX_COUNT = 2_147_483_647
ACCOUNT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REPORT_LOCK = threading.Lock()
_ACCOUNT_CACHE_LOCK = threading.Lock()
_ACCOUNT_CACHE_KEY: tuple[str, str] | None = None
_ACCOUNT_CACHE_VALUE: BoundAccount | None = None


@dataclass(frozen=True)
class UploadSettings:
    mode: str
    token: str = field(repr=False)
    base_url: str
    report_filename: str | None = None


@dataclass(frozen=True)
class BoundAccount:
    id: str
    name: str
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class UploadResult:
    success: bool
    status_code: int | None
    message: str


def inventory_recovery_hint(result: UploadResult) -> str:
    entry = "YuanHub「库存追踪 → 操作历史 → 导入本地报告」"
    if (
        result.status_code is not None
        and 400 <= result.status_code < 500
        and result.status_code not in {408, 429}
    ):
        return (
            f"请先解决上述请求拒绝原因，再在 {entry} 中选择此 TXT 手动补传；"
            "仅等待或重复上传不会解除拒绝"
        )
    return f"网络或服务恢复后，可在 {entry} 中选择此 TXT 手动补传"


def _load_remembered_token() -> str:
    try:
        payload = json.loads(TOKEN_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return ""
    token = payload.get("token") if isinstance(payload, dict) else None
    return token.strip() if isinstance(token, str) else ""


def _remember_token(token: str) -> None:
    if _load_remembered_token() == token:
        return
    temporary = None
    try:
        TOKEN_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=TOKEN_STORE_PATH.parent,
            prefix=f".{TOKEN_STORE_PATH.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(
                {"token": token, "saved_at": datetime.now(timezone.utc).isoformat()},
                file, ensure_ascii=False,
            )
        temporary.replace(TOKEN_STORE_PATH)
    except (OSError, UnicodeError):
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _instance_ids_from_app_settings() -> set[str] | None:
    # GUI 新建、删除实例时就会写入实例列表，还没保存过设置的新实例也能计入。
    path = TOKEN_STORE_PATH.parent.parent / "appsettings.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return set()
    except (OSError, UnicodeError, ValueError):
        return None
    ids = payload.get("Instances.List") if isinstance(payload, dict) else None
    if not isinstance(ids, str):
        return set()
    return {item.strip().lower() for item in ids.split(",") if item.strip()}


def _token_from_mfa_config() -> tuple[set[str], int] | None:
    instance_ids = _instance_ids_from_app_settings()
    if instance_ids is None:
        return None
    try:
        paths = list(TOKEN_STORE_PATH.parent.iterdir())
    except FileNotFoundError:
        return set(), len(instance_ids)
    except OSError:
        return None

    tokens: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            data = value.get("data")
            if value.get("name") in ("连接码", "据点日常-连接码") and isinstance(data, dict):
                token = data.get("token")
                if isinstance(token, str) and token.strip():
                    tokens.add(token.strip())
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    try:
        for path in paths:
            if not (fnmatch(path.name, "config.json") or fnmatch(path.name, "mfa_*.json")):
                continue
            if path.is_dir():
                continue
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, (dict, list)):
                return None
            if isinstance(payload, dict):
                for key in payload:
                    match = re.fullmatch(r"(?:Instance|instance)\.([^.]+)\..+", key)
                    if match:
                        instance_ids.add(match[1].lower())
            collect(payload)
    except (OSError, UnicodeError, ValueError, RecursionError):
        # 任一配置读不到，就无法确认连接码是否唯一。
        return None
    return tokens, len(instance_ids)


def resolve_token(token: str) -> str:
    token = token.strip()
    if token:
        _remember_token(token)
        return token
    config = _token_from_mfa_config()
    if config is None:
        return ""
    tokens, instance_count = config
    # 多开时配置里的码和记住的码都可能属于另一个实例，只能用本实例显式填写的码。
    if len(tokens) > 1 or instance_count > 1:
        return ""
    if tokens:
        token = next(iter(tokens))
        _remember_token(token)
        return token
    return _load_remembered_token()


def read_upload_settings(
    context: Any, token_required_hint: str = TOKEN_REQUIRED_HINT,
) -> UploadSettings:
    try:
        node_data = context.get_node_data(AUTH_NODE_NAME)
    except Exception as exc:
        raise RuntimeError(f"读取节点 {AUTH_NODE_NAME} 失败: {exc}") from exc
    if not isinstance(node_data, dict):
        raise ValueError(f"未找到运行时节点: {AUTH_NODE_NAME}")
    attach = node_data.get("attach")
    if not isinstance(attach, dict):
        raise ValueError(f"节点 {AUTH_NODE_NAME} 缺少 attach 对象")

    mode = str(attach.get("mode", "")).strip()
    if mode not in {AUTO_UPLOAD_MODE, LOCAL_ONLY_MODE}:
        raise ValueError(
            f"{AUTH_NODE_NAME}.attach.mode 必须为 "
            f"{AUTO_UPLOAD_MODE} 或 {LOCAL_ONLY_MODE}"
        )
    token = str(attach.get("token", "")).strip()
    if mode == AUTO_UPLOAD_MODE:
        token = resolve_token(token)
        if not token:
            raise ValueError(token_required_hint)

    base_url = str(attach.get("base_url", DEFAULT_BASE_URL)).strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"无效的库存服务 base_url: {base_url}")
    raw_report_filename = attach.get("inventory_report_filename")
    report_filename = (
        None if raw_report_filename in (None, "") else str(raw_report_filename).strip()
    )
    return UploadSettings(
        mode=mode,
        token=token,
        base_url=base_url,
        report_filename=report_filename,
    )


def get_bound_account(
    settings: UploadSettings,
    timeout_seconds: float = 10.0,
    max_attempts: int = 3,
) -> BoundAccount:
    if settings.mode != AUTO_UPLOAD_MODE or not settings.token:
        raise ValueError("只有配置 Token 的自动上报模式可以查询绑定账号")
    if timeout_seconds <= 0:
        raise ValueError("查询绑定账号超时必须大于 0")
    if max_attempts < 1:
        raise ValueError("查询绑定账号尝试次数必须至少为 1")

    cache_key = (settings.base_url, settings.token)
    global _ACCOUNT_CACHE_KEY, _ACCOUNT_CACHE_VALUE
    with _ACCOUNT_CACHE_LOCK:
        if _ACCOUNT_CACHE_KEY == cache_key and _ACCOUNT_CACHE_VALUE is not None:
            return _ACCOUNT_CACHE_VALUE

        url = f"{settings.base_url}{ACCOUNT_PATH}"
        request = urllib_request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {settings.token}",
                "User-Agent": "MaaYuan-Inventory/2",
            },
            method="GET",
        )
        for attempt in range(max_attempts):
            try:
                with urllib_request.urlopen(
                    request, timeout=timeout_seconds
                ) as response:
                    status_code = int(getattr(response, "status", response.getcode()))
                    response_body = response.read().decode("utf-8", errors="replace")
                if 200 <= status_code < 300:
                    account = _parse_bound_account(response_body)
                    _ACCOUNT_CACHE_KEY = cache_key
                    _ACCOUNT_CACHE_VALUE = account
                    return account
                if status_code >= 500 and attempt + 1 < max_attempts:
                    _retry_wait(attempt)
                    continue
                raise RuntimeError(
                    f"查询 Token 绑定账号失败：{_error_message(status_code, response_body)}"
                )
            except urllib_error.HTTPError as exc:
                response_body = exc.read().decode("utf-8", errors="replace")
                if exc.code >= 500 and attempt + 1 < max_attempts:
                    _retry_wait(attempt)
                    continue
                raise RuntimeError(
                    f"查询 Token 绑定账号失败：{_error_message(exc.code, response_body)}"
                ) from exc
            except (urllib_error.URLError, TimeoutError, OSError) as exc:
                if attempt + 1 < max_attempts:
                    _retry_wait(attempt)
                    continue
                reason = getattr(exc, "reason", exc)
                raise RuntimeError(
                    f"查询 Token 绑定账号失败：网络连接失败：{reason}"
                ) from exc

    raise RuntimeError("查询 Token 绑定账号重试循环异常结束")


def _parse_bound_account(response_body: str) -> BoundAccount:
    try:
        payload = json.loads(response_body)
        data = payload.get("data")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise RuntimeError("绑定账号接口返回了无效 JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("绑定账号接口响应缺少 data")
    account_id = str(data.get("id", "")).strip()
    if not ACCOUNT_ID_PATTERN.fullmatch(account_id):
        raise RuntimeError("绑定账号接口返回了无效 account_id")
    return BoundAccount(
        id=account_id,
        name=str(data.get("name", "")).strip(),
        created_at=_optional_string(data.get("created_at")),
        updated_at=_optional_string(data.get("updated_at")),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_inventory_report_path(value: Any, repo_root: Path) -> Path:
    return resolve_inventory_report_destination(value, None, repo_root)


def resolve_inventory_report_destination(
    path_value: Any,
    filename_part: Any,
    repo_root: Path,
    *,
    filename_prefix: str = "DailyRewards",
) -> Path:
    if filename_part not in (None, ""):
        part = str(filename_part).strip()
        if not part:
            raise ValueError("本地报告文件名不能为空")
        if len(part) > 160:
            raise ValueError("本地报告文件名不能超过 160 个字符")
        if re.search(r'[<>:"/\\|?*]', part) or part.endswith((".", " ")):
            raise ValueError("本地报告文件名包含 Windows 不允许的字符")
        path_value = f"{filename_prefix}-{part}.txt"

    value = path_value
    if value in (None, ""):
        path = repo_root / "DailyRewards.txt"
    else:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
    if not path.suffix:
        path = path.with_suffix(".txt")
    if path.suffix.lower() != ".txt":
        raise ValueError("inventory_report_path 必须是 .txt 文件")
    return path.resolve()


def bound_account_report_filename(account: BoundAccount) -> str:
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", account.name.strip()).rstrip(". ")
    return f"{safe_name}-{account.id}" if safe_name else account.id


def parse_record_options(params: dict) -> tuple[str, str | None]:
    record_type = str(params.get("record_type", "reward_delta")).strip()
    if record_type not in {"reward_delta", "stock_snapshot"}:
        raise ValueError("record_type 必须为 reward_delta 或 stock_snapshot")

    raw_scope = params.get("snapshot_scope")
    snapshot_scope = None if raw_scope in (None, "") else str(raw_scope).strip()
    if record_type == "reward_delta":
        if snapshot_scope is not None:
            raise ValueError("reward_delta 不得设置 snapshot_scope")
    elif snapshot_scope not in {"full", "listed"}:
        raise ValueError("stock_snapshot 必须设置 snapshot_scope: full 或 listed")
    return record_type, snapshot_scope


def is_dispatch_reward(record_type: Any, acquisition_channel: Any) -> bool:
    return (
        record_type == "reward_delta"
        and isinstance(acquisition_channel, str)
        and "派遣" in acquisition_channel
    )


def validate_stamina_cost(value: Any, *, record_id: str = "") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        location = f" record_id={record_id}" if record_id else ""
        raise ValueError(f"派遣奖励{location} 的 stamina_cost 必须是整数")
    stamina_cost = int(value)
    if not 0 <= stamina_cost <= MAX_COUNT:
        location = f" record_id={record_id}" if record_id else ""
        raise ValueError(
            f"派遣奖励{location} 的 stamina_cost 超出允许范围: {stamina_cost}"
        )
    return stamina_cost


def validate_exchange_document_stamina(document: dict) -> None:
    records = document.get("records")
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("record_id", ""))
        dispatch_reward = is_dispatch_reward(
            record.get("record_type"), record.get("acquisition_channel")
        )
        if dispatch_reward:
            if "stamina_cost" not in record:
                raise ValueError(f"派遣奖励 record_id={record_id} 缺少 stamina_cost")
            validate_stamina_cost(record["stamina_cost"], record_id=record_id)
        elif "stamina_cost" in record:
            raise ValueError(f"非派遣记录 record_id={record_id} 不得携带 stamina_cost")


def build_exchange_document(
    results: list[dict],
    invocation_id: str,
    effective_at: str,
    exported_at: str,
    acquisition_channel: str,
    record_type: str,
    snapshot_scope: str | None,
    account_id: str | None,
    stamina_cost: Any = None,
) -> dict:
    if not results:
        raise ValueError("无法为零条识别结果生成库存记录")
    if not acquisition_channel:
        raise ValueError("生成库存记录时 acquisition_channel 不能为空")
    if account_id is not None and not ACCOUNT_ID_PATTERN.fullmatch(account_id):
        raise ValueError("生成库存记录时 account_id 格式无效")
    dispatch_reward = is_dispatch_reward(record_type, acquisition_channel)
    validated_stamina_cost = (
        validate_stamina_cost(stamina_cost) if dispatch_reward else None
    )

    grouped_results: dict[str, list[dict]] = {"agent": [], "item": []}
    populated_types: list[str] = []
    for result in results:
        entity_type = str(result.get("entity_type", ""))
        if entity_type not in grouped_results:
            raise ValueError(f"识别结果包含无效对象类型: {entity_type!r}")
        if not grouped_results[entity_type]:
            populated_types.append(entity_type)
        grouped_results[entity_type].append(result)
    mixed = len(populated_types) > 1
    records: list[dict] = []
    for entity_type in populated_types:
        entries_by_id: dict[str, dict] = {}
        for result in grouped_results[entity_type]:
            if entity_type == "agent":
                entity_id = str(result.get("operator_id", "")).strip()
                name = str(result.get("operator_name", "")).strip()
            else:
                entity_id = str(result.get("item_id", "")).strip()
                name = str(result.get("item_name", "")).strip()
            if not entity_id:
                raise ValueError(f"识别结果缺少 {entity_type} 稳定 ID")

            raw_count = result.get("count")
            if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
                raise ValueError(f"{name or entity_id} 缺少有效整数数量")
            count = int(raw_count)
            minimum = 1 if record_type == "reward_delta" else 0
            if not minimum <= count <= MAX_COUNT:
                raise ValueError(f"{name or entity_id} 数量超出允许范围: {count}")

            current = entries_by_id.get(entity_id)
            if current is not None:
                if record_type == "stock_snapshot":
                    raise ValueError(f"库存快照中出现重复对象: {name or entity_id}")
                count += int(current["count"])
                if count > MAX_COUNT:
                    raise ValueError(f"{name or entity_id} 合并数量超出允许范围")
            entry = {"id": entity_id, "count": count}
            if name:
                entry["name"] = name
            entries_by_id[entity_id] = entry

        record_id = f"myshare:{invocation_id}"
        if mixed:
            record_id = f"{record_id}:{entity_type}"
        record = {
            "record_id": record_id,
            "record_type": record_type,
            "entity_type": entity_type,
            "acquisition_channel": acquisition_channel,
            "effective_at": effective_at,
            "entries": list(entries_by_id.values()),
        }
        if account_id is not None:
            record["account_id"] = account_id
        if snapshot_scope is not None:
            record["snapshot_scope"] = snapshot_scope
        if dispatch_reward:
            record["stamina_cost"] = validated_stamina_cost
        records.append(record)
    document = {
        "format": "myshare-inventory-exchange",
        "version": 2,
        "exported_at": exported_at,
        "producer": {"platform": "myshare"},
        "records": records,
    }
    validate_exchange_document_stamina(document)
    return document


def append_inventory_report(path: Path, document: dict, upload_status: str) -> None:
    records = document["records"]
    if not records:
        raise ValueError("库存交换文档不包含记录")
    first_record = records[0]
    record_label = (
        "奖励增量" if first_record["record_type"] == "reward_delta" else "库存快照"
    )
    lines = [
        "========== MaaYuan 库存记录 ==========",
        f"时间：{first_record['effective_at']}",
        f"渠道：{first_record['acquisition_channel']}",
        f"类型：{record_label}",
        f"上报状态：{upload_status}",
    ]
    if "stamina_cost" in first_record:
        lines.insert(3, f"消耗体力：{first_record['stamina_cost']}")
    if "account_id" in first_record:
        lines.insert(4, f"子账号ID：{first_record['account_id']}")
    for record in records:
        entity_label = "密探" if record["entity_type"] == "agent" else "道具"
        lines.extend(["", f"{entity_label}："])
        lines.extend(
            f"  {entry.get('name', entry['id'])} × {entry['count']}"
            for entry in record["entries"]
        )
    reference = {"r": [record["record_id"] for record in records]}
    if "account_id" in first_record:
        reference["a"] = first_record["account_id"]
    if "snapshot_scope" in first_record:
        reference["s"] = first_record["snapshot_scope"]
    if "stamina_cost" in first_record:
        reference["c"] = first_record["stamina_cost"]
    reference_json = json.dumps(reference, ensure_ascii=False, separators=(",", ":"))
    lines.extend(
        [
            "",
            "以下内容供导入工具使用，无需修改：",
            f"{MACHINE_MARKER}{reference_json}",
            "========== 记录结束 ==========",
            "",
        ]
    )
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    with _REPORT_LOCK:
        with path.open("ab") as file:
            if file.tell() == 0:
                file.write(codecs.BOM_UTF8)
            file.write(payload)


def update_upload_status(path: Path, record_ids: str | list[str], status: str) -> None:
    expected_ids = [record_ids] if isinstance(record_ids, str) else record_ids
    if not expected_ids:
        raise ValueError("更新上报状态时 record_ids 不能为空")

    with _REPORT_LOCK:
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        status_index: int | None = None
        for marker_index in range(len(lines) - 1, -1, -1):
            line = lines[marker_index]
            if not line.startswith(MACHINE_MARKER):
                continue
            reference = json.loads(line[len(MACHINE_MARKER) :])
            actual_ids = [str(record_id) for record_id in reference.get("r", [])]
            if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(
                expected_ids
            ):
                continue
            for index in range(marker_index - 1, -1, -1):
                if lines[index].startswith("上报状态："):
                    status_index = index
                    break
                if lines[index] == "========== MaaYuan 库存记录 ==========":
                    break
            break

        if status_index is None:
            raise ValueError(f"库存 TXT 中未找到对应记录: {expected_ids}")
        lines[status_index] = f"上报状态：{status}"
        updated_text = "\n".join(lines)
        if text.endswith(("\n", "\r")):
            updated_text += "\n"
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(updated_text, encoding="utf-8-sig")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def upload_inventory_document(
    document: dict,
    settings: UploadSettings,
    timeout_seconds: float = 10.0,
    max_attempts: int = 3,
) -> UploadResult:
    if settings.mode != AUTO_UPLOAD_MODE:
        raise ValueError("仅自动上报模式可以调用上传接口")
    if timeout_seconds <= 0:
        raise ValueError("上传超时必须大于 0")
    if max_attempts < 1:
        raise ValueError("上传尝试次数必须至少为 1")
    validate_exchange_document_stamina(document)

    url = f"{settings.base_url}{IMPORT_PATH}"
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=payload,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {settings.token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "MaaYuan-Inventory/1",
        },
        method="POST",
    )

    for attempt in range(max_attempts):
        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                status_code = int(getattr(response, "status", response.getcode()))
                response_body = response.read().decode("utf-8", errors="replace")
            if 200 <= status_code < 300:
                return UploadResult(
                    success=True,
                    status_code=status_code,
                    message=_success_message(status_code, response_body),
                )
            message = f"HTTP {status_code}"
            if status_code >= 500 and attempt + 1 < max_attempts:
                _retry_wait(attempt)
                continue
            return UploadResult(False, status_code, message)
        except urllib_error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            message = _error_message(exc.code, response_body)
            if exc.code >= 500 and attempt + 1 < max_attempts:
                _retry_wait(attempt)
                continue
            return UploadResult(False, int(exc.code), message)
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            if attempt + 1 < max_attempts:
                _retry_wait(attempt)
                continue
            reason = getattr(exc, "reason", exc)
            return UploadResult(False, None, f"网络连接失败：{reason}")

    raise RuntimeError("上传重试循环异常结束")


def _retry_wait(attempt: int) -> None:
    time.sleep(min(0.5 * (2**attempt), 2.0))


def _success_message(status_code: int, response_body: str) -> str:
    try:
        payload = json.loads(response_body)
        data = payload.get("data", payload)
        if isinstance(data, dict):
            accepted = data.get("accepted")
            duplicates = data.get("duplicates")
            if accepted is not None and duplicates is not None:
                return (
                    f"HTTP {status_code}，accepted={accepted}，"
                    f"duplicates={duplicates}"
                )
    except (json.JSONDecodeError, AttributeError):
        pass
    return f"HTTP {status_code}"


def _error_message(status_code: int, response_body: str) -> str:
    try:
        payload = json.loads(response_body)
        error = payload.get("error", {})
        if isinstance(error, dict):
            code = str(error.get("code", "")).strip()
            message = str(error.get("message", "")).strip()
            if status_code == 401:
                return "HTTP 401（Token 无效或已被删除）"
            if status_code == 403 and code == "account_scope_mismatch":
                return (
                    "HTTP 403（account_scope_mismatch：Token 绑定账号与库存记录不一致）"
                )
            detail = "：".join(value for value in (code, message) if value)
            if detail:
                return f"HTTP {status_code}（{detail}）"
    except (json.JSONDecodeError, AttributeError):
        pass
    if status_code == 401:
        return "HTTP 401（Token 无效或已被删除）"
    return f"HTTP {status_code}"
