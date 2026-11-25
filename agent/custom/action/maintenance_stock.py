"""Read account-scoped V2 snapshots and reconcile sweeps across restarts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

from custom.action.inventory_reporting import (
    ACCOUNT_ID_PATTERN,
    MACHINE_MARKER,
    MAX_COUNT,
    UploadSettings,
    resolve_inventory_report_destination,
)


@dataclass
class Stock:
    counts: dict[str, int] = field(default_factory=dict)
    baselines: dict[str, datetime] = field(default_factory=dict)
    receipts: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ReportTarget:
    rewards: Path
    account: str | None

    @property
    def key(self) -> str:
        return self.account or f"file:{self.rewards.resolve()}"


@dataclass
class Record:
    time: datetime
    id: str
    owner: str | None
    kind: str
    channel: str
    scope: str | None
    counts: dict[str, int]


def timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("库存时间无效")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("库存时间缺少时区")
    return result


def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_item_block(path, block, names, reverse) -> Record:
    fields = dict(
        s.split("：", 1) for s in block if "：" in s and not s.startswith(" ")
    )
    markers = [s[len(MACHINE_MARKER) :] for s in block if s.startswith(MACHINE_MARKER)]
    if len(markers) != 1:
        raise ValueError(f"{path.name} 缺少有效库存记录标记")
    ref = json.loads(markers[0])
    sections = [s for s in block if s in ("道具：", "密探：")]
    ids = ref["r"]
    if (
        not isinstance(ids, list)
        or not all(isinstance(i, str) and i for i in ids)
        or len(ids) != len(sections)
        or len(set(sections)) != len(sections)
    ):
        raise ValueError(f"{path.name} 库存记录分区不一致")
    counts = {}
    for s in block[block.index("道具：") + 1 :]:
        if not s.startswith("  "):
            break
        name, sep, amount = s.strip().rpartition(" × ")
        key = reverse.get(name, name)
        if not sep or not re.fullmatch(r"[0-9]+", amount) or key in counts:
            raise ValueError(f"{path.name} 道具数量格式无效")
        if key in names:
            count = int(amount)
            if count > MAX_COUNT:
                raise ValueError(f"{path.name} 道具数量超出范围")
            counts[key] = count
    owner = ref.get("a")
    if owner != fields.get("子账号ID") or (
        owner is not None and not ACCOUNT_ID_PATTERN.fullmatch(owner)
    ):
        raise ValueError(f"{path.name} 库存记录账号不一致")
    kind = fields["类型"]
    if (
        kind not in ("奖励增量", "库存快照")
        or (kind == "库存快照" and ref.get("s") not in ("listed", "full"))
        or (kind == "奖励增量" and ref.get("s") is not None)
    ):
        raise ValueError(f"{path.name} 库存记录类型或快照范围无效")
    return Record(
        timestamp(fields["时间"]),
        ids[sections.index("道具：")],
        owner,
        fields["类型"],
        fields["渠道"],
        ref.get("s"),
        counts,
    )


def report_records(path: Path, names: dict[str, str]):
    """Read only complete item blocks written by append_inventory_report."""
    if not path.is_file():
        return
    reverse = {name: key for key, name in names.items()}
    block = None
    with path.open(encoding="utf-8-sig") as stream:
        for raw in stream:
            line = raw.rstrip("\r\n")
            if line == "========== MaaYuan 库存记录 ==========":
                block = []
            elif line == "========== 记录结束 ==========" and block is not None:
                if "道具：" in block:
                    yield parse_item_block(path, block, names, reverse)
                block = None
            elif block is not None:
                block.append(line)


def local_stock(root: Path, settings: UploadSettings, names: dict[str, str]):
    # Pair StockReport and DailyRewards only within the same configured profile.
    if settings.report_filename:
        paths = [
            resolve_inventory_report_destination(
                None, settings.report_filename, root, filename_prefix=p
            )
            for p in ("StockReport", "DailyRewards")
        ]
    else:
        paths = sorted(
            p
            for p in root.glob("*.txt")
            if re.fullmatch(r"(?:StockReport|DailyRewards)(?:-.+)?", p.stem)
        )
    records = {}
    targets = {}
    for path in paths:
        suffix = path.stem.partition("-")[2]
        for row in report_records(path, names):
            target = ReportTarget(
                resolve_inventory_report_destination("DailyRewards.txt", suffix, root),
                row.owner,
            )
            targets[target.key] = target
            prior = records.setdefault((target.key, row.id), row)
            if prior != row:
                raise ValueError("本地报告中同一记录 ID 的内容冲突")
    if len(targets) > 1:
        raise ValueError("找到多个账号的本地报告，无法确定当前账号")
    default = ReportTarget(
        resolve_inventory_report_destination(
            "DailyRewards.txt", settings.report_filename, root
        ),
        None,
    )
    target = next(iter(targets.values()), default)
    stock = Stock()
    for row in sorted(records.values(), key=lambda r: (r.time, r.kind == "库存快照")):
        stock.receipts.add(row.id)
        if row.kind == "库存快照":
            if row.scope == "full" or row.channel == "背包-道具":
                stock.counts.clear()
                stock.baselines.clear()
            values = dict.fromkeys(names, 0) if row.scope == "full" else {}
            values.update(row.counts)
            stock.counts.update(values)
            stock.baselines.update(dict.fromkeys(values, row.time))
        elif row.kind == "奖励增量":
            for key, count in row.counts.items():
                if key in stock.counts:
                    stock.counts[key] += count
                    if stock.counts[key] > MAX_COUNT:
                        raise ValueError("本地库存数量超出允许范围")
    return target, stock


def cloud_stock(
    account_id: str, settings: UploadSettings, names: dict[str, str]
) -> Stock:
    req = request.Request(
        settings.base_url + "/open-api/inventory/current?entity_type=item",
        headers={
            "Authorization": f"Bearer {settings.token}",
            "Accept": "application/json",
        },
    )
    with request.urlopen(req, timeout=10) as response:
        payload = json.load(response)
    if (
        not isinstance(payload, dict)
        or payload.get("status_code", payload.get("statusCode")) != 200
        or not isinstance(payload.get("data"), list)
    ):
        raise ValueError("YuanHub 库存响应无效")
    stock = Stock()
    rows = payload["data"]
    if not rows:
        return stock
    if (
        len(rows) != 1
        or not isinstance(rows[0], dict)
        or rows[0].get("entity_type") != "item"
        or rows[0].get("account_id") != account_id
    ):
        raise ValueError("YuanHub 返回的库存与连接码绑定账号不一致")
    row = rows[0]
    if not isinstance(row.get("entries"), dict):
        raise ValueError("YuanHub 缺少库存条目")
    full = timestamp(row["full_baseline_at"]) if row.get("full_baseline_at") else None
    for key in names:
        item = row["entries"].get(key, {})
        if not isinstance(item, dict):
            continue
        try:
            listed = (
                timestamp(item["listed_baseline_at"])
                if item.get("listed_baseline_at")
                else None
            )
        except ValueError:
            continue
        dates = [date for date in (full, listed) if date]
        if dates:
            count = item.get("count") if key in row["entries"] else 0
            if type(count) is not int or not 0 <= count <= MAX_COUNT:
                continue
            stock.counts[key] = count
            stock.baselines[key] = max(dates)
    return stock


@contextmanager
def maintenance_lock(directory: Path, key: str):
    """Serialize sweeps for a report profile; the OS releases the lock on exit."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (hashlib.sha256(key.encode()).hexdigest() + ".lock")).open(
        "a+b"
    ) as stream:
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (PermissionError, BlockingIOError) as exc:
            raise RuntimeError("此账号已有库存保持任务正在执行") from exc
        yield


class SweepJournal:
    """Remember only the last submitted batch for each affected item."""

    def __init__(self, directory: Path, key: str):
        self.path = directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        self.entries = (
            json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}
        )

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(self.entries, stream)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(self.path)

    def reconcile(self, stock: Stock, *, cloud: bool = False):
        for key, saved in self.entries.items():
            baseline = stock.baselines.get(key)
            if baseline and baseline > timestamp(saved["time"]):
                continue
            if saved["id"] in stock.receipts:
                continue  # local replay already includes this batch
            if cloud and saved.get("cloud_synced"):
                continue  # the cloud reading already accounts for the acknowledged batch
            if saved["confirmed"] and key in stock.counts:
                current = stock.counts[key]
                if current >= saved["after"]:
                    continue
                if (
                    baseline
                    and current == saved["source_count"]
                    and baseline.isoformat() == saved["source_time"]
                ):
                    stock.counts[key] = saved["after"]
                    continue
            stock.counts.pop(key, None)
