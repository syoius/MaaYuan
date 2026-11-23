"""Refresh the operator catalog before an information scan."""
from pathlib import Path

from utils import logger
from custom.action.autoformation import AutoFormation


class _CollectorCatalogSync(AutoFormation):
    def __init__(self, path: Path):
        # Only use the catalog transport helpers, not the custom action lifecycle.
        self.OPERATORS_LOCAL_PATH = path
        self.OPERATORS_SYNC_META_PATH = path.with_name("operators.sync.meta")
        self.diagnostic_source = "builtin"
        self.local_catalog = None
        self.sync_status = "error"

    def _read_operators_file(self, path=None):
        data = super()._read_operators_file(path)
        if data is not None:
            self.diagnostic_source = "cache" if path is not None else "builtin"
            self.local_catalog = data
        return data

    def _fetch_remote_operators(self, *args, **kwargs):
        result = super()._fetch_remote_operators(*args, **kwargs)
        self.sync_status = result[0]
        if result[0] in {"updated", "not_modified"}:
            self.diagnostic_source = "remote"
        return result


class _CatalogSnapshot(dict):
    """Keep scan metadata beside the catalog, outside its serialized schema."""


def refresh_operator_catalog(path: Path) -> dict:
    # Reuse the existing source, HTTPS setup, validation and atomic writer,
    # without constructing AutoFormation (whose initializer triggers a sync).
    sync = _CollectorCatalogSync(path)
    logger.bind(console=False).info("AgentInfoCollector: 正在检查远程 operators.json 更新")
    with logger.contextualize(console=False):
        data = sync._sync_operators_data(force=True)
    if not data or not data.get("OPERATORS"):
        logger.warning("密探名单同步失败：没有可用名单，请检查网络后重试")
        raise RuntimeError("本地和远程均无可用 operators.json，停止密探采集")
    catalog_version = data.get("catalog_version")
    version_suffix = f"（版本 {catalog_version}）" if catalog_version else ""
    if sync.sync_status == "error":
        logger.warning(f"密探名单同步失败：未能获取有效的远端名单，使用本地名单{version_suffix}")
    elif sync.sync_status == "updated" and data != sync.local_catalog:
        previous_version = (sync.local_catalog or {}).get("catalog_version")
        if previous_version and catalog_version:
            logger.info(f"已更新密探名单：版本 {previous_version} → {catalog_version}")
        else:
            logger.info(f"已更新密探名单{version_suffix}")
    else:
        logger.info(f"密探名单已是最新{version_suffix}")
    version = data.get("catalog_version") or sync._operators_hash(data)
    operators = [operator for operator in data["OPERATORS"] if operator["name"].strip()]
    if len(operators) != len(data["OPERATORS"]):
        skipped_ids = [operator["id"] for operator in data["OPERATORS"] if not operator["name"].strip()]
        logger.bind(console=False).warning(f"AgentInfoCollector: 跳过未命名的密探占位记录: {', '.join(skipped_ids)}")
        # Preserve the source file and its sync hash; only filter the scan view.
        data = {**data, "OPERATORS": operators}
    snapshot = _CatalogSnapshot(data)
    snapshot.diagnostic = {"version": version,
                           "source": sync.diagnostic_source}
    return snapshot
