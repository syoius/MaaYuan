"""Refresh the operator catalog before an information scan."""
from pathlib import Path

from utils import logger
from custom.action.autoformation import AutoFormation


class _CollectorCatalogSync(AutoFormation):
    def __init__(self, path: Path):
        # Only use the catalog transport helpers, not the custom action lifecycle.
        self.OPERATORS_LOCAL_PATH = path
        self.OPERATORS_SYNC_META_PATH = path.with_name("operators.sync.meta")

    def _is_valid_operators_data(self, data: dict) -> bool:
        if not super()._is_valid_operators_data(data) or not data["OPERATORS"]:
            return False
        ids = set()
        for operator in data["OPERATORS"]:
            if not isinstance(operator, dict):
                return False
            operator_id, name = operator.get("id"), operator.get("name")
            if not isinstance(operator_id, str) or not operator_id.strip() or not isinstance(name, str) or not name.strip():
                return False
            if operator_id in ids:
                return False
            ids.add(operator_id)
        return True


def refresh_operator_catalog(path: Path) -> dict:
    # Reuse the existing source, HTTPS setup, validation and atomic writer,
    # without constructing AutoFormation (whose initializer triggers a sync).
    sync = _CollectorCatalogSync(path)
    local = sync._read_operators_file()
    logger.info("AgentInfoCollector: 正在检查远程 operators.json 更新")
    data = sync._sync_operators_data(local, force=True)
    if not data or not data.get("OPERATORS"):
        raise RuntimeError("本地和远程均无可用 operators.json，停止密探采集")
    return data
