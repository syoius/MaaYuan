"""Refresh the operator catalog before an information scan."""
import re
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
        has_named_operator = False
        for operator in data["OPERATORS"]:
            if not isinstance(operator, dict):
                return False
            operator_id, name = operator.get("id"), operator.get("name")
            if not isinstance(operator_id, str) or not operator_id.strip() or not isinstance(name, str):
                return False
            if operator_id in ids:
                return False
            ids.add(operator_id)
            if not name.strip():
                # Reserved IDs for unreleased operators are not scan targets.
                if re.fullmatch(r"char_\d+_unknown", operator_id):
                    continue
                return False
            has_named_operator = True
        return has_named_operator


def refresh_operator_catalog(path: Path) -> dict:
    # Reuse the existing source, HTTPS setup, validation and atomic writer,
    # without constructing AutoFormation (whose initializer triggers a sync).
    sync = _CollectorCatalogSync(path)
    local = sync._read_operators_file()
    logger.info("AgentInfoCollector: 正在检查远程 operators.json 更新")
    data = sync._sync_operators_data(local, force=True)
    if not data or not data.get("OPERATORS"):
        raise RuntimeError("本地和远程均无可用 operators.json，停止密探采集")
    operators = [operator for operator in data["OPERATORS"] if operator["name"].strip()]
    if len(operators) != len(data["OPERATORS"]):
        skipped_ids = [operator["id"] for operator in data["OPERATORS"] if not operator["name"].strip()]
        logger.warning(f"AgentInfoCollector: 跳过未命名的密探占位记录: {', '.join(skipped_ids)}")
        # Preserve the source file and its sync hash; only filter the scan view.
        return {**data, "OPERATORS": operators}
    return data
