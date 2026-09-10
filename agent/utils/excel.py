from pathlib import Path
from typing import Any, Iterator, Sequence

from openpyxl import load_workbook


def read_sheet_rows(
    file_path: str | Path,
    sheet_index: int,
    *,
    skip_data_rows: int = 0,
) -> tuple[list[Any], list[tuple[Any, ...]]]:
    """Read an XLSX worksheet with pandas-compatible header/row skipping semantics."""
    workbook = load_workbook(file_path, read_only=True, data_only=True)
    try:
        worksheet = workbook.worksheets[sheet_index]
        rows: Iterator[Sequence[Any]] = worksheet.iter_rows(values_only=True)
        headers = list(next(rows, ()))
        data = [tuple(row) for row in rows]
        return headers, data[skip_data_rows:]
    finally:
        workbook.close()


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)
