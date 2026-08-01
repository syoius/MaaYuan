import json
import sys
import tempfile
import time
import unittest
from email.message import Message
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch


AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from utils.remote_file_sync import sync_remote_file


class _Response:
    def __init__(self, payload: bytes, etag: str = '"remote"'):
        self.payload = payload
        self.status = 200
        self.headers = Message()
        self.headers["ETag"] = etag

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def getcode(self):
        return self.status

    def read(self, size: int = -1):
        return self.payload if size < 0 else self.payload[:size]


def _is_valid(path: Path) -> bool:
    return path.read_bytes().startswith(b"valid:")


class RemoteFileSyncTests(unittest.TestCase):
    def test_updates_valid_local_file_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "database.xlsx"
            local_path.write_bytes(b"valid:old")

            with patch(
                "utils.remote_file_sync.urllib_request.urlopen",
                return_value=_Response(b"valid:new"),
            ):
                result = sync_remote_file(
                    local_path,
                    "https://example.invalid/database.xlsx",
                    _is_valid,
                    check_interval_sec=1,
                )

            self.assertEqual(result.path, local_path)
            self.assertFalse(result.temporary)
            self.assertTrue(result.updated)
            self.assertEqual(local_path.read_bytes(), b"valid:new")
            meta = json.loads(
                local_path.with_suffix(".sync.meta").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["etag"], '"remote"')
            self.assertIn("last_success_ts", meta)

    def test_rejects_invalid_remote_file_and_keeps_local_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "database.xlsx"
            local_path.write_bytes(b"valid:local")

            with patch(
                "utils.remote_file_sync.urllib_request.urlopen",
                return_value=_Response(b"not-an-xlsx"),
            ):
                result = sync_remote_file(
                    local_path,
                    "https://example.invalid/database.xlsx",
                    _is_valid,
                    check_interval_sec=1,
                )

            self.assertEqual(result.path, local_path)
            self.assertFalse(result.updated)
            self.assertEqual(local_path.read_bytes(), b"valid:local")
            self.assertFalse(local_path.with_suffix(".xlsx.tmp").exists())

    def test_network_failure_keeps_local_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "database.xlsx"
            local_path.write_bytes(b"valid:local")

            with patch(
                "utils.remote_file_sync.urllib_request.urlopen",
                side_effect=URLError("offline"),
            ):
                result = sync_remote_file(
                    local_path,
                    "https://example.invalid/database.xlsx",
                    _is_valid,
                    check_interval_sec=1,
                )

            self.assertEqual(result.path, local_path)
            self.assertFalse(result.updated)
            self.assertEqual(local_path.read_bytes(), b"valid:local")

    def test_identical_remote_file_does_not_report_an_update(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "database.xlsx"
            local_path.write_bytes(b"valid:same")

            with patch(
                "utils.remote_file_sync.urllib_request.urlopen",
                return_value=_Response(b"valid:same"),
            ):
                result = sync_remote_file(
                    local_path,
                    "https://example.invalid/database.xlsx",
                    _is_valid,
                    check_interval_sec=1,
                )

            self.assertFalse(result.updated)
            self.assertEqual(local_path.read_bytes(), b"valid:same")

    def test_forced_check_ignores_recent_check_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "database.xlsx"
            local_path.write_bytes(b"valid:same")
            local_path.with_suffix(".sync.meta").write_text(
                json.dumps(
                    {"last_check_ts": int(time.time()), "etag": '"previous"'}
                ),
                encoding="utf-8",
            )

            with patch(
                "utils.remote_file_sync.urllib_request.urlopen",
                return_value=_Response(b"valid:new"),
            ) as urlopen:
                result = sync_remote_file(
                    local_path,
                    "https://example.invalid/database.xlsx",
                    _is_valid,
                    force_remote_check=True,
                )

            urlopen.assert_called_once()
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_header("If-none-match"), '"previous"')
            self.assertTrue(result.updated)
            self.assertEqual(local_path.read_bytes(), b"valid:new")


if __name__ == "__main__":
    unittest.main()
