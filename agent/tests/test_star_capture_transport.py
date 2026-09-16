import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from custom.action import star_capture_transport as transport  # noqa: E402
from custom.action.inventory_reporting import (  # noqa: E402
    AUTO_UPLOAD_MODE,
    UploadSettings,
)


PNG = b"\x89PNG\r\n\x1a\nminimal"


def session():
    return {
        "success": True,
        "stop_reason": "bottom_no_move",
        "retained_images": ["capture-00.png", "capture-01.png"],
        "adjacent_relations": [
            {
                "previous_image": "capture-00.png",
                "current_image": "capture-01.png",
                "relation": "overlap",
            }
        ],
    }


class _Response:
    status = 200

    def getcode(self):
        return self.status

    def read(self):
        return b'{"status_code":200,"data":{"capture_id":"ignored"}}'

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class StarCaptureTransportTests(unittest.TestCase):
    def test_full_batch_maps_local_section_names_to_unique_wire_names(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            batch = {
                "schemaVersion": 1, "captureId": "capture-full", "source": "maayuan", "gameVersion": "如鸢",
                "sections": {
                    "main": {"images": [{"sourceImageId": "capture-full:main:000", "sourceOrder": 1, "fileName": "main/capture-00.png"}, {"sourceImageId": "capture-full:main:001", "sourceOrder": 2, "fileName": "main/capture-01.png"}], "adjacentRelations": [{"previousSourceImageId": "capture-full:main:000", "currentSourceImageId": "capture-full:main:001", "relation": "overlap"}], "complete": True, "stopReason": "bottom_no_move"},
                    "support": {"images": [{"sourceImageId": "capture-full:support:000", "sourceOrder": 3, "fileName": "support/capture-00.png"}], "adjacentRelations": [], "complete": True, "stopReason": "bottom_no_move"},
                    "experience": {"images": [{"sourceImageId": "capture-full:experience:000", "sourceOrder": 4, "fileName": "experience/capture-00.png"}], "adjacentRelations": [], "complete": True, "stopReason": "single_capture"},
                },
            }
            for section in ("main", "support", "experience"):
                path = run_dir / section / "capture-00.png"
                path.parent.mkdir()
                path.write_bytes(PNG)
            (run_dir / "main" / "capture-01.png").write_bytes(PNG)
            manifest, wire_images = transport.build_full_capture_manifest(run_dir, batch)
            body, _boundary = transport._multipart_body(manifest, wire_images)
        self.assertEqual(list(manifest["sections"]), ["main", "support", "experience"])
        self.assertEqual([image["source_order"] for section in manifest["sections"].values() for image in section["images"]], [1, 2, 3, 4])
        self.assertEqual([name for name, _path in wire_images], ["main-000.png", "main-001.png", "support-000.png", "experience-000.png"])
        self.assertIn(b'filename="support-000.png"', body)
        self.assertNotIn(b'filename="capture-00.png"', body)

    def test_manifest_uses_stable_capture_id_one_based_order_and_preserved_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in session()["retained_images"]:
                (run_dir / name).write_bytes(PNG)
            first, paths = transport.build_main_capture_manifest(run_dir, session(), "如鸢")
            second, _ = transport.build_main_capture_manifest(run_dir, session(), "如鸢")
        self.assertEqual(first["capture_id"], second["capture_id"])
        self.assertEqual([image["source_order"] for image in first["images"]], [1, 2])
        self.assertTrue(first["images"][0]["source_image_id"].endswith(":000"))
        self.assertTrue(first["images"][1]["source_image_id"].endswith(":001"))
        self.assertEqual(len(paths), 2)
        self.assertEqual(first["adjacent_relations"][0]["relation"], "overlap")

    def test_multipart_upload_reuses_token_and_sends_manifest_and_each_png(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "secret-token", "https://hub.example")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in session()["retained_images"]:
                (run_dir / name).write_bytes(PNG)
            manifest, paths = transport.build_main_capture_manifest(run_dir, session(), "如鸢")
            with patch.object(transport.urllib_request, "urlopen", return_value=_Response()) as urlopen:
                result = transport.upload_main_capture_manifest(manifest, paths, settings)
        self.assertTrue(result.success)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://hub.example/open-api/star/captures")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        body = request.data
        self.assertIn(b'name="manifest"', body)
        self.assertEqual(body.count(b'name="files"'), 2)
        self.assertIn(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), body)

if __name__ == "__main__":
    unittest.main()
