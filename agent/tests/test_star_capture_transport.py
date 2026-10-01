import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote
from io import BytesIO


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from custom.action import star_capture_transport as transport  # noqa: E402
from custom.action.inventory_reporting import (  # noqa: E402
    AUTO_UPLOAD_MODE,
    UploadSettings,
)


PNG = b"\x89PNG\r\n\x1a\nminimal"


def full_batch(run_dir):
    batch = {"schemaVersion": 1, "captureId": "capture-full", "source": "maayuan",
             "gameVersion": "代号鸢", "sections": {}}
    order = 1
    for section in ("main", "support", "experience"):
        images = []
        for index in range(2 if section == "main" else 1):
            name = f"{section}/capture-{index:02d}.png"
            path = run_dir / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(PNG + b"x" * order)
            images.append({"sourceImageId": f"capture-full:{section}:{index:03d}",
                           "sourceOrder": order, "fileName": name})
            order += 1
        batch["sections"][section] = {"images": images, "adjacentRelations": [],
                                     "complete": True,
                                     "stopReason": "single_capture" if section == "experience" else "bottom_no_move"}
    (run_dir / "capture-batch.json").write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
    return batch


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

    def __init__(self, data=None):
        self.data = data or {"capture_id": "capture-full"}

    def getcode(self):
        return self.status

    def read(self):
        return json.dumps({"status_code": self.status, "data": self.data}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def success_response(request, **_kwargs):
    if request.full_url.endswith("/init"):
        return _Response({"capture_id": json.loads(request.data)["capture_id"]})
    suffix = request.full_url.split("/captures/")[1]
    capture_id = unquote(suffix.split("/")[0])
    data = {"capture_id": capture_id}
    if "/images/" in suffix:
        data["source_image_id"] = unquote(suffix.split("/images/")[1])
    return _Response(data)


class StarCaptureTransportTests(unittest.TestCase):
    def test_full_batch_telemetry_counts_sizes_and_keeps_logs_free_of_secrets(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "SECRET_AUTH_TOKEN", "https://hub.example")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            batch = full_batch(run_dir)
            with patch.object(transport, "read_upload_settings", return_value=settings), \
                 patch.object(transport.urllib_request, "urlopen", side_effect=success_response) as urlopen, \
                 patch.object(transport.logger, "info") as log:
                result = transport.upload_full_capture_batch(object(), run_dir)
            logs = "\n".join(call.args[0] for call in log.call_args_list)
            self.assertTrue(result.success)
            self.assertIn("image_count=4", logs)
            self.assertIn(f"total_image_bytes={len(PNG) * 4 + 10} ", logs)
            self.assertIn(f"largest_image_bytes={len(PNG) + 4} ", logs)
            self.assertEqual(urlopen.call_count, 6)
            calls = urlopen.call_args_list
            self.assertTrue(calls[0].args[0].full_url.endswith("/init"))
            self.assertTrue(calls[-1].args[0].full_url.endswith("/finalize"))
            self.assertEqual([call.kwargs["timeout"] for call in calls], [10.0, 20.0, 20.0, 20.0, 20.0, 10.0])
            self.assertEqual([unquote(call.args[0].full_url.split("/images/")[1]) for call in calls[1:-1]],
                             [image["sourceImageId"] for section in batch["sections"].values() for image in section["images"]])
            for call, path in zip(calls[1:-1], [run_dir / image["fileName"] for section in batch["sections"].values() for image in section["images"]]):
                self.assertEqual(call.args[0].data.count(b'name="file";'), 1)
                self.assertIn(path.read_bytes(), call.args[0].data)
                self.assertNotIn(b'name="manifest"', call.args[0].data)
            for field in ("capture_id=capture-full", "stage=init", "stage=image", "stage=finalize", "attempt=1/3",
                          "source_order=4", "image_bytes=", "multipart_body_bytes=", "http_attempt_elapsed_seconds=",
                          "http_status=200", "result=success", "total_elapsed_seconds="):
                self.assertIn(field, logs)
            for secret in (settings.token, "Authorization", "Bearer", json.dumps(batch), "代号鸢", "PNG"):
                self.assertNotIn(secret, logs)

    def test_timeout_retry_reuses_request_id_body_and_retains_all_local_files(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "SECRET_AUTH_TOKEN", "https://hub.example")
        # The exception text intentionally contains the token: telemetry must
        # report classes only, including nested urllib timeout reasons.
        error = transport.urllib_error.URLError(TimeoutError(settings.token))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            full_batch(run_dir)
            original = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
            with patch.object(transport, "read_upload_settings", return_value=settings), \
                 patch.object(transport.urllib_request, "urlopen", side_effect=error) as urlopen, \
                 patch.object(transport, "_retry_wait") as retry_wait, \
                 patch.object(transport, "_image_multipart_body", wraps=transport._image_multipart_body) as build, \
                 patch.object(transport.logger, "info") as log:
                result = transport.upload_full_capture_batch(object(), run_dir)
            self.assertFalse(result.success)
            self.assertEqual(urlopen.call_count, 3)
            self.assertEqual(retry_wait.call_count, 2)
            build.assert_not_called()
            first = urlopen.call_args_list[0].args[0]
            for call in urlopen.call_args_list:
                self.assertIs(call.args[0], first)
                self.assertIs(call.args[0].data, first.data)
                self.assertIn(b'"capture_id":"capture-full"', call.args[0].data)
            self.assertEqual(original, {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()})
            logs = "\n".join(call.args[0] for call in log.call_args_list)
            self.assertIn("attempt=3/3", logs)
            self.assertIn("exception_class=URLError reason_class=TimeoutError", logs)
            self.assertIn("result=failure", logs)
            self.assertNotIn(settings.token, logs)

    def test_current_image_retries_without_resending_successful_images(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "secret", "https://hub.example")
        seen = []
        def upload(request, **kwargs):
            seen.append(request)
            if len(seen) == 3:
                raise TimeoutError("secret")
            return success_response(request, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            full_batch(run_dir)
            with patch.object(transport, "read_upload_settings", return_value=settings), \
                 patch.object(transport.urllib_request, "urlopen", side_effect=upload), patch.object(transport, "_retry_wait"):
                self.assertTrue(transport.upload_full_capture_batch(object(), run_dir).success)
        self.assertEqual(len(seen), 7)
        self.assertIs(seen[2], seen[3])
        self.assertEqual(sum("main%3A000" in request.full_url for request in seen), 1)

    def test_terminal_image_failure_stops_later_images_and_retains_batch(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "secret", "https://hub.example")
        seen = []
        def upload(request, **kwargs):
            seen.append(request)
            if "main%3A001" in request.full_url:
                raise TimeoutError("secret")
            return success_response(request, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            full_batch(run_dir)
            before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
            with patch.object(transport, "read_upload_settings", return_value=settings), \
                 patch.object(transport.urllib_request, "urlopen", side_effect=upload), patch.object(transport, "_retry_wait"):
                self.assertFalse(transport.upload_full_capture_batch(object(), run_dir).success)
            self.assertEqual(before, {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()})
        self.assertEqual(len(seen), 5)
        self.assertFalse(any("support" in request.full_url or "/finalize" in request.full_url for request in seen))
        self.assertEqual(sum("main%3A000" in request.full_url for request in seen), 1)

    def test_init_and_finalize_transient_failures_retry_only_current_request(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "secret", "https://hub.example")
        for stage in ("init", "finalize"):
            seen = []
            def upload(request, **kwargs):
                seen.append(request)
                if request.full_url.endswith("/" + stage) and sum(item.full_url == request.full_url for item in seen) == 1:
                    raise transport.urllib_error.HTTPError(request.full_url, 504, "secret", {}, None)
                return success_response(request, **kwargs)
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                full_batch(run_dir)
                with patch.object(transport, "read_upload_settings", return_value=settings), \
                     patch.object(transport.urllib_request, "urlopen", side_effect=upload), patch.object(transport, "_retry_wait"):
                    self.assertTrue(transport.upload_full_capture_batch(object(), run_dir).success)
            repeated = [request for request in seen if request.full_url.endswith("/" + stage)]
            self.assertEqual(len(seen), 7)
            self.assertIs(repeated[0], repeated[1])

    def test_missing_images_fail_without_repair_and_no_progress_file_is_written(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "secret", "https://hub.example")
        seen = []
        def upload(request, **kwargs):
            seen.append(request)
            if request.full_url.endswith("/finalize"):
                response = _Response({"capture_id": "capture-full", "missing_source_image_ids": ["capture-full:main:000"]})
                response.status = 409
                raise transport.urllib_error.HTTPError(request.full_url, 409, "missing", {}, BytesIO(response.read()))
            return success_response(request, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            full_batch(run_dir)
            before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
            with patch.object(transport, "read_upload_settings", return_value=settings), \
                 patch.object(transport.urllib_request, "urlopen", side_effect=upload), \
                 patch.object(transport, "_retry_wait") as retry, patch.object(transport.logger, "info") as log:
                result = transport.upload_full_capture_batch(object(), run_dir)
            self.assertFalse(result.success)
            self.assertIn("capture-full:main:000", result.message)
            self.assertIn("missing_count=1", "\n".join(call.args[0] for call in log.call_args_list))
            retry.assert_not_called()
            self.assertEqual(before, {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()})
        self.assertEqual(len(seen), 6)

    def test_invalid_success_response_is_not_upload_success(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "secret", "https://hub.example")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            full_batch(run_dir)
            with patch.object(transport, "read_upload_settings", return_value=settings), \
                 patch.object(transport.urllib_request, "urlopen", return_value=_Response({"capture_id": "wrong"})) as upload:
                self.assertFalse(transport.upload_full_capture_batch(object(), run_dir).success)
            self.assertEqual(upload.call_count, 1)

    def test_timeout_then_success_keeps_retry_behavior(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "secret", "https://hub.example")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            batch = full_batch(run_dir)
            manifest, paths = transport.build_full_capture_manifest(run_dir, batch)
            with patch.object(transport.urllib_request, "urlopen", side_effect=[TimeoutError("secret"), _Response()]) as urlopen, \
                 patch.object(transport, "_retry_wait") as wait, patch.object(transport.logger, "info") as log:
                result = transport.upload_main_capture_manifest(manifest, paths, settings)
            self.assertTrue(result.success)
            self.assertEqual(urlopen.call_count, 2)
            wait.assert_called_once_with(0)
            self.assertIn("exception_class=TimeoutError", "\n".join(c.args[0] for c in log.call_args_list))

    def test_http_rejection_and_server_retry_are_logged_without_response_body(self):
        settings = UploadSettings(AUTO_UPLOAD_MODE, "secret", "https://hub.example")
        class RejectedResponse(_Response):
            status = 401
            def read(self):
                return b'{"message":"secret"}'
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest, paths = transport.build_full_capture_manifest(run_dir, full_batch(run_dir))
            for outcomes, expected_calls, expected_success in (([RejectedResponse()], 1, False),
                    ([transport.urllib_error.HTTPError("https://hub.example", 503, "secret", {}, None), _Response()], 2, True)):
                with self.subTest(expected_success=expected_success), \
                     patch.object(transport.urllib_request, "urlopen", side_effect=outcomes) as urlopen, \
                     patch.object(transport, "_retry_wait"), patch.object(transport.logger, "info") as log:
                    result = transport.upload_main_capture_manifest(manifest, paths, settings)
                self.assertEqual(result.success, expected_success)
                self.assertEqual(urlopen.call_count, expected_calls)
                logs = "\n".join(c.args[0] for c in log.call_args_list)
                self.assertIn("http_status=503" if expected_success else "http_status=401", logs)
                self.assertNotIn("secret", logs)

    def test_local_only_full_batch_never_builds_or_uploads(self):
        with patch.object(transport, "read_upload_settings", return_value=UploadSettings("仅保存到本地", "", "https://hub.example")), \
             patch.object(transport, "_multipart_body") as build, patch.object(transport.urllib_request, "urlopen") as upload:
            self.assertIsNone(transport.upload_full_capture_batch(object(), Path("unused")))
        build.assert_not_called()
        upload.assert_not_called()

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
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 20.0)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://hub.example/open-api/star/captures")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        body = request.data
        self.assertIn(b'name="manifest"', body)
        self.assertEqual(body.count(b'name="files"'), 2)
        self.assertIn(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), body)

if __name__ == "__main__":
    unittest.main()
