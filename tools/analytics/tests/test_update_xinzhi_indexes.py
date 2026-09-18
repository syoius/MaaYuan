import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import update_xinzhi_indexes as updater


class XinzhiUpdateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=updater.ANALYTICS_DIR)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "input"
        self.archive = self.root / "archive"
        self.templates = self.root / "templates"
        self.outputs = self.root / "outputs"
        for directory in (self.inputs, self.templates, self.outputs):
            directory.mkdir()
        self.bag_output = self.outputs / "bag-agent-index.npz"
        self.dispatch_output = self.outputs / "dispatch-reward-index.npz"
        self.bag_output.write_bytes(b"old bag index")
        self.dispatch_output.write_bytes(b"old dispatch index")
        rng = np.random.default_rng(42)
        self.old_template = self.templates / "char_001_old-bag.png"
        updater.write_png(self.old_template, rng.integers(0, 256, (58, 70, 3), dtype=np.uint8))
        self.original = self.inputs / "士䵋.png"
        icon = rng.integers(0, 256, (140, 140, 4), dtype=np.uint8)
        icon[:, :, 3] = 255
        updater.write_png(self.original, icon)
        self.original_bytes = self.original.read_bytes()
        self.operators = self.root / "operators.json"
        self.write_json(self.operators, {"OPERATORS": [
            {"id": "char_001_old", "name": "旧密探"},
            {"id": "char_002_new", "name": "士䵋", "alt_name": "新密探"},
        ]})
        items_dir = self.root / "items"
        item_file = items_dir / "testitem-bag.png"
        updater.write_png(item_file, rng.integers(0, 256, (58, 70, 3), dtype=np.uint8))
        items_path = self.root / "items.json"
        self.write_json(items_path, {
            "schema": "myshare.items", "version": 1,
            "template_roots": [self.relative(items_dir)],
            "items": [{"id": "testitem", "name": "测试道具", "category": "测试", "template": self.relative(item_file)}],
        })
        self.manifest = self.root / "manifest.json"
        self.write_json(self.manifest, {
            "schema": "maay.dispatch-reward-index-source", "version": 1,
            "agent_templates": self.relative(self.templates),
            "operators": self.relative(self.operators),
            "items": self.relative(items_path),
            "item_ids": ["testitem"],
            "scales": [0.89, 0.90, 0.91], "feature_scale": 0.90,
            "output": self.relative(self.dispatch_output),
        })

    def relative(self, path):
        return path.relative_to(updater.bag.REPO_ROOT).as_posix()

    def write_json(self, path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def update(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return updater.update_indexes(
                self.inputs, self.archive,
                manifest_path=self.manifest, bag_output=self.bag_output, **kwargs,
            )

    def assert_unmodified(self):
        self.assertEqual(self.original_bytes, self.original.read_bytes())
        self.assertEqual(b"old bag index", self.bag_output.read_bytes())
        self.assertEqual(b"old dispatch index", self.dispatch_output.read_bytes())
        self.assertEqual([self.old_template], list(self.templates.iterdir()))

    def test_success_preserves_old_entries_and_archives_exact_original(self):
        report = self.update()
        self.assertEqual("complete", report["status"])
        self.assertEqual([], list(self.inputs.iterdir()))
        batch = next(self.archive.iterdir())
        self.assertEqual(self.original_bytes, (batch / "originals" / self.original.name).read_bytes())
        self.assertEqual("complete", json.loads((batch / "report.json").read_text(encoding="utf-8"))["status"])
        with np.load(self.bag_output, allow_pickle=False) as index:
            self.assertEqual(["char_001_old", "char_002_new"], index["agent_ids"].tolist())
            self.assertEqual([1.0], index["scales"].tolist())
        with np.load(self.dispatch_output, allow_pickle=False) as index:
            self.assertEqual(["char_001_old", "char_002_new", "testitem"], index["agent_ids"].tolist())
            self.assertEqual(["templates_089", "templates_090", "templates_091"], [key for key in index.files if key.startswith("templates_")])
            self.assertEqual(self.relative(self.old_template), index["source_files"][0])
            self.assertFalse(any(".xinzhi-stage-" in path for path in index["source_files"]))
        before = self.bag_output.read_bytes()
        self.assertEqual("empty", self.update()["status"])
        self.assertEqual(before, self.bag_output.read_bytes())

    def test_dry_run_performs_validation_without_mutation(self):
        self.assertEqual("validated", self.update(dry_run=True)["status"])
        self.assert_unmodified()
        self.assertFalse(self.archive.exists())

    def test_second_build_failure_keeps_both_old_indexes_and_inputs(self):
        with patch.object(updater.dispatch, "build_index", side_effect=RuntimeError("self-check failed")):
            with self.assertRaisesRegex(RuntimeError, "self-check failed"):
                self.update()
        self.assert_unmodified()
        self.assertFalse(self.archive.exists())

    def test_publish_failure_rolls_back_templates_and_first_index(self):
        original_replace = Path.replace

        def fail_second_index(source, destination):
            if destination == self.dispatch_output:
                raise OSError("index is locked")
            return original_replace(source, destination)

        with patch.object(Path, "replace", fail_second_index):
            with self.assertRaisesRegex(OSError, "index is locked"):
                self.update()
        self.assert_unmodified()
        batch = next(self.archive.iterdir())
        self.assertEqual(self.original_bytes, (batch / "originals" / self.original.name).read_bytes())
        self.assertEqual("publish-failed", json.loads((batch / "report.json").read_text(encoding="utf-8"))["status"])

    @unittest.skipUnless(sys.platform == "win32", "Windows ACL regression")
    def test_publish_does_not_carry_private_staging_acl(self):
        # TemporaryDirectory is private on current Windows Python versions.
        # The publication directory must instead have normal workspace inheritance.
        published = updater.ANALYTICS_DIR / f".xinzhi-acl-test-{uuid4().hex}"
        published.mkdir()
        self.addCleanup(shutil.rmtree, published)
        existing = published / "existing.npz"
        new = published / "new.png"
        existing.write_bytes(b"old index")

        def acl(path):
            environment = os.environ.copy()
            environment["XINZHI_ACL_TEST_PATH"] = str(path)
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "$ErrorActionPreference = 'Stop'; "
                 "[System.IO.File]::GetAccessControl($env:XINZHI_ACL_TEST_PATH)."
                 "GetSecurityDescriptorSddlForm("
                 "[System.Security.AccessControl.AccessControlSections]::Access)"],
                env=environment, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            return result.stdout.strip()

        expected_acl = acl(existing)
        updater.publish_files(
            [(self.bag_output, existing), (self.old_template, new)],
            self.root / "previous",
        )
        self.assertEqual(expected_acl, acl(existing))
        self.assertEqual(expected_acl, acl(new))
        self.assertEqual(b"old bag index", existing.read_bytes())

    def test_archive_failure_never_publishes_or_removes_inputs(self):
        original_copytree = updater.shutil.copytree

        def fail_archive(source, destination, *args, **kwargs):
            if Path(destination).is_relative_to(self.archive):
                raise OSError("archive is read-only")
            return original_copytree(source, destination, *args, **kwargs)

        with patch.object(updater.shutil, "copytree", fail_archive):
            with self.assertRaisesRegex(OSError, "archive is read-only"):
                self.update()
        self.assert_unmodified()

    def test_unknown_name_duplicate_id_and_corrupt_image_keep_inputs(self):
        unknown = self.inputs / "未知.png"
        unknown.write_bytes(self.original_bytes)
        with self.assertRaisesRegex(ValueError, "无法唯一匹配"):
            self.update()
        unknown.unlink()
        duplicate = self.inputs / "char_002_new.png"
        duplicate.write_bytes(self.original_bytes)
        with self.assertRaisesRegex(ValueError, "同一角色"):
            self.update()
        duplicate.unlink()
        self.assert_unmodified()
        self.original.write_bytes(b"broken PNG")
        with self.assertRaisesRegex(ValueError, "无法解码"):
            self.update()
        self.assertEqual(b"broken PNG", self.original.read_bytes())

    def test_existing_template_requires_overwrite_and_is_backed_up(self):
        destination = self.templates / "char_002_new-bag.png"
        old_bytes = self.old_template.read_bytes()
        destination.write_bytes(old_bytes)
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.update()
        self.assertEqual(self.original_bytes, self.original.read_bytes())
        self.update(overwrite=True)
        batch = next(self.archive.iterdir())
        self.assertEqual(old_bytes, next((batch / "previous").glob("*-char_002_new-bag.png")).read_bytes())

    def test_id_named_ready_template_and_identical_retry(self):
        self.original.unlink()
        source = self.inputs / "char_002_new-bag.png"
        rng = np.random.default_rng(23)
        updater.write_png(source, rng.integers(0, 256, (58, 70, 3), dtype=np.uint8))
        data = source.read_bytes()
        self.update()
        source.write_bytes(data)
        report = self.update()
        self.assertTrue(report["records"][0]["unchanged"])
        self.assertEqual(2, len(list(self.archive.iterdir())))

    def test_unrelated_files_and_overlapping_archive_are_rejected(self):
        note = self.inputs / "notes.txt"
        note.write_text("keep me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不支持"):
            self.update()
        self.assertEqual("keep me", note.read_text(encoding="utf-8"))
        note.unlink()
        self.archive = self.inputs / "archive"
        with self.assertRaisesRegex(ValueError, "不能重叠"):
            self.update()
        self.assert_unmodified()


if __name__ == "__main__":
    unittest.main()
