import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare


class AndroidPackagingTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(prepare.CONFIG.read_text(encoding="utf-8"))
        self.interface = {
            "version": "old", "custom_title": "Maa鸢", "github": "https://github.com/syoius/MaaYuan",
            "mirrorchyan_rid": "MaaYuan", "mirrorchyan_multiplatform": True,
            "controller": [{"name": "模拟器", "type": "Adb"}, {"name": "PlayCover", "type": "PlayCover"}],
            "agent": {"child_exec": "python.exe", "child_args": ["../agent/main.py"]},
            "task": [{"name": "daily", "entry": "Daily"}],
        }

    def test_version_matches_shared_tag_without_removing_prefix(self):
        for version in ("v2.2.1", "v2.2.1-beta.1", "v2.2.1-beta.1-ci.53-g192e24a3"):
            with self.subTest(version=version):
                metadata = prepare.metadata(version, 10101, self.config)
                interface = prepare.android_interface(self.interface, version, self.config)
                self.assertEqual(version, metadata["versionName"])
                self.assertEqual(version, interface["version"])
                self.assertTrue(metadata["outputName"].endswith(f"-{version}.apk"))

    def test_version_and_source_identity_reject_invalid_inputs(self):
        for version in ("", "latest", "v2.2.1\nother=value", "../v2.2.1"):
            with self.assertRaises(ValueError):
                prepare.metadata(version, 1, self.config)
        for code in (0, -1, 2_100_000_001):
            with self.assertRaises(ValueError):
                prepare.metadata("v2.2.1", code, self.config)
        self.config["sourceRevision"] = "maayuan"
        with self.assertRaises(ValueError):
            prepare.metadata("v2.2.1", 1, self.config)

    def test_desktop_interface_is_preserved_and_unconfigured_mirror_removed(self):
        before = copy.deepcopy(self.interface)
        android = prepare.android_interface(self.interface, "v2.2.1", self.config)
        self.assertEqual(before, self.interface)
        self.assertEqual(before["task"], android["task"])
        self.assertEqual([before["controller"][0]], android["controller"])
        self.assertNotIn("mirrorchyan_rid", android)
        self.assertNotIn("mirrorchyan_multiplatform", android)
        self.assertEqual("python3", android["agent"]["child_exec"])

    def test_explicit_android_mirror_channel_is_retained(self):
        self.config["mirrorchyanRid"] = "MaaYuanAndroid"
        android = prepare.android_interface(self.interface, "v2.2.1", self.config)
        self.assertEqual("MaaYuanAndroid", android["mirrorchyan_rid"])
        self.assertTrue(android["mirrorchyan_multiplatform"])

    def test_payload_imports_models_and_normalizes_only_staged_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "assets").mkdir()
            (root / "assets/interface.json").write_text(json.dumps(self.interface), encoding="utf-8")
            (root / "agent").mkdir()
            (root / "agent/main.py").write_text("print('agent')\n", encoding="utf-8")
            (root / "agent/tests").mkdir()
            (root / "agent/tests/test_private.py").write_text("", encoding="utf-8")
            (root / "agent/debug").mkdir()
            (root / "agent/debug/screenshot.png").write_bytes(b"local debug image")
            (root / "agent/local.log").write_text("local run log", encoding="utf-8")
            for name in ("logo.png", "CONTACT", "LICENSE", "README.md"):
                (root / name).write_bytes(b"fixture")
            nodes = {"Launch": {"action": {"type": "StartApp", "param": {"package": "game.package/Launcher"}}, "max_hit": 9}}
            for locale in ("base", "zh_tw"):
                pipeline = root / "assets/resource" / locale / "pipeline"
                pipeline.mkdir(parents=True)
                prepare.write_json(pipeline / "start_up.json", nodes)
            for model, names in (("ppocr_v6/small", ("det.onnx", "rec.onnx", "keys.txt")), ("ppocr_v4/en_us", ("rec.onnx", "keys.txt"))):
                path = root / "assets/MaaCommonAssets/OCR" / model
                path.mkdir(parents=True)
                for name in names:
                    (path / name).write_bytes(b"model fixture")
            work = root / "android/.build"
            work.mkdir(parents=True)
            prepare.write_json(work / "metadata.json", prepare.metadata("v2.2.1", 101, self.config))
            with patch.object(prepare, "ROOT", root):
                prepare.stage(work, "v2.2.1", self.config)
            for locale in ("base", "zh_tw"):
                staged = work / "payload/resource" / locale
                launch = json.loads((staged / "pipeline/start_up.json").read_text())
                self.assertEqual("game.package", launch["Launch"]["action"]["param"]["package"])
                self.assertEqual(1, launch["Launch"]["max_hit"])
                self.assertEqual(b"model fixture", (staged / "model/ocr/en/det.onnx").read_bytes())
                self.assertEqual(nodes, json.loads((root / "assets/resource" / locale / "pipeline/start_up.json").read_text()))
            self.assertFalse((work / "payload/agent/tests").exists())
            self.assertFalse((work / "payload/agent/debug").exists())
            self.assertFalse((work / "payload/agent/local.log").exists())
            profile = json.loads((work / "profile.yaml").read_text(encoding="utf-8"))
            self.assertEqual("{nativeLibs}", profile["agent"]["runtimes"][0]["env"]["MAA_LIBRARY_DIR"])
            self.assertIn("android-build.json", profile["include"])


if __name__ == "__main__":
    unittest.main()
