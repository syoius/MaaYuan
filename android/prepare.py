"""Prepare MaaYuan resources and a MaaFwApp profile without editing source assets."""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name("build-config.json")


def metadata(version: str, version_code: int, config: dict) -> dict:
    if not re.fullmatch(r"v?\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?(?:\+[A-Za-z0-9.-]+)?", version):
        raise ValueError(f"Invalid shared release version: {version!r}")
    if not 1 <= version_code <= 2_100_000_000:
        raise ValueError("Android versionCode must be between 1 and 2100000000")
    if not re.fullmatch(r"[0-9a-f]{40}", config["sourceRevision"]):
        raise ValueError("MaaFwApp must be pinned to a full commit SHA")
    return {
        "versionName": version,
        "versionCode": version_code,
        "applicationId": "com.aliothmoon.maafw." + config["applicationIdSuffix"],
        "label": config["label"],
        "outputName": f"{config['assetPrefix']}arm64-{version}.apk",
        "sourceRepository": config["sourceRepository"],
        "sourceRevision": config["sourceRevision"],
        "assetPrefix": config["assetPrefix"],
        "frameworkVersion": config["frameworkVersion"],
        "ndkVersion": config["ndkVersion"],
        "opencvRevision": config["opencvRevision"],
    }


def android_interface(source: dict, version: str, config: dict) -> dict:
    interface = copy.deepcopy(source)
    interface["version"] = version
    interface["custom_title"] = config["label"]
    interface["github"] = "https://github.com/" + config["githubRepository"]
    interface.pop("mirrorchyan_rid", None)
    interface.pop("mirrorchyan_multiplatform", None)
    if config.get("mirrorchyanRid"):
        interface["mirrorchyan_rid"] = config["mirrorchyanRid"]
        interface["mirrorchyan_multiplatform"] = True
    interface["controller"] = [c for c in interface["controller"] if c["type"] == "Adb"]
    if not interface["controller"]:
        raise ValueError("MaaYuan must declare an Adb controller")
    interface["agent"] = {"child_exec": "python3", "child_args": ["-u", "agent/main.py"]}
    return interface


def normalize_startup(nodes: dict) -> None:
    for node in nodes.values():
        action = node.get("action", {})
        if isinstance(action, dict) and action.get("type") == "StartApp":
            params = action.get("param", {})
            if isinstance(params.get("package"), str):
                params["package"] = params["package"].split("/", 1)[0]
            node["max_hit"] = 1


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage(work: Path, version: str, config: dict) -> None:
    work = work.resolve()
    if work == ROOT or not work.is_relative_to(ROOT):
        raise ValueError("Build work directory must be below the MaaYuan checkout")
    payload = work / "payload"
    if payload.resolve() != payload or not payload.resolve().is_relative_to(work):
        raise ValueError("Refusing redirected payload directory")
    if payload.exists():
        shutil.rmtree(payload)
    payload.mkdir(parents=True)
    for name in ("logo.png", "CONTACT", "LICENSE", "README.md"):
        shutil.copy2(ROOT / name, payload / name)
    excluded = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo", ".pytest_cache")
    shutil.copytree(ROOT / "assets/resource", payload / "resource", ignore=excluded)
    shutil.copytree(ROOT / "agent", payload / "agent", ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc", ".pytest_cache"))
    source = json.loads((ROOT / "assets/interface.json").read_text(encoding="utf-8"))
    write_json(payload / "interface.json", android_interface(source, version, config))
    shutil.copy2(work / "metadata.json", payload / "android-build.json")
    common = ROOT / "assets/MaaCommonAssets/OCR"
    for locale in ("base", "zh_tw"):
        ocr = payload / "resource" / locale / "model/ocr"
        # Import from the checked-out submodule, never modify shared source resources.
        for destination, origin in ((ocr, common / "ppocr_v6/small"), (ocr / "en", common / "ppocr_v4/en_us")):
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("det.onnx", "rec.onnx", "keys.txt"):
                if not (destination / name).is_file() and (origin / name).is_file():
                    shutil.copy2(origin / name, destination / name)
        # v5.12.3 constructs a detector even for some English only_rec paths.
        if not (ocr / "en/det.onnx").is_file():
            shutil.copy2(ocr / "det.onnx", ocr / "en/det.onnx")
        for directory in (ocr, ocr / "en"):
            for name in ("det.onnx", "rec.onnx", "keys.txt"):
                if not (directory / name).is_file():
                    raise FileNotFoundError(directory / name)
        startup = payload / "resource" / locale / "pipeline/start_up.json"
        nodes = json.loads(startup.read_text(encoding="utf-8"))
        normalize_startup(nodes)
        write_json(startup, nodes)
    # JSON is valid YAML and avoids introducing another parser into the build script.
    profile = {
        "assets": "payload",
        "app": {"id": config["applicationIdSuffix"], "label": config["label"], "icon": "payload/logo.png"},
        "include": ["interface.json", "android-build.json", "resource/**", "agent/**", "logo.png", "CONTACT", "LICENSE", "README.md"],
        "agent": {
            "sourceDir": "agent-dist", "abi": [config["abi"]],
            "runtimes": [{"location": "bundle", "executable": "bin/python3", "args": ["-u", "agent/main.py"],
                "env": {"PYTHONHOME": "{bundle}/prefix", "PYTHONPATH": "{bundle}/site-packages:{bundle}/site-packages/pure.zip",
                        "LD_LIBRARY_PATH": "{bundle}/prefix/lib:{bundle}/site-packages/chaquopy/lib:{nativeLibs}",
                        "MAAFW_BINARY_PATH": "{nativeLibs}", "MAA_LIBRARY_DIR": "{nativeLibs}"}}],
        },
    }
    write_json(work / "profile.yaml", profile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--version-code", type=int, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "android/.build")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--ci", action="store_true", help="Use a separate identity for unsigned-release/PR test builds")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.ci:
        config["applicationIdSuffix"] += ".ci"
        config["label"] += " · CI"
        config["assetPrefix"] += "ci-"
    values = metadata(args.version, args.version_code, config)
    work = args.work.resolve()
    if work == ROOT or not work.is_relative_to(ROOT):
        raise ValueError("Build work directory must be below the MaaYuan checkout")
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "metadata.json", values)
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            for key, value in values.items():
                output.write(f"{key}={value}\n")
    if not args.metadata_only:
        stage(work, args.version, config)
    print(json.dumps(values, ensure_ascii=False))


if __name__ == "__main__":
    main()
