"""Verify the built APK's embedded identity, resources and Agent before publishing."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from prepare import CONFIG, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("apk", type=Path)
    parser.add_argument("--work", type=Path, default=ROOT / "android/.build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, default=os.environ.get("ANDROID_HOME"))
    parser.add_argument("--certificate-sha256", default="")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads((args.work / "metadata.json").read_text(encoding="utf-8"))
    tools = args.sdk / "build-tools/36.0.0"
    exe = ".exe" if os.name == "nt" else ""
    java = Path(os.environ["JAVA_HOME"]) / "bin" / ("java" + exe)
    signature = subprocess.check_output([str(java), "-jar", str(tools / "lib/apksigner.jar"), "verify", "--verbose", "--print-certs", str(args.apk)], text=True, encoding="utf-8")
    certificate = re.search(r"Signer #1 certificate SHA-256 digest: (\w+)", signature)[1]
    if args.certificate_sha256 and certificate.lower() != args.certificate_sha256.lower().replace(":", ""):
        raise ValueError("APK signing certificate differs from the configured release certificate")
    subprocess.run([str(tools / ("zipalign" + exe)), "-c", "-P", "16", "4", str(args.apk)], check=True)
    badging = subprocess.check_output([str(tools / ("aapt2" + exe)), "dump", "badging", str(args.apk)], text=True, encoding="utf-8")
    for text in (f"name='{metadata['applicationId']}'", f"versionCode='{metadata['versionCode']}'", f"versionName='{metadata['versionName']}'",
                 "minSdkVersion:'28'", "native-code: 'arm64-v8a'", f"application-label:'{metadata['label']}'"):
        if text not in badging:
            raise ValueError(f"APK identity/ABI mismatch: {text}")
    with zipfile.ZipFile(args.apk) as apk:
        assert apk.testzip() is None
        for name in ("libMaaFramework.so", "libMaaAgentServer.so", "libMaaAndroidNativeControlUnit.so", "libbridge.so", "liblauncher.so", "libc++_shared.so"):
            assert f"lib/{config['abi']}/{name}" in apk.namelist(), name
        runtime = json.loads(apk.read("assets/agent/agent-runtime.json"))["runtimes"][0]
        assert runtime["args"] == ["-u", "agent/main.py"]
        assert runtime["env"]["MAA_LIBRARY_DIR"] == "{nativeLibs}"
        with zipfile.ZipFile(io.BytesIO(apk.read("assets/pi.zip"))) as pi:
            assert pi.testzip() is None
            assert json.loads(pi.read("android-build.json")) == metadata
            interface = json.loads(pi.read("interface.json"))
            assert interface["version"] == metadata["versionName"]
            assert interface["github"] == "https://github.com/" + config["githubRepository"]
            assert interface.get("mirrorchyan_rid") == config.get("mirrorchyanRid")
            assert "agent/main.py" in pi.namelist()
            for locale in ("base", "zh_tw"):
                for model_dir in ("", "en/"):
                    for name in ("det.onnx", "rec.onnx", "keys.txt"):
                        assert pi.getinfo(f"resource/{locale}/model/ocr/{model_dir}{name}").file_size > 0
            forbidden = re.compile(r"(^|/)(\.git|__pycache__|\.venv|tests)/|\.(exe|dll|pyd|pyc|pyo|keystore|jks|pfx|log)$", re.I)
            assert not any(forbidden.search(name) for name in pi.namelist()), "Unexpected local/build file in PI payload"
        with zipfile.ZipFile(io.BytesIO(apk.read("assets/agent/bundle.zip"))) as agent:
            assert agent.testzip() is None
            abi = config["abi"]
            for name in ("bin/python3", "prefix/lib/libpython3.13.so", "site-packages/cv2.so", "site-packages/pure.zip", "site-packages/zhconv/zhcdict.json"):
                assert agent.getinfo(f"{abi}/{name}").file_size > 0, name
            cv2 = agent.read(f"{abi}/site-packages/cv2.so")
            assert cv2[:4] == b"\x7fELF" and int.from_bytes(cv2[18:20], "little") == 183
            core = json.loads(agent.read(f"{abi}/agent-core.json"))
            assert core["python"] == config["pythonVersion"]
            assert core["provides"]["maafw"] == config["frameworkVersion"].removeprefix("v")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / metadata["outputName"]
    shutil.copy2(args.apk, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".apk.sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    report = {**metadata, "sha256": digest, "certificateSha256": certificate, "verified": True, "deviceRuntimeTested": False}
    (args.output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
