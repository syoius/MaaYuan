"""Build the pinned Android Python Agent and its OpenCV extension on a CI host."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from prepare import CONFIG, ROOT


def run(*args: object, **kwargs) -> None:
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def verified_download(url: str, destination: Path, digest: str) -> None:
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"SHA-256 mismatch: {url}")
    destination.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-source", type=Path, required=True)
    parser.add_argument("--opencv-source", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "android/.build")
    parser.add_argument("--sdk", type=Path, default=os.environ.get("ANDROID_HOME"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    work = args.work.resolve()
    if work == ROOT or not work.is_relative_to(ROOT):
        raise ValueError("Runtime work directory must be below the MaaYuan checkout")
    work.mkdir(parents=True, exist_ok=True)
    sdk = args.sdk.resolve()
    ndk = sdk / "ndk" / config["ndkVersion"]
    host = "windows-x86_64" if sys.platform == "win32" else "linux-x86_64"
    executable = ".exe" if sys.platform == "win32" else ""
    cmake = sdk / "cmake/3.22.1/bin" / ("cmake" + executable)
    ninja = cmake.with_name("ninja" + executable)
    strip = ndk / "toolchains/llvm/prebuilt" / host / "bin" / ("llvm-strip" + executable)
    wheels = work / "wheels"
    wheels.mkdir(exist_ok=True)
    # zhconv is published as an sdist; create its pure wheel before cross-platform pip resolution.
    run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", wheels, "zhconv==1.4.3")
    env = os.environ.copy()
    env["PIP_FIND_LINKS"] = str(wheels)
    agent = work / "agent-dist"
    run(sys.executable, args.shell_source.resolve() / "scripts/build_agent_bundle.py",
        "--out", agent, "--abi", config["abi"], "--core-tag", config["agentCoreTag"],
        "--requirements", ROOT / "android/requirements.txt",
        "--extra-index-url", "https://chaquo.com/pypi-13.1/", env=env)
    bundle = agent / config["abi"] / "bundle"
    manifest = json.loads((bundle / "agent-core.json").read_text(encoding="utf-8"))
    if manifest["python"] != config["pythonVersion"] or manifest["provides"]["maafw"] != config["frameworkVersion"].removeprefix("v"):
        raise ValueError("Agent core does not match the pinned Python / MaaFramework versions")
    site = bundle / "site-packages"
    archive = work / "python-android-dev.zip"
    verified_download(config["pythonHeadersUrl"], archive, config["pythonHeadersSha256"])
    headers = work / "python-android-dev"
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if not (headers / entry.filename).resolve().is_relative_to(headers.resolve()):
                raise ValueError("Unsafe Python development archive path")
        source.extractall(headers)
    opencv = args.opencv_source.resolve()
    revision = subprocess.check_output(["git", "-C", str(opencv), "rev-parse", "HEAD"], text=True).strip()
    if revision != config["opencvRevision"]:
        raise ValueError("OpenCV checkout does not match build-config.json")
    # OpenCV disables Python on Android despite supporting explicit cross-compilation inputs.
    module = opencv / "modules/python/CMakeLists.txt"
    text = module.read_text(encoding="utf-8")
    before, after = "if(ANDROID OR APPLE_FRAMEWORK OR WINRT)", "if(APPLE_FRAMEWORK OR WINRT)"
    if before in text:
        module.write_text(text.replace(before, after, 1), encoding="utf-8")
    elif after not in text:
        raise ValueError("OpenCV Python module guard changed")
    build = work / "opencv-build"
    options = {
        "CMAKE_MAKE_PROGRAM": ninja, "CMAKE_TOOLCHAIN_FILE": ndk / "build/cmake/android.toolchain.cmake",
        "ANDROID_ABI": config["abi"], "ANDROID_PLATFORM": "android-28", "ANDROID_STL": "c++_shared",
        "CMAKE_BUILD_TYPE": "Release", "BUILD_SHARED_LIBS": "OFF",
        "BUILD_LIST": "core,imgproc,imgcodecs,features2d,calib3d,python3",
        "BUILD_opencv_python3": "ON", "OPENCV_PYTHON_SKIP_DETECTION": "ON",
        "PYTHON3_VERSION_MAJOR": "3", "PYTHON3_VERSION_MINOR": "13", "PYTHON3_VERSION_STRING": "3.13.9",
        "PYTHON3_NUMPY_VERSION": "2.3.2", "PYTHON3_EXECUTABLE": sys.executable,
        "PYTHON_DEFAULT_EXECUTABLE": sys.executable, "PYTHON_DEFAULT_AVAILABLE": "TRUE",
        "PYTHON3_INCLUDE_PATH": headers / "include/python3.13",
        "PYTHON3_NUMPY_INCLUDE_DIRS": site / "numpy/_core/include",
        "PYTHON3_LIBRARIES": bundle / "prefix/lib/libpython3.13.so", "OPENCV_FORCE_PYTHON_LIBS": "ON",
        "PYTHON3_CVPY_SUFFIX": ".so", "OPENCV_SKIP_PYTHON_LOADER": "ON",
        "OPENCV_PYTHON3_INSTALL_PATH": work / "opencv-dist", "WITH_JPEG": "ON", "WITH_PNG": "ON",
    }
    for name in ("BUILD_TESTS", "BUILD_PERF_TESTS", "BUILD_EXAMPLES", "BUILD_JAVA", "BUILD_ANDROID_EXAMPLES",
                 "BUILD_ANDROID_PROJECTS", "BUILD_opencv_apps", "BUILD_ANDROID_SERVICE", "WITH_OPENCL",
                 "WITH_IPP", "WITH_ITT", "WITH_TBB", "WITH_FFMPEG", "WITH_GSTREAMER", "WITH_CUDA", "WITH_WEBP",
                 "WITH_TIFF", "WITH_OPENEXR", "WITH_OPENJPEG", "WITH_JASPER", "WITH_PROTOBUF", "WITH_FLATBUFFERS"):
        options[name] = "OFF"
    run(cmake, "-G", "Ninja", "-S", opencv, "-B", build, *[f"-D{key}={value}" for key, value in options.items()])
    run(cmake, "--build", build, "--target", "opencv_python3", "--parallel", "4")
    extension = build / "lib" / config["abi"] / "python3/cv2.so"
    run(strip, "--strip-unneeded", extension)
    if extension.read_bytes()[:4] != b"\x7fELF":
        raise ValueError("OpenCV output is not an ELF extension")
    shutil.copy2(extension, site / "cv2.so")
    info = site / f"opencv_python-{config['opencvVersion']}.dist-info"
    licenses = info / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: opencv-python\nVersion: {config['opencvVersion']}\nLicense: Apache-2.0\n", encoding="utf-8")
    for name, origin in {
        "LICENSE": "LICENSE", "libpng-LICENSE": "3rdparty/libpng/LICENSE",
        "libjpeg-LICENSE": "3rdparty/libjpeg-turbo/LICENSE.md", "libjpeg-README.ijg": "3rdparty/libjpeg-turbo/README.ijg",
        "cpufeatures-LICENSE": "3rdparty/cpufeatures/LICENSE",
    }.items():
        shutil.copy2(opencv / origin, licenses / name)
    for name, origin in {"carotene-BSD": "hal/carotene/src/common.cpp", "cpufeatures-BSD": "3rdparty/cpufeatures/cpu-features.c"}.items():
        (licenses / name).write_bytes((opencv / origin).read_bytes().split(b"*/", 1)[0] + b"*/\n")
    (licenses / "NOTICE").write_text("This software is based in part on the work of the Independent JPEG Group.\n", encoding="utf-8")
    # zhconv opens zhcdict.json relative to __file__; keep it outside zipimport.
    with zipfile.ZipFile(site / "pure.zip") as source:
        for name in source.namelist():
            if name.startswith("zhconv/"):
                source.extract(name, site)
    print(f"Android Agent ready: {agent}")


if __name__ == "__main__":
    main()
