from pathlib import Path

import shutil
import sys
import json

from configure import configure_ocr_model


working_dir = Path(__file__).parent
install_path = working_dir / Path("install")
version = len(sys.argv) > 1 and sys.argv[1] or "v0.0.1"


def install_deps():
    if not (working_dir / "deps" / "bin").exists():
        print('Please download the MaaFramework to "deps" first.')
        print('请先下载 MaaFramework 到 "deps"。')
        sys.exit(1)

    shutil.copytree(
        working_dir / "deps" / "bin",
        install_path,
        ignore=shutil.ignore_patterns(
            "*MaaDbgControlUnit*",
            "*MaaThriftControlUnit*",
            "*MaaRpc*",
            "*MaaHttp*",
        ),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        working_dir / "deps" / "share" / "MaaAgentBinary",
        install_path / "MaaAgentBinary",
        dirs_exist_ok=True,
    )


def install_resource():

    configure_ocr_model()

    shutil.copytree(
        working_dir / "assets" / "resource",
        install_path / "resource",
        dirs_exist_ok=True,
    )
    shutil.copy2(
        working_dir / "assets" / "interface.json",
        install_path,
    )

    with open(install_path / "interface.json", "r", encoding="utf-8") as f:
        interface = json.load(f)

    interface["version"] = version

    with open(install_path / "interface.json", "w", encoding="utf-8") as f:
        json.dump(interface, f, ensure_ascii=False, indent=4)


def install_chores():
    for file in [
        "README.md",
        "LICENSE",
        "自定义派遣脚本修改说明.md",
        "抄作业V2及洞窟抄作业必看.md",
        "install-deps.ps1",
        "requirements.txt",
    ]:
        shutil.copy2(
            working_dir / file,
            install_path,
        )

    (install_path / "config").mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        working_dir / "assets" / "presets" / "新版全部功能.json",
        install_path / "config" / "config.json",
    )
    shutil.copytree(
        working_dir / "assets" / "presets", install_path / "config", dirs_exist_ok=True
    )


def install_agent():
    shutil.copytree(
        working_dir / "agent",
        install_path / "agent",
        dirs_exist_ok=True,
    )

    with open(install_path / "interface.json", "r", encoding="utf-8") as f:
        interface = json.load(f)

    if sys.platform.startswith("win"):
        interface["agent"]["child_exec"] = r"{PROJECT_DIR}/python/python.exe"
    elif sys.platform.startswith("darwin"):
        interface["agent"]["child_exec"] = r"{PROJECT_DIR}/python/bin/python3"
    elif sys.platform.startswith("linux"):
        interface["agent"]["child_exec"] = r"python3"

    interface["agent"]["child_args"] = [r"{PROJECT_DIR}/agent/main.py", "-u"]

    with open(install_path / "interface.json", "w", encoding="utf-8") as f:
        json.dump(interface, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    install_deps()
    install_resource()
    install_chores()
    install_agent()
    print(f"Install to {install_path} successfully.")
