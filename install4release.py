from pathlib import Path

import shutil
import sys
import json

from configure import configure_ocr_model


working_dir = Path(__file__).parent
install_path = working_dir / Path("install")
version = len(sys.argv) > 1 and sys.argv[1] or "v0.0.1"


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
    shutil.copy2(
        working_dir / "README.md",
        install_path,
    )
    shutil.copy2(
        working_dir / "LICENSE",
        install_path,
    )
    shutil.copy2(
        working_dir / "自定义派遣脚本修改说明.md",
        install_path,
    )
    shutil.copy2(
        working_dir / "抄作业V2及洞窟抄作业必看.md",
        install_path,
    )
    shutil.copy2(
        working_dir / "install-deps.ps1",
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


if __name__ == "__main__":
    install_resource()
    install_chores()

    print(f"Install to {install_path} successfully.")
