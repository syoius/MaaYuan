import sys  # 1. 导入 sys 模块
from pathlib import Path
import shutil

assets_dir = Path(__file__).parent / "assets"


def configure_ocr_model():
    if not (assets_dir / "MaaCommonAssets" / "OCR").exists():
        print(
            'Please clone this repository completely, don’t miss "--recursive", and don’t download the zip package!'
        )
        print('请完整克隆本仓库，不要漏掉 "--recursive"，也不要下载 zip 包！')
        exit(1)

    if sys.platform == "darwin":
        zh_model_version = "ppocr_v4"
        print("Detected macOS, will use ppocr_v4 for Chinese OCR model.")
    else:
        zh_model_version = "ppocr_v5"
        print(f"Detected OS: {sys.platform}. Will use ppocr_v5 for Chinese OCR model.")

    zh_model_source_path = (
        assets_dir / "MaaCommonAssets" / "OCR" / zh_model_version / "zh_cn"
    )

    zh_cn_ocr_dir = assets_dir / "resource" / "base" / "model" / "ocr"
    if not zh_cn_ocr_dir.exists():
        print(f"Copying {zh_model_version} zh_cn model to: {zh_cn_ocr_dir}")
        shutil.copytree(
            zh_model_source_path,
            zh_cn_ocr_dir,
            dirs_exist_ok=True,
        )
    else:
        print("Found existing zh_cn OCR directory, skipping default OCR model import.")

    zh_tw_ocr_dir = assets_dir / "resource" / "zh_tw" / "model" / "ocr"
    if not zh_tw_ocr_dir.exists():
        print(f"Copying {zh_model_version} zh_cn model to: {zh_tw_ocr_dir}")
        shutil.copytree(
            zh_model_source_path,
            zh_tw_ocr_dir,
            dirs_exist_ok=True,
        )
    else:
        print("Found existing zh_tw OCR directory, skipping default OCR model import.")

    en_ocr_dir = assets_dir / "resource" / "base" / "model" / "ocr" / "en"
    if not en_ocr_dir.exists():
        shutil.copytree(
            assets_dir / "MaaCommonAssets" / "OCR" / "ppocr_v4" / "en_us",
            en_ocr_dir,
            dirs_exist_ok=True,
        )
    else:
        print(
            "Found existing en OCR directory in base, skipping default OCR model import."
        )

    en_ocr_dir2 = assets_dir / "resource" / "zh_tw" / "model" / "ocr" / "en"
    if not en_ocr_dir2.exists():
        shutil.copytree(
            assets_dir / "MaaCommonAssets" / "OCR" / "ppocr_v4" / "en_us",
            en_ocr_dir2,
            dirs_exist_ok=True,
        )
    else:
        print(
            "Found existing en OCR directory in zh_tw, skipping default OCR model import."
        )


if __name__ == "__main__":
    configure_ocr_model()

    print("OCR model configured.")
