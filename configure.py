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

    zh_cn_ocr_dir = assets_dir / "resource" / "base" / "model" / "ocr"
    if not zh_cn_ocr_dir.exists():  # copy default OCR model only if dir does not exist
        shutil.copytree(
            assets_dir / "MaaCommonAssets" / "OCR" / "ppocr_v4" / "zh_cn",
            zh_cn_ocr_dir,
            dirs_exist_ok=True,
        )
    else:
        print("Found existing zh_cn OCR directory, skipping default OCR model import.")

    zh_tw_ocr_dir = assets_dir / "resource" / "zh_tw" / "model" / "ocr"
    if not zh_tw_ocr_dir.exists():
        shutil.copytree(
            assets_dir / "MaaCommonAssets" / "OCR" / "ppocr_v3" / "zh_tw",
            zh_tw_ocr_dir,
            dirs_exist_ok=True,
        )
    else:
        print("Found existing zh_tw OCR directory, skipping default OCR model import.")


if __name__ == "__main__":
    configure_ocr_model()

    print("OCR model configured.")
