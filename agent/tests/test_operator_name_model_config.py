import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import configure

class OperatorNameModelConfigTests(unittest.TestCase):
    def test_both_resources_receive_name_model_without_replacing_default(self):
        with tempfile.TemporaryDirectory() as directory:
            assets=Path(directory)
            files=('det.onnx','rec.onnx','keys.txt')
            for model,value in (('ppocr_v6/small','default'),('ppocr_v5/zh_cn','names'),('ppocr_v4/en_us','english')):
                path=assets/'MaaCommonAssets/OCR'/model;path.mkdir(parents=True)
                for file in files:(path/file).write_text(value)
            with patch.object(configure,'assets_dir',assets):
                configure.configure_ocr_model()
                for resource in ('base','zh_tw'):
                    ocr=assets/'resource'/resource/'model/ocr'
                    for file in files:
                        self.assertEqual((ocr/file).read_text(),'default')
                        self.assertEqual((ocr/'operator_names'/file).read_text(),'names')
                (assets/'resource/base/model/ocr/keys.txt').write_text('user-default')
                configure.configure_ocr_model()
                self.assertEqual((assets/'resource/base/model/ocr/keys.txt').read_text(),'user-default')

if __name__=='__main__':unittest.main()
