"""Capacity-only tests; no device gestures or full inventory replay."""
import sys,unittest,tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'agent'))
from utils.star_bag_ocr import parse_bag_text,recognize_bag,BAG_ROI

class BagOcrTests(unittest.TestCase):
    def test_pair_format_and_real_zero(self):
        self.assertEqual(parse_bag_text(' 203 / 270 '),(203,270))
        self.assertEqual(parse_bag_text('0／500'),(0,500))
    def test_invalid_or_missing_never_becomes_zero(self):
        for text in ('','270','203','203/','/270','203/200','203/0','-1/270','1.5/270','2O3/270','203/270 extra','2147483648/2147483649',None):
            with self.subTest(text=text):self.assertIsNone(parse_bag_text(text))
    def image(self):return np.zeros((1280,720,3),np.uint8)
    def provider(self,values):
        values=iter(values)
        def call(image):
            text,score=next(values)
            self.assertEqual(image.shape,(BAG_ROI[3]*3,BAG_ROI[2]*3,3))
            return {'all_results':[{'text':text,'score':score}],'ocr_ms':1}
        return call
    def test_confident_reading_uses_one_shared_provider_call(self):
        image=self.image();result=recognize_bag(image,self.provider([('203/270',.99)]))
        self.assertTrue(result['complete']);self.assertEqual((result['currentCount'],result['capacity']),(203,270));self.assertEqual(len(result['variants']),1)
        self.assertFalse(image.any())
    def test_low_confidence_and_missing_stop_after_two_variants(self):
        for values in ([('203/270',.2),('203/270',.3)],[('bad',.99),('',.99)]):
            result=recognize_bag(self.image(),self.provider(values));self.assertFalse(result['complete']);self.assertIsNone(result['capacity']);self.assertIsNone(result['currentCount']);self.assertEqual(len(result['variants']),2)
    def test_conflicting_confident_variants_are_unknown(self):
        result=recognize_bag(self.image(),self.provider([('203/270',.92),('208/270',.99)]))
        self.assertFalse(result['complete']);self.assertIsNone(result['capacity']);self.assertIn('bag_numbers_conflict',result['warnings'])
    def test_wrong_frame_never_invokes_ocr(self):
        def forbidden(image):raise AssertionError('bad frame reached OCR')
        self.assertFalse(recognize_bag(np.zeros((20,20,3),np.uint8),forbidden)['complete'])

class BagCaptureTests(unittest.TestCase):
    def test_uncertain_frame_retries_are_bounded_and_identity_checked(self):
        from custom.action import star_full_local_ocr_shadow as session
        with tempfile.TemporaryDirectory() as name:
            events=[]
            owner=SimpleNamespace(out=Path(name),last_capture={'file':'original.png'},progress=lambda stage:None,ocr=object(),check=lambda:None,
                capture=lambda label:events.append(label) or np.zeros((1280,720,3),np.uint8),check_backpack=lambda image,section:events.append(section))
            with patch.object(session,'recognize_bag',side_effect=lambda *args:session.empty_bag('unconfirmed')) as reco, patch.object(session.time,'sleep'):
                session.FullShadowSession.read_bag(owner,np.zeros((1280,720,3),np.uint8))
            self.assertEqual(reco.call_count,3);self.assertEqual(events,['bag-stable-retry','main','bag-stable-retry','main'])
            self.assertIsNone(owner.bag['capacity']);self.assertFalse(owner.bag['complete'])

if __name__=='__main__':unittest.main()
