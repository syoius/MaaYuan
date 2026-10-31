import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.star_four_row_shadow import claim_once,stable_pair,audit_candidate,adaptive_suggestion
class FourRowTests(unittest.TestCase):
 def test_candidate_and_late_audit_must_be_measured(self):
  good={'measured':True,'signed_shift_y_px':2,'direct_score':.999,'same_position_score':.95}
  self.assertTrue(stable_pair(good));self.assertFalse(stable_pair({**good,'signed_shift_y_px':3}));self.assertFalse(stable_pair({**good,'measured':False}))
  self.assertEqual(audit_candidate({**good,'signed_shift_y_px':5}),'failed');self.assertEqual(audit_candidate({**good,'measured':False}),'unknown')
 def test_single_gesture_claim_cannot_be_reused(self):
  with tempfile.TemporaryDirectory() as d:
   claim_once(d)
   with self.assertRaises(FileExistsError):claim_once(d)
 def test_adaptation_is_bounded_metadata_only(self):
  g={'rowCenters':[390,556.5,723,889.5],'rowPitches':[166.5]*3,'profile':{'contentBounds':{'y':272,'height':772}}}
  r=adaptive_suggestion(g,633);self.assertEqual(r['desired_content_shift_px'],666);self.assertLessEqual(abs(r['suggested_touch_length_px']-630),30);self.assertGreaterEqual(r['suggested_swipe_end'][1],300);self.assertFalse(r['executed'])
  self.assertNotIn('suggested_swipe_end',adaptive_suggestion(g,None))
if __name__=='__main__':unittest.main()
