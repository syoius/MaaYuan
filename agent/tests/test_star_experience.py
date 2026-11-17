from pathlib import Path
from unittest import mock
from unittest.mock import patch
import numpy as np
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))

from star_fixture_inputs import FakeGeometry
from star_ocr_golden import golden
from utils import star_experience_geometry as geometry
from utils import star_experience_ocr as ocr
from utils import star_experience_postprocess as post
from utils.star_experience_ocr import recognize_experience
from utils.star_experience_postprocess import confirm_experience_count
from utils.star_geometry import create_screenshot_profile
from utils.star_local_ocr_shadow import hash_image


class ExperienceParityTests(unittest.TestCase):
    def geometry_parity(self, image):
        before = hash_image(image)
        found = {}
        real_find = geometry.find_experience_circles

        def capture_find(*args):
            found.update(real_find(*args))
            return found

        with patch.object(geometry,'find_experience_circles',side_effect=capture_find):
            actual = geometry.recognize_experience_geometry(image)
        expected = golden('session', 'experienceGeometry', image)
        self.assertEqual(set(actual),set(expected))
        # Elapsed wall clocks differ; the key and numeric type remain contractual.
        self.assertGreaterEqual(actual['geometryMs'],0)
        self.assertEqual({k:v for k,v in actual.items() if k!='geometryMs'},
                         {k:v for k,v in expected.items() if k!='geometryMs'})
        self.assertEqual(found,golden('experience', 'find',image,profile=expected['profile'],tab=expected['tab']))
        self.assertEqual(hash_image(image),before)
        return actual,found

    @staticmethod
    def empty_canvas():
        image = np.full((1280,720,3),(36,42,54),np.uint8)
        image[220:250,490:680] = (120,160,220)
        return image


    def test_synthetic_empty_unverified_and_coloured_foreground_coverage(self):
        empty = self.empty_canvas()
        # Keep the normal 9:16 phone family while exercising viewport offsets.
        empty[:,:8] = empty[:,-8:] = 0
        actual,found = self.geometry_parity(empty)
        self.assertEqual(actual['profile']['viewport']['x'],8)
        self.assertTrue(actual['coverage']['emptyConfirmed'])
        self.assertEqual((found['proposals'],found['row'],actual['candidates']),([],[],[]))
        unknown = empty.copy()
        unknown[330:390,200:260,0] = 150  # single BGR channel change
        actual,_ = self.geometry_parity(unknown)
        self.assertFalse(actual['coverage']['emptyConfirmed'])
        self.assertIn('experience_content_unaccounted',actual['coverage']['warnings'])

    def test_tab_8_connected_components_area_ties_and_search_bounds(self):
        for mode in ('diagonal','left-wins','wrong-shape','no-tab'):
            image = np.full((1280,720,3),50,np.uint8)
            colour = (120,160,220)
            if mode == 'diagonal':
                # Each component is too narrow alone; diagonal contact on the
                # downsampled lattice qualifies only with 8-connectivity.
                image[200:226,500:556] = colour
                image[226:252,556:620] = colour
            elif mode == 'left-wins':
                image[200:232,40:200] = image[200:232,500:660] = colour
            elif mode == 'wrong-shape':
                image[200:206,480:670] = colour
            profile = create_screenshot_profile(image)
            tab = geometry.selected_experience_tab_evidence(image,profile['viewport'])
            bounds = geometry.experience_search_bounds(profile,tab)
            with self.subTest(synthetic_tab=mode):
                self.assertEqual(dict(tab=tab,bounds=bounds),golden('experience', 'tab',image,profile=profile))
                self.assertEqual(tab['selected'],mode=='diagonal')

    def test_synthetic_anchors_all_completeness_and_coverage_warning_order(self):
        image = self.empty_canvas()
        profile = create_screenshot_profile(image)
        tab = geometry.selected_experience_tab_evidence(image,profile['viewport'])
        bounds = geometry.experience_search_bounds(profile,tab)
        circles = [dict(centerX=x,centerY=y,radius=r,score=31.) for x,y,r in
                   ((119,bounds['y']+60,48),(274,bounds['y']+10,48),
                    (428,bounds['y']+bounds['height']-10,48),(-.5,350,48.5))]
        candidates = geometry.build_experience_candidates(circles,bounds)
        self.assertEqual(candidates,golden('experience', 'candidates',circles=circles,bounds=bounds))
        self.assertEqual([c['completeness'] for c in candidates],['complete','partial_top','partial_bottom','invalid'])
        cases = [dict(profile=profile,tab=tab,row=[],bounds=bounds),
                 dict(profile=profile,tab={'selected':False},row=[],bounds=bounds),
                 dict(profile=profile,tab=tab,row=circles[::-1],bounds=bounds)]
        actual = [geometry.experience_coverage(image,c['profile'],c['tab'],c['row'],c['bounds']) for c in cases]
        self.assertEqual(actual,golden('experience', 'coverage',image,cases=cases))
        self.assertEqual(actual[-1]['warnings'][:3],['experience_too_many_icons','experience_row_not_left_compacted','experience_icon_outside_content'])
        self.assertTrue(actual[0]['emptyConfirmed'])
        self.assertFalse(actual[1]['complete'])

    def test_count_ascii_and_ecmascript_trim_kind_thresholds_and_status(self):
        texts = ['0','000000','123456','1234567','-1','+1','1.5','1,000','1 2','\t001\r\n',
                 '\ufeff42\u3000','\u20281\u2029','\u00851\u0085','\x1c1\x1c','\u180e1','\uff11','\u0661','', 'bad']
        image = self.empty_canvas()
        colours = [(0,85,255),(255,0,85),(255,170,0),(180,180,180),
                   (0,180,0),(60,61,82),(60,61,83),(0,153,255)]
        icons = []
        for i,colour in enumerate(colours):
            x = 20+i*50
            image[330:365,x:x+35] = colour
            icons.append(dict(x=x,y=330,width=35,height=35))
        icons.append(dict(x=720,y=1280,width=30,height=30))
        states = [[c,k,n] for c in ('complete','partial_top','partial_bottom','invalid')
                  for k,n in (('orange',0),('orange',None),(None,1))]
        actual = dict(parse=[post.parse_experience_count(t) for t in texts],
                      status=[post.experience_status(*s) for s in states],
                      kinds=[post.classify_experience_kind(image,r) for r in icons])
        self.assertEqual(actual,golden('experience', 'post',image,texts=texts,states=states,icons=icons))
        self.assertEqual([k['kind'] for k in actual['kinds'][:4]],['orange','purple','white',None])
        self.assertLess(actual['kinds'][4]['confidence'],ocr.MIN_KIND_CONFIDENCE)
        self.assertIsNone(post.parse_experience_count('1000000'))
        for candidates in ([],[dict(text='bad',confidence=.99)],
                           [dict(text=t,confidence=s) for t,s in [('bad',.99),('00042',.95),('2',.95),('5',.9)]],
                           [dict(text=t,confidence=.99) for t in ('\ufeff0','123456','1234567')]):
            self.assertEqual(post.resolve_experience_count(candidates),golden('session', 'experienceCount', candidates=candidates))

    def test_aggregate_all_warning_branches_and_absence_contract(self):
        def item(kind,count,status='accepted',score=.99):
            return dict(kind=kind,count=count,status=status,kindConfidence=score,countConfidence=score)
        cases = [[],[item('purple',3)],[item('orange',0),item('white',6)],
                 [item('purple',7),item('orange',2),item('purple',9,score=.98),item(None,None,'needs_review'),item('white',None,'needs_review')],
                 [item('orange',None,'excluded_partial')],
                 [item(k,1) for k in ('orange','orange','purple','purple','white','white')]]
        for results in cases:
            for selected,cropped,empty in ((True,False,False),(True,False,True),(False,True,False),(True,True,True)):
                options = dict(selectedTab=selected,viewportCropped=cropped,emptyConfirmed=empty)
                with self.subTest(results=results,options=options):
                    actual = post.aggregate_experience(results,options)
                    self.assertEqual(actual,golden('session', 'experienceAggregate', results=results, options=options))


    def test_invalid_native_capture(self):
        for image in (None,np.zeros((2844,1260,3),np.uint8),np.zeros((1280,720),np.uint8),np.zeros((1280,720,4),np.uint8),np.zeros((1280,720,3),np.float32)):
            with self.assertRaisesRegex(ValueError,'720x1280 uint8 BGR'):
                geometry.recognize_experience_geometry(image)

class ExperienceTests(unittest.TestCase):
 def image(self):return np.full((1280,720,3),220,np.uint8)
 def recognize(self,image,provider,geometry):
  # Stub only geometry for ROI/variant boundary tests; parse and aggregate natively.
  with mock.patch('utils.star_experience_ocr.recognize_experience_geometry',geometry.experience_geometry):
   return recognize_experience(image,provider)
 def provider(self,values,score=.99):
  values=iter(values)
  return lambda image:{'ocr_ms':1.,'all_results':[{'text':str(next(values)),'score':score,'box':[0,0,image.shape[1],image.shape[0]]}]}
 def test_three_counts_are_mapped_by_color_not_array_order(self):
  result=self.recognize(self.image(),self.provider([91,1,525]),FakeGeometry())
  self.assertTrue(result['complete']);self.assertEqual([result[k+'Count'] for k in ('orange','purple','white')],[1,525,91])
  self.assertEqual(result['timing']['ocr_calls'],3)
 def test_zero_is_confirmed_and_multidigit_preserved(self):
  result=self.recognize(self.image(),self.provider([0,123456,42]),FakeGeometry())
  self.assertTrue(result['complete']);self.assertEqual((result['whiteCount'],result['orangeCount']),(0,123456))
 def test_low_confidence_is_bounded_and_never_filled_with_zero(self):
  result=self.recognize(self.image(),self.provider(['bad']*9,.2),FakeGeometry())
  self.assertFalse(result['complete']);self.assertIsNone(result['orangeCount']);self.assertLessEqual(result['timing']['ocr_calls'],9)
 def test_bad_roi_and_partial_icon_are_not_recognized(self):
  b=FakeGeometry();b.candidates[0]['countRect']['x']=-1;b.candidates[1]['completeness']='partial_bottom'
  result=self.recognize(self.image(),self.provider([7]),b)
  self.assertFalse(result['complete']);self.assertEqual(result['timing']['ocr_calls'],1)
  self.assertIn('experience_invalid_count_roi',result['warnings']);self.assertIn('experience_icon_incomplete',result['warnings'])
  self.assertIn('experience_viewport_cropped',result['warnings']);self.assertIsNone(result['orangeCount'])


class CountParsingTests(unittest.TestCase):
 def test_confirmation_requires_independent_votes_and_rejects_conflicts(self):
  confirm=lambda *rows:confirm_experience_count([dict(text=t,confidence=s,variant=v) for t,s,v in rows])
  color,contrast,badge='color_3x','contrast_3x','badge_otsu_3x'
  a=('12',.75,color);b=('0012',.80,contrast)
  self.assertEqual(confirm(a,b),(12,None))
  self.assertEqual(confirm(('0',.75,color),('0',.75,badge)),(0,None))
  self.assertEqual(confirm(('12',.90,color),('13',.80,contrast)),(12,None))
  for rows in [(a,), (a,('12',.85,color)), (a,('12',.749,contrast)),
               (('1',.199496,color),('1',.226818,contrast)),
               (('-1',.99,color),('-1',.99,contrast)),
               (('1.5',.99,color),('1000000',.99,badge))]:
   with self.subTest(rows=rows):
    self.assertEqual(confirm(*rows),(None,'experience_count_unconfirmed'))
  for rows in [(a,('13',.80,contrast)), (a,b,('13',.75,badge)),
               (a,b,('13',.94,badge)), (('12',.92,color),('13',.99,contrast))]:
   with self.subTest(rows=rows):
    self.assertEqual(confirm(*rows),(None,'experience_count_conflict'))


if __name__ == "__main__":
    unittest.main()
