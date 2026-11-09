from pathlib import Path
import copy
import math
import numpy as np
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))

from star_fixture_inputs import FIXTURES
from star_fixture_inputs import load_page
from star_ocr_golden import golden
from utils import star_geometry as geometry
from utils import star_grid_geometry as grid
from utils import star_quality_postprocess as quality
from utils.star_local_ocr_shadow import read_image


class QualityParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=geometry.refine_star_content_bounds(
            geometry.create_screenshot_profile(np.full((1280,720,3),100,np.uint8)),None,'maayuan_mumu')

    def classify_parity(self,cases):
        original=copy.deepcopy(cases)
        actual=[quality.classify_quality_pixels(p) for p in cases]
        self.assertEqual(actual,golden('quality', 'classify',cases=cases))
        self.assertEqual(cases,original)
        return actual

    def cards_parity(self,image,cards,detect=False):
        original=image.copy(),copy.deepcopy(cards)
        expected=golden('quality', 'cards',image,cards=cards,detect=detect)
        actual=[dict(card=c,quality=quality.recognize_quality(image,c),source_rect=quality.quality_rect_for_card(c)) for c in cards]
        self.assertEqual(actual,expected)
        np.testing.assert_array_equal(image,original[0])
        self.assertEqual(cards,original[1])
        return actual

    @staticmethod
    def pixels(hue=12,colour=100,white=0,total=100,saturation=100,value=200):
        return ([dict(hue=hue,saturation=saturation,value=value)]*colour
                +[dict(hue=0,saturation=0,value=200)]*white
                +[dict(hue=0,saturation=0,value=20)]*(total-colour-white))

    def test_categories_unknown_empty_white_priority_and_confidence(self):
        cases=[self.pixels(hue=hue) for hue in (12,142,106,61)]
        cases += [self.pixels(colour=0,white=100),[],self.pixels(colour=0),
                  self.pixels(hue=12,colour=34),self.pixels(hue=35),
                  self.pixels(hue=12,colour=24,white=56)]
        results=self.classify_parity(cases)
        self.assertEqual([r['quality'] for r in results[:5]],list(quality.QUALITY_HUES)+['\u767d'])
        self.assertTrue(all(r['quality'] is None for r in results[5:7]))
        self.assertLess(results[7]['confidence'],results[0]['confidence'])
        self.assertEqual(results[-1]['quality'],'\u767d')

    def test_thresholds_medians_stable_hue_ties_and_evidence_rounding(self):
        cases=[]
        for count in (15,16,17,23,24,25,33,34,35):
            cases.append(self.pixels(colour=count))
        for white in (55,56,57):
            cases.append(self.pixels(colour=24,white=white))
        for saturation in (45,45.0000000001,51.9999999999,52):
            cases.append(self.pixels(saturation=saturation))
        for value in (149.9999999999,150):
            cases.append(self.pixels(saturation=45,value=value))
        for hue in (0,180,31,31.0000000001,87,87.0000000001,124,124.0000000001,
                    12.05,12.15,12.25,12.35):
            cases.append(self.pixels(hue=hue))
        for hues in ((11,13),(10,12,14),(12,12,142,142)):
            cases.append([dict(hue=h,saturation=100,value=200) for h in hues])
        # 1/8 = .125 exercises JS's away-from-zero evidence formatting.
        cases.append([{}])  # missing pixel channels are not positive evidence
        cases.append(self.pixels(colour=1,total=8))
        results=self.classify_parity(cases)
        self.assertEqual(results[-1]['evidence'],'saturated=0.13')

    def test_float_hsv_dependency_and_exact_js_to_fixed(self):
        rgb=[[r,g,b] for r in (0,127,255) for g in (0,127,255) for b in (0,127,255)]
        rgb += [[255,1,127],[255,127,1],[1,255,127],[127,1,255]]
        fixed=[[v,d] for v in (0.,-.0,.125,.145,.285,2.55,12.05,12.15,12.25) for d in (1,2)]
        actual=dict(hsv=[quality._rgb_to_opencv_hsv(*v) for v in rgb],
                    fixed=[quality._to_fixed(v,d) for v,d in fixed])
        self.assertEqual(actual,golden('quality', 'numbers',rgb=rgb,fixed=fixed))

    def test_vectorized_hsv_matches_scalar_at_pixel_boundaries(self):
        rgb = [(255, i, 0) for i in range(256)]
        rgb += [(255, 0, i) for i in range(256)]
        rgb += [(0, 255, i) for i in range(256)]
        rgb += [(i, 255, 0) for i in range(256)]
        rgb += [(i, 0, 255) for i in range(256)]
        rgb += [(0, i, 255) for i in range(256)]
        # Includes value 150 and both sides of saturation 45 and 52.
        rgb += [(peak, peak-delta, peak-delta) for peak in (149, 150, 151, 170, 255)
                for delta in (0, 29, 30, 31, 51, 52, 53)]
        rgb += [(0, 0, 0), (170, 170, 140), (255, 203, 255)]
        expected = np.array([quality._rgb_to_opencv_hsv(*pixel) for pixel in rgb])
        actual = quality._bgr_pixels_to_hsv(np.array(rgb, dtype=np.uint8)[:, ::-1])
        np.testing.assert_array_equal(actual, expected)

    def test_vectorized_annulus_matches_scalar_positions_in_row_order(self):
        # 50x1 hits .62 exactly; 1x25 hits .96 exactly. Both edges are included.
        for width, height in ((1, 1), (50, 1), (1, 25), (111, 108), (117, 114), (121, 119)):
            with self.subTest(width=width, height=height):
                expected = [(y, x) for y in range(height) for x in range(width)
                            if .62 <= math.sqrt(((x-(width-1)/2)/(width/2))**2
                                                + ((y-(height-1)/2)/(height/2))**2) <= .96]
                actual = np.argwhere(quality._quality_annulus_mask(width, height))
                self.assertEqual([tuple(point) for point in actual], expected)

    def test_annular_samples_clipping_fractional_roi_and_empty_regions(self):
        image=np.zeros((1280,720,3),np.uint8)
        image[250:290,50:100]=np.random.default_rng(14).integers(0,256,(40,50,3),dtype=np.uint8)
        rects=[dict(x=50,y=250,width=30,height=30),dict(x=50.5,y=250.5,width=20.1,height=18.1),
               dict(x=-2.5,y=-1.5,width=12,height=14),dict(x=715,y=1275,width=20,height=20),
               dict(x=800,y=1400,width=20,height=20),dict(x=10,y=10,width=0,height=10),
               dict(x=10,y=10,width=-1,height=10),dict(x=10,y=10,width=1,height=1),{}]
        expected=golden('quality', 'extract',image,rects=rects)
        for rect,pixels in zip(rects,expected):
            with self.subTest(rect=rect):
                self.assertEqual(quality.extract_relative_quality_pixels(image,rect),pixels)
        self.assertEqual(quality.extract_relative_quality_pixels(image,{}),[])
        # Non-contiguous view keeps BGR semantics; it is still 720x1280.
        reflected=image[:,::-1]
        rect=rects[0]
        self.assertEqual(quality.extract_relative_quality_pixels(reflected,rect),
                         golden('quality', 'extract',reflected,rects=[rect])[0])

    def test_stage_candidates_adjacency_duplicates_missing_roi_and_partial_gating(self):
        # Synthetic fixed-size canvas; ROI comes only from existing geometry.
        image=np.full((1280,720,3),20,np.uint8)
        circles=[dict(centerX=x,centerY=y,radius=56,score=30.)
                 for y in (450,617,784,951) for x in (119,274,430,585)]
        cards=grid.build_card_candidates(circles,self.base)
        colours=[(0,102,255),(187,0,255),(255,119,0),(9,255,0),(210,210,210)]
        for i,card in enumerate(cards):
            r=card['cardRect']
            image[r['y']:r['y']+r['height'],r['x']:r['x']+r['width']]=colours[i%5]
        supplied=[cards[1],cards[0],cards[0],*cards[2:]]
        supplied += [dict(cardId='top',completeness='partial_top'),dict(cardId='bottom',completeness='partial_bottom'),
                     dict(cardId='invalid',completeness='invalid'),dict(cardId='missing'),
                     dict(cardId='null-roi',completeness='complete',cardRect=None),
                     dict(cardId='missing-roi',completeness='complete')]
        result=self.cards_parity(image,supplied)
        self.assertEqual([r['card']['cardId'] for r in result],[c['cardId'] for c in supplied])
        self.assertEqual(result[1]['quality'],result[2]['quality'])
        self.assertEqual({r['quality']['quality'] for r in result[:len(cards)+1]},set(quality.QUALITY_HUES)|{'\u767d'})
        self.assertEqual(result[-1]['quality']['evidence'],'empty_relative_quality_region')
        self.assertEqual(self.cards_parity(image,[]),[])
        rect=quality.quality_rect_for_card(cards[0]); rect['x']=-1
        self.assertNotEqual(rect,cards[0]['cardRect'])

    def test_all_saved_geometry_on_restored_real_crops(self):
        count=0
        observed=set()
        for folder,name in (('star_full_tail','main-004'), ('star_boundary_rows','support-003')):
            saved=load_page(folder,name)
            crop_path=FIXTURES/folder/f'{name}-grid.png'
            crop=read_image(crop_path)
            image=np.zeros((1280,720,3),np.uint8)
            image[272:1044,40:680]=crop
            cards=[]
            profile={**self.base,**saved['geometry']['profile']}
            for i,old in enumerate(saved['geometry']['cards']):
                if 'cardRect' in old:
                    cards.append(copy.deepcopy(old))
                else:
                    # POSITION-only JSON omitted cardRect and radius: recover
                    # radius from saved disc, obtain ROI from Stage 1A, and keep
                    # saved completeness. No new image detection is claimed.
                    circle={**old['circle'],'radius':old['discRect']['width']/2}
                    adapted=geometry.candidate_from_circle(circle,profile,old['rowIndex'],old['columnIndex'],i+1)
                    adapted['completeness']=old['completeness']
                    cards.append(adapted)
            with self.subTest(real_crop=crop_path.name,saved_json=name):
                result=self.cards_parity(image,cards)
                observed.update(r['quality']['quality'] for r in result)
            count+=1
        self.assertEqual(count,2)
        print('Restored real crop qualities:',sorted(q for q in observed if q is not None))

    def test_invalid_image_and_incomplete_candidate_short_circuit(self):
        complete=dict(completeness='complete',cardRect=dict(x=50,y=300,width=100,height=100))
        for image in (None,np.zeros((1280,720),np.uint8),np.zeros((1280,720,4),np.uint8),
                      np.zeros((1280,720,3),np.float32),np.zeros((0,720,3),np.uint8)):
            with self.assertRaises(ValueError):
                quality.recognize_quality(image,complete)
        # JS checks completeness before it accesses either image or cardRect.
        self.assertEqual(quality.recognize_quality(None,dict(completeness='partial_top'))['evidence'],
                         'incomplete_card_quality_not_evaluated')


if __name__ == "__main__":
    unittest.main()
