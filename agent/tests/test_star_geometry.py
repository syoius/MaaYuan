from pathlib import Path
import copy
import cv2
import numpy as np
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))

from star_fixture_inputs import geometry_pages
from star_fixture_inputs import load_page
from star_ocr_golden import golden
from utils import star_geometry as geometry
from utils import star_geometry as native
from utils import star_grid_geometry as grid
from utils.star_local_ocr_shadow import read_image


class GeometryParityTests(unittest.TestCase):
    def image_parity(self, image):
        original = image.copy()
        expected = golden('geometry', 'image', image)
        profile = native.create_screenshot_profile(image)
        actual = dict(detected=native.detect_viewport(image), profile=profile,
                      refined=native.refine_star_content_bounds(profile, None, 'maayuan_mumu'),
                      layout=native.layout_spec(profile['profileId']))
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(image, original)
        return actual

    def cards_parity(self, items):
        original = copy.deepcopy(items)
        actual = [dict(boxes=native.boxes_for_circle(v['circle']),
                       classification=native.classify_completeness(v['circle'], v['profile']),
                       candidate=native.candidate_from_circle(v['circle'], v['profile'], v['row'], v['column'], v['index']),
                       inside=native.is_inside(native.boxes_for_circle(v['circle'])['discRect'], v['profile']['contentBounds']))
                  for v in items]
        self.assertEqual(actual, golden('geometry', 'cards', items=items))
        self.assertEqual(items, original)
        return actual

    def test_profile_families_ratio_edges_and_half_rounding(self):
        for width, height in [(38,100),(39,100),(49,100),(50,100),(56,100),
                              (57,100),(77,100),(78,100),(90,100),(60,125),(1,1)]:
            with self.subTest(size=(width,height)):
                self.image_parity(np.full((height,width,3), 120, np.uint8))
        ids = ['phone_portrait_v1','phone_9_16_v1','tablet_portrait_v1','unknown','']
        self.assertEqual([native.layout_spec(i) for i in ids], golden('geometry', 'layouts', ids=ids))

    def test_synthetic_black_bars_minimum_depth_limit_and_rgb_thresholds(self):
        for depth in (0,7,8,18,22):
            image = np.full((100,80,3), 120, np.uint8)
            if depth:
                image[:depth] = image[-depth:] = 0
                image[:,:depth] = image[:,-depth:] = 0
            with self.subTest(depth=depth):
                self.image_parity(image)
        # Mean/deviation exactly 10/8 must be accepted; RGB weighting must not
        # silently become OpenCV uint8 gray or treat BGR as RGB.
        for colour in ((10,10,10),(11,11,11),(50,0,0),(0,0,50)):
            image = np.full((100,80,3), 120, np.uint8)
            image[:8] = colour
            self.image_parity(image)
        image = np.full((100,80,3), 120, np.uint8)
        image[:8,:40] = 2
        image[:8,40:] = 18
        self.assertEqual(self.image_parity(image)['detected']['viewport']['y'], 8)
        self.image_parity(np.zeros((100,80,3), np.uint8))
        # Non-contiguous BGR views remain valid MaaYuan image inputs.
        self.image_parity(image[:, ::-1])

    def test_content_bounds_tab_gates_layout_gates_and_no_mutation(self):
        base = native.create_screenshot_profile(np.full((1280,720,3), 120, np.uint8))
        cases = []
        tab = dict(x=20., y=200., width=120., height=80.)
        for hint in (None, 'maayuan_mumu', 'other'):
            for selected in (None, tab, {**tab,'width':100}, {**tab,'height':0},
                             {**tab,'x':-1}, {**tab,'y':80}, {**tab,'height':120},
                             {**tab,'width':720}, {**tab,'y':200.25}):
                cases.append((base, selected, hint))
        for changes in (dict(profileId='tablet_portrait_v1'), dict(imageWidth=721),
                        dict(imageHeight=1281),
                        dict(viewport={**base['viewport'],'x':1}),
                        dict(contentBounds={**base['contentBounds'],'height':1})):
            cases.append(({**base, **changes}, tab, 'maayuan_mumu'))
        for profile, selected, hint in cases:
            original = copy.deepcopy(profile)
            with self.subTest(tab=selected, hint=hint, profile=profile):
                self.assertEqual(native.refine_star_content_bounds(profile, selected, hint),
                                 golden('geometry', 'refine', profile=profile, tab=selected, hint=hint))
                self.assertEqual(profile, original)

    def test_committed_real_grid_crops_and_saved_circle_candidates(self):
        fixture = ROOT / 'agent/tests/fixtures'
        states = set()
        for sample, page in geometry_pages():
            g = page['geometry']
            # Older POSITION-only fixtures omit radius and screenshot metadata.
            # Recover radius from their saved disc (no guessed pixels); the full
            # viewport below is the same synthetic canvas used by their tests.
            profile = dict(imageWidth=720, imageHeight=1280,
                           viewport=dict(x=0,y=0,width=720,height=1280), **g['profile']) if 'viewport' not in g['profile'] else g['profile']
            items = []
            for i, c in enumerate(g['cards']):
                circle = dict(c['circle'])
                if 'radius' not in circle:
                    self.assertEqual(c['discRect']['width'], c['discRect']['height'])
                    circle['radius'] = c['discRect']['width'] / 2
                    boxes = native.boxes_for_circle(circle)
                    for key in ('discRect','nameRect','levelRect'):
                        self.assertEqual(boxes[key], c[key])
                items.append(dict(circle=circle, profile=profile, row=c['rowIndex'],
                                  column=c['columnIndex'], index=i+1))
            with self.subTest(saved_page=sample):
                states.update(v['classification']['completeness'] for v in self.cards_parity(items))
        self.assertTrue({'complete','partial_top','partial_bottom'} <= states)
        # Real crop alone and a synthetic restored canvas are explicitly distinct.
        crop = read_image(fixture / 'star_boundary_rows/support-002-grid.png')
        self.image_parity(crop)
        canvas = np.zeros((1280,720,3), np.uint8)
        canvas[272:1044,40:680] = crop
        self.image_parity(canvas)

    def test_one_pixel_guard_each_precondition_and_two_pixel_rejection(self):
        page = load_page('star_boundary_rows', 'support-002')['original']
        profile = page['geometry']['profile']
        circle = next(c['circle'] for c in page['geometry']['cards'] if c['rowIndex'] == 3)
        def item(c=circle, p=profile):
            return dict(circle=c, profile=p, row=3, column=0, index=1)
        valid = self.cards_parity([item()])[0]['candidate']
        self.assertEqual(valid['completeness'], 'complete')
        self.assertEqual(valid['originalNameRect']['height'] - valid['nameRect']['height'], 1)
        self.assertEqual(valid['evidence'], ['disc_name_level_inside_content_bounds','name_bottom_padding_clipped_1px'])
        cases = [item({**circle, 'centerY':circle['centerY']+1}),
                 item(p={**profile,'imageWidth':721}), item(p={**profile,'imageHeight':1281}),
                 item(p={**profile,'contentBounds':{**profile['contentBounds'],'y':273,'height':771}}),
                 item(p={**profile,'contentBounds':{**profile['contentBounds'],'height':771}}),
                 item(p={**profile,'contentBounds':{**profile['contentBounds'],'x':70,'width':650}})]
        # The name still overruns by 1px but disc/level is outside horizontally.
        boxes = native.boxes_for_circle(circle)
        disc_left = boxes['discRect']['x'] + .25
        cases.append(item(p={**profile,'contentBounds':{**profile['contentBounds'],'x':disc_left,'width':720-disc_left}}))
        level_right = boxes['levelRect']['x'] + boxes['levelRect']['width'] - .25
        cases.append(item(p={**profile,'contentBounds':{**profile['contentBounds'],'x':0,'width':level_right}}))
        for result in self.cards_parity(cases):
            self.assertNotIn('originalNameRect', result['candidate'])
            self.assertNotIn('name_bottom_padding_clipped_1px', result['candidate']['evidence'])
        self.assertEqual(self.cards_parity(cases[:1])[0]['classification']['completeness'], 'partial_bottom')

    def test_four_states_fractional_negative_truncation_and_bounds_intersection(self):
        profile = dict(profileId='test', imageWidth=720, imageHeight=1280,
                       viewport=dict(x=0,y=0,width=720,height=1280),
                       contentBounds=dict(x=0,y=272,width=720,height=772), evidence=[])
        circles = [dict(centerX=120,centerY=y,radius=56) for y in (400,300,1000)]
        circles += [dict(centerX=-.5,centerY=400,radius=56)]
        rng = random.Random(20261108)
        circles += [dict(centerX=rng.uniform(-60,780),centerY=rng.uniform(220,1100),
                         radius=rng.uniform(1,90)) for _ in range(80)]
        items = [dict(circle=c,profile=profile,row=i//4,column=i%4,index=i+1) for i,c in enumerate(circles)]
        results = self.cards_parity(items)
        self.assertEqual({v['classification']['completeness'] for v in results},
                         {'complete','partial_top','partial_bottom','invalid'})
        narrowed = {**profile,'viewport':dict(x=80,y=350,width=500,height=600)}
        self.cards_parity([{**v,'profile':narrowed} for v in items[:4]])
        # Level can be clipped even while the disc is just inside the top edge.
        c = dict(centerX=120,centerY=328,radius=56)
        self.assertTrue(native.is_inside(native.boxes_for_circle(c)['discRect'], profile['contentBounds']))
        self.assertEqual(self.cards_parity([dict(circle=c,profile=profile,row=0,column=0,index=1000)])[0]['classification']['completeness'], 'partial_top')


    def test_native_input_contract_rejects_unsupported_images(self):
        for image in (None, np.zeros((2,2),np.uint8), np.zeros((2,2,4),np.uint8),
                      np.zeros((2,2,3),np.float32), np.zeros((0,2,3),np.uint8)):
            with self.assertRaises(ValueError):
                native.detect_viewport(image)

class GridParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = geometry.refine_star_content_bounds(
            geometry.create_screenshot_profile(np.full((1280,720,3),120,np.uint8)),
            None, 'maayuan_mumu')

    @staticmethod
    def normalized_rows(rows):
        # Preserve Map insertion order; only the test JSON adapter uses pairs.
        return [dict(y=r['y'],circles=[[col,c] for col,c in r['circles'].items()]) for r in rows]

    def group_parity(self, proposals, profile=None):
        profile = self.base if profile is None else profile
        original = copy.deepcopy((proposals,profile))
        actual = dict(rows=self.normalized_rows(grid.group_grid_rows(proposals,profile)),
                      cards=grid.build_card_candidates(proposals,profile))
        self.assertEqual(actual,golden('grid', 'group',proposals=proposals,profile=profile))
        self.assertEqual((proposals,profile),original)
        return actual

    def image_parity(self, image, profile=None):
        original = image.copy()
        expected = golden('grid', 'image', image, profile=profile)
        base = geometry.create_screenshot_profile(image) if profile is None else profile
        proposals = grid.find_circle_proposals(image,base)
        refined = geometry.refine_star_content_bounds(base,None,'maayuan_mumu')
        actual = dict(profile=refined,proposals=proposals,
                      rows=self.normalized_rows(grid.group_grid_rows(proposals,refined)),
                      cards=grid.build_card_candidates(proposals,refined))
        self.assertEqual(actual,expected)
        np.testing.assert_array_equal(image,original)
        return actual

    @staticmethod
    def synthetic_rows(ys=(450,617,784,951), sizes=(4,4,4,4)):
        return [dict(centerX=x,centerY=y,radius=56,score=30.,source='synthetic',tag=f'{row}:{col}')
                for row,(y,size) in enumerate(zip(ys,sizes))
                for col,x in enumerate((119,274,430,585)[:size])]

    def test_edge_map_rgb_rounding_saturation_and_degenerate_sizes(self):
        rng = np.random.default_rng(20261108)
        images = [rng.integers(0,256,(19,23,3),dtype=np.uint8),
                  np.zeros((1,1,3),np.uint8), np.full((2,5,3),255,np.uint8),
                  np.array([[[250,0,0],[0,250,0],[0,0,250]],
                            [[0,0,0],[255,255,255],[0,0,0]],
                            [[255,255,255],[0,0,0],[255,255,255]]],dtype=np.uint8)]
        images.append(images[0][:,::-1])
        for image in images:
            with self.subTest(shape=image.shape):
                actual = grid.edge_map(image)
                self.assertEqual(actual.dtype,np.uint8)
                self.assertEqual(actual.ndim,1)
                self.assertEqual(actual.tolist(),golden('grid', 'edges', image))

    def test_ring_score_sampling_rounding_outside_edges_and_short_buffer(self):
        rng = np.random.default_rng(13)
        edges = rng.integers(0,256,31*29,dtype=np.uint8)
        circles = [dict(centerX=x,centerY=y,radius=r)
                   for x,y,r in ((15,14,8),(0,0,3),(30,28,9),(-20,-20,2),
                                 (.5,.5,0),(15.5,14.5,4.5),(1,1,-2))]
        for data,width,height in ((edges,31,29),(edges[:40],31,29),(np.zeros(1,np.uint8),1,1)):
            self.assertEqual([grid.ring_score(data,width,height,c['centerX'],c['centerY'],c['radius']) for c in circles],
                             golden('grid', 'ring',edges=data.tolist(),width=width,height=height,circles=circles))

    def test_batched_ring_scores_keep_scalar_rounding_and_viewport_edges(self):
        edges = np.random.default_rng(27).integers(0, 256, 131*129, dtype=np.uint8)
        ys = np.arange(-8, 137, 3)
        for x, radius in ((0, 6), (119, 53), (65, 56), (130, 20)):
            scores = grid._ring_scores(edges, 131, 129, x, ys, radius)
            self.assertEqual(scores.tolist(),
                             [grid.ring_score(edges, 131, 129, x, int(y), radius) for y in ys])

    def test_all_saved_geometry_json_with_explicit_metadata_adapters(self):
        count = 0
        for sample, page in geometry_pages():
            saved = page['geometry']
            # Synthetic 720x1280 canvas metadata is used only where omitted.
            profile = {**self.base,**saved['profile']}
            proposals = []
            for card in saved['cards']:
                circle = dict(card['circle'])
                if 'radius' not in circle:
                    self.assertEqual(card['discRect']['width'],card['discRect']['height'])
                    circle['radius'] = card['discRect']['width']/2
                # Reduced POSITION fixtures did not retain scores. 30 is an
                # explicit synthetic tie score, not a recovered detector score.
                circle.setdefault('score',30.)
                proposals.append(circle)
            with self.subTest(saved_json=sample):
                self.group_parity(proposals,profile)
            count += 1
        self.assertEqual(count, 4)

    def test_stable_ties_duplicates_nearest_column_and_input_permutations(self):
        proposals = self.synthetic_rows()
        proposals.insert(0,{**proposals[0],'tag':'first_tie'})
        proposals.append({**proposals[1],'score':29.,'tag':'lower_duplicate'})
        result = self.group_parity(proposals)
        self.assertEqual(result['cards'][0]['circle']['tag'],'first_tie')
        order = [(c['rowIndex'],c['columnIndex']) for c in result['cards']]
        self.assertEqual(order,[(r,c) for r in range(4) for c in range(4)])
        shuffled = list(proposals)
        random.Random(41).shuffle(shuffled)
        self.group_parity(shuffled)
        columns = [720*v for v in geometry.layout_spec(self.base['profileId'])['columnCenters']]
        probes = [dict(centerX=x,centerY=450,radius=56,score=31.,tag=str(i))
                  for i,x in enumerate(((columns[0]+columns[1])/2,columns[2],-100,900))]
        self.group_parity(probes)

    def test_cadence_singleton_tail_radius_filter_and_score_thresholds(self):
        cases = [self.synthetic_rows(sizes=(4,4,4,n)) for n in (1,2,3)]
        cases += [self.synthetic_rows(ys=(283,450,617,784,951),sizes=(1,4,4,4,1))]
        base = self.synthetic_rows()
        # Off-cadence row, wrong-radius row and evidence threshold boundaries.
        cases.append(base+[dict(centerX=x,centerY=530,radius=20,score=50.) for x in (119,274)])
        for score in (18.599999999999998,18.6,18.600000000000005):
            cases.append(base+[dict(centerX=x,centerY=1100,radius=56,score=score) for x in (119,274)])
        cases.append([{**p,'radius':20} for p in base[:8]])  # fallback to populated rows
        for proposals in cases:
            with self.subTest(proposals=proposals):
                self.group_parity(proposals)
        # Deterministic small numeric boundary set, not a performance experiment.
        rng = random.Random(18)
        proposals = base+[dict(centerX=rng.uniform(-50,770),centerY=rng.uniform(200,1120),
                              radius=rng.choice((20,53,56,65)),score=rng.choice((18.6,30.,40.))) for _ in range(24)]
        self.group_parity(proposals)

    def test_candidate_roi_completeness_and_one_pixel_fix_are_reused(self):
        one = self.group_parity(self.synthetic_rows())['cards']
        clipped = [c for c in one if 'originalNameRect' in c]
        self.assertEqual(len(clipped),4)
        self.assertTrue(all(c['completeness']=='complete' and c['nameRect']['height']==33
                            and 'name_bottom_padding_clipped_1px' in c['evidence'] for c in clipped))
        two = self.group_parity(self.synthetic_rows(ys=(451,618,785,952)))['cards']
        self.assertTrue(all(c['completeness']=='partial_bottom' and 'originalNameRect' not in c for c in two[-4:]))
        profile = {**self.base,'contentBounds':dict(x=0,y=272,width=400,height=772)}
        # Keep all six rows on cadence; an off-cadence last row is correctly
        # filtered BEFORE completeness classification by both implementations.
        states = self.group_parity(self.synthetic_rows(
            ys=(280,447,614,781,948,1115),sizes=(4,)*6),profile)['cards']
        self.assertEqual({c['completeness'] for c in states},{'complete','partial_top','partial_bottom','invalid'})

    def test_empty_sparse_outside_candidates_and_bad_image_contract(self):
        for proposals in ([], self.synthetic_rows()[:1],
                          [dict(centerX=-999,centerY=400,radius=56,score=30.)]):
            self.assertEqual(self.group_parity(proposals)['cards'],[])
        for image in (None,np.zeros((0,2,3),np.uint8),np.zeros((2,2),np.uint8),
                      np.zeros((2,2,4),np.uint8),np.zeros((2,2,3),np.float32)):
            with self.assertRaises(ValueError):
                grid.edge_map(image)
            with self.assertRaises(ValueError):
                grid.find_circle_proposals(image,self.base)
        self.assertEqual(self.image_parity(np.zeros((1,1,3),np.uint8))['proposals'],[])

    def test_synthetic_drawn_rings_and_horizontal_search_profiles(self):
        image = np.full((320,180,3),70,np.uint8)
        for x in (30,68,107,146):
            for y in (110,150,190,230):
                cv2.circle(image,(x,y),15,(230,230,230),2)
        result = self.image_parity(image)
        self.assertGreaterEqual(len(result['cards']),8)
        # Small synthetic tablet/fallback cases exercise x-search without
        # claiming real-device support for either profile family.
        small = np.full((96,64,3),70,np.uint8)
        cv2.circle(small,(25,40),12,(250,250,250),2)
        for profile_id in ('tablet_portrait_v1','unknown_portrait_fallback'):
            profile = {**geometry.create_screenshot_profile(small),'profileId':profile_id}
            self.image_parity(small,profile)


if __name__ == "__main__":
    unittest.main()
