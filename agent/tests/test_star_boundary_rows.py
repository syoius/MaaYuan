"""Real support-002/003 boundary regression; field readings are saved actual Context OCR."""
import copy,json,subprocess,sys,unittest
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.star_local_ocr_shadow import bundled_node,PACKAGED_RUNTIME
from utils.star_four_row_inventory import PositionInventory,position_candidates,classify_scan_transition,layout,_row_records,_validate_grid,PositionUncertain
import test_star_page_transitions as fixtures
FIXTURE=Path(__file__).parent/'fixtures/star_boundary_rows'

def real(name):
    p=json.loads((FIXTURE/(name+'.json')).read_text(encoding='utf-8'))
    im=np.zeros((1280,720,3),np.uint8)
    im[272:1044,40:680]=cv2.imdecode(np.frombuffer((FIXTURE/(name.replace('-fixed','')+'-grid.png')).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    return p,im

class OnePixelGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p,_=real('support-002');q,_=real('support-003')
        c=next(c for c in p['geometry']['cards'] if c['rowIndex']==3)
        normal=p['geometry']['cards'][0];top=q['geometry']['cards'][0]
        query={'profile':p['geometry']['profile'],'circles':[c['circle'],{**c['circle'],'centerY':c['circle']['centerY']+1},normal['circle'],top['circle']]}
        js="import fs from 'node:fs';import {candidateFromCircle} from "+json.dumps((PACKAGED_RUNTIME/'card-completeness.js').as_uri())+";const q=JSON.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(q.circles.map((c,i)=>candidateFromCircle(c,q.profile,3,i,i+1))));"
        cls.cards=json.loads(subprocess.check_output([str(bundled_node()),'--input-type=module','-e',js],input=json.dumps(query).encode('utf-8')))
    def test_exact_one_pixel_name_padding_is_clipped_inside_original_bounds(self):
        c=self.cards[0];self.assertEqual(c['completeness'],'complete');self.assertEqual(c['nameRect'],{'x':67,'y':1011,'width':103,'height':33})
        self.assertEqual(c['originalNameRect']['height'],34);self.assertIn('name_bottom_padding_clipped_1px',c['evidence'])
    def test_two_pixels_or_top_missing_disc_stay_partial(self):
        self.assertEqual(self.cards[1]['completeness'],'partial_bottom');self.assertNotIn('originalNameRect',self.cards[1])
        self.assertEqual(self.cards[3]['completeness'],'partial_top')
    def test_normal_complete_card_is_unchanged(self):
        p,_=real('support-002');old=p['geometry']['cards'][0];new=self.cards[2]
        self.assertEqual(new['completeness'],old['completeness'])
        for k in ('nameRect','discRect','levelRect'):self.assertEqual(new[k],old[k])
        self.assertNotIn('originalNameRect',new)
    def test_python_layout_uses_the_same_clipped_field_bounds(self):
        p,_=real('support-002-fixed');self.assertTrue(layout(p['geometry'])['four_rows_valid'])
        bad=copy.deepcopy(p);bad['geometry']['cards'][-1]['nameRect']['height']+=1
        self.assertFalse(layout(bad['geometry'])['fields_valid'])

class BoundaryPositionTests(unittest.TestCase):
    def test_real_complete_rows_start_at_one_without_rewriting_source_rows(self):
        p,_=real('support-003');rows,_=_validate_grid(p)
        self.assertEqual([r[0]['row'] for r in rows],[1,2,3]);self.assertEqual(p['geometry']['cards'][0]['completeness'],'partial_top')
        self.assertEqual(p['geometry']['cards'][-1]['completeness'],'partial_bottom')
    def test_real_partial_to_partial_four_column_names_prove_only_position(self):
        p,a=real('support-002');q,b=real('support-003');d=position_candidates(p,q,a,b,660)
        self.assertEqual(len(d['candidates']),1);m=d['candidates'][0];self.assertEqual(m['row_offset'],4);self.assertAlmostEqual(m['anchor_shift_px'],638,delta=1)
        self.assertEqual({z['column'] for z in m['anchors']},{0,1,2,3})
        self.assertTrue(all(z['partial_row_anchor'] and not z['instances_from_anchor'] for z in m['anchors']))
    def test_original_unread_fire_row_blocks_atomic_merge_despite_unique_mapping(self):
        p,a=real('support-002');q,b=real('support-003');v=PositionInventory('support');v.accept_initial(p);before=copy.deepcopy(v.__dict__)
        with self.assertRaisesRegex(ValueError,'gap'):v.accept_moved(q,a,b,660)
        self.assertEqual(v.__dict__,before);self.assertEqual(len(v.items),12)
    def test_fixed_fire_row_is_read_once_then_following_nonzero_rows_are_contiguous(self):
        p,a=real('support-002-fixed');q,b=real('support-003');v=PositionInventory('support');v.accept_initial(p)
        result=v.accept_moved(q,a,b,660);self.assertEqual((result['new_count'],result['duplicate_count'],len(v.items)),(12,0,28))
        self.assertEqual(sorted(v.positions),[(i//4,i%4) for i in range(28)])
        fire=[i for i in v.items if i['physicalRow']==3];self.assertEqual(len(fire),4)
        self.assertTrue(all((i['name'],i['level'],i['quality'],i['sourceRow'])==('火星',1,'橙',3) for i in fire))
        self.assertEqual({i['sourceRow'] for i in v.items if i['sourceImageId']=='support-003'},{1,2,3})
        again=v.accept_moved(q,b,b,660);self.assertEqual((again['new_count'],again['duplicate_count'],len(v.items)),(0,12,28))
    def test_saved_real_ocr_preserves_fallback_and_confidence_flags(self):
        p,_=real('support-002-fixed');fire=[r for r in p['records'] if r['row']==3]
        self.assertEqual(len(fire),4)
        for r in fire:
            self.assertGreaterEqual(r['fields']['name']['max_raw_confidence'],.95)
            self.assertEqual(r['fields']['level']['fallback_calls'],2)
            self.assertEqual(r['fields']['level']['resolution']['reasons'],[])
        self.assertEqual([r['column'] for r in fire if r['fields']['level']['low_confidence']],[0,3])
    def test_public_discs_cannot_replace_missing_partial_name_evidence(self):
        p,a=real('support-002');q,b=real('support-003');b=b.copy()
        for c in q['geometry']['cards']:
            if c['rowIndex']==0:
                r=c['nameRect'];b[r['y']:r['y']+r['height'],r['x']:r['x']+r['width']]=0
        self.assertEqual(position_candidates(p,q,a,b,660)['candidates'],[])
    def test_initial_partial_top_cannot_hide_an_unseen_leading_row(self):
        p,_=real('support-003')
        with self.assertRaisesRegex(ValueError,'unseen leading'):PositionInventory('support').accept_initial(p)
    def test_nonconsecutive_geometric_indices_are_still_rejected(self):
        p,_=real('support-003')
        for r in p['records']:
            if r['row']==2:r['row']=4
        with self.assertRaisesRegex(ValueError,'consecutive'):_row_records(p)
    def test_one_two_three_card_tail_and_repeated_attributes_remain_independent(self):
        for n in (1,2,3):
            with self.subTest(n=n):
                p,a=fixtures.synthetic_page();q,b=fixtures.synthetic_page(1,lengths=(4,4,4,n));v=PositionInventory('main');v.accept_initial(p);v.accept_moved(q,a,b,630)
                self.assertEqual(len(v.items),16+n);self.assertEqual(len(v.positions),16+n)
    def test_raw_row_one_normal_full_page_keeps_source_index_and_physical_mapping(self):
        p,a=fixtures.synthetic_page();q,b=fixtures.synthetic_page(4)
        for r in q['records']:r['row']+=1;r['card']['rowIndex']+=1
        v=PositionInventory('main');v.accept_initial(p);v.accept_moved(q,a,b,630)
        self.assertEqual(len(v.items),32);self.assertEqual(v.items[16]['sourceRow'],1);self.assertEqual(v.items[16]['physicalRow'],4)

if __name__=='__main__':unittest.main()
