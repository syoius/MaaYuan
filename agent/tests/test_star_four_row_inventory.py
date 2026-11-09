from pathlib import Path
from unittest import mock
import copy
import numpy as np
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))

from star_fixture_inputs import boundary_page
from star_fixture_inputs import replay_page
from star_fixture_inputs import synthetic_page
from utils.star_four_row_inventory import PositionInventory
from utils.star_four_row_inventory import _row_records
from utils.star_four_row_inventory import _validate_grid
from utils.star_four_row_inventory import classify_scan_transition
from utils.star_four_row_inventory import confirmed_no_move
from utils.star_four_row_inventory import frame_stability
from utils.star_four_row_inventory import layout
from utils.star_four_row_inventory import locate_terminal_rows
from utils.star_four_row_inventory import position_candidates
from utils.star_four_row_inventory import terminal_reason
from utils.star_geometry import candidate_from_circle


class SavedFailureTests(unittest.TestCase):
 def test_real_30px_phase_full_page_has_four_new_rows(self):
  previous,before=replay_page('star_full_tail', 'main-003');current,after=replay_page('star_full_tail', 'main-004')
  mapping=locate_terminal_rows(previous,current,before,after,673)
  self.assertEqual(mapping['row_offset'],4)
  self.assertEqual(len(mapping['anchors']),4)
  self.assertTrue(all(a['ncc']>=.97 for a in mapping['anchors']))
  self.assertTrue(all(abs(a['upward_shift_px']-636)<=1 for a in mapping['anchors']))

 def test_saved_full_page_append_keeps_same_attributes_as_independent_positions(self):
  previous,before=replay_page('star_full_tail', 'main-003');current,after=replay_page('star_full_tail', 'main-004')
  for p in (previous,current):
   for r in p['records']:r['resolved']={'name':'天同','level':1,'quality':'橙'}
  v=PositionInventory('main');v.accept_initial(previous);x=v.accept_moved(current,before,after,673)
  self.assertEqual((x['new_count'],x['duplicate_count'],len(v.items)),(16,0,32))
  self.assertEqual(len(v.positions),32);self.assertIsNone(terminal_reason(before,after,current['geometry']))
 def test_saved_full_tail_has_no_move_confirmation(self):
  page,image=replay_page('star_full_tail', 'main-004');decision=classify_scan_transition(page,page,image,image,673)
  self.assertTrue(confirmed_no_move(decision,decision,decision))
  self.assertFalse(confirmed_no_move(image,image,image))

class TransitionTests(unittest.TestCase):
 def test_normal_full_page_and_30px_phase_advance(self):
  for first in (403,433):
   with self.subTest(first=first):
    a,im=synthetic_page();b,after=synthetic_page(4,first)
    v=PositionInventory('main');v.accept_initial(a);x=v.accept_moved(b,im,after,673)
    self.assertEqual((x['new_count'],x['duplicate_count'],len(v.items)),(16,0,32))
 def test_full_four_row_overlap_does_not_append_sixteen(self):
  for offset in (1,2,3):
   with self.subTest(offset=offset):
    a,im=synthetic_page();b,after=synthetic_page(offset,433)
    v=PositionInventory('main');v.accept_initial(a);x=v.accept_moved(b,im,after,673)
    self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),((4-offset)*4,offset*4,16+offset*4))
 def test_partial_last_row_one_two_three_and_small_first_page(self):
  for size in (1,2,3):
   with self.subTest(size=size):
    a,im=synthetic_page();b,after=synthetic_page(2,433,(4,4,size))
    v=PositionInventory('main');v.accept_initial(a);x=v.accept_moved(b,im,after,673)
    self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(8,size,16+size));self.assertEqual(len(v.positions),16+size)
    small,_=synthetic_page(0,403,(4,size));v=PositionInventory('main');v.accept_initial(small);self.assertEqual(len(v.items),4+size)
 def test_invalid_missing_cell_or_entire_row_is_rejected_atomically(self):
  a,im=synthetic_page();b,after=synthetic_page(2,433,(4,4,3))
  for missing in ('cell','row'):
   bad=copy.deepcopy(b);bad['records']=[r for r in bad['records'] if not (r['row']==1 and (missing=='row' or r['column']==1))]
   v=PositionInventory('main');v.accept_initial(a)
   with self.assertRaises(ValueError):v.accept_moved(bad,im,after,673)
   self.assertEqual(len(v.items),16)
 def test_absent_or_ambiguous_pixel_evidence_never_appends(self):
  a,im=synthetic_page();b,after=synthetic_page(4,433)
  v=PositionInventory('main');v.accept_initial(a)
  with self.assertRaises(ValueError):v.accept_moved(b,im*0,after*0,673)
  self.assertEqual(len(v.items),16)
 def test_boundary_survives_vertical_fade_but_different_pixels_are_rejected(self):
  a,im=synthetic_page();b,after=synthetic_page(4,433)
  gain=np.linspace(.3,.9,1280)[:,None,None];faded=np.clip(after*gain,0,255).astype(np.uint8)
  v=PositionInventory('main');v.accept_initial(a);x=v.accept_moved(b,im,faded,673);self.assertEqual(x['new_count'],16)
  wrong=after.copy();wrong[272:380,40:680]=np.random.default_rng(999).integers(0,255,(108,640,3),dtype=np.uint8)
  v=PositionInventory('main');v.accept_initial(a)
  with self.assertRaises(ValueError):v.accept_moved(b,im,wrong,673)
  self.assertEqual(len(v.items),16)
 def test_required_fields_and_quality_specific_level_limits_fail_before_accept(self):
  page,_=synthetic_page()
  for quality,maximum in (('蓝',50),('绿',40),('白',30)):
   valid=copy.deepcopy(page);valid['records'][0]['resolved'].update(quality=quality,level=maximum)
   v=PositionInventory('main');v.accept_initial(valid);self.assertEqual(len(v.items),16)
   invalid=copy.deepcopy(valid);invalid['records'][0]['resolved']['level']=maximum+1
   v=PositionInventory('main')
   with self.assertRaises(ValueError):v.accept_initial(invalid)
   self.assertEqual(len(v.items),0)
  for field,value in (('name',''),('name',None),('quality','unknown'),('level',True),('level',None)):
   invalid=copy.deepcopy(page);invalid['records'][0]['resolved'][field]=value;v=PositionInventory('main')
   with self.assertRaises(ValueError):v.accept_initial(invalid)
   self.assertEqual(len(v.items),0)
 def test_no_move_rejects_animated_frames(self):
  _,im=synthetic_page();animated=np.roll(im,4,axis=0)
  self.assertFalse(frame_stability(im,animated)['stable']);self.assertFalse(confirmed_no_move(im,im,animated))

class OnePixelGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p,_=boundary_page('support-002');q,_=boundary_page('support-003')
        c=next(c for c in p['geometry']['cards'] if c['rowIndex']==3)
        normal=p['geometry']['cards'][0];top=q['geometry']['cards'][0]
        query={'profile':p['geometry']['profile'],'circles':[c['circle'],{**c['circle'],'centerY':c['circle']['centerY']+1},normal['circle'],top['circle']]}
        cls.cards=[candidate_from_circle(c,query['profile'],3,i,i+1) for i,c in enumerate(query['circles'])]
    def test_exact_one_pixel_name_padding_is_clipped_inside_original_bounds(self):
        c=self.cards[0];self.assertEqual(c['completeness'],'complete');self.assertEqual(c['nameRect'],{'x':67,'y':1011,'width':103,'height':33})
        self.assertEqual(c['originalNameRect']['height'],34);self.assertIn('name_bottom_padding_clipped_1px',c['evidence'])
    def test_two_pixels_or_top_missing_disc_stay_partial(self):
        self.assertEqual(self.cards[1]['completeness'],'partial_bottom');self.assertNotIn('originalNameRect',self.cards[1])
        self.assertEqual(self.cards[3]['completeness'],'partial_top')
    def test_python_layout_uses_the_same_clipped_field_bounds(self):
        p,_=boundary_page('support-002', repaired=True);self.assertTrue(layout(p['geometry'])['four_rows_valid'])
        bad=copy.deepcopy(p);bad['geometry']['cards'][-1]['nameRect']['height']+=1
        self.assertFalse(layout(bad['geometry'])['fields_valid'])

class BoundaryPositionTests(unittest.TestCase):
    def test_real_complete_rows_start_at_one_without_rewriting_source_rows(self):
        p,_=boundary_page('support-003');rows,_=_validate_grid(p)
        self.assertEqual([r[0]['row'] for r in rows],[1,2,3]);self.assertEqual(p['geometry']['cards'][0]['completeness'],'partial_top')
        self.assertEqual(p['geometry']['cards'][-1]['completeness'],'partial_bottom')
    def test_real_partial_to_partial_four_column_names_prove_only_position(self):
        p,a=boundary_page('support-002');q,b=boundary_page('support-003');d=position_candidates(p,q,a,b,660)
        self.assertEqual(len(d['candidates']),1);m=d['candidates'][0];self.assertEqual(m['row_offset'],4);self.assertAlmostEqual(m['anchor_shift_px'],638,delta=1)
        self.assertEqual({z['column'] for z in m['anchors']},{0,1,2,3})
        self.assertTrue(all(z['partial_row_anchor'] and not z['instances_from_anchor'] for z in m['anchors']))
    def test_original_unread_fire_row_blocks_atomic_merge_despite_unique_mapping(self):
        p,a=boundary_page('support-002');q,b=boundary_page('support-003');v=PositionInventory('support');v.accept_initial(p);before=copy.deepcopy(v.__dict__)
        with self.assertRaisesRegex(ValueError,'gap'):v.accept_moved(q,a,b,660)
        self.assertEqual(v.__dict__,before);self.assertEqual(len(v.items),12)
    def test_fixed_fire_row_is_read_once_then_following_nonzero_rows_are_contiguous(self):
        p,a=boundary_page('support-002', repaired=True);q,b=boundary_page('support-003');v=PositionInventory('support');v.accept_initial(p)
        result=v.accept_moved(q,a,b,660);self.assertEqual((result['new_count'],result['duplicate_count'],len(v.items)),(12,0,28))
        self.assertEqual(sorted(v.positions),[(i//4,i%4) for i in range(28)])
        fire=[i for i in v.items if i['physicalRow']==3];self.assertEqual(len(fire),4)
        restored = [r for r in p['records'] if r['row'] == 3]
        self.assertEqual([r['column'] for r in restored if r['fields']['level']['low_confidence']], [0, 3])
        self.assertTrue(all((i['name'],i['level'],i['quality'],i['sourceRow'])==('火星',1,'橙',3) for i in fire))
        self.assertEqual({i['sourceRow'] for i in v.items if i['sourceImageId']=='support-003'},{1,2,3})
        again=v.accept_moved(q,b,b,660);self.assertEqual((again['new_count'],again['duplicate_count'],len(v.items)),(0,12,28))
    def test_public_discs_cannot_replace_missing_partial_name_evidence(self):
        p,a=boundary_page('support-002');q,b=boundary_page('support-003');b=b.copy()
        for c in q['geometry']['cards']:
            if c['rowIndex']==0:
                r=c['nameRect'];b[r['y']:r['y']+r['height'],r['x']:r['x']+r['width']]=0
        self.assertEqual(position_candidates(p,q,a,b,660)['candidates'],[])
    def test_initial_partial_top_cannot_hide_an_unseen_leading_row(self):
        p,_=boundary_page('support-003')
        with self.assertRaisesRegex(ValueError,'unseen leading'):PositionInventory('support').accept_initial(p)
    def test_nonconsecutive_geometric_indices_are_still_rejected(self):
        p,_=boundary_page('support-003')
        for r in p['records']:
            if r['row']==2:r['row']=4
        with self.assertRaisesRegex(ValueError,'consecutive'):_row_records(p)
    def test_raw_row_one_normal_full_page_keeps_source_index_and_physical_mapping(self):
        p,a=synthetic_page();q,b=synthetic_page(4)
        for r in q['records']:r['row']+=1;r['card']['rowIndex']+=1
        v=PositionInventory('main');v.accept_initial(p);v.accept_moved(q,a,b,630)
        self.assertEqual(len(v.items),32);self.assertEqual(v.items[16]['sourceRow'],1);self.assertEqual(v.items[16]['physicalRow'],4)

def saved_pair(section):
    before, after = (replay_page('star_overlap_tail', f'{section}-{phase}')
                     for phase in ('before', 'after'))
    return (before[0], after[0]), (before[1], after[1])

def unit_prefix(predecessor,count):
 """Synthetic earlier pages establish inventory size; no historical full scan."""
 pages=[]
 for index in range(count-1):
  page=copy.deepcopy(predecessor);page['sourceImageId']=f'unit-prefix-{index}'
  for r in page['records']:
   r['sourceImageId']=page['sourceImageId'];r['occurrenceId']=f"{page['sourceImageId']}:r{r['row']}c{r['column']}"
   r['resolved']={'name':'天同' if page['section']=='main' else '天府','quality':'橙','level':1}
   r['fields']={field:{'confidence':1.,'max_raw_confidence':1.} for field in ('name','level')}
  pages.append(page)
 return pages+[predecessor]

class TailTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  (predecessor,tail),(cls.before,cls.after)=saved_pair('main')
  cls.pages=unit_prefix(predecessor,4)+[tail]
 def inventory(self,pages=None):
  result=PositionInventory('main')
  for page in (pages or self.pages)[:4]:result.accept_normal(page)
  return result
 def test_real_tail_keeps_order_and_purple_last_item(self):
  v=self.inventory();first=copy.deepcopy(v.items)
  result=v.merge_terminal(self.pages[4],self.before,self.after,669)
  self.assertEqual((result['duplicate_count'],result['new_count'],len(v.items)),(8,5,69));self.assertEqual(v.items[:64],first)
  self.assertEqual([(x['name'],x['level'],x['quality']) for x in v.items[-5:]],[('太阴',1,'橙'),('天机',1,'橙'),('天机',1,'橙'),('天梁',1,'橙'),('破军',1,'紫')])
  self.assertFalse(any('instanceId' in x or 'instance_id' in x for x in v.items))
 def test_normal_pages_append_without_tail_image_matching(self):
  v=PositionInventory('main')
  with mock.patch('utils.star_four_row_inventory._tail_anchor',side_effect=AssertionError('normal scan entered tail matching')):
   v.accept_normal(self.pages[0]);v.accept_normal(self.pages[1])
  self.assertEqual(len(v.items),32);self.assertEqual([x['captureOrder'] for x in v.items],list(range(1,33)))
  self.assertEqual([x['sourceImageId'] for x in v.items],[self.pages[0]['sourceImageId']]*16+[self.pages[1]['sourceImageId']]*16)
  self.assertFalse(any(x['fromTail'] for x in v.items))
 def test_support_tail_duplicates_four_adds_nine(self):
  predecessor,before=synthetic_page()
  tail,after=synthetic_page(3,433,(4,4,4,1))
  predecessor['section']=tail['section']='support'
  v=PositionInventory('support');v.accept_initial(predecessor)
  result=v.merge_terminal(tail,before,after,656)
  self.assertEqual((result['duplicate_count'],result['new_count'],len(v.items)),(4,9,25))
  self.assertEqual(len(v.positions),25)

 def test_terminal_rows_one_two_three_four_cells(self):
  for size in range(1,5):
   with self.subTest(size=size):
    tail=copy.deepcopy(self.pages[4]);last=tail['records'][-1]
    for col in range(1,size):
     cell=copy.deepcopy(last);cell['column']=col;cell['occurrenceId']+=f'-unit-col-{col}';cell['card']['columnIndex']=col;cell['card']['circle']['centerX']=119+col*156;tail['records'].append(cell)
    v=self.inventory();x=v.merge_terminal(tail,self.before,self.after,669)
    self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(8,4+size,68+size))
 def test_identical_name_level_quality_are_not_position_keys(self):
  pages=copy.deepcopy(self.pages)
  for page in pages:
   for record in page['records']:record['resolved']={'name':'天同','level':1,'quality':'橙'}
  v=self.inventory(pages);x=v.merge_terminal(pages[4],self.before,self.after,669)
  self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(8,5,69))
 def test_no_move_and_no_new_terminal_do_not_append(self):
  decision=classify_scan_transition(self.pages[4],self.pages[4],self.after,self.after,669)
  self.assertTrue(confirmed_no_move(decision,decision,decision));v=self.inventory();v.merge_terminal(self.pages[4],self.before,self.after,669)
  x=v.merge_terminal(self.pages[4],self.before,self.after,669)
  self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(13,0,69))
 def test_ambiguous_positions_stop_without_field_guessing(self):
  with self.assertRaisesRegex(ValueError,'ambiguous'):
   locate_terminal_rows(self.pages[3],self.pages[4],self.before*0,self.after*0,669)
 def test_conflicting_duplicate_preserves_accepted_inventory(self):
  v=self.inventory();tail=copy.deepcopy(self.pages[4]);tail['records'][0]['resolved']['level']=59
  with self.assertRaisesRegex(ValueError,'conflicting'):v.merge_terminal(tail,self.before,self.after,669)
  self.assertEqual(len(v.items),64)
 def test_missing_nonfinal_cell_stops_before_mutating_inventory(self):
  v=self.inventory();tail=copy.deepcopy(self.pages[4]);tail['records']=[r for r in tail['records'] if (r['row'],r['column'])!=(2,1)]
  with self.assertRaisesRegex(ValueError,'missing nonfinal'):v.merge_terminal(tail,self.before,self.after,669)
  self.assertEqual(len(v.items),64)


if __name__ == "__main__":
    unittest.main()
