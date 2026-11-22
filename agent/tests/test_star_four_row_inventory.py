from pathlib import Path
from unittest import mock
import copy
import numpy as np
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))

from star_fixture_inputs import synthetic_page
from utils.star_four_row_inventory import PositionInventory
from utils.star_four_row_inventory import _text_anchor
from utils.star_four_row_inventory import classify_scan_transition
from utils.star_four_row_inventory import confirmed_no_move
from utils.star_four_row_inventory import frame_stability
from utils.star_four_row_inventory import layout
from utils.star_four_row_inventory import initial_forward_plan
from utils.star_geometry import candidate_from_circle


def merge_boundary(inventory, page, before, after):
 plan=initial_forward_plan(inventory.last_page['geometry'])
 decision=classify_scan_transition(inventory.last_page,page,before,after,plan)
 if decision['basis']!='boundary':
  raise AssertionError('expected boundary evidence')
 return inventory.merge_terminal(page,decision['row_offset'])


class TransitionTests(unittest.TestCase):
 def test_offscreen_name_anchor_is_rejected_before_color_conversion(self):
  image=np.zeros((1280,720,3),dtype=np.uint8)
  record={'column':0,'card':{'nameRect':{'x':70,'y':914,'width':97,'height':32}}}
  with mock.patch('utils.star_four_row_inventory.cv2.cvtColor',side_effect=AssertionError('empty anchor converted')):
   self.assertIsNone(_text_anchor(image,image,record,681.5))
 def test_normal_full_page_and_30px_phase_advance(self):
  for first in (403,433):
   with self.subTest(first=first):
    a,im=synthetic_page();b,after=synthetic_page(4,first)
    v=PositionInventory('main');v.accept_initial(a);x=v.accept_moved(b,4)
    self.assertEqual((x['new_count'],x['duplicate_count'],len(v.items)),(16,0,32))
 def test_full_four_row_overlap_does_not_append_sixteen(self):
  for offset in (1,2,3):
   with self.subTest(offset=offset):
    a,im=synthetic_page();b,after=synthetic_page(offset,433)
    v=PositionInventory('main');v.accept_initial(a);x=merge_boundary(v,b,im,after)
    self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),((4-offset)*4,offset*4,16+offset*4))
 def test_partial_last_row_one_two_three_and_small_first_page(self):
  for size in (1,2,3):
   with self.subTest(size=size):
    a,im=synthetic_page();b,after=synthetic_page(2,433,(4,4,size))
    v=PositionInventory('main');v.accept_initial(a);x=merge_boundary(v,b,im,after)
    self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(8,size,16+size));self.assertEqual(len(v.positions),16+size)
    small,_=synthetic_page(0,403,(4,size));v=PositionInventory('main');v.accept_initial(small);self.assertEqual(len(v.items),4+size)
 def test_invalid_missing_cell_or_entire_row_is_rejected_atomically(self):
  a,im=synthetic_page();b,after=synthetic_page(2,433,(4,4,3))
  for missing in ('cell','row'):
   bad=copy.deepcopy(b);bad['records']=[r for r in bad['records'] if not (r['row']==1 and (missing=='row' or r['column']==1))]
   v=PositionInventory('main');v.accept_initial(a)
   with self.assertRaises(ValueError):merge_boundary(v,bad,im,after)
   self.assertEqual(len(v.items),16)
 def test_boundary_overlap_survives_vertical_fade(self):
  a,im=synthetic_page();b,after=synthetic_page(2,433,(4,4,3))
  gain=np.linspace(.3,.9,1280)[:,None,None];faded=np.clip(after*gain,0,255).astype(np.uint8)
  v=PositionInventory('main');v.accept_initial(a);x=merge_boundary(v,b,im,faded)
  self.assertEqual((x['duplicate_count'],x['new_count']),(8,3))
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

 def test_raw_row_one_normal_full_page_keeps_source_index_and_physical_mapping(self):
     p,a=synthetic_page();q,b=synthetic_page(4)
     for r in q['records']:r['row']+=1;r['card']['rowIndex']+=1
     v=PositionInventory('main');v.accept_initial(p);v.accept_moved(q,4)
     self.assertEqual(len(v.items),32);self.assertEqual(v.items[16]['sourceRow'],1);self.assertEqual(v.items[16]['physicalRow'],4)

class OnePixelGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profile = {'imageWidth':720, 'imageHeight':1280,
                   'viewport':{'x':0,'y':0,'width':720,'height':1280},
                   'contentBounds':{'x':0,'y':272,'width':720,'height':772}}
        circles = [dict(centerX=119,centerY=y,radius=56) for y in (951,952,450,300)]
        cls.cards = [candidate_from_circle(c,profile,3,i,i+1) for i,c in enumerate(circles)]
    def test_exact_one_pixel_name_padding_is_clipped_inside_original_bounds(self):
        c=self.cards[0];self.assertEqual(c['completeness'],'complete');self.assertEqual(c['nameRect'],{'x':67,'y':1011,'width':103,'height':33})
        self.assertEqual(c['originalNameRect']['height'],34);self.assertIn('name_bottom_padding_clipped_1px',c['evidence'])
    def test_two_pixels_or_top_missing_disc_stay_partial(self):
        self.assertEqual(self.cards[1]['completeness'],'partial_bottom');self.assertNotIn('originalNameRect',self.cards[1])
        self.assertEqual(self.cards[3]['completeness'],'partial_top')
    def test_python_layout_uses_the_same_clipped_field_bounds(self):
        p,_=synthetic_page(first=450)
        for record in p['records']:
            card=record['card'];r=card['circle']
            card.update(candidate_from_circle({**r,'radius':56},p['geometry']['profile'],record['row'],record['column'],1))
        self.assertTrue(layout(p['geometry'])['four_rows_valid'])
        bad=copy.deepcopy(p);bad['geometry']['cards'][-1]['nameRect']['height']+=1
        self.assertFalse(layout(bad['geometry'])['fields_valid'])


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
  predecessor,cls.before=synthetic_page()
  tail,cls.after=synthetic_page(2,433,(4,4,4,1))
  cls.pages=unit_prefix(predecessor,4)+[tail]
 def inventory(self,pages=None):
  result=PositionInventory('main')
  for page in (pages or self.pages)[:4]:result.accept_normal(page)
  return result
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
  result=merge_boundary(v,tail,before,after)
  self.assertEqual((result['duplicate_count'],result['new_count'],len(v.items)),(4,9,25))
  self.assertEqual(len(v.positions),25)

 def test_terminal_rows_one_two_three_four_cells(self):
  for size in range(1,5):
   with self.subTest(size=size):
    tail,after=synthetic_page(2,433,(4,4,4,size))
    v=self.inventory();x=merge_boundary(v,tail,self.before,after)
    self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(8,4+size,68+size))
 def test_identical_name_level_quality_are_not_position_keys(self):
  pages=copy.deepcopy(self.pages)
  for page in pages:
   for record in page['records']:record['resolved']={'name':'天同','level':1,'quality':'橙'}
  v=self.inventory(pages);x=merge_boundary(v,pages[4],self.before,self.after)
  self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(8,5,69))
 def test_no_move_and_no_new_terminal_do_not_append(self):
  page=self.pages[4];plan=initial_forward_plan(page['geometry'])
  decision=classify_scan_transition(page,page,self.after,self.after,plan)
  self.assertTrue(confirmed_no_move(decision,decision,decision));v=self.inventory();merge_boundary(v,page,self.before,self.after)
  x=v.merge_terminal(page,decision['row_offset'])
  self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(13,0,69))
 def test_partial_boundary_requires_visual_evidence(self):
  v=self.inventory()
  with self.assertRaisesRegex(ValueError,'partial boundary position unconfirmed'):
   merge_boundary(v,self.pages[4],self.before*0,self.after*0)
  self.assertEqual(len(v.items),64)
 def test_conflicting_duplicate_preserves_accepted_inventory(self):
  v=self.inventory();tail=copy.deepcopy(self.pages[4]);tail['records'][0]['resolved']['level']=59
  with self.assertRaisesRegex(ValueError,'conflicting'):v.merge_terminal(tail,2)
  self.assertEqual(len(v.items),64)
 def test_missing_nonfinal_cell_stops_before_mutating_inventory(self):
  v=self.inventory();tail=copy.deepcopy(self.pages[4]);tail['records']=[r for r in tail['records'] if (r['row'],r['column'])!=(2,1)]
  with self.assertRaisesRegex(ValueError,'missing nonfinal'):v.merge_terminal(tail,2)
  self.assertEqual(len(v.items),64)


if __name__ == "__main__":
    unittest.main()
