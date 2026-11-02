"""Focused saved-real-image tail regression; synthetic extensions are unit fixtures only."""
import sys,json,copy,unittest
import cv2,numpy as np
from unittest import mock
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.star_four_row_inventory import PositionInventory,confirmed_no_move,locate_terminal_rows
from utils.star_backpack_navigation import prepare_star_backpack_entry
FIXTURE=Path(__file__).parent/'fixtures/star_overlap_tail'

def saved_pair(section):
 pages=[];images=[]
 for phase in ('before','after'):
  stem=f'{section}-{phase}'
  pages.append(json.loads((FIXTURE/f'{stem}.json').read_text(encoding='utf-8')))
  image=np.zeros((1280,720,3),np.uint8)
  image[272:1044,40:680]=cv2.imread(str(FIXTURE/f'{stem}-grid.png'))
  images.append(image)
 return pages,images

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
 def test_saved_full_support_tail_duplicates_four_adds_nine(self):
  (predecessor,tail),(before,after)=saved_pair('support')
  v=PositionInventory('support')
  for page in unit_prefix(predecessor,6):v.accept_normal(page)
  x=v.merge_terminal(tail,before,after,656)
  self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(4,9,105))
  self.assertEqual((v.items[-1]['name'],v.items[-1]['level']),('红鸾',1))
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
  self.assertTrue(confirmed_no_move(self.after,self.after,self.after));v=self.inventory();v.merge_terminal(self.pages[4],self.before,self.after,669)
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

class Detail:
 hit=True
 class status:succeeded=True

class NavigationTests(unittest.TestCase):
 def test_three_starting_routes_share_entry_without_reverse_swipes(self):
  for initial in ('home','observatory','backpack'):
   class C:
    class tasker:stopping=False
    def __init__(self):self.page=initial;self.nodes=[]
    def run_recognition(self,node,image):
     result=Detail();result.hit=(node=='星石本地OCR-已在背包' and self.page=='backpack') or (node=='已在界面-观星' and self.page=='observatory') or (node=='已在界面-左半屏' and self.page=='home');return result
    def run_task(self,node,override):
     self.nodes.append(node)
     if node=='星石本地OCR-点击背包入口':self.page='backpack'
     elif node!='星石本地OCR-已在背包':self.page='observatory'
     return Detail()
   import numpy as np
   c=C();records=[];prepare_star_backpack_entry(c,lambda _:np.full((1280,720,3),200,dtype=np.uint8),records.append)
   expected={'home':['进入界面-观星','星石本地OCR-点击背包入口'],'observatory':['星石本地OCR-点击背包入口'],'backpack':['返回上级界面','已在界面-观星','星石本地OCR-点击背包入口']}[initial]
   self.assertEqual(c.nodes,expected+['星石本地OCR-已在背包']);self.assertTrue(records[0]['success']);self.assertEqual(records[0]['reverse_swipes'],0)

if __name__=='__main__':unittest.main()
