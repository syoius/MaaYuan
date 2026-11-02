"""Saved real full-tail regression; synthetic cases are explicitly unit-only."""
import sys,json,unittest,copy,tempfile
from types import SimpleNamespace,MethodType
from unittest.mock import patch
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.star_four_row_inventory import locate_terminal_rows,PositionInventory,terminal_reason,confirmed_no_move,frame_stability
FIXTURE=Path(__file__).parent/'fixtures/star_full_tail'
def real_page(number):
 p=json.loads((FIXTURE/f'main-{number}.json').read_text(encoding='utf-8'))
 im=np.zeros((1280,720,3),np.uint8);im[272:1044,40:680]=cv2.imread(str(FIXTURE/f'main-{number}-grid.png'))
 return p,im
class SavedFailureTests(unittest.TestCase):
 def test_real_30px_phase_full_page_has_four_new_rows(self):
  previous,before=real_page('003');current,after=real_page('004')
  mapping=locate_terminal_rows(previous,current,before,after,673)
  self.assertEqual(mapping['row_offset'],4)
  self.assertEqual(len(mapping['anchors']),4)
  self.assertTrue(all(a['ncc']>=.97 for a in mapping['anchors']))
  self.assertTrue(all(abs(a['upward_shift_px']-636)<=1 for a in mapping['anchors']))

 def test_saved_full_page_append_keeps_same_attributes_as_independent_positions(self):
  previous,before=real_page('003');current,after=real_page('004')
  for p in (previous,current):
   for r in p['records']:r['resolved']={'name':'天同','level':1,'quality':'橙'}
  v=PositionInventory('main');v.accept_initial(previous);x=v.accept_moved(current,before,after,673)
  self.assertEqual((x['new_count'],x['duplicate_count'],len(v.items)),(16,0,32))
  self.assertEqual(len(v.positions),32);self.assertIsNone(terminal_reason(before,after,current['geometry']))
 def test_saved_full_tail_has_no_move_confirmation(self):
  _,image=real_page('004');self.assertTrue(confirmed_no_move(image,image,image))


def synthetic_page(base=0,first=403,lengths=(4,4,4,4)):
 """Synthetic physical-row texture fixtures, never device evidence."""
 template,_=real_page('003');p={'sourceImageId':f'unit-{base}-{first}-{lengths}','section':'main','geometry':copy.deepcopy(template['geometry']),'records':[],'timing':{}}
 image=np.zeros((1280,720,3),np.uint8);last=base+len(lengths)-1
 for physical in range(max(0,base-1),base+5):
  n=lengths[-1] if physical==last else 4
  if physical>last:continue
  center=first+(physical-base)*167
  for col in range(n):
   r=copy.deepcopy(template['records'][col]);old_y=r['card']['circle']['centerY'];dy=center-old_y
   for key in ('discRect','nameRect','levelRect'):r['card'][key]['y']+=dy
   r['card']['circle']['centerY']=center;r['card']['rowIndex']=physical-base;r['row']=physical-base;r['sourceImageId']=p['sourceImageId'];r['occurrenceId']=f"{p['sourceImageId']}:r{r['row']}c{col}";r['resolved']={'name':'天同','level':1,'quality':'橙'}
   rects=[r['card'][k] for k in ('discRect','nameRect','levelRect')];x=min(z['x'] for z in rects);y=min(z['y'] for z in rects);x2=max(z['x']+z['width'] for z in rects);y2=max(z['y']+z['height'] for z in rects)
   patch_image=np.random.default_rng(physical*4+col).integers(0,255,(y2-y,x2-x,3),dtype=np.uint8)
   top=max(272,y);bottom=min(1044,y2)
   if bottom>top:image[top:bottom,x:x2]=patch_image[top-y:bottom-y]
   if physical>=base:p['records'].append(r)
 p['geometry']['cards']=[r['card'] for r in p['records']]
 return p,image


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
    self.assertEqual((x['duplicate_count'],x['new_count'],len(v.items)),(8,size,16+size))
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


class LoopTests(unittest.TestCase):
 def test_full_overlap_then_full_advance_then_confirmed_bottom(self):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  pages=[synthetic_page(base,433 if base else 403) for base in (0,1,5)]
  lookup={id(im):p for p,im in pages};frames=iter([pages[1][1],pages[2][1],pages[2][1],pages[2][1]]);saved=[]
  owner=SimpleNamespace(inventories={'main':PositionInventory('main')},sections={},progress=lambda stage:None,
   recognize=lambda im,sid:copy.deepcopy(lookup[id(im)]),save_page=lambda page,*args:saved.append(page),last_capture={'file':'unit.png'})
  def swipe(summary,touch,label):summary['swipes']+=1;return next(frames),{}
  owner.swipe_capture=swipe;owner.confirm_bottom=MethodType(FullShadowSession.confirm_bottom,owner)
  result=FullShadowSession.scan(owner,'main',pages[0][1])
  self.assertTrue(result['complete']);self.assertEqual(result['stop_reason'],'bottom_no_move_confirmed');self.assertEqual(result['final_count'],36);self.assertEqual(len(saved),3)
 def test_stabilization_is_bounded_and_requires_two_consecutive_stable_pairs(self):
  from custom.action import star_full_local_ocr_shadow as session
  _,image=synthetic_page();moved=np.roll(image,5,axis=0)
  for frames,success in (([moved,image,moved,image,moved,image],False),([moved,moved,image,image,image],True)):
   source=iter(frames);calls=[]
   owner=SimpleNamespace(check=lambda:None,check_backpack=lambda *args:None,capture=lambda label:calls.append(label) or next(source))
   with patch.object(session.time,'sleep'):
    if success:session.FullShadowSession.stabilize(owner,image,'unit','main');self.assertEqual(len(calls),5)
    else:
     with self.assertRaisesRegex(RuntimeError,'settle'):session.FullShadowSession.stabilize(owner,image,'unit','main')
     self.assertEqual(len(calls),6)
 def test_complete_scan_unknown_or_mismatched_game_count_is_not_uploadable(self):
  from custom.action import star_full_local_ocr_shadow as session
  from custom.action import star_inventory_sync as sync
  for bag in ({'complete':True,'currentCount':3,'capacity':270,'warnings':[]},{'complete':False,'currentCount':None,'capacity':None,'warnings':['unconfirmed']}):
   with tempfile.TemporaryDirectory() as name:
    base=Path(name);runtime=base/'runtime';runtime.mkdir();fake=SimpleNamespace(resolve=lambda *args:None,close=lambda:None)
    with patch.object(session,'REPO',base),patch.object(session,'PACKAGED_RUNTIME',runtime):a=session.FullShadowSession(object(),{'shadow':True})
    image=np.zeros((1280,720,3),np.uint8);a.prepare=lambda:image;a.stabilize=lambda im,*args:im
    def read(im):a.bag=copy.deepcopy(bag);return im
    a.read_bag=read
    def scan(section,im):a.sections[section]={'complete':True};a.inventories[section].items=[{'name':'unit'}]
    a.scan=scan
    def switch(section):a.last_capture={'file':'unit.png'};return image
    a.switch=switch
    with patch.object(session,'Bridge',return_value=fake),patch.object(session,'recognize_experience',return_value={'complete':True,'structured':True,'orangeCount':0,'purpleCount':0,'whiteCount':0}):self.assertFalse(a.execute())
    local=json.loads((a.out/'inventory.json').read_text(encoding='utf-8'));self.assertFalse(local['complete'])
    error=json.loads((a.out/'error.json').read_text(encoding='utf-8'))['error'];self.assertIn('differs' if bag['complete'] else 'unconfirmed',error)
    with patch.object(sync,'post_inventory') as post:
     with self.assertRaises(ValueError):sync.sync_saved_inventory(object(),a.out)
     post.assert_not_called()

if __name__=='__main__':unittest.main()
