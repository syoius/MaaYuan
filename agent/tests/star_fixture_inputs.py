"""Synthetic geometry and pixel textures; no captured account inventory."""
import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np


def synthetic_page(base=0,first=403,lengths=(4,4,4,4)):
 """Deterministic physical-row textures with an explicit four-column layout."""
 sid=f'unit-{base}-{first}-{lengths}'
 p={'sourceImageId':sid,'section':'main','geometry':{'profile':{
  'imageWidth':720,'imageHeight':1280,
  'viewport':{'x':0,'y':0,'width':720,'height':1280},
  'contentBounds':{'x':0,'y':272,'width':720,'height':772}},'cards':[]},'records':[],'timing':{}}
 image=np.zeros((1280,720,3),np.uint8);last=base+len(lengths)-1
 for physical in range(max(0,base-1),base+5):
  n=lengths[-1] if physical==last else 4
  if physical>last:continue
  center=first+(physical-base)*167
  for col in range(n):
   x=(119,274,430,585)[col]
   card={'rowIndex':physical-base,'columnIndex':col,'completeness':'complete',
    'circle':{'centerX':x,'centerY':center},
    'discRect':{'x':x-56,'y':center-56,'width':112,'height':112},
    'nameRect':{'x':x-52,'y':center+60,'width':103,'height':34},
    'levelRect':{'x':x+1,'y':center-60,'width':60,'height':32}}
   r={'sourceImageId':sid,'occurrenceId':f'{sid}:r{physical-base}c{col}',
    'row':physical-base,'column':col,'card':card,
    'resolved':{'name':'天同','level':1,'quality':'橙'},
    'fields':{k:{'confidence':.99,'max_raw_confidence':.99} for k in ('name','level')}}
   rects=[card[k] for k in ('discRect','nameRect','levelRect')]
   left=min(z['x'] for z in rects);y=min(z['y'] for z in rects)
   right=max(z['x']+z['width'] for z in rects);bottom=max(z['y']+z['height'] for z in rects)
   texture=np.random.default_rng(physical*4+col).integers(0,255,(bottom-y,right-left,3),dtype=np.uint8)
   top=max(272,y);stop=min(1044,bottom)
   if stop>top:image[top:stop,left:right]=texture[top-y:stop-y]
   if physical>=base:p['records'].append(r)
 p['geometry']['cards']=[r['card'] for r in p['records']]
 return p,image


def replay_owner(page_frames, swipes, root):
    from custom.action.star_full_local_ocr_shadow import FullShadowSession
    from utils.star_four_row_inventory import PositionInventory

    source = iter(swipes)
    lookup = {id(image): page for page, image in page_frames}
    saved, gestures, recognized = [], [], []
    owner = SimpleNamespace(inventories={'main': PositionInventory('main')}, sections={},
        progress=lambda stage: None, last_capture={'file': 'unit.png'}, out=Path(root),
        recovery_checkpoints=set(), compensation_checkpoints=set(), position_recoveries=[], events=[])
    def recognize(image, sid):
        recognized.append(sid)
        page = copy.deepcopy(lookup[id(image)])
        page['sourceImageId'] = sid
        page.setdefault('timing', {})
        return page
    def swipe(summary, plan, label):
        summary['swipes'] += 1
        gestures.append((plan, label))
        return next(source), {}
    owner.recognize = recognize
    owner.swipe_capture = swipe
    owner.save_page = lambda page, *args: saved.append(copy.deepcopy(page))
    owner.confirm_bottom = MethodType(FullShadowSession.confirm_bottom, owner)
    owner.compensate_name = MethodType(FullShadowSession.compensate_name, owner)
    owner.recover_position = MethodType(FullShadowSession.recover_position, owner)
    return owner, saved, gestures, recognized



class FakeGeometry:
 def __init__(self,kinds=('white','orange','purple')):
  self.candidates=[{'quality':{'kind':k,'confidence':.99},'countRect':{'x':{'orange':0,'purple':20,'white':40}[k],'y':0,'width':18,'height':18},'completeness':'complete'} for i,k in enumerate(kinds)]
 def experience_geometry(self,image):return {'tab':{'selected':True},'coverage':{'complete':True,'emptyConfirmed':not self.candidates},'candidates':self.candidates,'geometryMs':0.}
