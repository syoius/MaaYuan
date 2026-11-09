"""Restore committed pixel crops in their original coordinate system."""
import copy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from utils.star_local_ocr_shadow import read_image

FIXTURES = Path(__file__).with_name('fixtures')
EXPERIENCE_CROPS = ('purple-white-region.png', 'simplified-three-region.png')


def experience_canvas(name):
    canvas = np.full((1280, 720, 3), 50, np.uint8)
    canvas[208:528] = read_image(FIXTURES / 'star_experience_presence' / name)
    return canvas


def grid_canvas(path, section):
    canvas = np.full((1280, 720, 3), 80, np.uint8)
    canvas[272:1044, 40:680] = read_image(path)
    # Synthetic title and selected tab for the real identity check.
    canvas[130:150, 270:340] = 180
    tab = {'main': 90, 'support': 290}[section]
    canvas[225:245, tab:tab+12, 2] = 240
    return canvas



def load_page(folder, name):
    return json.loads((FIXTURES / folder / f'{name}.json').read_text(encoding='utf-8'))


def replay_crop(folder, name):
    canvas = np.zeros((1280, 720, 3), np.uint8)
    canvas[272:1044, 40:680] = read_image(FIXTURES / folder / f'{name}-grid.png')
    return canvas


def replay_page(folder, name):
    return load_page(folder, name), replay_crop(folder, name)


def synthetic_page(base=0,first=403,lengths=(4,4,4,4)):
 """Synthetic physical-row texture fixtures, never device evidence."""
 template=load_page('star_full_tail', 'main-003');p={'sourceImageId':f'unit-{base}-{first}-{lengths}','section':'main','geometry':copy.deepcopy(template['geometry']),'records':[],'timing':{}}
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
 p['geometry']['profile'].update(imageWidth=720, imageHeight=1280)
 return p,image


def replay_owner(page_frames, swipes, root):
    from custom.action.star_full_local_ocr_shadow import FullShadowSession
    from utils.star_four_row_inventory import PositionInventory

    source = iter(swipes)
    lookup = {id(image): page for page, image in page_frames}
    saved, gestures, recognized = [], [], []
    owner = SimpleNamespace(inventories={'main': PositionInventory('main')}, sections={},
        progress=lambda stage: None, last_capture={'file': 'unit.png'}, out=Path(root),
        recovery_checkpoints=set(), position_recoveries=[], events=[])
    def recognize(image, sid):
        recognized.append(sid)
        page = copy.deepcopy(lookup[id(image)])
        page['sourceImageId'] = sid
        page.setdefault('timing', {})
        return page
    def swipe(summary, touch, label, **kwargs):
        summary['swipes'] += 1
        gestures.append((touch, label, kwargs))
        return next(source), {}
    owner.recognize = recognize
    owner.swipe_capture = swipe
    owner.save_page = lambda page, *args: saved.append(copy.deepcopy(page))
    owner.confirm_bottom = MethodType(FullShadowSession.confirm_bottom, owner)
    owner.recover_position = MethodType(FullShadowSession.recover_position, owner)
    return owner, saved, gestures, recognized



def boundary_page(name, *, repaired=False):
    saved = load_page('star_boundary_rows', name)
    if name == 'support-002':
        page = saved['original']
        if repaired:
            changes = saved['repair']
            for index, card in changes['cards'].items():
                page['geometry']['cards'][int(index)] = card
            records = page['records']
            for index, record in changes['records'].items():
                index = int(index)
                if index == len(records):
                    records.append(record)
                else:
                    records[index] = record
    else:
        if repaired:
            raise ValueError('no repaired version for this page')
        page = saved
    return page, replay_crop('star_boundary_rows', name)



def geometry_pages():
    """Representative saved pages for migration golden comparisons."""
    yield 'boundary support-002', load_page('star_boundary_rows', 'support-002')['original']
    for folder, name in (
        ('star_boundary_rows', 'support-003'),
        ('star_full_tail', 'main-004'),
        ('star_overlap_tail', 'main-after'),
    ):
        yield f'{folder}/{name}', load_page(folder, name)


class FakeGeometry:
 def __init__(self,kinds=('white','orange','purple')):
  self.candidates=[{'quality':{'kind':k,'confidence':.99},'countRect':{'x':{'orange':0,'purple':20,'white':40}[k],'y':0,'width':18,'height':18},'completeness':'complete'} for i,k in enumerate(kinds)]
 def experience_geometry(self,image):return {'tab':{'selected':True},'coverage':{'complete':True,'emptyConfirmed':not self.candidates},'candidates':self.candidates,'geometryMs':0.}
