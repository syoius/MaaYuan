"""Position-based four-row inventory; pixel correspondence is confined to terminal pages."""
from __future__ import annotations
import json,time,copy
from pathlib import Path
import cv2
import numpy as np

CALIBRATED_GAIN=((637/630)+(654/654))/2
NO_MOVE_SIMILARITY=.97

def layout(geometry):
    by_row={}
    for card in geometry['cards']:
        if card['completeness']=='complete':by_row.setdefault(card['rowIndex'],[]).append(card)
    groups=[sorted(v,key=lambda c:c['columnIndex']) for _,v in sorted(by_row.items())]
    centers=[float(np.mean([c['circle']['centerY'] for c in row])) for row in groups]
    pitches=np.diff(centers).tolist();pitch=float(np.median(pitches)) if pitches else None
    b=geometry['profile']['contentBounds'];right=b['x']+b['width'];bottom=b['y']+b['height']
    field_regions=all(r['x']>=b['x'] and r['y']>=b['y'] and r['x']+r['width']<=right and r['y']+r['height']<=bottom for row in groups for c in row for r in [c[k] for k in ('discRect','nameRect','levelRect')])
    columns=all(len(row)==4 and [c['columnIndex'] for c in row]==[0,1,2,3] for row in groups)
    return {'groups':groups,'row_centers':centers,'row_pitches':pitches,'median_pitch_px':pitch,'complete_count':sum(map(len,groups)),'fields_valid':field_regions,'four_rows_valid':len(groups)==4 and columns and field_regions and pitch is not None and 145<=pitch<=185 and all(abs(v-pitch)<=8 for v in pitches)}

def validate_swipe(begin,end):
    if begin!=[360,1000] or end[0]!=360 or not 300<=end[1]<=400:
        raise ValueError('four-row swipe outside calibrated safe region')

def next_swipe(geometry,current_touch):
    t=time.perf_counter();s=layout(geometry)
    if not s['four_rows_valid']:raise ValueError('adaptive control requires four complete rows')
    desired=s['row_centers'][0]+4*s['median_pitch_px']-390
    touch=int(round(np.clip(desired/CALIBRATED_GAIN,max(600,current_touch-30),min(700,current_touch+30))))
    end=[360,1000-touch];validate_swipe([360,1000],end)
    return {'desired_content_shift_px':desired,'calibrated_gain':CALIBRATED_GAIN,'touch_length_px':touch,'end':end,'max_correction_px':30,'calculation_ms':(time.perf_counter()-t)*1000}

def same_position(a,b):
    a=cv2.GaussianBlur(cv2.cvtColor(a[280:1040,40:680],cv2.COLOR_BGR2GRAY).astype(np.float32),(3,3),0)
    b=cv2.GaussianBlur(cv2.cvtColor(b[280:1040,40:680],cv2.COLOR_BGR2GRAY).astype(np.float32),(3,3),0)
    a-=float(a.mean());b-=float(b.mean());den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip((a.ravel()@b.ravel())/den,-1,1)) if den>1e-6 else float(np.array_equal(a,b))

def terminal_reason(before,after,geometry):
    """Cheap existing no-move check and current-frame edge geometry only."""
    if same_position(before,after)>=NO_MOVE_SIMILARITY:return 'no_move_candidate'
    s=layout(geometry)
    if not s['four_rows_valid']:return 'terminal_incomplete_row'
    # A full last row can still be terminal: the game's bottom layout places it
    # lower than an ordinary 390/556/723/889 four-row viewport.
    if s['row_centers'][-1]>920:return 'terminal_edge_phase'
    if abs(s['row_centers'][0]-390)>60:return 'abnormal_phase_requires_review'
    return None

def confirmed_no_move(before,first,confirmation):
    return same_position(before,first)>=NO_MOVE_SIMILARITY and same_position(first,confirmation)>=NO_MOVE_SIMILARITY

def validate_fields(page):
    if any(r['resolved'].get(k) is None for r in page['records'] for k in ('name','level','quality')):
        raise ValueError('unconfirmed required OCR field')
    positions=[(r['row'],r['column']) for r in page['records']]
    if len(set(positions))!=len(positions):raise ValueError('duplicate position inside a page')

def _row_records(page):
    rows={}
    for r in page['records']:rows.setdefault(r['row'],[]).append(r)
    return [sorted(v,key=lambda x:x['column']) for _,v in sorted(rows.items())]

def _tail_anchor(before,after,old,new):
    """Reuse the established .97 per-card anchor criterion, terminal stage only."""
    rects=[old['card'][k] for k in ('discRect','nameRect','levelRect')]
    x=min(r['x'] for r in rects);y=min(r['y'] for r in rects);x2=max(r['x']+r['width'] for r in rects);y2=max(r['y']+r['height'] for r in rects)
    dx=new['card']['circle']['centerX']-old['card']['circle']['centerX'];dy=new['card']['circle']['centerY']-old['card']['circle']['centerY']
    tx=int(round(x+dx));ty=int(round(y+dy));margin=6
    left=max(0,tx-margin);top=max(272,ty-margin);right=min(720,tx+(x2-x)+margin);bottom=min(1044,ty+(y2-y)+margin)
    template=cv2.cvtColor(before[y:y2,x:x2],cv2.COLOR_BGR2GRAY);target=cv2.cvtColor(after[top:bottom,left:right],cv2.COLOR_BGR2GRAY)
    if target.shape[0]<template.shape[0] or target.shape[1]<template.shape[1]:return None
    matrix=cv2.matchTemplate(target,template,cv2.TM_CCOEFF_NORMED);_,ncc,_,pos=cv2.minMaxLoc(matrix)
    return {'column':old['column'],'ncc':float(ncc),'upward_shift_px':float(y-top-pos[1]),'horizontal_shift_px':float(x-left-pos[0])}

def locate_terminal_rows(previous,current,before,after,touch_length):
    """Find a single positional row offset. Names/levels/qualities are not keys."""
    started=time.perf_counter();old_rows=_row_records(previous);new_rows=_row_records(current)
    if len(old_rows)!=4 or not new_rows:raise ValueError('terminal mapping needs retained four-row predecessor and visible rows')
    if any([r['column'] for r in row]!=[0,1,2,3] for row in old_rows):raise ValueError('invalid predecessor columns')
    if any([r['column'] for r in row]!=(list(range(len(row))) if j==len(new_rows)-1 else [0,1,2,3]) for j,row in enumerate(new_rows)):
        raise ValueError('terminal has a missing nonfinal cell or noncontiguous last-row columns')
    pitch=float(np.median(np.diff([np.mean([r['card']['circle']['centerY'] for r in row]) for row in old_rows])))
    old_first=float(np.mean([r['card']['circle']['centerY'] for r in old_rows[0]]));new_first=float(np.mean([r['card']['circle']['centerY'] for r in new_rows[0]]))
    possibilities=[]
    for offset in range(4):
        geometric=old_first+offset*pitch-new_first
        if not -3<=geometric<=touch_length*1.2+10:continue
        anchors=[];pairs=[];ok=True
        for j,row in enumerate(new_rows):
            if offset+j>=len(old_rows):break
            old_by_col={r['column']:r for r in old_rows[offset+j]}
            for record in row:
                old=old_by_col.get(record['column'])
                if old is None:ok=False;break
                anchor=_tail_anchor(before,after,old,record)
                if anchor is None or anchor['ncc']<.97 or abs(anchor['upward_shift_px']-geometric)>6 or abs(anchor['horizontal_shift_px'])>3:
                    ok=False;break
                anchors.append(anchor);pairs.append((old['occurrenceId'],record['occurrenceId']))
            if not ok:break
        shifts=[a['upward_shift_px'] for a in anchors]
        if ok and len(anchors)>=4 and max(shifts)-min(shifts)<=3:
            possibilities.append({'row_offset':offset,'geometric_shift_px':geometric,'anchor_shift_px':float(np.median(shifts)),'anchors':anchors,'pairs':pairs})
    if len(possibilities)!=1:
        raise ValueError(f'ambiguous terminal row position: {len(possibilities)} possible offsets; no field-based guessing')
    return {**possibilities[0],'mapping_ms':(time.perf_counter()-started)*1000,'scope':'terminal positional anchors only; no ordinary-stage matching'}

class PositionInventory:
    def __init__(self,section):
        if section not in ('main','support'):raise ValueError('invalid section')
        self.section=section;self.items=[];self.positions={};self.next_row=0;self.last_page=None;self.last_base=None;self.observation_count=0;self.tail_diagnostics=[]
    def _append(self,record,physical_row,tail):
        key=(physical_row,record['column'])
        if key in self.positions:raise ValueError('position already accepted')
        value={'section':self.section,'captureOrder':len(self.items)+1,'sourceImageId':record['sourceImageId'],'sourceRow':record['row'],'sourceColumn':record['column'],'physicalRow':physical_row,'name':record['resolved']['name'],'level':record['resolved']['level'],'quality':record['resolved']['quality'],'confidence':{k:record['fields'][k]['confidence'] for k in ('name','level')},'fromTail':tail}
        value['ocrConfidence']={k:record['fields'][k]['max_raw_confidence'] for k in ('name','level')}
        self.positions[key]=value;self.items.append(value)
    def accept_normal(self,page):
        validate_fields(page);s=layout(page['geometry'])
        if not s['four_rows_valid'] or len(page['records'])!=16:raise ValueError('normal page must contain four complete four-column rows')
        base=self.next_row
        for j,row in enumerate(_row_records(page)):
            for r in row:self._append(r,base+j,False)
        self.next_row+=4;self.last_base=base;self.last_page=page;self.observation_count+=len(page['records'])
        return {'accepted':16,'duplicates':0,'tail_new':0}
    def merge_terminal(self,page,before,after,touch_length):
        validate_fields(page)
        if self.last_page is None:raise ValueError('no predecessor for terminal merge')
        mapping=locate_terminal_rows(self.last_page,page,before,after,touch_length);base=self.last_base+mapping['row_offset'];duplicates=[];new=[]
        # Validate all repeat observations before mutating accepted state.
        for j,row in enumerate(_row_records(page)):
            for r in row:
                key=(base+j,r['column']);prior=self.positions.get(key)
                if prior is not None:
                    if any(prior[k]!=r['resolved'][k] for k in ('name','level','quality')):raise ValueError('terminal positional duplicate has conflicting OCR fields')
                    duplicates.append({'observation':r['occurrenceId'],'prior_capture_order':prior['captureOrder'],'physical_row':key[0],'column':key[1]})
                else:new.append((r,key[0]))
        for r,row in new:self._append(r,row,True)
        self.observation_count+=len(page['records']);self.next_row=max([x['physicalRow'] for x in self.items],default=-1)+1
        result={'sourceImageId':page['sourceImageId'],'row_offset':mapping['row_offset'],'duplicate_count':len(duplicates),'new_count':len(new),'duplicates':duplicates,'position_mapping':mapping,'final_count':len(self.items)}
        self.tail_diagnostics.append(result);return result

class PageStore:
    def __init__(self,out):self.out=Path(out);self.out.mkdir(exist_ok=True,parents=True);self.index=[]
    def save(self,page):
        started=time.perf_counter();name=page['sourceImageId']+'.json'
        with (self.out/name).open('x',encoding='utf-8') as f:json.dump(page,f,ensure_ascii=False,separators=(',',':'))
        entry={'file':name,'image':page['sourceImageId'],'status':page['status'],'records':len(page.get('records',[]))}
        with (self.out/'checkpoints.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        entry['save_ms']=(time.perf_counter()-started)*1000;self.index.append(entry);return entry
