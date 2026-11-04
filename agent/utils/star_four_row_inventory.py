"""Position-based inventory with verified row transitions and separate bottom confirmation."""
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

def frame_stability(a,b):
    """Motion evidence on the inventory viewport; field values are never identities."""
    x=cv2.cvtColor(a[280:1040,40:680],cv2.COLOR_BGR2GRAY).astype(np.float32)
    y=cv2.cvtColor(b[280:1040,40:680],cv2.COLOR_BGR2GRAY).astype(np.float32)
    shift,response=cv2.phaseCorrelate(x,y,cv2.createHanningWindow((x.shape[1],x.shape[0]),cv2.CV_32F))
    ncc=same_position(a,b)
    return {'stable':bool(ncc>=.99 and response>=.5 and max(map(abs,shift))<=2),
            'ncc':ncc,'shift_px':list(shift),'response':float(response)}


def terminal_reason(before,after,geometry):
    """Geometry describes a moved page, never proves that it is the bottom."""
    if same_position(before,after)>=NO_MOVE_SIMILARITY:return 'no_move_candidate'
    return None if layout(geometry)['four_rows_valid'] else 'partial_page_requires_confirmation'


def confirmed_no_move(before,first,confirmation):
    # Also compare the endpoints: two small animated shifts must not accumulate.
    return all(frame_stability(a,b)['stable'] for a,b in ((before,first),(first,confirmation),(before,confirmation)))


def validate_fields(page):
    limits={'橙':60,'紫':60,'蓝':50,'绿':40,'白':30,'orange':60,'purple':60,'blue':50,'green':40,'white':30}
    for record in page['records']:
        fields=record['resolved'];name=fields.get('name');quality=fields.get('quality');level=fields.get('level')
        if not isinstance(name,str) or not name.strip() or quality not in limits or type(level) is not int or not 1<=level<=limits[quality]:
            raise ValueError('unconfirmed or invalid required OCR field')
    positions=[(r['row'],r['column']) for r in page['records']]
    if len(set(positions))!=len(positions):raise ValueError('duplicate position inside a page')

def _row_records(page):
    rows={}
    for r in page['records']:rows.setdefault(r['row'],[]).append(r)
    if not rows or sorted(rows)!=list(range(len(rows))) or len(rows)>4:
        raise ValueError('page must have consecutive visible rows')
    result=[sorted(v,key=lambda x:x['column']) for _,v in sorted(rows.items())]
    if any([r['column'] for r in row]!=(list(range(len(row))) if j==len(result)-1 else [0,1,2,3]) or not 1<=len(row)<=4 for j,row in enumerate(result)):
        raise ValueError('terminal has a missing nonfinal cell or noncontiguous last-row columns')
    return result


def _validate_grid(page):
    validate_fields(page);rows=_row_records(page);s=layout(page['geometry'])
    if not s['fields_valid'] or s['complete_count']!=len(page['records']):raise ValueError('page has invalid or missing field geometry')
    if len(rows)>1 and (not 145<=s['median_pitch_px']<=185 or any(abs(v-s['median_pitch_px'])>8 for v in s['row_pitches'])):
        raise ValueError('page row spacing is not a regular grid')
    return rows,s


def _tail_anchor(before,after,old,new):
    """Only name glyphs anchor complete cards; shared discs never identify them."""
    geometric=old['card']['circle']['centerY']-new['card']['circle']['centerY']
    dx=new['card']['circle']['centerX']-old['card']['circle']['centerX']
    return _text_anchor(before,after,old,geometric,dx)


def _row_contrast_ncc(template,target):
    # The fixed top/bottom viewport fade changes gain and offset by scanline.
    # Remove that photometric change while preserving every horizontal pixel.
    a=template.astype(np.float32);b=target.astype(np.float32)
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    sa=np.sqrt(np.mean(a*a,1));sb=np.sqrt(np.mean(b*b,1))
    valid=(sa>max(5,.25*sa.max()))&(sb>max(5,.25*sb.max()))
    if int(valid.sum())<8:return -1.
    scores=np.mean(a[valid]*b[valid],1)/(sa[valid]*sb[valid])
    return float(np.average(scores,weights=sa[valid]))


def _text_anchor(before,after,record,geometric,dx=0):
    r=record['card']['nameRect'];x,y,w,h=(int(r[k]) for k in ('x','y','width','height'))
    tx=int(round(x+dx));ty=int(round(y-geometric));margin=6
    left=max(0,tx-margin);top=max(272,ty-margin);right=min(720,tx+w+margin);bottom=min(1044,ty+h+margin)
    template=cv2.cvtColor(before[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    target=cv2.cvtColor(after[top:bottom,left:right],cv2.COLOR_BGR2GRAY)
    if template.size==0 or float(template.std())<5 or target.shape[0]<h or target.shape[1]<w:return None
    raw=cv2.matchTemplate(target,template,cv2.TM_CCOEFF_NORMED)
    _,ncc,_,(px,py)=cv2.minMaxLoc(raw);metric='raw_name_ncc'
    if ncc<.97:
        # The independently tested boundary criterion tolerates scanline fade.
        # These scores are not ranked against raw NCC or against other offsets.
        template=cv2.GaussianBlur(template,(3,3),0);target=cv2.GaussianBlur(target,(3,3),0)
        ncc,px,py=max((_row_contrast_ncc(template,target[yy:yy+h,xx:xx+w]),xx,yy)
                      for yy in range(target.shape[0]-h+1) for xx in range(target.shape[1]-w+1))
        metric='row_contrast_name_ncc'
    return {'column':record['column'],'ncc':float(ncc),'raw_ncc':float(raw[py,px]),'metric':metric,'threshold':.97,
            'upward_shift_px':float(y-top-py),'horizontal_shift_px':float(x-left-px),'evidence':'name_pixels_only'}


def _boundary_name_anchor(before,after,record,geometric):
    """Clipped predecessor text is an anchor, never an additional instance."""
    return _text_anchor(before,after,record,geometric)


def _reliable_field(record,key):
    value=record.get('resolved',{}).get(key)
    if value is None:return False
    if key=='quality':
        evidence=record.get('quality',record.get('card',{}).get('qualityEvidence',{}))
        return evidence.get('quality')==value and evidence.get('confidence',0)>=.5
    evidence=record.get('fields',{}).get(key,{})
    # Same raw-confidence floor used by the ROI recognizer; its resolver also
    # scales confidence, so a raw .99 single reading normally becomes .89/.90.
    return (evidence.get('confidence',0)>=.8 and evidence.get('max_raw_confidence',0)>=.9
            and not evidence.get('low_confidence',False))


def field_conflicts(old,new):
    """Normalized fields can veto identity, never establish it."""
    return [key for key in ('name','level','quality')
            if _reliable_field(old,key) and _reliable_field(new,key) and old['resolved'][key]!=new['resolved'][key]]


class PositionUncertain(ValueError):
    def __init__(self,diagnostics):
        self.diagnostics=diagnostics
        conflicts=any(c['reason']=='conflicting OCR fields' for c in diagnostics['rejected'])
        super().__init__(f"ambiguous terminal row position: {len(diagnostics['candidates'])} possible offsets; "
                         + ('conflicting OCR fields rejected; ' if conflicts else '')+'no unique text/geometry correspondence')


def position_candidates(previous,current,before,after,touch_length,*,offsets=range(5),shift_bounds=None,boundary=True):
    """Generate physical offsets; all surviving offsets need local text evidence."""
    started=time.perf_counter();old_rows=_row_records(previous);new_rows=_row_records(current)
    if len(old_rows)!=4 or any([r['column'] for r in row]!=[0,1,2,3] for row in old_rows):
        raise ValueError('terminal mapping needs retained four-row predecessor and visible rows')
    pitch=float(np.median(np.diff([np.mean([r['card']['circle']['centerY'] for r in row]) for row in old_rows])))
    old_first=float(np.mean([r['card']['circle']['centerY'] for r in old_rows[0]]));new_first=float(np.mean([r['card']['circle']['centerY'] for r in new_rows[0]]))
    minimum,maximum=shift_bounds if shift_bounds is not None else (-3,touch_length*1.2+10)
    possibilities=[];rejected=[]
    for offset in offsets:
        geometric=old_first+offset*pitch-new_first
        if not minimum<=geometric<=maximum:continue
        pairs=[(old_rows[offset+j][r['column']],r) for j,row in enumerate(new_rows) if 0<=offset+j<len(old_rows) for r in row]
        conflicts=[{'previous':old['occurrenceId'],'current':new['occurrenceId'],'fields':keys}
                   for old,new in pairs if (keys:=field_conflicts(old,new))]
        if conflicts:
            rejected.append({'row_offset':offset,'reason':'conflicting OCR fields','conflicts':conflicts});continue
        anchors=[];ok=True
        for old,new in pairs:
            anchor=_tail_anchor(before,after,old,new)
            if anchor is None or anchor['ncc']<anchor['threshold'] or abs(anchor['upward_shift_px']-geometric)>6 or abs(anchor['horizontal_shift_px'])>3:
                ok=False;break
            anchors.append(anchor)
        if not pairs and boundary and offset==4:
            for old in old_rows[-1]:
                anchor=_boundary_name_anchor(before,after,old,geometric)
                if anchor is None or anchor['ncc']<anchor['threshold'] or abs(anchor['upward_shift_px']-geometric)>6 or abs(anchor['horizontal_shift_px'])>3:
                    ok=False;break
                anchors.append(anchor)
        shifts=[a['upward_shift_px'] for a in anchors]
        if ok and len(anchors)>=4 and max(shifts)-min(shifts)<=3:
            possibilities.append({'row_offset':offset,'geometric_shift_px':geometric,'anchor_shift_px':float(np.median(shifts)),
                                  'anchors':anchors,'pairs':[(a['occurrenceId'],b['occurrenceId']) for a,b in pairs]})
        else:rejected.append({'row_offset':offset,'reason':'name pixels or geometry unconfirmed'})
    return {'candidates':possibilities,'rejected':rejected,'mapping_ms':(time.perf_counter()-started)*1000,
            'scope':'physical row correspondence; normalized fields veto only, local name pixels required'}


def locate_terminal_rows(previous,current,before,after,touch_length):
    result=position_candidates(previous,current,before,after,touch_length)
    if len(result['candidates'])!=1:raise PositionUncertain(result)
    return {**result['candidates'][0],'mapping_ms':result['mapping_ms'],'rejected_candidates':result['rejected'],'scope':result['scope']}


def recovery_swipe(geometry,predecessor):
    """Two rows from measured pitch, with the existing gain, inside this viewport."""
    current=layout(geometry);prior=layout(predecessor)
    pitch=current['median_pitch_px'] if current['median_pitch_px'] is not None else prior['median_pitch_px']
    if pitch is None or not 145<=pitch<=185:raise ValueError('recovery needs calibrated row spacing')
    profile=geometry['profile'];bounds=profile['contentBounds']
    if (profile.get('imageWidth'),profile.get('imageHeight'))!=(720,1280):raise ValueError('recovery requires the supported 720x1280 frame')
    touch=int(round(2*pitch/CALIBRATED_GAIN));x=int(round(bounds['x']+bounds['width']/2));y=int(round(bounds['y']+.7*pitch))
    if not bounds['y']<y<y+touch<bounds['y']+bounds['height']:raise ValueError('two-row recovery exceeds the content viewport')
    return {'swipe_begin':[x,y],'swipe_end':[x,y+touch],'swipe_duration_ms':700,'swipe_end_hold_ms':250,
            'row_pitch_px':pitch,'target_rows':2,'calibrated_gain':CALIBRATED_GAIN,'touch_length_px':touch}


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
    def accept_initial(self,page):
        if self.items:raise ValueError('initial page already accepted')
        rows,s=_validate_grid(page)
        if s['four_rows_valid']:return self.accept_normal(page)
        for j,row in enumerate(rows):
            for r in row:self._append(r,j,False)
        self.next_row=len(rows);self.last_base=0;self.last_page=page;self.observation_count+=len(page['records'])
        return {'accepted':len(page['records']),'duplicates':0,'tail_new':0}
    def accept_moved(self,page,before,after,touch_length):
        _,s=_validate_grid(page)
        if self.last_page is None:raise ValueError('no predecessor for moved page')
        mapping=locate_terminal_rows(self.last_page,page,before,after,touch_length)
        base=self.last_base+mapping['row_offset']
        # After recovery the visible page can lag the accepted frontier. A four-
        # row swipe then includes known positions and cannot blindly append 16.
        if mapping['row_offset']==4 and s['four_rows_valid'] and base==self.next_row:
            self.accept_normal(page)
            return {'new_count':16,'duplicate_count':0,'final_count':len(self.items),'position_mapping':mapping}
        result=self._merge_at(page,base,mapping)
        self.last_page=page;self.last_base=base
        return result
    def _merge_at(self,page,base,mapping):
        validate_fields(page);duplicates=[];new=[]
        if base<0:raise ValueError('physical row cannot precede the initial checkpoint')
        for j,row in enumerate(_row_records(page)):
            for r in row:
                key=(base+j,r['column']);prior=self.positions.get(key)
                if prior is not None:
                    if any(prior[k]!=r['resolved'][k] for k in ('name','level','quality')):raise ValueError('terminal positional duplicate has conflicting OCR fields')
                    duplicates.append({'observation':r['occurrenceId'],'prior_capture_order':prior['captureOrder'],'physical_row':key[0],'column':key[1]})
                else:new.append((r,key[0]))
        # Check the whole proposed footprint before changing a single instance.
        keys=set(self.positions)|{(row,r['column']) for r,row in new}
        last=max((row for row,col in keys),default=-1)
        for row in range(last+1):
            cols=sorted(col for rr,col in keys if rr==row)
            if cols!=(list(range(len(cols))) if row==last else [0,1,2,3]) or not cols:
                raise ValueError('position recovery would leave a gap in the accepted inventory')
        for r,row in new:self._append(r,row,True)
        self.observation_count+=len(page['records']);self.next_row=last+1
        result={'sourceImageId':page['sourceImageId'],'row_offset':mapping['row_offset'],'duplicate_count':len(duplicates),'new_count':len(new),
                'duplicates':duplicates,'position_mapping':mapping,'final_count':len(self.items)}
        self.tail_diagnostics.append(result);return result
    def merge_terminal(self,page,before,after,touch_length,mapping=None):
        if self.last_page is None:raise ValueError('no predecessor for terminal merge')
        mapping=mapping or locate_terminal_rows(self.last_page,page,before,after,touch_length)
        return self._merge_at(page,self.last_base+mapping['row_offset'],mapping)
    def recover_page(self,unresolved,recovery,before,uncertain_image,recovery_image,reverse_touch):
        """Resolve both overlaps, then atomically commit; resume at the real view."""
        _validate_grid(unresolved);_validate_grid(recovery)
        if not layout(recovery['geometry'])['four_rows_valid']:raise ValueError('recovery needs four complete visible rows')
        left=position_candidates(self.last_page,recovery,before,recovery_image,reverse_touch,
                                 offsets=range(-2,4),shift_bounds=(-reverse_touch*1.5-10,reverse_touch*1.5+10),boundary=False)
        right=position_candidates(recovery,unresolved,recovery_image,uncertain_image,reverse_touch,
                                  offsets=range(4),shift_bounds=(-3,reverse_touch*1.5+10),boundary=False)
        solutions={}
        for a in left['candidates']:
            for b in right['candidates']:
                recovered=self.last_base+a['row_offset'];pending=recovered+b['row_offset']
                if recovered<0 or not self.last_base<=pending<=self.last_base+4:continue
                solutions[(recovered,pending)]=(a,b)
        if len(solutions)!=1:raise ValueError(f'two-row recovery has {len(solutions)} reliable position solutions; checkpoint preserved')
        (base,pending),(a,b)=next(iter(solutions.items()))
        trial=copy.deepcopy(self)
        recovered=trial._merge_at(recovery,base,a)
        chained={'row_offset':pending-self.last_base,'scope':'two verified overlaps through recovery','checkpoint_to_recovery':a,'recovery_to_unresolved':b}
        resolved=trial._merge_at(unresolved,pending,chained)
        trial.last_page=recovery;trial.last_base=base
        self.__dict__.update(trial.__dict__)
        return {'resumed_base':base,'unresolved_base':pending,'new_count':recovered['new_count']+resolved['new_count'],
                'duplicate_count':recovered['duplicate_count']+resolved['duplicate_count'],'final_count':len(self.items),
                'recovery_page':recovered,'unresolved_page':resolved,'mapping_ms':left['mapping_ms']+right['mapping_ms']}


class PageStore:
    def __init__(self,out):self.out=Path(out);self.out.mkdir(exist_ok=True,parents=True);self.index=[]
    def save(self,page):
        started=time.perf_counter();name=page['sourceImageId']+'.json'
        with (self.out/name).open('x',encoding='utf-8') as f:json.dump(page,f,ensure_ascii=False,separators=(',',':'))
        entry={'file':name,'image':page['sourceImageId'],'status':page['status'],'records':len(page.get('records',[]))}
        with (self.out/'checkpoints.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        entry['save_ms']=(time.perf_counter()-started)*1000;self.index.append(entry);return entry
