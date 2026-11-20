"""Position-based inventory with planned row transitions and independently verified boundaries."""
from __future__ import annotations
import json,time,copy,math
from pathlib import Path
import cv2
import numpy as np
from utils.star_frame_stability import StabilityFrame
from utils.star_geometry import is_inside

CALIBRATED_GAIN=((637/630)+(654/654))/2
TARGET_FIRST_ROW_CENTER_PX = 390

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

def plan_forward(geometry):
    """Plan a safe landing using the existing calibrated gain."""
    s = layout(geometry)
    pitch = s['median_pitch_px']
    groups = s['groups']
    if (pitch is None or not 145 <= pitch <= 185 or not s['fields_valid']
            or len(groups) < 3 or any(len(row) != 4 for row in groups)
            or any(abs(v-pitch) > 8 for v in s['row_pitches'])):
        raise ValueError('forward planning requires at least three complete regular rows')
    first = s['row_centers'][0]
    bounds = geometry['profile']['contentBounds']
    relative = [(c[k]['y']-center, c[k]['y']+c[k]['height']-center)
                for row,center in zip(groups,s['row_centers']) for c in row
                for k in ('discRect','nameRect','levelRect')]
    top = min(v[0] for v in relative)
    bottom = max(v[1] for v in relative)
    gain = CALIBRATED_GAIN
    low, high = gain-.025, gain+.025

    def safe_interval(rows, minimum, maximum):
        # Leave all four destination rows readable throughout the gain range.
        minimum = max(minimum, math.ceil((first+(rows+3)*pitch+bottom
                                        -(bounds['y']+bounds['height']-4))/low))
        maximum = min(maximum, math.floor((first+rows*pitch+top-(bounds['y']+4))/high))
        return minimum, maximum

    for rows in (4, 2, 1):
        if rows == 4 and not s['four_rows_valid']:
            continue
        minimum, maximum = safe_interval(rows, 600, 700) if rows == 4 else safe_interval(rows, 80, 599)
        if minimum <= maximum:
            break
    else:
        raise ValueError('no safe four-row or bridge swipe for the current geometry; checkpoint preserved')

    # The observed first-row phase shifts the next landing by the same amount
    # in content pixels. Apply it once and keep the existing safe touch limits.
    phase_adjustment = first - TARGET_FIRST_ROW_CENTER_PX
    touch = max(minimum, min(maximum, round((rows*pitch+phase_adjustment)/gain)))
    mode = 'normal' if rows == 4 else 'bridge'
    return {'mode': mode, 'target_rows': rows, 'swipe_begin': [360,1000], 'swipe_end': [360,1000-touch],
            'touch_length_px': touch, 'calibrated_gain': gain, 'safe_touch_interval_px': [minimum,maximum],
            'first_row_phase_adjustment_px': phase_adjustment,
            'landing_center_interval_px': [first+rows*pitch-touch*high,first+rows*pitch-touch*low]}


def initial_forward_plan(geometry):
    s = layout(geometry)
    if len(s['groups']) >= 3 and all(len(row) == 4 for row in s['groups']):
        return plan_forward(geometry)
    # A short initial list still uses the existing normal swipe to probe the boundary.
    return {'mode': 'normal', 'target_rows': 4, 'swipe_begin': [360,1000], 'swipe_end': [360,370],
            'touch_length_px': 630}


def frame_stability(a,b):
    """Animation settling only; this does not establish inventory position."""
    return StabilityFrame(a).compare(StabilityFrame(b))


def confirmed_no_move(*decisions):
    """Both gestures and their endpoints need unique, unchanged physical positions."""
    return len(decisions)==3 and all(isinstance(d,dict) and d.get('state')=='NO_MOVE_CANDIDATE' for d in decisions)


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
    indices=sorted(rows)
    if (not rows or any(type(i) is not int or i<0 for i in indices)
            or indices!=list(range(indices[0],indices[0]+len(rows))) or len(rows)>4):
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


def name_compensation_candidate(page):
    """Describe a four-row name-only obstruction without relaxing inventory validation."""
    cards = page['geometry']['cards']
    bounds = page['geometry']['profile']['contentBounds']
    complete = {(r['row'], r['column']) for r in page['records']}
    if not complete:
        return None
    start = min(row for row, _ in complete)
    final = start + 3
    first_three = {(row, col) for row in range(start, final) for col in range(4)}
    positions = {(c['rowIndex'], c['columnIndex']): c for c in cards if start <= c['rowIndex'] <= final}
    fourth = {(final, col) for col in range(4)}
    if set(positions) != first_three | fourth or not first_three <= complete <= set(positions):
        return None
    clipped = []
    bottom = bounds['y'] + bounds['height']
    for key, card in positions.items():
        if not all(is_inside(card[k], bounds) for k in ('discRect', 'levelRect')):
            return None
        name = card.get('originalNameRect', card['nameRect'])
        if key in complete:
            if card['completeness'] != 'complete' or not is_inside(card['nameRect'], bounds):
                return None
        elif (key not in fourth or card['completeness'] != 'partial_bottom'
              or 'disc_or_ocr_roi_below_content_bottom' not in card.get('evidence', ())
              or name['y'] < bounds['y'] or name['x'] < bounds['x']
              or name['x']+name['width'] > bounds['x']+bounds['width']
              or not 1 < name['y']+name['height']-bottom <= name['height']):
            return None
        else:
            clipped.append(name)
    if not clipped:
        return None
    centers = [np.mean([positions[row,col]['circle']['centerY'] for col in range(4)])
               for row in range(start, final+1)]
    pitches = np.diff(centers); pitch = np.median(pitches)
    if not 145 <= pitch <= 185 or np.any(np.abs(pitches-pitch) > 8):
        return None
    return max(r['height'] for r in clipped)


def name_compensation_swipe(name_height):
    touch = math.ceil(name_height / CALIBRATED_GAIN)
    return {'mode': 'compensation', 'target_rows': 0, 'swipe_begin': [360,1000],
            'swipe_end': [360,1000-touch], 'touch_length_px': touch,
            'target_content_shift_px': name_height, 'calibrated_gain': CALIBRATED_GAIN}


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
    if bottom<=top or right<=left:return None
    template=before[y:y+h,x:x+w];target=after[top:bottom,left:right]
    if template.size==0 or target.shape[0]<h or target.shape[1]<w:return None
    template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    target=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
    if float(template.std())<5:return None
    raw=cv2.matchTemplate(target,template,cv2.TM_CCOEFF_NORMED)
    _,ncc,_,(px,py)=cv2.minMaxLoc(raw)
    if ncc<.97:
        # The independently tested boundary criterion tolerates scanline fade.
        # These scores are not ranked against raw NCC or against other offsets.
        template=cv2.GaussianBlur(template,(3,3),0);target=cv2.GaussianBlur(target,(3,3),0)
        ncc,px,py=max((_row_contrast_ncc(template,target[yy:yy+h,xx:xx+w]),xx,yy)
                      for yy in range(target.shape[0]-h+1) for xx in range(target.shape[1]-w+1))
    return {'ncc':float(ncc),'threshold':.97,'upward_shift_px':float(y-top-py),
            'horizontal_shift_px':float(x-left-px)}


def _boundary_name_anchor(before,after,record,geometric):
    """Clipped predecessor text is an anchor, never an additional instance."""
    return _text_anchor(before,after,record,geometric)


def _partial_row_anchors(previous,current,before,after,offset,geometric):
    """Raw geometry retains partial rows; anchors never supply instance fields.

    Offsets count from each page's first *complete* row, not geometry row zero.
    A common partial row must have all four name columns independently verified.
    """
    old_first=min(r['row'] for r in previous['records'])
    new_first=min(r['row'] for r in current['records'])
    old={};new={}
    for card in previous['geometry']['cards']:
        old.setdefault(card['rowIndex']-old_first,{})[card['columnIndex']]=card
    for card in current['geometry']['cards']:
        new.setdefault(card['rowIndex']-new_first,{})[card['columnIndex']]=card
    b=previous['geometry']['profile']['contentBounds']
    for row,columns in old.items():
        targets=new.get(row-offset,{})
        if sorted(columns)!=[0,1,2,3] or sorted(targets)!=[0,1,2,3]:continue
        if all(columns[c]['completeness']=='complete' and targets[c]['completeness']=='complete' for c in columns):continue
        anchors=[]
        for col,card in sorted(columns.items()):
            # Clip source text to the existing content bounds for positional
            # evidence only. This does not upgrade the card's completeness.
            rect=card['nameRect'];top=max(rect['y'],b['y']);bottom=min(rect['y']+rect['height'],b['y']+b['height'])
            if bottom-top<8:break
            clipped={**card,'nameRect':{**rect,'y':top,'height':bottom-top}}
            dx=targets[col]['circle']['centerX']-card['circle']['centerX']
            anchor=_text_anchor(before,after,{'card':clipped},geometric,dx)
            if (anchor is None or anchor['ncc']<anchor['threshold']
                    or abs(anchor['upward_shift_px']-geometric)>6 or abs(anchor['horizontal_shift_px'])>3):break
            anchors.append(anchor)
        if len(anchors)==4 and max(a['upward_shift_px'] for a in anchors)-min(a['upward_shift_px'] for a in anchors)<=3:
            return anchors
    return []


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


class BoundaryPositionError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _boundary_matches(previous, current, before, after, touch_length, offsets, *, evidence=None):
    """Find visual correspondence for shortened swipes and stationary views."""
    old_rows = _row_records(previous)
    new_rows = _row_records(current)
    old_centers = layout(previous['geometry'])['row_centers']
    new_centers = layout(current['geometry'])['row_centers']
    pitches = np.diff(old_centers if len(old_centers) > 1 else new_centers)
    pitch = float(np.median(pitches)) if len(pitches) else None
    required = min(4, len(previous['records']), len(current['records']))
    matches = []
    for offset in offsets:
        candidate={'row_offset':offset}
        if evidence is not None:
            evidence.append(candidate)
        if pitch is None and offset != 0:
            candidate['reason']='row_pitch_unavailable'
            continue
        geometric = old_centers[0]+offset*(pitch or 0)-new_centers[0]
        candidate['geometry_shift_px']=geometric
        if not -3 <= geometric <= touch_length*1.2+10:
            candidate['reason']='motion_out_of_range'
            continue
        pairs = [(old,new) for j,row in enumerate(new_rows) if offset+j < len(old_rows)
                 for new in row for old in old_rows[offset+j] if old['column'] == new['column']]
        # Even a high-confidence OCR disagreement on a visibly stationary
        # page must be reported, rather than turning that swipe into an advance.
        if offset and any(field_conflicts(old,new) for old,new in pairs):
            candidate['reason']='overlap_field_conflict'
            continue
        if not pairs and offset == 4 and len(old_rows) == 4 and len(old_rows[-1]) == 4:
            candidates = (_text_anchor(before,after,old,geometric) for old in old_rows[-1])
        else:
            candidates = (_tail_anchor(before,after,old,new) for old,new in pairs)
        anchors = []
        for anchor in candidates:
            if (anchor is None or anchor['ncc'] < anchor['threshold']
                    or abs(anchor['upward_shift_px']-geometric) > 6
                    or abs(anchor['horizontal_shift_px']) > 3):
                candidate['reason']='name_anchor_unavailable' if anchor is None else 'name_anchor_mismatch'
                if anchor is not None:
                    candidate['rejected_anchor']=anchor
                break
            anchors.append(anchor)
        else:
            if len(anchors) < required:
                anchors.extend(_partial_row_anchors(previous,current,before,after,offset,geometric))
            shifts = [a['upward_shift_px'] for a in anchors]
            if len(anchors) < required:
                candidate['reason']='insufficient_name_anchors'
            elif max(shifts)-min(shifts) > 3:
                candidate['reason']='inconsistent_name_shifts'
            else:
                name_shift=float(np.median(shifts))
                candidate.update(reason='matched',name_shift_px=name_shift)
                matches.append((offset,geometric,name_shift))
        candidate['anchor_count']=len(anchors)
    return matches


def classify_boundary(previous, current, before, after, plan, *, extra_touch=0):
    """Verify shortened or stationary swipes against the existing tail evidence."""
    target_rows = plan['target_rows']
    touch_length = plan['touch_length_px'] + extra_touch
    current_layout = layout(current['geometry'])
    partial = not current_layout['four_rows_valid']
    candidates=[]
    short = _boundary_matches(previous, current, before, after, touch_length, range(target_rows), evidence=candidates)
    if not short and not partial:
        return None

    matches = short + _boundary_matches(previous, current, before, after, touch_length, range(target_rows,5), evidence=candidates)
    evidence={'previous_page':previous['sourceImageId'],'current_page':current['sourceImageId'],
              'matched_offsets':[offset for offset,_,_ in matches],'candidates':candidates}
    old = {(r['row'],r['column']):r['resolved'] for r in previous['records']}
    new = {(r['row'],r['column']):r['resolved'] for r in current['records']}
    stationary = (any(offset == 0 and max(abs(geometric),abs(anchor)) <= 2
                      for offset,geometric,anchor in short) and old.keys() == new.keys())
    if stationary and (len(matches) != 1 or old != new):
        raise BoundaryPositionError('stationary boundary position unconfirmed; checkpoint preserved',evidence)
    if partial and len(matches) != 1:
        # A short final row of fully visible cards can be a terminal page.
        # A clipped row still needs the existing unique position evidence.
        if (len(matches) > 1 and len(current_layout['groups'][-1]) < 4
                and all(c['completeness'] != 'partial_bottom' for c in current['geometry']['cards'])):
            return {'state': 'TERMINAL_CANDIDATES', 'basis': 'boundary',
                    'row_offsets':evidence['matched_offsets'],'position_evidence':evidence}
        raise BoundaryPositionError('partial boundary position unconfirmed; checkpoint preserved',evidence)
    if len(matches) == 1:
        return {'state': 'NO_MOVE_CANDIDATE' if stationary else 'MOVED',
                'basis': 'boundary', 'row_offset': matches[0][0]}
    return None


def classify_scan_transition(previous, current, before, after, plan, *, extra_touch=0):
    """Apply planned rows outside verified list boundaries."""
    _, current_layout = _validate_grid(current)
    boundary = classify_boundary(previous, current, before, after, plan, extra_touch=extra_touch)
    if boundary is not None:
        return boundary

    decision = {'state': 'MOVED', 'basis': 'plan', 'row_offset': plan['target_rows']}
    landing = plan.get('landing_center_interval_px')
    if not extra_touch and landing is not None:
        first = current_layout['row_centers'][0]
        decision['landing'] = {
            'first_row_center_px': first,
            'expected_center_interval_px': landing,
            'error_from_expected_center_px': first - (landing[0]+landing[1])/2,
        }
    return decision


def recovery_swipe(geometry,predecessor,*,target_rows):
    """A bounded reverse probe; its response is independent of forward calibration."""
    current=layout(geometry);prior=layout(predecessor)
    pitch=current['median_pitch_px'] if current['median_pitch_px'] is not None else prior['median_pitch_px']
    if pitch is None or not 145<=pitch<=185:raise ValueError('recovery needs calibrated row spacing')
    profile=geometry['profile'];bounds=profile['contentBounds']
    if (profile.get('imageWidth'),profile.get('imageHeight'))!=(720,1280):raise ValueError('recovery requires the supported 720x1280 frame')
    if target_rows not in (1,2):raise ValueError('bounded recovery requires one or two rows')
    # This is a reverse probe estimate, not a sample of the normal forward gain.
    touch=int(round(target_rows*pitch/CALIBRATED_GAIN));x=int(round(bounds['x']+bounds['width']/2));y=int(round(bounds['y']+.7*pitch))
    if not bounds['y']<y<y+touch<bounds['y']+bounds['height']:raise ValueError('two-row recovery exceeds the content viewport')
    return {'mode':'recovery','swipe_begin':[x,y],'swipe_end':[x,y+touch],'swipe_duration_ms':600,'swipe_end_hold_ms':600,
            'row_pitch_px':pitch,'target_rows':target_rows,'reverse_probe_gain_estimate':CALIBRATED_GAIN,'touch_length_px':touch}


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
        if rows[0][0]['row']!=0:raise ValueError('initial checkpoint has an unseen leading physical row')
        if s['four_rows_valid']:return self.accept_normal(page)
        for j,row in enumerate(rows):
            for r in row:self._append(r,j,False)
        self.next_row=len(rows);self.last_base=0;self.last_page=page;self.observation_count+=len(page['records'])
        return {'accepted':len(page['records']),'duplicates':0,'tail_new':0}
    def accept_moved(self,page,target_rows):
        if target_rows not in (1,2,4):
            raise ValueError('forward plan requires one, two or four rows')
        _,s=_validate_grid(page)
        if self.last_page is None:raise ValueError('no predecessor for moved page')
        base=self.last_base+target_rows
        # A four-row swipe after recovery can still overlap the accepted frontier.
        if target_rows==4 and s['four_rows_valid'] and base==self.next_row:
            self.accept_normal(page)
            return {'new_count':16,'duplicate_count':0,'final_count':len(self.items),
                    'physical_base':base,'row_offset':target_rows}
        return self._accept_at(page,base,target_rows)

    def _merge_footprint(self,page,base):
        # base is the physical first COMPLETE row. Keep source geometry indices
        # unchanged; enumerate consecutive complete rows only after mapping.
        validate_fields(page);duplicates=[];new=[]
        if base<0:raise ValueError('physical row cannot precede the initial checkpoint')
        for j,row in enumerate(_row_records(page)):
            for r in row:
                key=(base+j,r['column']);prior=self.positions.get(key)
                if prior is not None:
                    if any(prior[k]!=r['resolved'][k] for k in ('name','level','quality')):raise ValueError('positional duplicate has conflicting OCR fields')
                    duplicates.append({'observation':r['occurrenceId'],'prior_capture_order':prior['captureOrder'],'physical_row':key[0],'column':key[1]})
                else:new.append((r,key[0]))
        # Check the whole proposed footprint before changing a single instance.
        keys=set(self.positions)|{(row,r['column']) for r,row in new}
        last=max((row for row,col in keys),default=-1)
        for row in range(last+1):
            cols=sorted(col for rr,col in keys if rr==row)
            if cols!=(list(range(len(cols))) if row==last else [0,1,2,3]) or not cols:
                raise ValueError('page would leave a gap in the accepted inventory')
        return duplicates,new,last

    def preview_terminal(self,page,row_offset):
        """Count a possible terminal merge using the same checks as the actual merge."""
        rows,_=_validate_grid(page)
        if self.last_page is None:raise ValueError('no predecessor for terminal merge')
        base=self.last_base+row_offset
        duplicates,new,last=self._merge_footprint(page,base)
        # A confirmed short tail must cover the accepted frontier, including
        # the rightmost occupied column; a prior recovery can put it off-screen.
        if last!=base+len(rows)-1 or any((last,col) in self.positions for col in range(len(rows[-1]),4)):
            raise ValueError('terminal candidate ends before the accepted inventory frontier')
        return {'row_offset':row_offset,'duplicate_count':len(duplicates),'new_count':len(new),
                'section_count':len(self.items)+len(new)}

    def _merge_at(self,page,base,row_offset):
        duplicates,new,last=self._merge_footprint(page,base)
        for r,row in new:self._append(r,row,True)
        self.observation_count+=len(page['records']);self.next_row=last+1
        result={'sourceImageId':page['sourceImageId'],'physical_base':base,'row_offset':row_offset,'duplicate_count':len(duplicates),'new_count':len(new),
                'duplicates':duplicates,'final_count':len(self.items)}
        self.tail_diagnostics.append(result);return result
    def merge_terminal(self,page,row_offset):
        _validate_grid(page)
        if self.last_page is None:raise ValueError('no predecessor for terminal merge')
        base=self.last_base+row_offset
        return self._accept_at(page,base,row_offset)

    def _accept_at(self,page,base,row_offset):
        result=self._merge_at(page,base,row_offset)
        self.last_page=page;self.last_base=base
        return result

    def recover_page(self,unresolved,recovery,*,forward_rows,reverse_rows):
        """Merge both page footprints atomically and resume at the reverse view."""
        if self.last_base is None:
            raise ValueError('no predecessor for recovery')
        if forward_rows not in (1,2,4) or reverse_rows not in (1,2):
            raise ValueError('invalid forward or reverse recovery plan')
        pending = self.last_base + forward_rows
        base = pending - reverse_rows
        if base < 0:
            raise ValueError('physical row cannot precede the initial checkpoint')
        _validate_grid(unresolved)
        recovery_rows,_ = _validate_grid(recovery)
        if len(recovery_rows)<3 or any(len(row)!=4 for row in recovery_rows):
            raise ValueError('recovery needs at least three complete four-column rows')
        trial = copy.deepcopy(self)
        recovered = trial._merge_at(recovery,base,forward_rows-reverse_rows)
        resolved = trial._merge_at(unresolved,pending,forward_rows)
        trial.last_page = recovery
        trial.last_base = base
        self.__dict__.update(trial.__dict__)
        return {'resumed_base':base,'unresolved_base':pending,
                'new_count':recovered['new_count']+resolved['new_count'],
                'duplicate_count':recovered['duplicate_count']+resolved['duplicate_count'],
                'final_count':len(self.items),'recovery_page':recovered,'unresolved_page':resolved}


class PageStore:
    def __init__(self,out):self.out=Path(out);self.out.mkdir(exist_ok=True,parents=True);self.index=[]
    def save(self,page):
        started=time.perf_counter();name=page['sourceImageId']+'.json'
        with (self.out/name).open('x',encoding='utf-8') as f:json.dump(page,f,ensure_ascii=False,separators=(',',':'))
        entry={'file':name,'image':page['sourceImageId'],'status':page['status'],'records':len(page.get('records',[]))}
        with (self.out/'checkpoints.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        entry['save_ms']=(time.perf_counter()-started)*1000;self.index.append(entry);return entry
