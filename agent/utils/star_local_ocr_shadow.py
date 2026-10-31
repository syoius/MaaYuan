"""Synchronous Shadow helpers. No OCR Resource, Tasker, upload, or background thread."""
from __future__ import annotations
import ast,base64,hashlib,json,subprocess,sys,time,types
from pathlib import Path
import cv2
import numpy as np
REPO=Path(__file__).resolve().parents[2]

def now_ms(start):return (time.perf_counter()-start)*1000

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def read_image(path):
    image=cv2.imdecode(np.frombuffer(Path(path).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f'Cannot read {path}')
    return image

def save_image(path,image):
    ok,data=cv2.imencode('.png',image);assert ok
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(data.tobytes())

def hash_image(image):return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()

class Bridge:
    def __init__(self,runtime,stderr):
        self.err=Path(stderr).open('w',encoding='utf-8')
        self.process=subprocess.Popen(['node',str(REPO/'tools/star_local_ocr_bridge.mjs'),str(runtime)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def call(self,q):
        self.process.stdin.write(json.dumps(q,ensure_ascii=False)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('YuanStar bridge closed; inspect bridge stderr')
        reply=json.loads(line)
        if not reply['ok']:raise RuntimeError(reply)
        return reply['result']
    def geometry(self,image):
        return self.call({'op':'geometry','width':image.shape[1],'height':image.shape[0],'rgba':base64.b64encode(cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)).decode('ascii')})
    def resolve(self,kind,variants,section='main'):
        return self.call({'op':'resolve','kind':kind,'section':section,'candidates':[{'text':t['text'],'confidence':t['score'],'variant':v['variant']} for v in variants for t in v['all_results']]})
    def close(self):
        self.process.stdin.close();self.process.wait(timeout=10);self.err.close()

class Recognizer:
    def __init__(self,provider,bridge,output,*,save_debug_crops=True):
        self.provider=provider;self.bridge=bridge;self.output=Path(output);self.calls=0;self.save_debug_crops=save_debug_crops
    def field(self,image,rect,kind,source_id,card):
        start=time.perf_counter();x,y,w,h=[int(rect[k]) for k in ['x','y','width','height']]
        if min(x,y,w,h)<0 or not w or not h or x+w>image.shape[1] or y+h>image.shape[0]:raise ValueError('invalid field ROI')
        crop=image[y:y+h,x:x+w].copy();large=cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        prep=now_ms(start);variants=[]
        def invoke(label,prepared):
            result=self.provider(np.ascontiguousarray(prepared));self.calls+=1
            for token in result['all_results']:
                bx,by,bw,bh=token['box'];token['source_box']=[x+bx/3,y+by/3,bw/3,bh/3]
            variants.append({'variant':label,'prepared_shape':list(prepared.shape),**result})
        invoke('color_3x',large);res=self.bridge.resolve(kind,variants)
        best=max(variants[0]['all_results'],key=lambda z:z['score'],default={'text':'','score':0})
        value=res.get('normalized') if kind=='name' else res.get('level')
        fallback=value is None or best['score']<.95 or (kind=='level' and '级' not in best['text'])
        if fallback:
            t=time.perf_counter();gray=cv2.cvtColor(large,cv2.COLOR_BGR2GRAY);contrast=np.clip(gray.astype(np.float32)*1.7-35,0,255).astype(np.uint8);_,otsu=cv2.threshold(contrast,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);prep+=now_ms(t)
            for label,variant in [('contrast_3x',contrast),('otsu_3x',otsu)]:invoke(label,cv2.cvtColor(variant,cv2.COLOR_GRAY2BGR))
            res=self.bridge.resolve(kind,variants)
        value=res.get('normalized') if kind=='name' else res.get('level')
        max_score=max([t['score'] for v in variants for t in v['all_results']],default=0.)
        # Raw variants and Web resolver output remain separate. No hand correction.
        if self.save_debug_crops and card['rowIndex'] in [0,3]:save_image(self.output/'debug-crops'/f'{source_id}-r{card["rowIndex"]}c{card["columnIndex"]}-{kind}.png',crop)
        return {'source_rect':rect,'variants':variants,'resolution':res,'resolved':value,'confidence':res['confidence'],'max_raw_confidence':max_score,'low_confidence':max_score<.9,'fallback':fallback,'fallback_calls':len(variants)-1,'fallback_ocr_ms':sum(v['ocr_ms'] for v in variants[1:]),'preprocess_ms':prep,'ocr_ms':sum(v['ocr_ms'] for v in variants),'wall_ms':now_ms(start)}
    def image(self,image,source_id,geometry=None,geometry_wall_ms=0.0):
        image=np.ascontiguousarray(image).copy();image.flags.writeable=False;original=hash_image(image);start=time.perf_counter();t=time.perf_counter();reused=geometry is not None
        g=geometry if reused else self.bridge.geometry(image);geometry_wall=geometry_wall_ms if reused else now_ms(t);records=[]
        for c in g['cards']:
            if c['completeness']!='complete':continue
            fields={k:self.field(image,c[k+'Rect'],k,source_id,c) for k in ['name','level']}
            q=c['qualityEvidence'];occ=f'{source_id}:r{c["rowIndex"]}c{c["columnIndex"]}'
            record={'occurrenceId':occ,'sourceImageId':source_id,'section':'main','row':c['rowIndex'],'column':c['columnIndex'],'card':c,'fields':fields,'quality':q,'resolved':{'name':fields['name']['resolved'],'level':fields['level']['resolved'],'quality':q['quality']},
              'businessCandidate':{'card_id':occ,'source_image':source_id,'source_position':f'r{c["rowIndex"]}c{c["columnIndex"]}','page_type':'main','recognized_name':fields['name']['resolved'],'recognized_level':fields['level']['resolved'],'recognized_quality':q['quality'],'final_name':fields['name']['resolved'],'final_level':fields['level']['resolved'],'final_quality':q['quality'],'is_complete_card':True,'manual_override':False,'overlap_duplicate_of':None}}
            records.append(record)
        assert hash_image(image)==original
        fs=[v for r in records for v in r['fields'].values()]
        return {'sourceImageId':source_id,'section':'main','shape':list(image.shape),'pixel_sha256':original,'geometry':g,'records':records,'timing':{'geometry_compute_ms':g['geometryMs'],'geometry_transport_wall_ms':geometry_wall,'quality_ms':g['qualityMs'],'name_ocr_ms':sum(r['fields']['name']['ocr_ms'] for r in records),'level_ocr_ms':sum(r['fields']['level']['ocr_ms'] for r in records),'preprocess_ms':sum(f['preprocess_ms'] for f in fs),'fallback_count':sum(f['fallback'] for f in fs),'fallback_calls':sum(f['fallback_calls'] for f in fs),'fallback_ocr_ms':sum(f['fallback_ocr_ms'] for f in fs),'page_wall_ms':now_ms(start)+(geometry_wall if reused else 0.0),'geometry_reused':reused}}

def motion_functions():
    sys.path.insert(0,str(REPO/'tools'));import star_swipe_endhold
    return star_swipe_endhold.load_motion_algorithms(REPO)

def image_anchor(before,after,cards):
    rows=[]
    for c in cards:
        rects=[c[k] for k in ['discRect','nameRect','levelRect']];x=min(z['x'] for z in rects);y=min(z['y'] for z in rects);x2=max(z['x']+z['width'] for z in rects);y2=max(z['y']+z['height'] for z in rects)
        scores=cv2.matchTemplate(cv2.cvtColor(after[280:1040,x:x2],cv2.COLOR_BGR2GRAY),cv2.cvtColor(before[y:y2,x:x2],cv2.COLOR_BGR2GRAY),cv2.TM_CCOEFF_NORMED).reshape(-1);pos=int(np.argmax(scores));rows.append({'column':c['columnIndex'],'source_rect':[x,y,x2-x,y2-y],'target_y':pos+280,'shift_y_px':float(y-pos-280),'score':float(scores[pos])})
    shifts=[z['shift_y_px'] for z in rows];accepted=len(rows)==4 and all(z['score']>=.97 for z in rows) and max(shifts)-min(shifts)<=2
    return {'accepted':accepted,'shift_y_px':float(np.median(shifts)) if accepted else None,'columns':rows}

def pair_status(old,new,geometry_ok):
    conflicts=[k for k in ['name','level','quality'] if old[k] is not None and new[k] is not None and old[k]!=new[k]]
    unknown=[k for k in ['name','level','quality'] if old[k] is None or new[k] is None]
    return ('conflict' if conflicts else 'unconfirmed' if unknown else 'matched' if geometry_ok else 'geometry_failed'),conflicts,unknown

def match_pair(a,b,im_a,im_b):
    probe=motion_functions();crop=lambda im:im[280:1040,40:680];ga=a['geometry'];gb=b['geometry']
    raw=probe.estimate_vertical_motion(crop(im_a),crop(im_b),(250,700),(30,700))
    source=[r['card'] for r in a['records'] if r['row']==max(x['row'] for x in a['records'])]
    anchor=image_anchor(im_a,im_b,source);shift=anchor['shift_y_px'];pairs=[];sem=None;default=None
    if shift is not None:
        default=probe.evaluate_semantic_row_overlap(crop(im_a),crop(im_b),shift,166.5)
        sem=probe.evaluate_semantic_row_overlap(crop(im_a),crop(im_b),shift,166.5,prev_row_centers=[v-280 for v in ga['rowCenters']],candidate_row_centers=[v-280 for v in gb['rowCenters']])
        for old in a['records']:
            matches=[new for new in b['records'] if new['column']==old['column'] and abs(new['card']['circle']['centerY']-(old['card']['circle']['centerY']-shift))<=3]
            if len(matches)!=1:continue
            new=matches[0]
            rects=[new['card'][k] for k in ['discRect','nameRect','levelRect']];clearance=min(min(r['y']-280,1040-r['y']-r['height']) for r in rects)
            geometry_ok=a['section']==b['section'] and sem['semantic_overlap_state']=='definitely_full_row' and clearance>=10
            status,conflicts,unknown=pair_status(old['resolved'],new['resolved'],geometry_ok)
            pairs.append({'a':old['occurrenceId'],'b':new['occurrenceId'],'section':a['section'],'column':old['column'],'row_a':old['row'],'row_b':new['row'],'values_a':old['resolved'],'values_b':new['resolved'],'status':status,'conflict_fields':conflicts,'unknown_fields':unknown,'field_clearance_px':clearance,'geometry_ok':geometry_ok})
    matched={p['b']:p['a'] for p in pairs if p['status']=='matched'}
    groups=[{'observations':[r['occurrenceId']],'resolved':r['resolved'],'businessCandidate':dict(r['businessCandidate'])} for r in a['records']]
    for item in b['records']:
        if item['occurrenceId'] in matched:
            next(g for g in groups if matched[item['occurrenceId']] in g['observations'])['observations'].append(item['occurrenceId']);item['businessCandidate']['overlap_duplicate_of']=matched[item['occurrenceId']]
        else:groups.append({'observations':[item['occurrenceId']],'resolved':item['resolved'],'businessCandidate':dict(item['businessCandidate'])})
    # No global name/level de-duplication and no generated business instanceId.
    return {'sourceImageIdA':a['sourceImageId'],'sourceImageIdB':b['sourceImageId'],'adjacent_pair_only':True,'raw_motion':{'selected_shift_y_px':raw.shift_y,'confidence':raw.confidence,'selected_hypothesis':raw.selected_hypothesis,'motion_hypotheses':raw.motion_hypotheses},'image_anchor':anchor,'semantic_default_lattice':default,'semantic_detected_centers':sem,'pairs':pairs,'deduplicated_count':len(matched),'candidate_count_before_merge':len(a['records'])+len(b['records']),'different_position_candidate_count':len(groups),'conflict_count':sum(p['status']=='conflict' for p in pairs),'groups':groups,'no_business_instance_ids_generated':True}
