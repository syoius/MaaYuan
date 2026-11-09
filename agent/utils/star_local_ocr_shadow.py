"""Synchronous Shadow helpers. No OCR Resource, Tasker, upload, or background thread."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
from utils.star_geometry import create_screenshot_profile, refine_star_content_bounds
from utils.star_grid_geometry import build_card_candidates, find_circle_proposals
from utils.star_ocr_postprocess import StarFieldResolver, _has_suffix
from utils.star_quality_postprocess import quality_rect_for_card, recognize_quality
REPO=Path(__file__).resolve().parents[2]
PACKAGED_RUNTIME=REPO/'agent/runtime/star_ocr'


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

def _field_candidates(variants):
    return [{'text': t['text'], 'confidence': t['score'], 'variant': v['variant']}
            for v in variants for t in v['all_results']]


class NativeStarOcr:
    """Main/support recognition backend using the existing grid and field contracts."""

    def __init__(self, runtime, section):
        self.field_resolver = StarFieldResolver(runtime)
        self.section = section

    def resolve(self, kind, variants):
        return self.field_resolver.resolve(kind, _field_candidates(variants), self.section())

    @staticmethod
    def geometry(image):
        if (not isinstance(image, np.ndarray) or image.shape != (1280, 720, 3)
                or image.dtype != np.uint8):
            raise ValueError('720x1280 uint8 BGR image required')
        start = time.perf_counter()
        phase = start
        base = create_screenshot_profile(image)
        timings = {'profile_viewport_ms': now_ms(phase)}
        # The detector searches from the base profile; refined bounds apply to the cards.
        proposals = find_circle_proposals(image, base, timings)
        phase = time.perf_counter()
        profile = refine_star_content_bounds(base, layout_hint='maayuan_mumu')
        timings['content_bounds_ms'] = now_ms(phase)
        phase = time.perf_counter()
        cards = build_card_candidates(proposals, profile)
        timings['row_cluster_and_rois_ms'] = now_ms(phase)
        geometry_ms = now_ms(start)
        phase = time.perf_counter()
        for card in cards:
            card['qualityEvidence'] = {**recognize_quality(image, card),
                                       'source_rect': quality_rect_for_card(card)}
        timings['quality_ms'] = now_ms(phase)
        rows = {}
        counts = dict.fromkeys(('complete', 'partial_top', 'partial_bottom', 'invalid'), 0)
        for card in cards:
            rows.setdefault(card['rowIndex'], []).append(card['circle']['centerY'])
            counts[card['completeness']] += 1
        centers = [sum(ys) / len(ys) for ys in rows.values()]
        return {'profile': profile, 'cards': cards, 'geometryMs': geometry_ms,
                'qualityMs': timings['quality_ms'], 'rowCenters': centers,
                'timingBreakdown': {**timings, 'full_geometry_ms': now_ms(start)},
                'rowPitches': [b - a for a, b in zip(centers, centers[1:])],
                'completenessCounts': counts}


def _can_finish_field(backend, kind, variants, resolution, min_raw):
    """Stop only when the latest variant independently supports the resolved value."""
    key = 'normalized' if kind == 'name' else 'level'
    value = resolution.get(key)
    # Keep the same parse confidence floor as inventory's _reliable_field.
    if value is None or resolution['confidence'] < .80 or resolution.get('reasons'):
        return False
    latest = variants[-1]
    latest_resolution = resolution if len(variants) == 1 else backend.resolve(kind, [latest])
    if latest_resolution.get(key) != value or latest_resolution['confidence'] < .80:
        return False
    if kind == 'level' and not _has_suffix(latest_resolution['raw']):
        return False

    reliable_read = False
    for variant in variants:
        for token in variant['all_results']:
            candidate = backend.resolve(kind, [{'variant': variant['variant'], 'all_results': [token]}])
            parsed = candidate.get(key)
            if parsed is None:
                continue
            if parsed != value:
                return False
            if (variant is latest and token['score'] >= min_raw
                    and (kind == 'name' or _has_suffix(token['text']))):
                reliable_read = True
    return reliable_read


class Recognizer:
    def __init__(self,provider,backend,output,*,save_debug_crops=True):
        self.provider=provider;self.backend=backend;self.output=Path(output);self.calls=0;self.save_debug_crops=save_debug_crops
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
        invoke('color_3x',large);res=self.backend.resolve(kind,variants)
        fallback=not _can_finish_field(self.backend,kind,variants,res,.90)
        if fallback:
            t=time.perf_counter();gray=cv2.cvtColor(large,cv2.COLOR_BGR2GRAY);contrast=np.clip(gray.astype(np.float32)*1.7-35,0,255).astype(np.uint8);prepared=cv2.cvtColor(contrast,cv2.COLOR_GRAY2BGR);prep+=now_ms(t)
            invoke('contrast_3x',prepared);res=self.backend.resolve(kind,variants)
            if not _can_finish_field(self.backend,kind,variants,res,.95):
                t=time.perf_counter();_,otsu=cv2.threshold(contrast,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);prepared=cv2.cvtColor(otsu,cv2.COLOR_GRAY2BGR);prep+=now_ms(t)
                invoke('otsu_3x',prepared);res=self.backend.resolve(kind,variants)
        value=res.get('normalized') if kind=='name' else res.get('level')
        max_score=max([t['score'] for v in variants for t in v['all_results']],default=0.)
        # Raw variants and native resolver output remain separate. No hand correction.
        if self.save_debug_crops and card['rowIndex'] in [0,3]:save_image(self.output/'debug-crops'/f'{source_id}-r{card["rowIndex"]}c{card["columnIndex"]}-{kind}.png',crop)
        return {'source_rect':rect,'variants':variants,'resolution':res,'resolved':value,'confidence':res['confidence'],'max_raw_confidence':max_score,'low_confidence':max_score<.9,'fallback':fallback,'fallback_calls':len(variants)-1,'fallback_ocr_ms':sum(v['ocr_ms'] for v in variants[1:]),'preprocess_ms':prep,'ocr_ms':sum(v['ocr_ms'] for v in variants),'wall_ms':now_ms(start)}
    def image(self,image,source_id,geometry=None,geometry_wall_ms=0.0):
        image=np.ascontiguousarray(image).copy();image.flags.writeable=False;original=hash_image(image);start=time.perf_counter();t=time.perf_counter();reused=geometry is not None
        g=geometry if reused else self.backend.geometry(image);geometry_wall=geometry_wall_ms if reused else now_ms(t);records=[]
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

def pair_status(old,new,geometry_ok):
    conflicts=[k for k in ['name','level','quality'] if old[k] is not None and new[k] is not None and old[k]!=new[k]]
    unknown=[k for k in ['name','level','quality'] if old[k] is None or new[k] is None]
    return ('conflict' if conflicts else 'unconfirmed' if unknown else 'matched' if geometry_ok else 'geometry_failed'),conflicts,unknown
