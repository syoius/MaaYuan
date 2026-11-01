"""Experience counts via YuanStar geometry/postprocess and the shared Context provider."""
from __future__ import annotations
import time
import cv2
import numpy as np
from utils.star_local_ocr_shadow import now_ms,hash_image

KINDS=('orange','purple','white')
# Match the existing local ROI recognizer: retry below .95, review below .9.
MIN_COUNT_CONFIDENCE=.9
RETRY_COUNT_CONFIDENCE=.95
# YuanStar hue confidence is a distance score, not a model probability.
# A half-scale hue match plus one unique icon per kind is required.
MIN_KIND_CONFIDENCE=.5


def empty_experience(reason):
    return {'structured':False,'complete':False,'captured':False,
            **{k+'Count':None for k in KINDS},'warnings':[reason],'items':[]}


def experience_summary(result):
    """Keep raw fields once, in experience-results.json, like star page checkpoints."""
    return {**{k:v for k,v in result.items() if k not in ('items','geometry')},
            'detailsFile':'experience-results.json'}


def badge_otsu(image):
    """White count glyphs over a colored disc: exclude color before tight Otsu.

    Used only after the ordinary color/contrast recognition failed its retry gate.
    The source ROI and inferred foreground box remain in the result evidence.
    """
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,1]<=40)&(hsv[:,:,2]>=180)).astype(np.uint8)*255
    ys,xs=np.where(mask>0)
    if not len(xs):return None,None
    x,y,x2,y2=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
    glyph=mask[y:y2,x:x2]
    border=max(4,glyph.shape[0]//5)
    # The model receives dark text on a light, padded background.
    prepared=cv2.copyMakeBorder(255-glyph,border,border,border,border,cv2.BORDER_CONSTANT,value=255)
    return cv2.cvtColor(prepared,cv2.COLOR_GRAY2BGR),[x,y,x2-x,y2-y]


def recognize_experience(image,provider,bridge):
    """No zero inference: a count requires one complete, classified, confident icon."""
    started=time.perf_counter();original=hash_image(image)
    geometry=bridge.experience_geometry(image);items=[];warnings=[]
    if not geometry['tab']['selected']:warnings.append('experience_tab_selection_unverified')
    for candidate in geometry['candidates']:
        quality=candidate['quality'];kind=quality['kind'];variants=[];resolved={'count':None,'confidence':0.}
        item={'candidate':candidate,'kind':kind,'canonicalName':candidate.get('canonicalName'),'kindConfidence':quality['confidence'],
              'count':None,'countConfidence':0.,'variants':variants,'warnings':[],'fallbackCalls':0}
        items.append(item)
        if candidate['completeness']!='complete':item['warnings'].append('experience_icon_incomplete')
        if kind not in KINDS or quality['confidence']<MIN_KIND_CONFIDENCE:item['warnings'].append('experience_icon_unclassified')
        r=candidate['countRect'];x,y,w,h=(int(r[k]) for k in ('x','y','width','height'))
        if min(x,y)<0 or min(w,h)<=0 or x+w>image.shape[1] or y+h>image.shape[0]:item['warnings'].append('experience_invalid_count_roi')
        if item['warnings']:continue
        crop=image[y:y+h,x:x+w].copy();large=cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        def invoke(label,prepared):
            variants.append({'variant':label,**provider(np.ascontiguousarray(prepared))})
            return bridge.experience_count(variants)
        def accepted(value):
            return value.get('count') is not None and value.get('confidence',0)>=MIN_COUNT_CONFIDENCE
        resolved=invoke('color_3x',large)
        if resolved.get('count') is None or resolved.get('confidence',0)<RETRY_COUNT_CONFIDENCE:
            gray=cv2.cvtColor(large,cv2.COLOR_BGR2GRAY)
            contrast=np.clip(gray.astype(np.float32)*1.7-35,0,255).astype(np.uint8)
            resolved=invoke('contrast_3x',cv2.cvtColor(contrast,cv2.COLOR_GRAY2BGR));item['fallbackCalls']+=1
            if resolved.get('count') is None or resolved.get('confidence',0)<RETRY_COUNT_CONFIDENCE:
                otsu,foreground=badge_otsu(large)
                item['foregroundBox3x']=foreground
                if otsu is not None:
                    resolved=invoke('badge_otsu_3x',otsu);item['fallbackCalls']+=1
        # Conflicting confident readings remain reviewable rather than selecting silently.
        readings={bridge.experience_count([v])['count'] for v in variants
                  if accepted(bridge.experience_count([v]))}
        item.update(resolution=resolved,countConfidence=resolved['confidence'])
        if len(readings)>1:item['warnings'].append('experience_count_conflict')
        elif not accepted(resolved):item['warnings'].append('experience_count_unconfirmed')
        else:item['count']=resolved['count']
    counts={}
    for kind in KINDS:
        matches=[i for i in items if i['kind']==kind]
        counts[kind+'Count']=matches[0]['count'] if len(matches)==1 else None
        if len(matches)!=1:warnings.append('experience_kind_missing_or_duplicate:'+kind)
    warnings.extend(w for i in items for w in i['warnings'])
    if any(v is None for v in counts.values()):warnings.append('experience_counts_incomplete')
    complete=not warnings
    assert hash_image(image)==original
    return {'structured':complete,'complete':complete,'captured':True,**counts,'warnings':warnings,
            'items':items,'geometry':geometry,'pixel_sha256':original,
            'timing':{'wall_ms':now_ms(started),'geometry_ms':geometry['geometryMs'],
                      'ocr_ms':sum(v['ocr_ms'] for i in items for v in i['variants']),
                      'ocr_calls':sum(len(i['variants']) for i in items),
                      'fallback_calls':sum(i['fallbackCalls'] for i in items)}}
