"""Native experience counts with the shared Context provider."""
from __future__ import annotations
import time
import cv2
import numpy as np
from utils.star_local_ocr_shadow import now_ms,hash_image
from utils.star_experience_geometry import recognize_experience_geometry
from utils.star_experience_postprocess import aggregate_experience,confirm_experience_count,resolve_experience_count

KINDS=('orange','purple','white')
# Keep the existing retry gate even when two weaker variants agree.
RETRY_COUNT_CONFIDENCE=.95
# YuanStar hue confidence is a distance score, not a model probability.
# A half-scale hue match is required for every present icon; absence needs page coverage.
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


def _count_candidates(variants):
    return [{'text':t['text'],'confidence':t['score'],'variant':v['variant']}
            for v in variants for t in v['all_results']]


def recognize_experience(image,provider):
    """Present items need OCR; absent kinds become zero only on a confirmed page."""
    started=time.perf_counter();original=hash_image(image)
    geometry=recognize_experience_geometry(image);items=[];warnings=[]
    coverage=geometry.get('coverage',{})
    selected_tab=geometry['tab']['selected']
    coverage_complete=coverage.get('complete',False)
    if not selected_tab:warnings.append('experience_tab_selection_unverified')
    if not coverage_complete:warnings.append('experience_page_coverage_unconfirmed')
    warnings.extend(coverage.get('warnings',[]))
    page_valid=selected_tab and coverage_complete
    viewport_cropped=not coverage_complete
    for candidate in geometry['candidates']:
        quality=candidate['quality'];kind=quality['kind'];variants=[];resolved={'count':None,'confidence':0.}
        item={'candidate':candidate,'kind':kind,'canonicalName':candidate.get('canonicalName'),'kindConfidence':quality['confidence'],
              'count':None,'countConfidence':0.,'variants':variants,'warnings':[],'fallbackCalls':0}
        items.append(item)
        if candidate['completeness']!='complete':item['warnings'].append('experience_icon_incomplete')
        r=candidate['countRect'];x,y,w,h=(int(r[k]) for k in ('x','y','width','height'))
        if min(x,y)<0 or min(w,h)<=0 or x+w>image.shape[1] or y+h>image.shape[0]:item['warnings'].append('experience_invalid_count_roi')
        bounds=geometry.get('bounds')
        if bounds and not (x>=bounds['x'] and y>=bounds['y'] and x+w<=bounds['x']+bounds['width'] and y+h<=bounds['y']+bounds['height']):
            item['warnings'].append('experience_count_outside_content')
        icon=candidate.get('iconRect')
        if bounds and icon and not (icon['x']>=bounds['x'] and icon['y']>=bounds['y'] and icon['x']+icon['width']<=bounds['x']+bounds['width'] and icon['y']+icon['height']<=bounds['y']+bounds['height']):
            item['warnings'].append('experience_icon_outside_content')
        # Only geometry warnings above are evidence of incomplete coverage.
        viewport_cropped=viewport_cropped or bool(item['warnings'])
        if not page_valid:item['warnings'].append('experience_page_unverified')
        if kind not in KINDS or quality['confidence']<MIN_KIND_CONFIDENCE:item['warnings'].append('experience_icon_unclassified')
        if item['warnings']:continue
        crop=image[y:y+h,x:x+w].copy();large=cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        def invoke(label,prepared):
            variants.append({'variant':label,**provider(np.ascontiguousarray(prepared))})
            return resolve_experience_count(_count_candidates(variants))
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
        count,warning=confirm_experience_count(_count_candidates(variants))
        item.update(resolution=resolved,countConfidence=resolved['confidence'],count=count)
        if warning:item['warnings'].append(warning)
    # Reuse the native YuanStar aggregate rather than a second three-kind
    # cardinality rule. Sort by observed position, never by the expected kind.
    ordered=sorted(items,key=lambda i:i['candidate']['countRect']['x'])
    aggregate=aggregate_experience([
        {'kind':i['kind'] if i['kindConfidence']>=MIN_KIND_CONFIDENCE else None,
         'count':i['count'],'kindConfidence':i['kindConfidence'],'countConfidence':i['countConfidence'],
         'status':'excluded_partial' if i['candidate']['completeness']!='complete' else ('accepted' if not i['warnings'] else 'needs_review')}
        for i in ordered],{'selectedTab':selected_tab,
        'viewportCropped':selected_tab and viewport_cropped,
        'emptyConfirmed':coverage.get('emptyConfirmed',False)})
    counts={k+'Count':aggregate[k+'Count'] for k in KINDS}
    warnings.extend(w for i in items for w in i['warnings'])
    warnings.extend(aggregate['warnings'])
    if any(v is None for v in counts.values()):warnings.append('experience_counts_incomplete')
    warnings=list(dict.fromkeys(warnings));complete=aggregate['complete'] and not warnings
    assert hash_image(image)==original
    return {'structured':complete,'complete':complete,'captured':True,**counts,'warnings':warnings,
            'items':items,'geometry':geometry,'pixel_sha256':original,
            'timing':{'wall_ms':now_ms(started),'geometry_ms':geometry['geometryMs'],
                      'ocr_ms':sum(v['ocr_ms'] for i in items for v in i['variants']),
                      'ocr_calls':sum(len(i['variants']) for i in items),
                      'fallback_calls':sum(i['fallbackCalls'] for i in items)}}
