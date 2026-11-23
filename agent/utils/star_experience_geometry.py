"""Native experience-page geometry for MaaYuan's 720x1280 BGR capture."""
from __future__ import annotations

import math
import time

import numpy as np

from utils.star_experience_postprocess import EXPERIENCE_CANONICAL, classify_experience_kind
from utils.star_geometry import _js_round, create_screenshot_profile
from utils.star_grid_geometry import _median, edge_map, ring_score
from utils.star_quality_postprocess import _rgb_to_opencv_hsv


COUNT_X_OFFSET = .40
COUNT_Y_OFFSET = .78
COUNT_WIDTH = .60
COUNT_HEIGHT = .24
_COLUMN_RATIOS = (.165, .38, .595)


def experience_count_rect(icon):
    return dict(x=icon['x']+math.trunc(icon['width']*COUNT_X_OFFSET),
                y=icon['y']+math.trunc(icon['height']*COUNT_Y_OFFSET),
                width=max(1,math.trunc(icon['width']*COUNT_WIDTH)),
                height=max(1,math.trunc(icon['height']*COUNT_HEIGHT)))


def classify_experience_completeness(rect, bounds):
    if rect['x'] < bounds['x'] or rect['x']+rect['width'] > bounds['x']+bounds['width']:
        return 'invalid'
    if rect['y'] < bounds['y']:
        return 'partial_top'
    if rect['y']+rect['height'] > bounds['y']+bounds['height']:
        return 'partial_bottom'
    return 'complete'


def selected_experience_tab_evidence(image, viewport):
    image_height, image_width = image.shape[:2]
    top = viewport['y'] + math.trunc(viewport['height']*.07)
    bottom = viewport['y'] + math.trunc(viewport['height']*.24)
    step = 2
    width = max(1,math.ceil(viewport['width']/step))
    height = max(1,math.ceil((bottom-top)/step))
    mask = bytearray(width*height)
    for ly in range(height):
        y = min(image_height-1,top+ly*step)
        for lx in range(width):
            x = min(image_width-1,viewport['x']+lx*step)
            blue, green, red = map(int,image[y,x])
            hue, saturation, value = _rgb_to_opencv_hsv(red,green,blue)
            if 8 <= hue <= 42 and 25 <= saturation <= 210 and value >= 120:
                mask[ly*width+lx] = 1

    visited = bytearray(len(mask))
    choices = []
    for seed, present in enumerate(mask):
        if not present or visited[seed]:
            continue
        queue = [seed]
        visited[seed] = 1
        min_x, min_y, max_x, max_y = width, height, 0, 0
        # Appending to the list during iteration preserves the JS BFS order.
        for point in queue:
            y, x = divmod(point,width)
            min_x, max_x = min(min_x,x), max(max_x,x)
            min_y, max_y = min(min_y,y), max(max_y,y)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    nx, ny = x+dx, y+dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    neighbor = ny*width+nx
                    if mask[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        queue.append(neighbor)
        cw, ch = (max_x-min_x+1)*step, (max_y-min_y+1)*step
        if cw >= viewport['width']*.14 and 1.4 <= cw/max(1,ch) <= 8 and ch <= viewport['height']*.10:
            choices.append(dict(area=cw*ch, center=(min_x+max_x+1)*step/2,
                                rect=dict(x=viewport['x']+min_x*step,y=top+min_y*step,width=cw,height=ch)))
    best = max(choices,key=lambda c:c['area'],default=None)
    if best is None or best['center'] < viewport['width']*.65:
        return dict(selected=False,confidence=0,evidence=[])
    return dict(selected=True,confidence=.82,rect=best['rect'],evidence=['selected_tab_visual:experience'])


def experience_search_bounds(profile, selected_tab):
    v = profile['viewport']
    rect = (selected_tab or {}).get('rect')
    tab_bottom = rect['y']+rect['height'] if rect is not None else v['y']+v['height']*.18
    top = max(profile['contentBounds']['y'],math.ceil(tab_bottom+v['width']*.025))
    bottom = min(v['y']+v['height']*.75,math.ceil(top+v['width']*.32))
    return dict(x=v['x'],y=top,width=v['width'],height=max(0,bottom-top))


def _anchors(viewport):
    return [viewport['x']+viewport['width']*ratio for ratio in _COLUMN_RATIOS]


def _nearest_slot(anchors, x):
    return min(range(len(anchors)),key=lambda i:abs(anchors[i]-x))


def normalize_experience_row(row, profile):
    if not row:
        return []
    v = profile['viewport']
    anchors = _anchors(v)
    slots = {}
    for circle in row:
        slot = _nearest_slot(anchors,circle['centerX'])
        if abs(anchors[slot]-circle['centerX']) > v['width']*.075:
            continue
        if slot not in slots or circle['score'] > slots[slot]['score']:
            slots[slot] = circle
    selected = sorted(slots.items())
    radius = _js_round(_median(c['radius'] for _,c in selected))
    center_y = _js_round(_median(c['centerY'] for _,c in selected))
    return [{**c,'centerX':_js_round(anchors[slot]),'centerY':center_y,'radius':radius} for slot,c in selected]


def dominant_row(circles, profile):
    clusters = []
    for circle in sorted(circles,key=lambda c:c['centerY']):
        cluster = next((row for row in clusters
                        if abs(circle['centerY']-_median(c['centerY'] for c in row))
                        <= max(18,_median(c['radius'] for c in row)*.38)),None)
        if cluster is None:
            clusters.append([circle])
        else:
            cluster.append(circle)

    def rank(row):
        # JS reduce adds scores in normalized slot order.
        score = 0
        for circle in row:
            score += circle['score']
        return len(row),score

    selected = max((normalize_experience_row(r,profile) for r in clusters),key=rank,default=[])
    separated = []
    for circle in sorted(selected,key=lambda c:-c['score']):
        if any(abs(c['centerX']-circle['centerX']) <= min(c['radius'],circle['radius'])*1.15 for c in separated):
            continue
        separated.append(circle)
    return sorted(separated,key=lambda c:c['centerX'])


def find_experience_circles(image, profile, selected_tab):
    bounds = experience_search_bounds(profile,selected_tab)
    edges = edge_map(image)
    height,width = image.shape[:2]
    expected_radius = profile['viewport']['width']*.068
    min_radius = max(18,_js_round(expected_radius*.82))
    max_radius = max(min_radius+2,_js_round(expected_radius*1.18))
    radius_step = max(3,_js_round(expected_radius*.08))
    x_step = max(6,_js_round(expected_radius*.14))
    y_step = max(4,_js_round(expected_radius*.10))
    raw = []
    for y in range(bounds['y']+min_radius,math.floor(bounds['y']+bounds['height']-min_radius)+1,y_step):
        for x in range(bounds['x']+min_radius,bounds['x']+bounds['width']-min_radius+1,x_step):
            best_radius, best_score = min_radius, 0
            for radius in range(min_radius,max_radius+1,radius_step):
                score = ring_score(edges,width,height,x,y,radius)
                if score > best_score:
                    best_radius, best_score = radius,score
            if best_score >= 21:
                raw.append(dict(centerX=x,centerY=y,radius=best_radius,score=_js_round(best_score*100)/100,source='canvas_ring'))
    proposals = []
    for proposal in sorted(raw,key=lambda c:-c['score']):
        if any(math.hypot(c['centerX']-proposal['centerX'],c['centerY']-proposal['centerY'])
               <= min(c['radius'],proposal['radius'])*1.15 for c in proposals):
            continue
        proposals.append(proposal)
        if len(proposals) >= 48:
            break
    row = dominant_row(proposals,profile)
    return dict(proposals=proposals,row=row,bounds=bounds,
                coverage=experience_coverage(image,profile,selected_tab,row,bounds))


def build_experience_candidates(circles, bounds):
    candidates = []
    for index,circle in enumerate(circles):
        r = circle['radius']
        size = max(1,math.trunc(r*2))
        icon = dict(x=math.trunc(circle['centerX']-r),y=math.trunc(circle['centerY']-r),width=size,height=size)
        completeness = classify_experience_completeness(icon,bounds)
        candidates.append(dict(itemId=f'experience-{index+1}',index=index,iconRect=icon,countRect=experience_count_rect(icon),
                               circle=circle,completeness=completeness,
                               evidence=['experience_icon_complete' if completeness=='complete' else completeness]))
    return candidates


def experience_coverage(image, profile, tab, row, bounds):
    warnings = []
    v = profile['viewport']
    if not tab or not tab.get('selected') or not tab.get('rect'):
        warnings.append('experience_tab_bounds_unverified')
    # Preserve the oracle's evidence gate if black bars alter the viewport.
    if profile['profileId'] not in ('phone_portrait_v1','phone_9_16_v1'):
        warnings.append('experience_portrait_profile_unverified')
    if len(row) > 3:
        warnings.append('experience_too_many_icons')
    anchors = _anchors(v)
    if any(_nearest_slot(anchors,c['centerX']) != i for i,c in enumerate(row)):
        warnings.append('experience_row_not_left_compacted')
    bottom = bounds['y']+bounds['height']
    if any(c['centerY']-c['radius'] < bounds['y'] or c['centerY']+c['radius'] > bottom for c in row):
        warnings.append('experience_icon_outside_content')
    unexplained = 0
    # All three channels are compared independently. Their permutation from
    # RGB to BGR leaves max(abs(channel - channel_median)) unchanged.
    right = v['x']+v['width']
    for y in range(bounds['y'],math.ceil(bottom),2):
        samples = image[y,math.floor(v['x']+v['width']*.75):right:4]
        bg = [_median(samples[:,channel].tolist()) for channel in range(3)]
        for x in range(bounds['x'],bounds['x']+bounds['width'],2):
            if any(abs(x-c['centerX'])<=c['radius']*1.2 and abs(y-c['centerY'])<=c['radius']*1.2 for c in row):
                continue
            if max(abs(int(pixel)-value) for pixel,value in zip(image[y,x],bg)) > 45:
                unexplained += 4
    limit = max(80,v['width']*v['width']*.0015)
    if unexplained > limit:
        warnings.append('experience_content_unaccounted')
    complete = not warnings
    return dict(complete=complete,emptyConfirmed=complete and not row,unexplainedPixels=unexplained,
                unexplainedLimit=limit,warnings=warnings,
                evidence=['selected_tab_bounds','complete_compacted_row','no_unaccounted_foreground'] if complete else [])


def recognize_experience_geometry(image):
    if not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.shape != (1280,720,3):
        raise ValueError('720x1280 uint8 BGR image required')
    started = time.perf_counter()
    profile = create_screenshot_profile(image)
    tab = selected_experience_tab_evidence(image,profile['viewport'])
    found = find_experience_circles(image,profile,tab)
    candidates = build_experience_candidates(found['row'],found['bounds'])
    for candidate in candidates:
        quality = classify_experience_kind(image,candidate['iconRect'])
        candidate.update(quality=quality,canonicalName=EXPERIENCE_CANONICAL.get(quality['kind']))
    # Keep the existing payload; raw proposals and row evidence are available via find.
    return dict(profile=profile,tab=tab,bounds=found['bounds'],coverage=found['coverage'],
                candidates=candidates,geometryMs=(time.perf_counter()-started)*1000)
