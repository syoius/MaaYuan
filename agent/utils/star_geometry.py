"""Native profiles, content bounds and card ROIs from the retained star OCR JS.

Images follow MaaYuan's existing nonempty uint8 BGR ndarray convention. Output
keys and evidence retain the original contract. Ring detection and quality are separate.
"""
from __future__ import annotations

import math

import numpy as np


PHONE_PROFILE = dict(
    profileId="phone_portrait_v1", deviceKind="phone", ratioMin=.39, ratioMax=.50,
    columnCenters=[.165, .380, .597, .812], gridRegion=[.05, .18, .90, .94],
    rowSpacingRange=[.075, .135], bottomSafeY=.89, radiusRatio=.088,
    horizontalSearchRatio=0,
)
PHONE_9_16_PROFILE = {
    **PHONE_PROFILE, "profileId": "phone_9_16_v1", "ratioMin": .50,
    "ratioMax": .57, "radiusRatio": .079,
}
TABLET_PROFILE = {
    **PHONE_PROFILE, "profileId": "tablet_portrait_v1", "deviceKind": "tablet",
    "ratioMin": .57, "ratioMax": .78, "rowSpacingRange": [.075, .145],
    "radiusRatio": .068, "horizontalSearchRatio": .09,
}
FALLBACK_PROFILE = {
    **PHONE_PROFILE, "profileId": "unknown_portrait_fallback", "deviceKind": "unknown",
    "ratioMin": .25, "ratioMax": .90, "rowSpacingRange": [.070, .150],
    "bottomSafeY": .87, "radiusRatio": .078, "horizontalSearchRatio": .09,
}
MAAYUAN_MUMU_LAYOUT = dict(
    imageWidth=720, imageHeight=1280,
    contentTopRatio=272 / 1280, contentBottomRatio=1044 / 1280,
)


def _js_round(value):
    # Math.round chooses the upper integer at a tie, unlike Python round.
    lower = math.floor(value)
    return lower + (value - lower >= .5)


def _number_text(value):
    # Evidence interpolates JS Numbers (e.g. 272, not Python's 272.0).
    return str(int(value)) if value == int(value) else str(value)


def layout_spec(profile_id):
    for spec in (PHONE_PROFILE, PHONE_9_16_PROFILE, TABLET_PROFILE):
        if profile_id == spec['profileId']:
            return spec
    return FALLBACK_PROFILE


def _edge_depth(image, axis, reverse):
    height, width = image.shape[:2]
    size = width if axis == 'x' else height
    limit = max(1, math.floor(size * .18))
    depth = 0
    for offset in range(limit):
        position = size - 1 - offset if reverse else offset
        line = image[:, position] if axis == 'x' else image[position, :]
        # Float64 RGB weights, not OpenCV's quantized uint8 grayscale. Keep
        # left-to-right accumulation exactly as JS rather than np.sum/fsum.
        pixels = line.astype(np.float64)
        gray = (pixels[:, 2] * 299 + pixels[:, 1] * 587 + pixels[:, 0] * 114) / 1000
        total = squared = 0.0
        for value in gray.tolist():
            total += value
            squared += value * value
        mean = total / len(gray)
        deviation = math.sqrt(max(0, squared / len(gray) - mean * mean))
        if mean <= 10 and deviation <= 8:
            depth += 1
        else:
            break
    return depth if depth >= 8 else 0


def detect_viewport(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
        raise ValueError('nonempty uint8 BGR image required')
    height, width = image.shape[:2]
    left = _edge_depth(image, 'x', False)
    right = _edge_depth(image, 'x', True)
    top = _edge_depth(image, 'y', False)
    bottom = _edge_depth(image, 'y', True)
    cropped_width, cropped_height = width - left - right, height - top - bottom
    if cropped_width < width * .45 or cropped_height < height * .45:
        return dict(viewport=dict(x=0, y=0, width=width, height=height),
                    confidence=.6, evidence=['black_bar_detection_rejected'])
    cropped = left + right + top + bottom > 0
    return dict(
        viewport=dict(x=left, y=top, width=cropped_width, height=cropped_height),
        confidence=.8 if cropped else .95,
        evidence=[f'black_bars:{left},{top},{right},{bottom}' if cropped else 'black_bars:none'],
    )


def create_screenshot_profile(image):
    detected = detect_viewport(image)
    viewport = detected['viewport']
    height, width = image.shape[:2]
    ratio = viewport['width'] / max(1, viewport['height'])
    spec = next((s for s in (PHONE_PROFILE, PHONE_9_16_PROFILE, TABLET_PROFILE)
                 if s['ratioMin'] <= ratio < s['ratioMax']), FALLBACK_PROFILE)
    top = viewport['y'] + _js_round(viewport['height'] * spec['gridRegion'][1])
    bottom = viewport['y'] + _js_round(viewport['height'] * spec['bottomSafeY'])
    return dict(
        profileId=spec['profileId'], deviceKind=spec['deviceKind'],
        imageWidth=width, imageHeight=height, viewport=viewport,
        contentBounds=dict(x=viewport['x'], y=top, width=viewport['width'], height=max(0, bottom-top)),
        columnCount=4, confidence=detected['confidence'],
        evidence=[*detected['evidence'], f"profile:{spec['profileId']}",
                  f"content_top:profile_grid_{spec['gridRegion'][1]}",
                  f"content_bottom:profile_safe_{spec['bottomSafeY']}"],
    )


def refine_star_content_bounds(profile, selected_tab=None, layout_hint=None):
    viewport = profile['viewport']
    bounds = profile['contentBounds']
    top, bottom = bounds['y'], bounds['y'] + bounds['height']
    evidence = []
    if (selected_tab and selected_tab['width'] >= viewport['width'] * .14
            and selected_tab['height'] > 0 and selected_tab['x'] >= viewport['x']
            and selected_tab['x'] + selected_tab['width'] <= viewport['x'] + viewport['width']
            and selected_tab['y'] >= viewport['y'] + viewport['height'] * .07
            and selected_tab['y'] + selected_tab['height'] <= viewport['y'] + viewport['height'] * .24):
        top = max(top, selected_tab['y'] + selected_tab['height'])
        evidence.append('content_top:selected_tab_bottom:' + _number_text(top))
    layout_matches = (
        profile['profileId'] == 'phone_9_16_v1'
        and profile['imageWidth'] == MAAYUAN_MUMU_LAYOUT['imageWidth']
        and profile['imageHeight'] == MAAYUAN_MUMU_LAYOUT['imageHeight']
        and viewport['x'] == viewport['y'] == 0
        and viewport['width'] == profile['imageWidth']
        and viewport['height'] == profile['imageHeight']
    )
    if layout_hint == 'maayuan_mumu' and layout_matches:
        top = max(top, viewport['y'] + _js_round(viewport['height'] * MAAYUAN_MUMU_LAYOUT['contentTopRatio']))
        bottom = min(bottom, viewport['y'] + _js_round(viewport['height'] * MAAYUAN_MUMU_LAYOUT['contentBottomRatio']))
        evidence.extend([
            f"content_top:maayuan_mumu_layout:{MAAYUAN_MUMU_LAYOUT['contentTopRatio']}",
            f"content_bottom:maayuan_mumu_layout:{MAAYUAN_MUMU_LAYOUT['contentBottomRatio']}",
        ])
    if bottom <= top:
        return profile
    return {**profile, 'contentBounds': {**bounds, 'y': top, 'height': bottom-top},
            'evidence': [*profile['evidence'], *evidence]}


def is_inside(rect, bounds):
    return (rect['x'] >= bounds['x'] and rect['y'] >= bounds['y']
            and rect['x'] + rect['width'] <= bounds['x'] + bounds['width']
            and rect['y'] + rect['height'] <= bounds['y'] + bounds['height'])


def boxes_for_circle(circle):
    x, y, r = (circle[key] for key in ('centerX', 'centerY', 'radius'))
    return dict(
        cardRect=dict(x=math.trunc(x-r*1.05), y=math.trunc(y-r), width=math.trunc(r*2.1), height=math.trunc(r*2.05)),
        discRect=dict(x=x-r, y=y-r, width=r*2, height=r*2),
        nameRect=dict(x=math.trunc(x-r*.92), y=math.trunc(y+r*1.08), width=math.trunc(r*1.84), height=max(12, math.trunc(r*.62))),
        levelRect=dict(x=math.trunc(x+r*.02), y=math.trunc(y-r*1.06), width=math.trunc(r*1.08), height=max(12, math.trunc(r*.58))),
    )


def _bounded_boxes(circle, profile):
    boxes = boxes_for_circle(circle)
    bounds = profile['contentBounds']
    bottom = bounds['y'] + bounds['height']
    name = boxes['nameRect']
    if (profile['imageWidth'] == 720 and profile['imageHeight'] == 1280
            and bounds['y'] == 272 and bottom == 1044
            and is_inside(boxes['discRect'], bounds) and is_inside(boxes['levelRect'], bounds)
            and name['x'] >= bounds['x']
            and name['x'] + name['width'] <= bounds['x'] + bounds['width']
            and name['y'] >= bounds['y'] and name['y'] + name['height'] == bottom + 1):
        boxes['originalNameRect'] = dict(name)
        boxes['nameRect'] = {**name, 'height': name['height'] - 1}
    return boxes


def classify_completeness(circle, profile):
    boxes = _bounded_boxes(circle, profile)
    viewport, bounds = profile['viewport'], profile['contentBounds']
    left = max(0, viewport['x'], bounds['x'])
    top = max(0, viewport['y'], bounds['y'])
    right = min(profile['imageWidth'], viewport['x']+viewport['width'], bounds['x']+bounds['width'])
    bottom = min(profile['imageHeight'], viewport['y']+viewport['height'], bounds['y']+bounds['height'])
    parts = [boxes[key] for key in ('discRect', 'nameRect', 'levelRect')]
    if any(rect['x'] < left or rect['x']+rect['width'] > right for rect in parts):
        return dict(completeness='invalid', evidence=['horizontal_bounds_exceeded'])
    if any(rect['y'] < top for rect in parts):
        return dict(completeness='partial_top', evidence=['disc_or_ocr_roi_above_content_top'])
    if any(rect['y']+rect['height'] > bottom for rect in parts):
        return dict(completeness='partial_bottom', evidence=['disc_or_ocr_roi_below_content_bottom'])
    evidence = ['disc_name_level_inside_content_bounds']
    if 'originalNameRect' in boxes:
        evidence.append('name_bottom_padding_clipped_1px')
    return dict(completeness='complete', evidence=evidence)


def candidate_from_circle(circle, profile, row_index, column_index, index):
    return dict(cardId='card_' + str(index).rjust(3, '0'), rowIndex=row_index,
                columnIndex=column_index, circle=circle,
                **_bounded_boxes(circle, profile), **classify_completeness(circle, profile))
