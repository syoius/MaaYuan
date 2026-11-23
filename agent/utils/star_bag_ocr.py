"""Read the fixed backpack count/capacity badge with the shared Context OCR provider."""
from __future__ import annotations
import re
import cv2
import numpy as np

# 720x1280 portrait game frame: digits at bottom right, excluding the icon and + button.
BAG_ROI = (540, 1144, 116, 42)
MIN_CONFIDENCE = .9
RETRY_CONFIDENCE = .95
MAX_COUNT = 2_147_483_647


def empty_bag(reason):
    return dict(complete=False, currentCount=None, capacity=None, warnings=[reason], roi=list(BAG_ROI), variants=[])


def parse_bag_text(text):
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"\s*([0-9]{1,10})\s*[/／]\s*([0-9]{1,10})\s*", text)
    if not match:
        return None
    current, capacity = map(int, match.groups())
    return (current, capacity) if 0 <= current <= capacity <= MAX_COUNT and capacity > 0 else None


def recognize_bag(image, provider):
    result = empty_bag('bag_numbers_unconfirmed')
    if image.shape != (1280, 720, 3):
        result['warnings'] = ['bag_frame_size_invalid']
        return result
    x, y, w, h = BAG_ROI
    crop = image[y:y+h, x:x+w].copy()
    large = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    readings = set()
    def invoke(label, prepared):
        detail = provider(np.ascontiguousarray(prepared))
        result['variants'].append(dict(variant=label, **detail))
        confidence = 0.
        for item in detail.get('all_results', []):
            pair = parse_bag_text(item.get('text'))
            score = item.get('score', 0.)
            if pair is not None and score >= MIN_CONFIDENCE:
                readings.add(pair)
                confidence = max(confidence, score)
        return confidence
    confidence = invoke('color_3x', large)
    if confidence < RETRY_CONFIDENCE:
        gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
        contrast = np.clip(gray.astype(np.float32)*1.7-35, 0, 255).astype(np.uint8)
        invoke('contrast_3x', cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR))
    if len(readings) == 1:
        current, capacity = readings.pop()
        result.update(complete=True, currentCount=current, capacity=capacity, warnings=[])
    elif len(readings) > 1:
        result['warnings'] = ['bag_numbers_conflict']
    return result


def bag_summary(result):
    return {k:v for k,v in result.items() if k != 'variants'}
