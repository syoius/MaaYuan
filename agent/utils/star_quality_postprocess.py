"""Native quality-postprocess.js for MaaYuan's fixed 720x1280 BGR capture.

Consume Stage 1A/1B1 cardRect/completeness unchanged. The only dependency ported
from experience-geometry.js is its scalar RGB->floating HSV conversion, required
by quality sampling. Experience quality is handled separately.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from utils.star_grid_geometry import _median


QUALITY_HUES = {'\u6a59': 12, '\u7d2b': 142, '\u84dd': 106, '\u7eff': 61}


def _to_fixed(value, digits):
    # JS toFixed rounds the exact binary Number, with ties away from zero.
    # Python format uses ties-to-even; multiplying first can create false ties.
    value = float(value)
    numerator, denominator = abs(value).as_integer_ratio()
    factor = 10 ** digits
    rounded, remainder = divmod(numerator * factor, denominator)
    if remainder * 2 >= denominator:
        rounded += 1
    sign = '-' if value < 0 else ''
    return f'{sign}{rounded // factor}.{rounded % factor:0{digits}d}'


def _rgb_to_opencv_hsv(red, green, blue):
    r, g, b = red / 255, green / 255, blue / 255
    maximum, minimum = max(r, g, b), min(r, g, b)
    delta = maximum - minimum
    hue = 0
    if delta:
        if maximum == r:
            # JS remainder preserves the sign; Python % would wrap too early.
            hue = 60 * math.fmod((g-b) / delta, 6)
        elif maximum == g:
            hue = 60 * ((b-r) / delta + 2)
        else:
            hue = 60 * ((r-g) / delta + 4)
    if hue < 0:
        hue += 360
    return [hue / 2, delta / maximum * 255 if maximum else 0, maximum * 255]


def _bgr_pixels_to_hsv(pixels):
    # Match the scalar operation order in float64; OpenCV uint8 HSV quantizes.
    b, g, r = (pixels.astype(np.float64) / 255).T
    maximum = np.maximum(np.maximum(r, g), b)
    minimum = np.minimum(np.minimum(r, g), b)
    delta = maximum - minimum
    chromatic = delta != 0
    red = chromatic & (maximum == r)
    green = chromatic & ~red & (maximum == g)
    blue = chromatic & ~red & ~green
    hue = np.zeros_like(delta)
    hue[red] = 60 * np.fmod((g[red] - b[red]) / delta[red], 6)
    hue[green] = 60 * ((b[green] - r[green]) / delta[green] + 2)
    hue[blue] = 60 * ((r[blue] - g[blue]) / delta[blue] + 4)
    hue[hue < 0] += 360
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum != 0)
    return np.column_stack((hue / 2, saturation * 255, maximum * 255))


@lru_cache(maxsize=16)
def _quality_annulus_mask(width, height):
    def squared_axis(length):
        center, scale = (length - 1) / 2, length / 2
        # Keep scalar **2 rounding on each axis, once per size, for edge pixels.
        return np.fromiter((((i - center) / scale) ** 2 for i in range(length)),
                           dtype=np.float64, count=length)

    radius = np.sqrt(squared_axis(width)[None, :] + squared_axis(height)[:, None])
    mask = (radius >= .62) & (radius <= .96)
    mask.flags.writeable = False
    return mask


def quality_rect_for_card(card):
    return dict(card.get('cardRect') or {})


def _hue_distance(left, right):
    distance = abs(left-right)
    return min(distance, 180-distance)


def classify_quality_pixels(pixels):
    if not pixels:
        return dict(quality=None, confidence=0, evidence='empty_relative_quality_region',
                    warnings=['quality_unknown'], source='unknown')
    colourful = [p for p in pixels if p.get('saturation', math.nan) >= 52]
    neutral = [p for p in pixels if p.get('saturation', math.nan) <= 45 and p.get('value', math.nan) >= 150]
    colour_ratio, white_ratio = len(colourful)/len(pixels), len(neutral)/len(pixels)
    if white_ratio >= .56 and colour_ratio <= .24:
        return dict(quality='\u767d', confidence=min(.92, .55+white_ratio*.42),
                    evidence=f'neutral={_to_fixed(white_ratio,2)};saturated={_to_fixed(colour_ratio,2)}',
                    warnings=[], source='visual_background')
    if colour_ratio < .16:
        return dict(quality=None, confidence=0, evidence=f'saturated={_to_fixed(colour_ratio,2)}',
                    warnings=['quality_low_saturation','quality_unknown'], source='unknown')
    hue = _median(p['hue'] for p in colourful)
    # Stable first-min follows QUALITY_HUES insertion order, including ties.
    quality, target = min(QUALITY_HUES.items(), key=lambda item: _hue_distance(hue,item[1]))
    distance = _hue_distance(hue,target)
    evidence = f'hue={_to_fixed(hue,1)};distance={_to_fixed(distance,1)};saturated={_to_fixed(colour_ratio,2)}'
    if distance > 19 or colour_ratio < .34:
        return dict(quality=None, confidence=0, evidence=evidence,
                    warnings=['quality_visual_conflict','quality_unknown'], source='unknown')
    return dict(quality=quality, confidence=min(.93, .40+colour_ratio*.38+(19-distance)/19*.25),
                evidence=evidence, warnings=[], source='visual_background')


def extract_relative_quality_pixels(image, rect):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (1280,720,3):
        raise ValueError('720x1280 uint8 BGR image required')
    # Absent/nonfinite ROI coordinates produce no samples in the JS loop.
    values = [rect.get(key, math.nan) for key in ('x','y','width','height')]
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
        return []
    x, y, rect_width, rect_height = values
    left, top = max(0,math.floor(x)), max(0,math.floor(y))
    width, height = max(0,min(720-left,math.ceil(rect_width))), max(0,min(1280-top,math.ceil(rect_height)))
    if not width or not height:
        return []
    roi = image[top:top+height, left:left+width]
    # Boolean indexing visits selected pixels in the original row-major order.
    hsv = _bgr_pixels_to_hsv(roi[_quality_annulus_mask(width, height)])
    return [dict(hue=hue, saturation=saturation, value=value)
            for hue, saturation, value in hsv.tolist()]


def recognize_quality(image, card):
    if card.get('completeness') != 'complete':
        return dict(quality=None, confidence=0, evidence='incomplete_card_quality_not_evaluated',
                    warnings=['quality_incomplete_card','quality_unknown'], source='unknown')
    return classify_quality_pixels(extract_relative_quality_pixels(image,quality_rect_for_card(card)))
