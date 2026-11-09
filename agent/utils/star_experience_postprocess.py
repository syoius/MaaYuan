"""Native experience icon colour, count resolution and inventory aggregation."""
from __future__ import annotations

import math
import re

from utils.star_grid_geometry import _median
from utils.star_quality_postprocess import _rgb_to_opencv_hsv


EXPERIENCE_CANONICAL = {'orange': '\u6a59\u661f\u66dc', 'purple': '\u7d2b\u661f\u66dc', 'white': '\u767d\u661f\u66dc'}
ORDER = {'orange': 0, 'purple': 1, 'white': 2}
HUE_TARGETS = {'orange': 10, 'purple': 130, 'white': 100}
# ECMAScript trim includes BOM and excludes Python-only whitespace like U+0085.
_JS_TRIM = '\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff'
_COUNT = re.compile(r'[0-9]{1,6}')
MIN_COUNT_CONFIDENCE = .90
MIN_CONSENSUS_CONFIDENCE = .75


def classify_experience_kind(image, icon):
    left = max(0, icon['x'] + math.trunc(icon['width'] * .22))
    top = max(0, icon['y'] + math.trunc(icon['height'] * .22))
    width = max(1, math.trunc(icon['width'] * .56))
    height = max(1, math.trunc(icon['height'] * .56))
    hues = []
    for y in range(top, min(image.shape[0], top+height)):
        for x in range(left, min(image.shape[1], left+width)):
            blue, green, red = map(int, image[y, x])
            hue, saturation, _ = _rgb_to_opencv_hsv(red, green, blue)
            if saturation > 70:
                hues.append(hue)
    if not hues:
        return dict(kind=None, confidence=0, hue=None)
    hue = _median(hues)
    kind = min(HUE_TARGETS, key=lambda k: abs(hue-HUE_TARGETS[k]))
    return dict(kind=kind, confidence=max(0, 1-abs(hue-HUE_TARGETS[kind])/35), hue=hue)


def parse_experience_count(text):
    match = _COUNT.fullmatch(text.strip(_JS_TRIM))
    return int(match[0]) if match else None


def resolve_experience_count(candidates):
    best = raw_best = None
    for candidate in candidates:
        if raw_best is None or candidate['confidence'] > raw_best['confidence']:
            raw_best = candidate
        count = parse_experience_count(candidate['text'])
        if count is not None and (best is None or candidate['confidence'] > best['confidence']):
            best = {**candidate, 'count': count}
    if best is not None:
        return dict(raw=best['text'], count=best['count'], confidence=best['confidence'], reasons=[])
    return dict(raw=raw_best['text'] if raw_best is not None else '', count=None, confidence=0,
                reasons=['experience_count_unparsed'])


def confirm_experience_count(candidates):
    """Confirm a strong read or agreement between distinct preprocessing variants."""
    strong = set()
    votes = {}
    for candidate in candidates:
        count = parse_experience_count(candidate['text'])
        if count is None:
            continue
        score = candidate['confidence']
        if score >= MIN_COUNT_CONFIDENCE:
            strong.add(count)
        if score >= MIN_CONSENSUS_CONFIDENCE and candidate.get('variant'):
            votes.setdefault(count, set()).add(candidate['variant'])

    consensus = {count for count, variants in votes.items() if len(variants) >= 2}
    if strong:
        # Preserve the single strong read unless another strong read or a
        # competing two-version consensus contradicts it.
        if len(strong | consensus) == 1:
            return next(iter(strong)), None
    elif len(votes) == 1 and consensus:
        return next(iter(consensus)), None
    elif len(votes) < 2:
        return None, 'experience_count_unconfirmed'
    return None, 'experience_count_conflict'


def experience_status(completeness, kind, count):
    if completeness != 'complete':
        return 'excluded_partial'
    return 'accepted' if kind is not None and count is not None else 'needs_review'


def aggregate_experience(results, options):
    warnings = []
    visible = [item for item in results if item.get('status') != 'excluded_partial']
    if not visible and not options.get('emptyConfirmed'):
        warnings.append('experience_icons_not_found')
    classified = [item for item in visible if item.get('kind') is not None]
    multiplicity = {kind: sum(item['kind'] == kind for item in classified) for kind in ORDER}
    for kind, amount in multiplicity.items():
        if amount > 1:
            warnings.append(f'experience_kind_duplicate:{kind}')
    kinds = [item['kind'] for item in classified]
    if any(a in ORDER and b in ORDER and ORDER[b] < ORDER[a] for a, b in zip(kinds, kinds[1:])):
        warnings.append('experience_order_conflict')
    if any(item.get('kind') is None for item in visible):
        warnings.append('experience_icon_unclassified')
    for item in classified:
        if item.get('count') is None:
            warnings.append(f"experience_count_unparsed:{item['kind']}")
    if not options.get('selectedTab'):
        warnings.append('experience_tab_selection_unverified')
    if options.get('viewportCropped'):
        warnings.append('experience_viewport_cropped')
    complete = bool(options.get('selectedTab') and not options.get('viewportCropped') and not warnings)

    by_kind = {}
    for item in classified:
        kind = item['kind']
        score = min(item['kindConfidence'], item['countConfidence'])
        if kind not in by_kind or score > by_kind[kind][0]:
            by_kind[kind] = (score, item)

    def field(kind):
        if multiplicity[kind] > 1:
            return None
        count = by_kind.get(kind, (None, {}))[1].get('count')
        return count if count is not None else (0 if complete else None)

    return {**{kind+'Count': field(kind) for kind in ORDER}, 'complete': complete, 'warnings': warnings}
