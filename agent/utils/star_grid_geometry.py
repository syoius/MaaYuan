"""Native main-grid.js algorithms, using Stage 1A profiles and card ROIs.

Accept nonempty uint8 BGR images as elsewhere in MaaYuan. edge_map returns a
flat uint8 array like the JS Uint8Array. Grouped rows contain an insertion-ordered
column->proposal dict in place of JS Map; final candidates keep the JS contract.
OCR, quality classification and scan state are handled by the callers.
"""
from __future__ import annotations

from functools import lru_cache
import math
import time

import numpy as np

from utils.star_geometry import _js_round, candidate_from_circle, layout_spec


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return 0
    return ordered[middle] if len(ordered) % 2 else (ordered[middle-1] + ordered[middle]) / 2


def edge_map(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
        raise ValueError('nonempty uint8 BGR image required')
    # The JS first rounds RGB-weighted gray into uint8, then differences it.
    # Integer weights keep the exact numerator; neither cvtColor nor Sobel is
    # equivalent. int32 also prevents uint8 wrap in sums and differences.
    pixels = image.astype(np.int32)
    weighted = pixels[:, :, 2] * 299 + pixels[:, :, 1] * 587 + pixels[:, :, 0] * 114
    gray = ((weighted + 500) // 1000).astype(np.int32)
    edges = np.zeros(image.shape[:2], dtype=np.uint8)
    gx = np.abs(gray[1:-1, 2:] - gray[1:-1, :-2])
    gy = np.abs(gray[2:, 1:-1] - gray[:-2, 1:-1])
    edges[1:-1, 1:-1] = np.minimum(255, gx + gy)
    return edges.reshape(-1)


def ring_score(edges, width, height, x, y, radius):
    score = count = 0
    for index in range(40):
        angle = index / 40 * math.pi * 2
        for offset in (-2, 0, 2):
            px = _js_round(x + math.cos(angle) * (radius + offset))
            py = _js_round(y + math.sin(angle) * (radius + offset))
            if px < 1 or py < 1 or px >= width-1 or py >= height-1:
                continue
            at = py * width + px
            score += int(edges[at]) if at < len(edges) else 0
            count += 1
    return score / count if count else 0


@lru_cache(maxsize=64)
def _ring_offsets(radius):
    offsets = []
    for index in range(40):
        angle = index / 40 * math.pi * 2
        for offset in (-2, 0, 2):
            offsets.append((math.cos(angle)*(radius+offset), math.sin(angle)*(radius+offset)))
    return np.asarray(offsets).T


def _ring_scores(edges, width, height, x, centers_y, radius):
    """The scalar ring_score samples, evaluated together across one column."""
    dx, dy = _ring_offsets(radius)
    def round_like_js(values):
        lower = np.floor(values)
        return (lower + (values-lower >= .5)).astype(np.intp)
    # Round AFTER adding the centers, preserving the scalar float tie behavior.
    px = round_like_js(x + dx)
    py = round_like_js(centers_y[:, None] + dy)
    valid = (px >= 1) & (px < width-1) & (py >= 1) & (py < height-1)
    indices = np.where(valid, py*width+px, 0)
    return (edges[indices] * valid).sum(axis=1) / np.maximum(1, valid.sum(axis=1))


def _suppress_column(peaks):
    selected = []
    for proposal in sorted(peaks, key=lambda p: -p['score']):
        if any(abs(item['centerY'] - proposal['centerY']) <= min(item['radius'], proposal['radius']) * 1.15
               for item in selected):
            continue
        selected.append(proposal)
    return sorted(selected, key=lambda p: p['centerY'])


def find_circle_proposals(image, profile, timings=None):
    started = time.perf_counter()
    spec = layout_spec(profile['profileId'])
    edges = edge_map(image)
    if timings is not None:timings['edge_map_ms']=(time.perf_counter()-started)*1000
    started=time.perf_counter();ring_seconds=0.
    height, width = image.shape[:2]
    viewport = profile['viewport']
    expected_radius = viewport['width'] * spec['radiusRatio']
    min_radius = max(12, _js_round(expected_radius * .82))
    max_radius = max(min_radius+2, _js_round(expected_radius * 1.18))
    radius_step = max(3, _js_round(expected_radius * .055))
    x_search = _js_round(viewport['width'] * spec['horizontalSearchRatio'])
    x_step = max(4, _js_round(expected_radius * .12))
    y_step = max(2, _js_round(viewport['height'] / 900))
    proposals = []
    for normalized_x in spec['columnCenters']:
        expected_x = _js_round(viewport['x'] + viewport['width'] * normalized_x)
        first_y = profile['contentBounds']['y'] - max_radius
        last_y = math.floor(viewport['y'] + viewport['height'] - min_radius / 2)
        centers_y = np.arange(first_y, last_y+1, y_step)
        best_scores = np.zeros(len(centers_y))
        best_radii = np.full(len(centers_y), min_radius)
        best_xs = np.full(len(centers_y), expected_x)
        for radius in range(min_radius, max_radius+1, radius_step):
            for center_x in range(expected_x-x_search, expected_x+x_search+1, x_step):
                ring_start = time.perf_counter() if timings is not None else 0.
                scores = _ring_scores(edges, width, height, center_x, centers_y, radius)
                if timings is not None:
                    ring_seconds += time.perf_counter()-ring_start
                better = scores > best_scores  # Preserve the first candidate on ties.
                best_scores[better] = scores[better]
                best_radii[better] = radius
                best_xs[better] = center_x
        peaks = [dict(centerX=int(best_xs[i]), centerY=int(centers_y[i]), radius=int(best_radii[i]),
                      score=_js_round(float(best_scores[i])*100)/100, source='canvas_ring')
                 for i in np.flatnonzero(best_scores >= 21)]
        proposals.extend(_suppress_column(peaks))
    if proposals:
        radius_mode = _median(p['radius'] for p in proposals)
        proposals = [p for p in proposals if abs(p['radius']-radius_mode) <= max(4, radius_mode*.22)]
    if timings is not None:
        timings['ring_score_ms'] = ring_seconds*1000
        timings['proposal_generation_ms'] = (time.perf_counter()-started-ring_seconds)*1000
    return proposals


def _score_sum(row):
    # Python's compensated sum (3.12+) can differ from JS's left-fold reduce.
    total = 0
    for item in row['circles'].values():
        total += item['score']
    return total


def _row_score(row):
    return _score_sum(row) / max(1, len(row['circles']))


def group_grid_rows(proposals, profile):
    if not proposals:
        return []
    spec = layout_spec(profile['profileId'])
    viewport = profile['viewport']
    columns = [viewport['x'] + viewport['width'] * value for value in spec['columnCenters']]
    radius = _median(p['radius'] for p in proposals)
    tolerance = max(12, radius*.7)
    rows = []
    for proposal in sorted(proposals, key=lambda p: (p['centerY'], -p['score'])):
        # min returns the first column on ties, as the JS strict-< reducer does.
        column = min(range(len(columns)), key=lambda i: abs(columns[i]-proposal['centerX']))
        if abs(columns[column]-proposal['centerX']) > viewport['width'] * .11:
            continue
        row = next((r for r in rows if abs(r['y']-proposal['centerY']) <= tolerance), None)
        if row is None:
            row = dict(y=proposal['centerY'], circles={})
            rows.append(row)
        previous = row['circles'].get(column)
        if previous is None or proposal['score'] > previous['score']:
            row['circles'][column] = proposal
        row['y'] = _median(p['centerY'] for p in row['circles'].values())
    expected_radius = viewport['width'] * spec['radiusRatio']
    eligible = [r for r in rows if abs(_median(p['radius'] for p in r['circles'].values())-expected_radius) <= expected_radius*.15]
    populated = [r for r in (eligible if len([r for r in eligible if len(r['circles']) >= 2]) >= 2 else rows)
                 if len(r['circles']) >= 2]
    best_score = max([_row_score(r) for r in populated] + [0])
    evidence_threshold = best_score * .62
    evidence_rows = [r for r in populated if _row_score(r) >= evidence_threshold]
    strong = sorted(evidence_rows if len(evidence_rows) >= 2 else populated, key=lambda r: r['y'])
    if not strong:
        return []
    full_anchors = [r for r in strong if len(r['circles']) == 4]
    cadence_source = full_anchors if len(full_anchors) >= 3 else strong
    gaps = [r['y']-cadence_source[i]['y'] for i, r in enumerate(cadence_source[1:])]
    gaps = [g for g in gaps if viewport['height']*spec['rowSpacingRange'][0] <= g <= viewport['height']*spec['rowSpacingRange'][1]]
    cadence = _median(gaps) if gaps else 0
    accepted = list(strong)
    if cadence and len(full_anchors) >= 3:
        alignment_rows = [r for r in eligible if _row_score(r) >= evidence_threshold] if len(evidence_rows) >= 2 else rows

        def aligned_count(candidate):
            return sum(len(r['circles']) for r in alignment_rows
                       if abs(r['y']-(candidate['y']+_js_round((r['y']-candidate['y'])/cadence)*cadence)) <= cadence*.18)

        base = sorted(full_anchors, key=lambda r: (-aligned_count(r), -_row_score(r)))[0]['y']
        aligned = {}
        for row in alignment_rows:
            offset = _js_round((row['y']-base)/cadence)
            if abs(row['y']-(base+offset*cadence)) > cadence*.18:
                continue
            previous = aligned.get(offset)
            score = _score_sum(row)
            previous_score = _score_sum(previous) if previous is not None else -1
            if (previous is None or len(row['circles']) > len(previous['circles'])
                    or (len(row['circles']) == len(previous['circles']) and score > previous_score)):
                aligned[offset] = row
        accepted = [row for offset, row in aligned.items()
                    if len(row['circles']) >= 2 or offset < 0 or offset > len(full_anchors)-1]
    if cadence:
        for row in (r for r in rows if len(r['circles']) == 1 and _row_score(r) >= evidence_threshold):
            last = accepted[-1] if accepted else None
            first = accepted[0] if accepted else None
            if ((last is not None and row['y'] > last['y'] and abs(row['y']-last['y']-cadence) <= cadence*.35)
                    or (first is not None and row['y'] < first['y'] and abs(first['y']-row['y']-cadence) <= cadence*.35)):
                accepted.append(row)
    return sorted(accepted, key=lambda r: r['y'])


def build_card_candidates(proposals, profile):
    cards = []
    for row_index, row in enumerate(group_grid_rows(proposals, profile)):
        for column_index, circle in sorted(row['circles'].items()):
            cards.append(candidate_from_circle(circle, profile, row_index, column_index, len(cards)+1))
    return cards
