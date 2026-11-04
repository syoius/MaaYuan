"""Name-only motion evidence for settling; never identifies inventory instances.

Same bidirectional corner tracker validated by the local swipe experiments.
The full scanner supports 720x1280 frames and a four-column inventory viewport.
"""
import cv2
import numpy as np


def name_motion(before, after, geometry, shift=0.):
    """Measure glyph movement; shared disc pixels are excluded from the mask."""
    gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    other = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for card in geometry['cards']:
        rect = card.get('nameRect')
        if not rect:
            continue
        x, y, w, h = [int(rect[k]) for k in ('x', 'y', 'width', 'height')]
        y = int(round(y + shift))
        top, bottom = max(280, y), min(1040, y + h)
        left, right = max(40, x), min(680, x + w)
        if bottom > top and right > left:
            mask[top:bottom, left:right] = 255
    mask[gray < 125] = 0
    points = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=.03,
                                    minDistance=3, mask=mask, blockSize=3)
    if points is None:
        return {'reliable': False, 'reason': 'no glyph corners'}
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    target, forward, _ = cv2.calcOpticalFlowPyrLK(gray, other, points, None, **options)
    if target is None or forward is None:
        return {'reliable': False, 'reason': 'forward tracking failed'}
    back, backward, _ = cv2.calcOpticalFlowPyrLK(other, gray, target, None, **options)
    if back is None or backward is None:
        return {'reliable': False, 'reason': 'backward tracking failed'}
    good = (forward.ravel().astype(bool) & backward.ravel().astype(bool)
            & (np.linalg.norm((back - points).reshape(-1, 2), axis=1) <= 1.))
    delta = (target - points).reshape(-1, 2)
    valid = delta[good]
    if len(valid) < 8:
        return {'reliable': False, 'tracked': int(len(valid)), 'reason': 'few consistent name corners'}
    median = np.median(valid, axis=0)
    inliers = good & (np.max(np.abs(delta - median), axis=1) <= 1.5)
    positions = points.reshape(-1, 2)[inliers]
    movement = delta[inliers]
    columns = {int(x // 180) for x, y in positions}
    if len(movement) < 8 or len(columns) < 2:
        return {'reliable': False, 'tracked': int(len(movement)),
                'columns': len(columns), 'reason': 'insufficient spatial consensus'}
    dx, dy = np.median(movement, axis=0)
    return {'reliable': True, 'dx': float(dx), 'dy': float(dy),
            'corners': int(len(points)), 'inliers': int(len(movement)),
            'columns': len(columns),
            'mad_px': float(np.median(np.abs(movement - np.array([dx, dy])))),
            'evidence': 'name glyph ROI corners only; bidirectional optical flow'}
