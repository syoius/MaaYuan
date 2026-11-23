"""Viewport and name-glyph motion evidence; never proves inventory position."""
from functools import lru_cache
import cv2
import numpy as np

LIST_X, LIST_Y = 40, 280
LIST_WIDTH, LIST_HEIGHT = 640, 760


@lru_cache(maxsize=1)
def _window():
    return cv2.createHanningWindow((LIST_WIDTH, LIST_HEIGHT), cv2.CV_32F)


class StabilityFrame:
    def __init__(self, image):
        self.gray = cv2.cvtColor(image[280:1040, 40:680], cv2.COLOR_BGR2GRAY)
        self.phase = self.gray.astype(np.float32)
        centered = cv2.GaussianBlur(self.phase, (3, 3), 0)
        centered -= float(centered.mean())
        self.ncc_vector = centered.ravel()
        self.norm = float(np.linalg.norm(self.ncc_vector))
        self.name_points = None

    def compare(self, other):
        # OpenCV multiplies float32 inputs by the window in place on some builds.
        shift, response = cv2.phaseCorrelate(self.phase.copy(), other.phase.copy(), _window())
        den = self.norm * other.norm
        ncc = (float(np.clip((self.ncc_vector @ other.ncc_vector) / den, -1, 1))
               if den > 1e-6 else float(np.array_equal(self.ncc_vector, other.ncc_vector)))
        return {'stable': bool(ncc >= .99 and response >= .5 and max(map(abs, shift)) <= 2),
                'ncc': ncc, 'shift_px': list(shift), 'response': float(response)}

    def prepare_names(self, rects):
        mask = np.zeros_like(self.gray)
        for rect in rects:
            x, y, w, h = (int(rect[k]) for k in ('x', 'y', 'width', 'height'))
            top, bottom = max(0, y - LIST_Y), min(LIST_HEIGHT, y + h - LIST_Y)
            left, right = max(0, x - LIST_X), min(LIST_WIDTH, x + w - LIST_X)
            if bottom > top and right > left:
                mask[top:bottom, left:right] = 255
        mask[self.gray < 125] = 0
        self.name_points = cv2.goodFeaturesToTrack(self.gray, maxCorners=300, qualityLevel=.03,
                                                   minDistance=3, mask=mask, blockSize=3)


def _name_rects(frame, reference):
    """Project the previous page's name bands onto the visible list phase.

    Only a sparse one-dimensional ring response at known columns/radius is
    needed to find that phase. No profiles, candidates, completeness or
    qualities are computed here. Rings locate the mask; only glyph pixels
    inside that mask are used by LK to decide stability.
    """
    cards = reference.get('cards', ()) if reference else ()
    defaults = (119, 274, 430, 585)
    columns = [[c for c in cards if c['columnIndex'] == col and c.get('nameRect')]
               for col in range(4)]
    # Median geometry is a layout prior, not evidence of which stars are visible.
    radius = int(round(np.median([c['circle']['radius'] for col in columns for c in col]))) if any(columns) else 54
    pitch = np.diff(reference.get('rowCenters', ())) if reference else ()
    pitch = [x for x in pitch if 145 <= x <= 185]
    pitch = int(round(np.median(pitch))) if len(pitch) else 167
    gray = frame.gray
    edge = np.abs(cv2.Sobel(gray, cv2.CV_16S, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_16S, 0, 1))
    angles = np.arange(24) * (2 * np.pi / 24)
    radii = radius + np.array([-4, -2, 0, 2, 4])[:, None]
    dx = np.rint(radii * np.cos(angles)).astype(int)
    dy = np.rint(radii * np.sin(angles)).astype(int)
    ys = np.arange(-radius, LIST_HEIGHT)
    yy = ys[:, None, None] + dy
    valid_y = (yy >= 0) & (yy < LIST_HEIGHT)
    yy = np.clip(yy, 0, LIST_HEIGHT - 1)
    responses = []
    specs = []
    for default, col in zip(defaults, columns):
        med = lambda values: int(round(np.median(values)))
        cx = med([c['circle']['centerX'] for c in col]) if col else default
        x = cx - LIST_X + dx
        valid = valid_y & (x >= 0) & (x < LIST_WIDTH)
        samples = np.minimum(edge[yy, np.clip(x, 0, LIST_WIDTH - 1)], 255)
        score = (samples * valid).sum(axis=2) / np.maximum(1, valid.sum(axis=2))
        score[valid.sum(axis=2) < 12] = 0
        responses.append(score.max(axis=1))
        specs.append((cx, med([c['nameRect']['x']-c['circle']['centerX'] for c in col]) if col else -49,
                      med([c['nameRect']['width'] for c in col]) if col else 98,
                      med([c['nameRect']['y']-c['circle']['centerY'] for c in col]) if col else 58,
                      med([c['nameRect']['height'] for c in col]) if col else 33))
    # A common phase must have support in more than one column. Use the second
    # strongest response so a single decoration cannot establish a name band.
    support = np.sort(responses, axis=0)[-2]
    # Prefer a repeating row phase over an isolated peak. A short list can
    # still supply a single supported row.
    phases = np.bincount(np.arange(len(ys)) % pitch, weights=np.maximum(0, support - 36), minlength=pitch)
    phase = int(np.argmax(phases + np.roll(phases, 1) + np.roll(phases, -1)))
    rects = []
    for idx in range(phase, len(ys), pitch):
        lo, hi = max(0, idx-8), min(len(ys), idx+9)
        local = lo + int(np.argmax(support[lo:hi]))
        if support[local] < 36:
            continue
        for cx, xoff, w, yoff, h in specs:
            rects.append(dict(x=cx+xoff, y=int(ys[local])+LIST_Y+yoff, width=w, height=h))
    return rects


class NameStability:
    """Prepare each screenshot once and reuse feature points across comparisons."""
    def __init__(self, reference=None):
        self.reference = reference

    def prepare(self, image, previous=None):
        frame = StabilityFrame(image)
        if previous is not None and np.array_equal(frame.gray, previous.gray):
            frame.name_points = previous.name_points
        else:
            frame.prepare_names(_name_rects(frame, self.reference))
        return frame


def glyph_motion(before, after):
    """Bidirectional LK on name glyph corners, independent of frame NCC."""
    gray, other, points = before.gray, after.gray, before.name_points
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
    columns = {int((x+LIST_X)//180) for x, y in positions}
    if len(movement) < 8 or len(columns) < 2:
        return {'reliable': False, 'tracked': int(len(movement)),
                'columns': len(columns), 'reason': 'insufficient spatial consensus'}
    dx, dy = np.median(movement, axis=0)
    return {'reliable': True, 'dx': float(dx), 'dy': float(dy),
            'corners': int(len(points)), 'inliers': int(len(movement)), 'columns': len(columns),
            'mad_px': float(np.median(np.abs(movement - np.array([dx, dy])))),
            'evidence': 'name glyph ROI corners only; bidirectional optical flow'}


def name_motion(before, after, geometry, shift=0.):
    """Diagnostic entry for explicitly known name rectangles."""
    source, target = StabilityFrame(before), StabilityFrame(after)
    source.prepare_names([{**r, 'y': round(r['y'] + shift)} for card in geometry['cards']
                          if (r := card.get('nameRect'))])
    return glyph_motion(source, target)
