"""Offline-testable vision and policy for the DOT team's conditional action."""
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def choose_action(vein, qi, skill_ready=None):
    if vein not in ("inside", "outside"):
        return None
    if vein == "outside":
        if skill_ready is None:
            return None
        return "ultimate" if skill_ready else "normal"
    if qi is None:
        return None
    return "normal" if qi == 1 else "defense"


class DragonVision:
    def __init__(self, resource):
        resource = Path(resource)
        self.config = json.loads((resource / "dragon_smart.json").read_text(encoding="utf-8"))
        self.icon = read_image(resource / "image/copilot/dragon.png")
        self.backgrounds = [(entry, [read_image(resource / "image" / name)
                                    for name in [entry["template"], *entry.get("variants", [])]])
                            for entry in self.config["backgrounds"]]
        self.energy = self.config["energy"]
        if self.energy["skill_cost"] != 4:
            raise ValueError("Current energy templates require a four-energy skill")
        self.energy_ready = read_image(resource / "image" / self.energy["ready_template"])
        self.energy_ready_variants = [self.energy_ready] + [read_image(resource / "image" / name)
            for name in self.energy.get("ready_variants", [])]
        self.energy_unready = read_image(resource / "image" / self.energy["unready_template"])

    def inspect(self, image, position=1):
        if position not in range(1, 6):
            raise ValueError("position must be 1..5")
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR screenshot")
        height, width = image.shape[:2]
        if not 1.6 <= height / width <= 1.9:
            raise ValueError("Expected a portrait battle screenshot")
        image = cv2.resize(image, (720, 1280))
        cards = image[820:1190]
        usable = cards.std() >= 15 and cards.mean() >= 20
        scores = []
        for entry, templates in self.backgrounds:
            x, y, w, h = entry["roi"]
            patch = image[y:y+h, x:x+w]
            # Correlation alone accepts a darkened transition frame. Reject
            # featureless or heavily faded patches before matching.
            candidates = []
            for template in templates:
                ratio = float(patch.mean()) / max(float(template.mean()), 1)
                score = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0]) if patch.std() >= 8 else 0.0
                candidates.append(None if score >= self.config["outside_threshold"] and not 0.55 < ratio < 1.8 else score)
            # Variants belong to ONE background region, so they cannot count as
            # multiple agreeing regions. Ambiguous variants cannot prove outside.
            valid_candidates = [s for s in candidates if s is not None]
            best = max(valid_candidates, default=None)
            scores.append(None if None in candidates and (best is None or best < self.config["inside_threshold"]) else best)
        valid = [s for s in scores if s is not None]
        if not usable:
            vein = "unknown"
        elif sum(s >= self.config["inside_threshold"] for s in valid) >= 2:
            vein = "inside"
        elif len(valid) == len(scores) and all(s < self.config["outside_threshold"] for s in valid):
            vein = "outside"
        else:
            vein = "unknown"
        qi, peaks = self.count_qi(image, position)
        skill_ready, energy_scores = self.skill_readiness(image, position)
        return {"vein": vein, "qi": qi, "scores": scores, "peaks": peaks,
                "skill_ready": skill_ready, "energy_scores": energy_scores,
                "action": choose_action(vein, qi, skill_ready)}

    def skill_readiness(self, image, position):
        """Four lit energy diamonds => ready; a matched unlit diamond => not.

        A failed ready match alone is never evidence of insufficient energy.
        This estimates energy sufficiency, not silence/other skill restrictions.
        """
        x, y, w, h = self.energy["roi"]
        x += (position - 1) * 140
        region = image[y:y+h, x:x+w]
        if region.std() < 8 or region.mean() < 15:
            return None, {}
        ready_matches = [(self.energy_match(region, template, cv2.TM_SQDIFF_NORMED, x, position),
                         self.energy_match(region, template, cv2.TM_CCOEFF_NORMED, x, position))
                         for template in self.energy_ready_variants]
        ready = max(r[0] for r in ready_matches)
        unready = self.energy_match(region, self.energy_unready, cv2.TM_SQDIFF_NORMED, x, position)
        shape = max(r[1] for r in ready_matches)
        scores = {"four_lit": ready, "unlit": unready, "four_lit_shape": shape}
        threshold = self.energy["threshold"]
        if any(color >= threshold and structure >= threshold for color, structure in ready_matches) and unready < threshold:
            return True, scores
        # An intact full-energy pattern with changed brightness is a transition
        # or calibration mismatch, not proof that energy was spent.
        if unready >= threshold and shape < threshold:
            return False, scores
        return None, scores

    def stable_background(self, current, previous):
        for x, y, w, h in self.config["stability_rois"]:
            change = np.abs(current[y:y+h, x:x+w].astype(float) - previous[y:y+h, x:x+w].astype(float)).mean()
            if change >= self.config["stability_threshold"]:
                return False
        return True

    def energy_match(self, region, template, method, x, position):
        scores = cv2.matchTemplate(region, template, method)
        # The broader ROI accommodates desktop crops and native captures, but
        # must not accept card artwork beside the energy column as an unlit gem.
        low, high = self.energy["glyph_center_x"]
        centers = x + np.arange(scores.shape[1]) + (template.shape[1]-1)/2
        valid = (centers >= low + (position-1)*140) & (centers <= high + (position-1)*140)
        if not valid.any():
            return 0.0
        scores = scores[:, valid]
        return 1-float(scores.min()) if method == cv2.TM_SQDIFF_NORMED else float(scores.max())

    def count_qi(self, image, position):
        # Includes the bottom icon omitted by DragonRestart's old 160px ROI.
        x, y, w, h = self.config["qi_roi"]
        x += (position - 1) * 140
        region = image[y:y+h, x:x+w]
        if region.std() < 8:
            return None, []
        scores = cv2.matchTemplate(region, self.icon, cv2.TM_CCOEFF_NORMED)
        peaks = []
        th, tw = self.icon.shape[:2]
        while len(peaks) <= 6:
            _, score, _, (px, py) = cv2.minMaxLoc(scores)
            if score < self.config["qi_threshold"]:
                break
            peaks.append({"score": float(score), "x": x+px, "y": y+py})
            # Suppress duplicate matches of the same icon, not adjacent icons.
            scores[max(0, py-th//2):py+th//2+1, max(0, px-tw//2):px+tw//2+1] = -1
        count = len(peaks)
        if count > 6:
            return None, peaks
        # For 0/1, weak additional candidates could change the policy. Retry
        # instead of silently calling a missed icon zero. >=2 is one category.
        if count < 2 and float(scores.max()) >= self.config["qi_uncertain_threshold"]:
            return None, peaks
        return count, peaks
