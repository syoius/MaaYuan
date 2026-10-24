"""Ensure relative click areas stay inside the visible part of their templates."""

import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]


class YechengClickTargetTests(unittest.TestCase):
    def test_masked_template_click_areas(self):
        checked = 0
        for locale in ("base", "zh_tw"):
            resource = ROOT / "assets/resource" / locale
            for path in (resource / "pipeline/sp/yecheng").glob("*.json"):
                for name, node in json.loads(path.read_text(encoding="utf-8")).items():
                    reco = node.get("recognition", {})
                    action = node.get("action", {})
                    param = reco.get("param", {})
                    offset = action.get("param", {}).get("target_offset")
                    if (reco.get("type") != "TemplateMatch" or not param.get("green_mask")
                            or action.get("type") != "Click" or offset is None):
                        continue
                    with self.subTest(locale=locale, name=name):
                        with Image.open(resource / "image" / param["template"]) as image:
                            rgba = np.array(image.convert("RGBA"))
                        height, width = rgba.shape[:2]
                        x, y, dw, dh = offset
                        w, h = width + dw, height + dh
                        self.assertGreater(w, 0)
                        self.assertGreater(h, 0)
                        self.assertGreaterEqual(x, 0)
                        self.assertGreaterEqual(y, 0)
                        self.assertLessEqual(x + w, width)
                        self.assertLessEqual(y + h, height)
                        area = rgba[y:y + h, x:x + w]
                        self.assertFalse(np.any(np.all(area[:, :, :3] == [0, 255, 0], axis=2)))
                        self.assertTrue(np.all(area[:, :, 3] == 255))
                        checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
