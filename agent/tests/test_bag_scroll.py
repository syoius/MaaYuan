import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import bag_scroll


class Clock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class BagScrollTests(unittest.TestCase):
    def setUp(self):
        self.roi = (0, 0, 240, 320)
        self.image = np.random.default_rng(17).integers(0, 256, (320, 240, 3), np.uint8)
        self.context = Mock()
        self.context.run_task.return_value = SimpleNamespace(
            status=SimpleNamespace(succeeded=True),
            nodes=[SimpleNamespace(action=SimpleNamespace(success=True))],
        )
        self.clock = Clock()
        self.job = self.context.tasker.controller.post_screencap.return_value.wait.return_value
        self.job.succeeded = True
        self.job.get.return_value = self.image

    def scroll(self, **kwargs):
        with patch.object(bag_scroll, "time", self.clock):
            bag_scroll.scroll_backpack(self.context, (120, 270, 120, 100, 500), self.roi,
                                       stopped=lambda: False, **kwargs)

    def test_native_swipe_holds_before_release_and_waits_for_stable_frames(self):
        self.scroll()
        node, pipeline = self.context.run_task.call_args.args
        params = pipeline[node]["action"]["param"]
        self.assertEqual(params["end_hold"], 600)
        self.assertEqual(params["begin"], [120, 270, 1, 1])
        self.assertEqual(params["end"], [120, 100, 1, 1])
        self.assertFalse(params["only_hover"])
        self.context.tasker.controller.post_swipe.assert_not_called()
        self.assertGreaterEqual(self.clock.now, .7)
        self.assertGreaterEqual(self.job.get.call_count, 3)

    def test_moving_frames_are_not_accepted_until_three_still_samples(self):
        moving = [np.roll(self.image, pixels, axis=0) for pixels in (0, 12, 24, 36)]
        self.job.get.side_effect = [*moving, moving[-1], moving[-1]]
        self.scroll()
        self.assertEqual(self.job.get.call_count, 6)

    def test_list_translation_is_not_considered_still(self):
        gray = bag_scroll._content_frame(self.image, self.roi)
        shifted = lambda pixels: cv2.warpAffine(gray, np.float32([[1, 0, 0], [0, 1, pixels]]),
                                                (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REFLECT)
        self.assertFalse(bag_scroll.frames_are_still(shifted(0), shifted(8)))

    def test_local_animation_does_not_hide_a_stationary_majority(self):
        gray = bag_scroll._content_frame(self.image, self.roi)
        animated = gray.copy()
        animated[:gray.shape[0] // 2, :gray.shape[1] // 2] = np.random.default_rng(23).uniform(
            0, 255, (gray.shape[0] // 2, gray.shape[1] // 2))
        self.assertTrue(bag_scroll.frames_are_still(gray, animated))

    def test_slow_drift_cannot_accumulate_across_the_stability_window(self):
        frames = [np.roll(self.image, pixels, axis=0) for pixels in (0, 2, 4, 6, 8)]
        self.job.get.side_effect = [*frames, frames[-1], frames[-1]]
        self.scroll()
        self.assertEqual(self.job.get.call_count, 7)

    def test_timeout_does_not_proceed_with_a_moving_frame(self):
        self.job.get.side_effect = lambda: np.roll(self.image, int(self.clock.now * 100), axis=0)
        with self.assertRaises(TimeoutError):
            self.scroll()

    def test_failed_swipe_never_reads_a_frame(self):
        self.context.run_task.return_value.status.succeeded = False
        with self.assertRaises(RuntimeError):
            self.scroll()
        self.context.tasker.controller.post_screencap.assert_not_called()

    def test_failed_capture_rejects_cached_image(self):
        self.job.succeeded = False
        with self.assertRaisesRegex(RuntimeError, "拒绝使用缓存"):
            self.scroll()
        self.job.get.assert_not_called()

    def test_stop_while_waiting_prevents_next_capture(self):
        checks = iter([False, False, False, True])
        with patch.object(bag_scroll, "time", self.clock), self.assertRaises(InterruptedError):
            bag_scroll.wait_backpack_still(self.context, self.roi, stopped=lambda: next(checks),
                                          minimum_wait_seconds=.7)

    def test_invalid_swipe_sends_no_action(self):
        with self.assertRaises(ValueError):
            bag_scroll.scroll_backpack(self.context, (120, 270, 120, 320, 500), self.roi,
                                       stopped=lambda: False)
        self.context.run_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
