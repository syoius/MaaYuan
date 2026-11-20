import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.action.paged_item_recognition import (  # noqa: E402
    BagLayoutError,
    PageScan,
    _bag_earlier_base,
    _bag_page_alignment,
    _bag_same_position,
    _bag_scan_alignment,
    _recover_bag_page,
    _resume_bag_from_checkpoint,
    _seek_bag_top,
)
from custom.reco.agent_item import detect_auto_layout  # noqa: E402


class BagLayoutTests(unittest.TestCase):
    columns = (118., 286., 454., 622.)

    def setUp(self):
        self.full_rows = [(x, y, 50) for y in (420, 610, 800, 990) for x in self.columns]

    def detect(self, circles, *, fixed=True):
        roi = (34, 245, 672, 940)
        detected = np.array([[(x - roi[0], y - roi[1], radius) for x, y, radius in circles]], dtype=np.float32)
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        with patch("custom.reco.agent_item.cv2.HoughCircles", return_value=detected):
            return detect_auto_layout(
                frame, None, roi, (None, None), {"auto_circle_radius": [42, 69]},
                fixed_columns=self.columns if fixed else None,
            )

    def test_off_column_circle_does_not_create_an_extra_row(self):
        circles = [*self.full_rows, (171, 700, 50)]
        cells, layout = self.detect(circles)
        self.assertEqual(len(cells), 16)
        self.assertEqual(layout["row_centers"], [420, 610, 800, 990])
        self.assertEqual(layout["discarded_centers"], [(171, 700)])
        self.assertEqual(len(self.detect(circles, fixed=False)[0]), 17)

    def test_off_column_circle_cannot_hide_missing_or_ambiguous_cells(self):
        off_column = (171, 700, 50)
        for circles in (
            [*self.full_rows[1:], off_column],
            [*self.full_rows[1:], (119, 420, 50), (120, 420, 50), off_column],
            [*self.full_rows[:4], *self.full_rows[8:], off_column],
            [*self.full_rows[4:], (171, 420, 50)],
        ):
            with self.subTest(circles=circles):
                cells, layout = self.detect(circles)
                self.assertFalse(cells)
                self.assertEqual(layout["reason"], "auto-layout-ambiguous-after-filtering-off-grid-circles")

    def test_off_column_circle_between_clipped_and_complete_rows(self):
        complete_y = [473.9, 662.0, 851.0, 1039.1]
        full_rows = [(x, y, 54) for y in complete_y for x in self.columns]
        clipped_row = [(x, 286, 54) for x in self.columns]
        cells, layout = self.detect([*full_rows, *clipped_row, (382.6, 360.8, 68.3)])
        self.assertEqual(len(cells), 16)
        self.assertEqual(layout["row_centers"], complete_y)

        # An off-column circle at the projected row height could still be a
        # real cell. It cannot establish coverage of that row.
        cells, layout = self.detect([*full_rows, *clipped_row, (382.6, 302, 50)])
        self.assertFalse(cells)
        self.assertEqual(layout["reason"], "auto-layout-ambiguous-after-filtering-off-grid-circles")

    def test_on_column_circle_between_rows_does_not_add_a_row(self):
        for complete_y, outlier_y in (
            ([439.4, 627.2, 816.2, 1004.0], 704.0),
            ([425.3, 613.1, 802.4, 990.5], 1065.2),
        ):
            with self.subTest(outlier_y=outlier_y):
                circles = [(x, y, 54) for y in complete_y for x in self.columns]
                cells, layout = self.detect([*circles, (495, outlier_y, 68)])
                self.assertEqual(len(cells), 16)
                self.assertEqual(layout["row_centers"], complete_y)
                [(discarded_x, discarded_y)] = layout["discarded_centers"]
                self.assertEqual(discarded_x, 495)
                self.assertAlmostEqual(discarded_y, outlier_y, places=2)

    def test_unobstructed_sparse_page_keeps_its_existing_layout(self):
        circles = self.full_rows[:13]
        self.assertEqual(self.detect(circles), self.detect(circles, fixed=False))
        cells, layout = self.detect([*circles, (454, 705, 50)])
        self.assertEqual(len(cells), 13)
        self.assertEqual(layout["row_centers"], [420, 610, 800, 990])

    def test_row_spacing_drift_preserves_a_sparse_final_row(self):
        complete_y = [363, 553, 750, 947]
        full_rows = [(x, y, 50) for y in complete_y for x in self.columns]
        cells, layout = self.detect([*full_rows, (118, 1123, 50)])
        self.assertEqual(len(cells), 17)
        self.assertEqual(layout["row_centers"], [*complete_y, 1123])
        self.assertEqual((cells[-1].row, cells[-1].column, cells[-1].center), (4, 0, (118, 1123)))


def item_pages():
    # The same four icons occur on every row, as with the story fragment boxes.
    # Distinct label strips distinguish them even when no icon has an item ID.
    icons = dict(enumerate(np.eye(4, dtype=np.float32)))
    labels = np.random.default_rng(0).integers(0, 2, (12, 26, 110)).astype(np.float32)

    def page(start, *, gap=False):
        return PageScan(
            results=[],
            rejected=[],
            row_features=[icons] * 4,
            column_count=4,
            layout={"row_centers": [350, 535, 720, 905], "coverage_gap": gap},
            row_entities=[{} for _ in range(4)],
            row_labels=list(labels[start:start + 4]),
        )

    return page


class BagPageRecoveryTests(unittest.TestCase):
    def test_item_labels_disambiguate_overlapping_identical_icons(self):
        page = item_pages()
        earlier, later = page(0), page(1)
        self.assertEqual(_bag_page_alignment(earlier, later)["shift"], 0)
        self.assertEqual(
            _bag_scan_alignment(earlier, later, .88, "tab1")["rows"],
            {1: 0, 2: 1, 3: 2},
        )
        self.assertEqual(_bag_scan_alignment(earlier, later, .88, "tab3-1")["shift"], 0)
        self.assertFalse(_bag_same_position(earlier, later, "tab1"))
        self.assertTrue(_bag_same_position(earlier, earlier, "tab1"))

    def test_backward_overlap_checks_known_pages_before_accepting_the_alignment(self):
        page = item_pages()
        previous, earlier = page(3), page(2)
        # The named labels identify an earlier row even though the icons all match.
        self.assertEqual(_bag_scan_alignment(previous, earlier, .88, "tab1")["rows"], {0: 1, 1: 2, 2: 3})
        bridge = _bag_scan_alignment(previous, previous, .88, "tab1")
        seen = []

        def resume(frame, current, threshold):
            seen.append((frame, current, threshold))
            return "rejoined", previous, bridge

        kwargs = {"resume_from_checkpoint": resume}
        result = _recover_bag_page(None, previous, "earlier", earlier, None, (), (None, None), {},
                                  None, "tab1", None, 500, None, {}, **kwargs)
        self.assertEqual(seen, [("earlier", earlier, .88)])
        self.assertEqual(result, ("rejoined", previous, bridge, True))

    def test_visual_recovery_requires_one_position_and_a_complete_grid(self):
        page = item_pages()
        prior, current = page(2), page(3)
        self.assertEqual(_bag_earlier_base({2: prior}, current, 8, .88, visual=True), 3)
        self.assertIsNone(_bag_earlier_base({2: prior}, current, 8, .88))
        self.assertIsNone(_bag_earlier_base({2: prior, 4: prior}, current, 8, .88, visual=True))
        self.assertIsNone(_bag_earlier_base({2: prior}, page(3, gap=True), 8, .88, visual=True))

    def test_recovery_rejoins_last_position_and_rejects_a_nonmoving_capture(self):
        page = item_pages()
        earlier, last = page(1), page(5)
        frames = iter((n, page(n)) for n in (3, 5))
        scrolled = []
        image, resumed, alignment = _resume_bag_from_checkpoint(
            last, 5, 1, 1, earlier, frames.__next__, scrolled.append, .88, visual=True,
        )
        self.assertEqual(image, 5)
        self.assertEqual(alignment["rows"], {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(len(scrolled), 2)
        _, _, alignment = _resume_bag_from_checkpoint(
            last, 5, 1, 1, earlier, lambda: (1, earlier), scrolled.append, .88, visual=True,
        )
        self.assertIsNone(alignment)
        self.assertEqual(len(scrolled), 3)


class BagTopRetryTests(unittest.TestCase):
    def setUp(self):
        self.page = item_pages()(0)
        self.page.layout["column_centers"] = [118, 286, 454, 622]
        self.roi = (34, 245, 672, 940)
        self.module = "custom.action.paged_item_recognition"

    def seek_top(self):
        _seek_bag_top(None, None, self.roi, (None, None), {}, (118, 286, 454, 622),
                      10, 500, .7, "tab3-2", None)

    def test_layout_loss_clears_stationary_confirmation(self):
        good = (None, self.page)
        bad = BagLayoutError("invalid column")
        with (
            patch(self.module + "._bag_probe", side_effect=[good, good, bad, good, good, good]),
            patch(self.module + "._bag_scroll") as swipe,
        ):
            self.seek_top()

        # One stable swipe before the error plus a fresh baseline and two stable swipes afterward.
        self.assertEqual(swipe.call_count, 5)
        self.assertEqual(swipe.call_args_list[1], swipe.call_args_list[2])
        _, (start_x, start_y, end_x, end_y, _), _, _ = swipe.call_args.args
        self.assertEqual(start_x, end_x)
        self.assertLess(start_y, end_y)

    def test_persistent_layout_failure_stops_after_two_extra_swipes(self):
        good = (None, self.page)
        bad = BagLayoutError("invalid column")
        with (
            patch(self.module + "._bag_probe", side_effect=[good, bad, bad, bad]),
            patch(self.module + "._bag_scroll") as swipe,
        ):
            with self.assertRaises(BagLayoutError):
                self.seek_top()
        self.assertEqual(swipe.call_count, 3)

    def test_non_layout_failure_is_not_retried(self):
        good = (None, self.page)
        with (
            patch(self.module + "._bag_probe", side_effect=[good, RuntimeError("popup")]),
            patch(self.module + "._bag_scroll") as swipe,
        ):
            with self.assertRaisesRegex(RuntimeError, "popup"):
                self.seek_top()
        self.assertEqual(swipe.call_count, 1)


if __name__ == "__main__":
    unittest.main()
