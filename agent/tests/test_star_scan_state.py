"""Shared scan state regressions: real grids plus explicitly synthetic loops."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import test_star_position_recovery as recovery_tests
synthetic_page = recovery_tests.synthetic_page
from custom.action.star_full_local_ocr_shadow import FullShadowSession
from utils import star_four_row_inventory as rows

FIXTURE = Path(__file__).parent / 'fixtures/star_scan_state'


def real_pages():
    pages = []
    for sid in ('support-005', 'support-006', 'support-007-confirmation'):
        page = json.loads((FIXTURE / f'{sid}.json').read_text(encoding='utf-8'))
        image = np.zeros((1280, 720, 3), np.uint8)
        image[272:1044, 40:680] = cv2.imread(str(FIXTURE / f'{sid}-grid.png'))
        pages.append((page, image))
    return pages


def owner_for(section, pairs, frames, directory):
    owner, saved, gestures, recognized = recovery_tests.RecoveryLoopTests().owner(pairs, frames, directory)
    owner.inventories = {section: rows.PositionInventory(section)}
    return owner, saved, gestures, recognized


def repeated_page(base, first):
    """All rows share exactly the same observable name pixels, unlike unique textures."""
    page, image = synthetic_page(base, first)
    image[:] = 0
    for relative in range(-1 if base else 0, 4):
        for col in range(4):
            card = page['records'][col]['card']
            rect = card['nameRect']
            x, y, w, h = (rect[k] for k in ('x', 'y', 'width', 'height'))
            y += relative * 167
            texture = np.random.default_rng(col).integers(0, 255, (h, w, 3), dtype=np.uint8)
            top, bottom = max(272, y), min(1044, y+h)
            if bottom > top:
                image[top:bottom, x:x+w] = texture[top-y:bottom-y]
    return page, image


class ScanStateTests(unittest.TestCase):
    def test_real_059_064_067_adds_sixteen_then_one_without_guessing_capacity(self):
        pages = real_pages()
        self.assertGreater(rows.same_position(pages[0][1], pages[1][1]), .97)
        for section in ('main', 'support'):
            inventory = rows.PositionInventory(section)
            inventory.accept_initial(pages[0][0])
            for index, expected in enumerate(((4, 666, 16, 0), (1, 118, 1, 12))):
                previous, before = pages[index]
                current, after = pages[index+1]
                decision = rows.classify_scan_transition(previous, current, before, after, 655)
                self.assertEqual(decision['state'], 'MOVED')
                mapping = decision['mapping']
                result = inventory.accept_moved(current, before, after, 655, mapping=mapping)
                self.assertEqual((mapping['row_offset'], round(mapping['anchor_shift_px']),
                                  result['new_count'], result['duplicate_count']), expected)
            self.assertEqual(len(inventory.items), 33)
            self.assertEqual(len(inventory.positions), 33)
            last, image = pages[-1]
            proof = rows.classify_scan_transition(last, last, image, image, 655)
            self.assertTrue(rows.confirmed_no_move(proof, proof, proof))
            self.assertEqual(len(last['records']), 13)  # Partial discs are not instances.

    def test_no_move_needs_unique_position_proof_independent_of_whole_image_ncc(self):
        a, before = synthetic_page()
        b, after = synthetic_page(4, 433)
        with patch.object(rows, 'same_position', return_value=1.):
            decision = rows.classify_scan_transition(a, b, before, after, 673)
        self.assertEqual(decision['state'], 'MOVED')
        with patch.object(rows, 'same_position', return_value=0.):
            decision = rows.classify_scan_transition(a, a, before, before, 673)
        self.assertEqual(decision['state'], 'NO_MOVE_CANDIDATE')
        self.assertFalse(rows.confirmed_no_move(before, before, before))

    def test_low_confidence_changed_reading_is_not_silently_discarded_at_bottom(self):
        a, image = synthetic_page()
        b = copy.deepcopy(a)
        b['records'][0]['resolved']['name'] = '巨门'
        b['records'][0]['fields']['name']['confidence'] = .1
        self.assertEqual(rows.classify_scan_transition(a, b, image, image, 673)['state'], 'UNCERTAIN')

    def test_confirmation_move_is_consumed_once_and_resumes_normal_scan_for_both_sections(self):
        pairs = real_pages()
        for section in ('main', 'support'):
            # First swipe really stays put; confirmation then advances four
            # rows. The next normal swipe reveals the 13-card terminal page.
            frames = [pairs[0][1], pairs[1][1], pairs[2][1], pairs[2][1], pairs[2][1]]
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                owner, saved, gestures, recognized = owner_for(section, pairs, frames, directory)
                result = FullShadowSession.scan(owner, section, pairs[0][1])
                self.assertTrue(result['complete'])
                self.assertEqual(result['final_count'], 33)
                self.assertEqual((result['bottom_attempts'], result['bottom_resumes'], result['recovery_attempts']), (2, 1, 0))
                self.assertEqual(len(recognized), 6)
                self.assertEqual(len(set(recognized)), 6)
                self.assertEqual([p['scan_transition']['state'] for p in saved[1:]],
                    ['NO_MOVE_CANDIDATE', 'MOVED', 'MOVED', 'NO_MOVE_CANDIDATE', 'NO_MOVE_CANDIDATE'])
                self.assertEqual([p['acceptance'].get('new_count', 0) for p in saved[1:]], [0,16,1,0,0])
                self.assertEqual(len(gestures), 5)

    def test_partial_last_rows_and_short_predecessors_confirm_without_recovery(self):
        for section in ('main', 'support'):
            for size in (1, 2, 3):
                for lengths in ((size,), (4, size), (4, 4, 4, size)):
                    page, image = synthetic_page(0, 403, lengths)
                    with self.subTest(section=section, lengths=lengths), tempfile.TemporaryDirectory() as directory:
                        owner, _, gestures, recognized = owner_for(section, [(page,image)], [image,image], directory)
                        result = FullShadowSession.scan(owner, section, image)
                        self.assertTrue(result['complete'])
                        self.assertEqual(result['final_count'], sum(lengths))
                        self.assertEqual(result['recovery_attempts'], 0)
                        self.assertEqual(len(gestures), 2)
                        self.assertEqual(len(recognized), 3)

    def test_indistinguishable_repetition_stops_partial_without_pointless_reverse(self):
        for scenario, target in (('unknown_advance', repeated_page(4,433)), ('same_pixels', repeated_page(0,403))):
            pairs = [repeated_page(0,403), target]
            for section in ('main', 'support'):
                with self.subTest(section=section, scenario=scenario), tempfile.TemporaryDirectory() as directory:
                    owner, saved, gestures, _ = owner_for(section, pairs, [pairs[1][1]], directory)
                    with self.assertRaisesRegex(RuntimeError, 'cannot add evidence'):
                        FullShadowSession.scan(owner, section, pairs[0][1])
                    result = owner.sections[section]
                    self.assertFalse(result['complete'])
                    self.assertEqual(len(owner.inventories[section].items), 16)
                    self.assertEqual(result['recovery_attempts'], 0)
                    self.assertEqual(len(gestures), 1)
                    self.assertGreater(len(saved[-1]['position_diagnostics']['candidates']), 1)

    def test_far_history_does_not_make_a_pointless_reverse_meaningful(self):
        older, _ = repeated_page(0,403)
        older['records'][0]['resolved']['name'] = '太阴'
        current, before = repeated_page(4,433)
        pending, after = repeated_page(8,433)
        inventory = rows.PositionInventory('support')
        inventory.accept_initial(older)
        inventory.accept_normal(current)
        decision = rows.classify_scan_transition(current, pending, before, after, 673)
        self.assertEqual(decision['state'], 'UNCERTAIN')
        self.assertFalse(rows.recovery_has_new_evidence(inventory,pending,before,decision['diagnostics']))
        # A different known row within reverse reach can add a real constraint.
        inventory.items[12]['name'] = '太阴'
        self.assertTrue(rows.recovery_has_new_evidence(inventory,pending,before,decision['diagnostics']))

    def test_two_small_shifts_cannot_accumulate_into_false_bottom(self):
        pairs = [synthetic_page(0,403), synthetic_page(0,402), synthetic_page(0,400)]
        with tempfile.TemporaryDirectory() as directory:
            owner, _, gestures, _ = owner_for('main', pairs, [pairs[1][1],pairs[2][1]], directory)
            with self.assertRaisesRegex(RuntimeError, 'endpoint position unconfirmed'):
                FullShadowSession.scan(owner, 'main', pairs[0][1])
            self.assertFalse(owner.sections['main']['complete'])
            self.assertEqual(len(gestures), 2)
            self.assertEqual(len(owner.inventories['main'].items), 16)


if __name__ == '__main__':
    unittest.main()
