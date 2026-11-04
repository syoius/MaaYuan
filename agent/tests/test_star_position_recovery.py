"""Shared name/geometry veto and bounded recovery; synthetic frames are unit evidence only."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import star_four_row_inventory as rows
from custom.action.star_full_local_ocr_shadow import FullShadowSession
from test_star_page_transitions import synthetic_page as _synthetic_page

FIXTURE = Path(__file__).parent / 'fixtures/star_traditional_transition'


def synthetic_page(*args, **kwargs):
    page, image = _synthetic_page(*args, **kwargs)
    # The older cropped test fixture omits whole-frame metadata.
    page['geometry']['profile'].update(imageWidth=720, imageHeight=1280)
    return page, image


def real_pair():
    result = []
    for phase in ('before', 'after'):
        page = json.loads((FIXTURE / f'support-{phase}.json').read_text(encoding='utf-8'))
        image = np.zeros((1280, 720, 3), np.uint8)
        image[272:1044, 40:680] = cv2.imread(str(FIXTURE / f'support-{phase}-grid.png'))
        result.append((page, image))
    return result


def checkpoint(inventory):
    return copy.deepcopy((inventory.items, inventory.positions, inventory.next_row,
                          inventory.last_page, inventory.last_base, inventory.observation_count,
                          inventory.tail_diagnostics))


class SharedMatchingTests(unittest.TestCase):
    def test_real_false_three_row_candidate_is_vetoed_for_both_sections(self):
        for section in ('main', 'support'):
            (a, before), (b, after) = real_pair()
            for page in (a, b):
                page['section'] = section
            mapping = rows.locate_terminal_rows(a, b, before, after, 668)
            self.assertEqual(mapping['row_offset'], 4)
            rejected = next(c for c in mapping['rejected_candidates'] if c['row_offset'] == 3)
            self.assertEqual(rejected['reason'], 'conflicting OCR fields')
            self.assertTrue(all('name' in c['fields'] for c in rejected['conflicts']))
            self.assertEqual(len(mapping['anchors']), 4)
            self.assertTrue(all(c['evidence'] == 'name_pixels_only' for c in mapping['anchors']))
            inventory = rows.PositionInventory(section)
            inventory.accept_initial(a)
            accepted = copy.deepcopy(inventory.items)
            transition = inventory.accept_moved(b, before, after, 668)
            self.assertEqual((transition['new_count'], transition['duplicate_count']), (16, 0))
            self.assertEqual(inventory.items[:16], accepted)
            self.assertEqual(len(inventory.items), 32)

    def test_known_conflicts_veto_but_unknown_and_low_confidence_do_not(self):
        page, _ = synthetic_page()
        old = page['records'][0]
        for key, value in (('name', '太阳'), ('level', 2), ('quality', '紫')):
            new = copy.deepcopy(old)
            new['resolved'][key] = value
            if key == 'quality':
                old['quality'] = dict(quality='橙', confidence=.9)
                new['quality'] = dict(quality='紫', confidence=.9)
            self.assertIn(key, rows.field_conflicts(old, new))
            if key == 'quality':
                new['quality']['confidence'] = .1
            else:
                new['fields'][key]['confidence'] = .1
            self.assertNotIn(key, rows.field_conflicts(old, new))
            new['resolved'][key] = None
            self.assertNotIn(key, rows.field_conflicts(old, new))
        self.assertEqual(rows.field_conflicts(old, copy.deepcopy(old)), [])

    def test_public_disc_pixels_cannot_rescue_different_name_pixels(self):
        page, image = synthetic_page()
        old = page['records'][0]
        # The whole card is unchanged except its small name ROI.
        changed = image.copy()
        r = old['card']['nameRect']
        changed[r['y']:r['y']+r['height'], r['x']:r['x']+r['width']] = 0
        anchor = rows._tail_anchor(image, changed, old, old)
        self.assertTrue(anchor is None or anchor['ncc'] < anchor['threshold'])

    def test_reliable_level_and_quality_conflicts_reject_an_image_supported_offset(self):
        for field, value in (('level', 2), ('quality', '紫')):
            a, before = synthetic_page()
            b, after = synthetic_page(3, 433)
            b['records'][0]['resolved'][field] = value
            if field == 'quality':
                a['records'][12]['quality'] = dict(quality='橙', confidence=.9)
                b['records'][0]['quality'] = dict(quality='紫', confidence=.9)
            inventory = rows.PositionInventory('support');inventory.accept_initial(a)
            saved = checkpoint(inventory)
            with self.assertRaises(rows.PositionUncertain) as caught:
                inventory.accept_moved(b, before, after, 673)
            rejected = next(c for c in caught.exception.diagnostics['rejected'] if c['row_offset'] == 3)
            self.assertIn(field, rejected['conflicts'][0]['fields'])
            self.assertEqual(checkpoint(inventory), saved)

    def test_repeated_name_pixels_remain_ambiguous_and_do_not_prove_identity(self):
        def repeated(base, first):
            page, _ = synthetic_page(base, first)
            image = np.zeros((1280,720,3), np.uint8)
            for relative in range(0 if base == 0 else -1, 4):
                for col in range(4):
                    card = page['records'][col]['card']
                    rects = [card[k] for k in ('discRect','nameRect','levelRect')]
                    x=min(r['x'] for r in rects);y=min(r['y'] for r in rects)+relative*167
                    x2=max(r['x']+r['width'] for r in rects)
                    height=max(r['y']+r['height'] for r in rects)-min(r['y'] for r in rects)
                    texture=np.random.default_rng(col).integers(0,255,(height,x2-x,3),dtype=np.uint8)
                    top=max(272,y);bottom=min(1044,y+height)
                    if bottom>top:image[top:bottom,x:x2]=texture[top-y:bottom-y]
            return page,image
        a,before=repeated(0,403);b,after=repeated(4,433);r,reverse=repeated(2,433)
        with self.assertRaises(rows.PositionUncertain) as caught:
            rows.locate_terminal_rows(a,b,before,after,673)
        self.assertGreater(len(caught.exception.diagnostics['candidates']),1)
        for section in ('main','support'):
            inventory=rows.PositionInventory(section);inventory.accept_initial(a);saved=checkpoint(inventory)
            with self.assertRaisesRegex(ValueError,'position solutions'):
                inventory.recover_page(b,r,before,after,reverse,332)
            self.assertEqual(checkpoint(inventory),saved)

    def test_unknown_fields_do_not_make_empty_pixels_a_position_proof(self):
        a, image = synthetic_page()
        b, after = synthetic_page(4, 433)
        for page in (a, b):
            for record in page['records']:
                record['resolved']['name'] = None
        with self.assertRaises(rows.PositionUncertain):
            rows.locate_terminal_rows(a, b, image*0, after*0, 673)


class RecoveryDataTests(unittest.TestCase):
    def test_measured_one_two_or_three_row_reverse_is_verified_not_assumed(self):
        for section in ('main', 'support'):
            for recovered_base in (1, 2, 3):
                with self.subTest(section=section, recovered_base=recovered_base):
                    a, before = synthetic_page()
                    unresolved, image = synthetic_page(4, 433)
                    recovery, reverse_image = synthetic_page(recovered_base, 433)
                    inventory = rows.PositionInventory(section)
                    inventory.accept_initial(a)
                    accepted = copy.deepcopy(inventory.items)
                    result = inventory.recover_page(unresolved, recovery, before, image, reverse_image, 332)
                    self.assertEqual((result['resumed_base'], result['unresolved_base']), (recovered_base, 4))
                    self.assertEqual(inventory.items[:16], accepted)
                    self.assertEqual((len(inventory.items), inventory.next_row), (32, 8))
                    self.assertEqual(len(inventory.positions), 32)
                    # Four rows from the actual recovered screen overlap the known
                    # frontier. Re-appending a full 16 here would double count.
                    forward, forward_image = synthetic_page(recovered_base+4, 433)
                    transition = inventory.accept_moved(forward, reverse_image, forward_image, 673)
                    self.assertEqual(transition['duplicate_count'], (4-recovered_base)*4)
                    self.assertEqual(transition['new_count'], recovered_base*4)
                    self.assertEqual(len(inventory.items), (8+recovered_base)*4)
                    self.assertEqual(set(inventory.positions), {(row, col) for row in range(8+recovered_base) for col in range(4)})
                    self.assertTrue(all(x['name'] == '天同' for x in inventory.items))

    def test_failed_recovery_preserves_every_part_of_the_checkpoint(self):
        a, before = synthetic_page()
        unresolved, image = synthetic_page(4, 433)
        recovery, reverse_image = synthetic_page(2, 433)
        for mode in ('no_pixels', 'conflict_after_trial_merge', 'unknown_field'):
            inventory = rows.PositionInventory('support')
            inventory.accept_initial(a)
            saved = checkpoint(inventory)
            pending = copy.deepcopy(unresolved)
            if mode == 'conflict_after_trial_merge':
                pending['records'][0]['resolved']['name'] = '巨门'
                pending['records'][0]['fields']['name']['confidence'] = .1
            if mode == 'unknown_field':
                pending['records'][0]['resolved']['name'] = None
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                inventory.recover_page(pending, recovery, before,
                    image*0 if mode == 'no_pixels' else image, reverse_image, 332)
            self.assertEqual(checkpoint(inventory), saved)

    def test_two_row_plan_uses_measured_pitch_and_safe_downward_finger_motion(self):
        for pitch in (155, 175):
            page, _ = synthetic_page()
            for record in page['records']:
                dy=record['row']*(pitch-167)
                record['card']['circle']['centerY'] += dy
                for name in ('discRect', 'nameRect', 'levelRect'):
                    record['card'][name]['y'] += dy
            page['geometry']['cards'] = [r['card'] for r in page['records']]
            plan = rows.recovery_swipe(page['geometry'], page['geometry'])
            self.assertEqual(plan['row_pitch_px'], pitch)
            self.assertEqual(plan['touch_length_px'], round(2*pitch/rows.CALIBRATED_GAIN))
            self.assertLess(plan['swipe_begin'][1], plan['swipe_end'][1])
            self.assertLess(plan['swipe_end'][1], 1044)


class RecoveryLoopTests(unittest.TestCase):
    def owner(self, page_frames, swipes, root):
        source = iter(swipes)
        lookup = {id(image): page for page, image in page_frames}
        saved, gestures, recognized = [], [], []
        owner = SimpleNamespace(inventories={'main': rows.PositionInventory('main')}, sections={},
            progress=lambda stage: None, last_capture={'file': 'unit.png'}, out=Path(root),
            recovery_checkpoints=set(), position_recoveries=[], events=[])
        def recognize(image, sid):
            recognized.append(sid)
            page = copy.deepcopy(lookup[id(image)])
            page['sourceImageId'] = sid
            return page
        def swipe(summary, touch, label, **kwargs):
            summary['swipes'] += 1
            gestures.append((touch, label, kwargs))
            return next(source), {}
        owner.recognize = recognize
        owner.swipe_capture = swipe
        owner.save_page = lambda page, *args: saved.append(copy.deepcopy(page))
        owner.confirm_bottom = MethodType(FullShadowSession.confirm_bottom, owner)
        owner.recover_position = MethodType(FullShadowSession.recover_position, owner)
        return owner, saved, gestures, recognized

    def test_true_uncertainty_gets_one_reverse_then_four_row_scan_and_bottom(self):
        pages = [synthetic_page(0), synthetic_page(4,433), synthetic_page(2,433), synthetic_page(6,433)]
        original = rows.locate_terminal_rows
        calls = []
        def once(a, b, *args):
            calls.append(b['sourceImageId'])
            if len(calls) == 1:
                raise rows.PositionUncertain(dict(candidates=[], rejected=[], mapping_ms=.1))
            return original(a, b, *args)
        with tempfile.TemporaryDirectory() as directory:
            owner, saved, gestures, recognized = self.owner(pages,
                [pages[1][1], pages[2][1], pages[3][1], pages[3][1], pages[3][1]], directory)
            with patch.object(rows, 'locate_terminal_rows', side_effect=once):
                result = FullShadowSession.scan(owner, 'main', pages[0][1])
            self.assertTrue(result['complete'])
            self.assertEqual(result['final_count'], 40)
            self.assertEqual(result['recovery_attempts'], 1)
            self.assertEqual(sum('recovery_plan' in kwargs for _, _, kwargs in gestures), 1)
            self.assertEqual(len(recognized), 4)  # No repeated OCR of the checkpoint.
            self.assertEqual([p['status'] for p in saved], ['accepted', 'position_unconfirmed', 'recovery_merged', 'position_merged'])
            self.assertTrue(owner.position_recoveries[0]['success'])
            self.assertEqual(len(owner.inventories['main'].positions), 40)

    def test_real_false_ambiguity_needs_no_reverse_gesture(self):
        pairs = real_pair()
        with tempfile.TemporaryDirectory() as directory:
            owner, _, gestures, _ = self.owner(pairs, [pairs[1][1]]*3, directory)
            result = FullShadowSession.scan(owner, 'main', pairs[0][1])
            self.assertEqual(result['final_count'], 32)
            self.assertEqual(result['recovery_attempts'], 0)
            self.assertFalse(any(kwargs for _, _, kwargs in gestures))
            self.assertEqual(result['stop_reason'], 'bottom_no_move_confirmed')

    def test_failed_recovery_does_not_loop_or_merge_unresolved_instances(self):
        a, before = synthetic_page()
        b, after = synthetic_page(4,433)
        bad, reverse = synthetic_page(2,433)
        with tempfile.TemporaryDirectory() as directory:
            empty = reverse*0
            owner, _, gestures, _ = self.owner([(a,before),(b,after),(bad,empty)], [after,empty], directory)
            with patch.object(rows, 'locate_terminal_rows', side_effect=rows.PositionUncertain(dict(candidates=[],rejected=[],mapping_ms=0))):
                with self.assertRaisesRegex(RuntimeError, 'recovery has 0 reliable position solutions'):
                    FullShadowSession.scan(owner, 'main', before)
            self.assertEqual(len(gestures), 2)
            self.assertEqual(len(owner.inventories['main'].items), 16)
            self.assertEqual(owner.sections['main']['recovery_attempts'], 1)
            self.assertFalse(owner.position_recoveries[0]['success'])

    def test_same_unresolved_frontier_cannot_trigger_another_reverse(self):
        a, image = synthetic_page()
        with tempfile.TemporaryDirectory() as directory:
            owner, _, gestures, _ = self.owner([(a,image)], [], directory)
            owner.section = 'main'
            inventory = owner.inventories['main'];inventory.accept_initial(a)
            owner.recovery_checkpoints.add(('main',inventory.next_row))
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                owner.recover_position({},inventory,image,a,image,673,rows.PositionUncertain(dict(candidates=[],rejected=[],mapping_ms=0)))
            self.assertEqual(gestures, [])


if __name__ == '__main__':
    unittest.main()
