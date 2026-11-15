from pathlib import Path
from types import MethodType
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import numpy as np
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))

from custom.action.star_full_local_ocr_shadow import FullShadowSession
from star_fixture_inputs import replay_owner
from star_fixture_inputs import replay_page
from star_fixture_inputs import synthetic_page
from utils import star_four_row_inventory as rows
from utils.star_four_row_inventory import PositionInventory


def real_pair():
    return [replay_page('star_traditional_transition', f'support-{phase}')
            for phase in ('before', 'after')]

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

    def test_recovery_prefers_four_rows_and_merges_only_new_positions(self):
        pages = [synthetic_page(base,first) for base,first in
                 ((0,358),(4,390),(2,390),(6,400),(10,402))]
        original = rows.locate_terminal_rows
        calls = []
        def once(a,b,*args):
            calls.append(b['sourceImageId'])
            if len(calls)==1:
                raise rows.PositionUncertain(dict(candidates=[],rejected=[],mapping_ms=.1))
            return original(a,b,*args)
        with tempfile.TemporaryDirectory() as directory:
            owner,saved,gestures,recognized=replay_owner(pages,
                [pages[i][1] for i in (1,2,3,4,4,4)],directory)
            with patch.object(rows,'locate_terminal_rows',side_effect=once):
                result=FullShadowSession.scan(owner,'main',pages[0][1])
            self.assertTrue(result['complete'])
            self.assertEqual(result['final_count'],56)
            self.assertEqual(result['recovery_attempts'],1)
            self.assertEqual(sum('recovery_plan' in kwargs for _,_,kwargs in gestures),1)
            self.assertEqual(len(recognized),7)
            self.assertEqual(len(set(recognized)),7)
            self.assertEqual([p['status'] for p in saved],['accepted','position_unconfirmed',
                'recovery_merged','position_merged','accepted','position_merged','position_merged'])
            self.assertTrue(owner.position_recoveries[0]['success'])
            self.assertNotIn('adaptive',saved[1])
            self.assertEqual(saved[2]['adaptive']['mode'],'normal')
            self.assertNotEqual(gestures[0][0],gestures[2][0])
            self.assertEqual(saved[2]['adaptive']['target_rows'],4)
            self.assertNotIn('forward_plan',gestures[2][2])
            self.assertEqual(saved[3]['adaptive']['mode'],'normal')
            self.assertEqual(saved[3]['acceptance']['new_count'],8)
            self.assertEqual(saved[3]['acceptance']['duplicate_count'],8)
            self.assertEqual(saved[4]['acceptance']['new_count'],16)
            self.assertEqual(set(owner.inventories['main'].positions),
                             {(row,col) for row in range(14) for col in range(4)})

    def test_real_false_ambiguity_needs_no_reverse_gesture(self):
        pairs = real_pair()
        with tempfile.TemporaryDirectory() as directory:
            owner, _, gestures, _ = replay_owner(pairs, [pairs[1][1]]*3, directory)
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
            owner, _, gestures, _ = replay_owner([(a,before),(b,after),(bad,empty)], [after,empty], directory)
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
            owner, _, gestures, _ = replay_owner([(a,image)], [], directory)
            owner.section = 'main'
            inventory = owner.inventories['main'];inventory.accept_initial(a)
            owner.recovery_checkpoints.add(('main',inventory.next_row))
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                owner.recover_position({},inventory,image,a,image,673,rows.PositionUncertain(dict(candidates=[],rejected=[],mapping_ms=0)))
            self.assertEqual(gestures, [])

def with_name_obstruction(page, part='nameRect'):
 page=copy.deepcopy(page)
 # Fixture shifts all circles to the same y; make the other names fit so the
 # scenario contains exactly one overrun, as opposed to mixed invalid geometry.
 for other in page['records']:
  rect=other['card']['nameRect']
  rect['height']=min(rect['height'],1044-rect['y'])
 record=next(r for r in page['records'] if (r['row'],r['column'])==(3,2))
 card=record['card'];rect=card[part]
 rect['height']=1047-rect['y']
 card.update(completeness='partial_bottom',evidence=['disc_or_ocr_roi_below_content_bottom'])
 page['records'].remove(record)
 return page

class ScanControlTests(unittest.TestCase):
 def owner(self, pairs, swipes):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  lookup={id(im):p for p,im in pairs};frames=iter(swipes);saved=[];gestures=[]
  owner=SimpleNamespace(inventories={'main':PositionInventory('main')},sections={},
   progress=lambda stage:None,last_capture={'file':'unit.png'},compensation_checkpoints=set())
  def recognize(im,sid):
   page=copy.deepcopy(lookup[id(im)]);page['sourceImageId']=sid;return page
  def swipe(summary,touch,label,**kwargs):
   summary['swipes']+=1;gestures.append((touch,label,kwargs));return next(frames),{}
  owner.recognize=recognize;owner.swipe_capture=swipe
  owner.save_page=lambda page,*args:saved.append(copy.deepcopy(page))
  owner.confirm_bottom=MethodType(FullShadowSession.confirm_bottom,owner)
  owner.compensate_name=MethodType(FullShadowSession.compensate_name,owner)
  return owner,saved,gestures

 def test_feedback_replans_current_phase_without_step_clamp(self):
  previous,_=synthetic_page(first=358);current,after=synthetic_page(4,400)
  _,before=synthetic_page(first=358)
  mapping=rows.locate_terminal_rows(previous,current,before,after,630)
  evidence=rows.measured_forward(previous['geometry'],current['geometry'],630,mapping)
  result=rows.next_swipe(current['geometry'],630,evidence)
  self.assertAlmostEqual(evidence['observed_gain'],626/630)
  self.assertEqual(result['mode'],'normal')
  self.assertGreater(result['touch_length_px'],660)
  self.assertFalse(result['clamped'])
  self.assertGreaterEqual(400+668-result['touch_length_px']*evidence['observed_gain'],390)
  self.assertEqual(result['reason'],'four_row_safe_landing_interval')

 def test_consecutive_normal_moves_calibrate_without_learning_no_move(self):
  pairs=[synthetic_page(base,first) for base,first in ((0,358),(4,425),(8,390))]
  start,first,last=[im for _,im in pairs]
  owner,saved,gestures=self.owner(pairs,[first,last,last,last])
  result=FullShadowSession.scan(owner,'main',start)
  self.assertEqual(result['final_count'],48)
  learned=[p['adaptive'] for p in saved if 'adaptive' in p]
  self.assertEqual([p['measurement']['actual_anchor_shift_px'] for p in learned],[601,703])
  self.assertEqual([p['mode'] for p in learned],['normal','normal'])
  self.assertAlmostEqual(learned[1]['observed_gain'],703/gestures[1][0])
  self.assertEqual(gestures[2][0],gestures[3][0])
  self.assertNotIn('adaptive',saved[-2]);self.assertNotIn('adaptive',saved[-1])

 def test_compensation_uses_verified_move_and_commits_cumulative_position_once(self):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  a,b,c=synthetic_page(first=425),synthetic_page(4,453),synthetic_page(4,418)
  clipped=(with_name_obstruction(b[0]),b[1]);owner,saved,gestures=self.owner([a,clipped,c],[b[1],c[1],c[1],c[1]])
  result=FullShadowSession.scan(owner,'main',a[1])
  self.assertTrue(result['complete']);self.assertEqual(result['final_count'],32)
  self.assertEqual(len(owner.inventories['main'].positions),32)
  self.assertEqual(sum('compensation_plan' in kwargs for _,_,kwargs in gestures),1)
  distance,_,kwargs=gestures[1];plan=kwargs['compensation_plan']
  self.assertEqual(plan['swipe_begin'][1]-plan['swipe_end'][1],distance)
  self.assertGreater(plan['target_content_shift_px'],10)
  self.assertEqual(saved[1]['status'],'name_compensation_required')
  self.assertEqual(saved[2]['name_compensation'],dict(normal_shift_px=640.,compensation_shift_px=35.,combined_shift_px=675.))
  self.assertEqual(saved[2]['adaptive']['observed_gain'],640/gestures[0][0])
  self.assertEqual(saved[2]['acceptance']['new_count'],16)
  self.assertEqual(saved[2]['acceptance']['position_mapping']['anchor_shift_px'],675.)

 def test_only_name_obstruction_can_trigger_and_failed_compensation_stops(self):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  a,b=synthetic_page(first=425),synthetic_page(4,453)
  for part,expected in [('discRect',1),('levelRect',1),('nameRect',2)]:
   with self.subTest(part=part):
    clipped=(with_name_obstruction(b[0],part),b[1])
    owner,saved,gestures=self.owner([a,clipped],[b[1]]*2)
    with self.assertRaises(ValueError if expected==1 else RuntimeError):
     FullShadowSession.scan(owner,'main',a[1])
    self.assertEqual(len(gestures),expected);self.assertEqual(len(owner.inventories['main'].items),16)
    self.assertFalse(owner.sections['main']['complete'])

 def test_ambiguous_or_stationary_clip_cannot_authorize_compensation(self):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  a,b=synthetic_page(first=425),synthetic_page(4,453)
  for clipped_image in (b[1]*0,a[1]):
   clipped=(with_name_obstruction(b[0]),clipped_image.copy())
   owner,_,gestures=self.owner([a,clipped],[clipped[1]])
   with self.assertRaises(ValueError):FullShadowSession.scan(owner,'main',a[1])
   self.assertEqual(len(gestures),1);self.assertEqual(len(owner.inventories['main'].items),16)

 def test_confirmation_with_name_obstruction_stops_without_compensation(self):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  a,b=synthetic_page(first=425),synthetic_page(4,453)
  clipped=(with_name_obstruction(b[0]),b[1]);owner,_,gestures=self.owner([a,clipped],[a[1],b[1]])
  with self.assertRaisesRegex(RuntimeError,'incomplete name on confirmation'):
   FullShadowSession.scan(owner,'main',a[1])
  self.assertEqual([label for _,label,_ in gestures],['page-001','bottom-confirmation'])
  self.assertEqual(len(owner.inventories['main'].items),16)

class LoopTests(unittest.TestCase):
 def test_overlap_bridge_then_full_advance_then_confirmed_bottom(self):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  pages=[synthetic_page(base,first) for base,first in ((0,403),(1,433),(3,390),(7,390))]
  lookup={id(im):p for p,im in pages};frames=iter([pages[1][1],pages[2][1],pages[3][1],pages[3][1],pages[3][1]]);saved=[]
  owner=SimpleNamespace(inventories={'main':PositionInventory('main')},sections={},progress=lambda stage:None,
   recognize=lambda im,sid:copy.deepcopy(lookup[id(im)]),save_page=lambda page,*args:saved.append(page),last_capture={'file':'unit.png'})
  def swipe(summary,touch,label,**kwargs):summary['swipes']+=1;return next(frames),{}
  owner.swipe_capture=swipe;owner.confirm_bottom=MethodType(FullShadowSession.confirm_bottom,owner)
  result=FullShadowSession.scan(owner,'main',pages[0][1])
  self.assertTrue(result['complete']);self.assertEqual(result['stop_reason'],'bottom_no_move_confirmed');self.assertEqual(result['final_count'],44);self.assertEqual(len(saved),6)
  self.assertTrue(all(p.get("scan_transition",{}).get("state")=="NO_MOVE_CANDIDATE" for p in saved[-2:]))
 def test_stabilization_is_bounded_and_requires_two_consecutive_stable_pairs(self):
  from custom.action import star_full_local_ocr_shadow as session
  _,image=synthetic_page();moved=np.roll(image,5,axis=0)
  for frames,success in (([moved,image,moved,image,moved,image],False),([moved,moved,image,image,image],True)):
   source=iter(frames);calls=[]
   owner=SimpleNamespace(check=lambda:None,check_backpack=lambda *args:None,capture=lambda label:calls.append(label) or next(source))
   with patch.object(session.time,'sleep'):
    if success:session.FullShadowSession.stabilize(owner,image,'unit','main');self.assertEqual(len(calls),5)
    else:
     with self.assertRaisesRegex(RuntimeError,'settle'):session.FullShadowSession.stabilize(owner,image,'unit','main')
     self.assertEqual(len(calls),6)
 def test_complete_scan_unknown_or_mismatched_game_count_is_not_uploadable(self):
  from custom.action import star_full_local_ocr_shadow as session
  from custom.action import star_inventory_sync as sync
  for bag in ({'complete':True,'currentCount':3,'capacity':270,'warnings':[]},{'complete':False,'currentCount':None,'capacity':None,'warnings':['unconfirmed']}):
   with tempfile.TemporaryDirectory() as name:
    base=Path(name);runtime=base/'runtime';runtime.mkdir()
    with patch.object(session,'REPO',base),patch.object(session,'PACKAGED_RUNTIME',runtime):a=session.FullShadowSession(SimpleNamespace(tasker=SimpleNamespace(stopping=False)),{'shadow':True})
    image=np.zeros((1280,720,3),np.uint8);a.prepare=lambda:image;a.stabilize=lambda im,*args:im
    def read(im):a.bag=copy.deepcopy(bag);return im
    a.read_bag=read
    def scan(section,im):a.sections[section]={'complete':True};a.inventories[section].items=[{'name':'unit'}]
    a.scan=scan
    def switch(section):a.last_capture={'file':'unit.png'};return image
    a.switch=switch
    with patch.object(session,'NativeStarOcr',return_value=SimpleNamespace()),patch.object(session,'recognize_experience',return_value={'complete':True,'structured':True,'orangeCount':0,'purpleCount':0,'whiteCount':0}):self.assertFalse(a.execute())
    local=json.loads((a.out/'inventory.json').read_text(encoding='utf-8'));self.assertFalse(local['complete'])
    error=json.loads((a.out/'error.json').read_text(encoding='utf-8'))['error'];self.assertIn('differs' if bag['complete'] else 'unconfirmed',error)
    with patch.object(sync,'post_inventory') as post:
     with self.assertRaises(ValueError):sync.sync_saved_inventory(object(),a.out)
     post.assert_not_called()

def real_pages():
    return [replay_page('star_scan_state', name)
            for name in ('support-005', 'support-006', 'support-007-confirmation')]

def owner_for(section, pairs, frames, directory):
    owner, saved, gestures, recognized = replay_owner(pairs, frames, directory)
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

    def test_indistinguishable_repetition_gets_one_small_probe_then_stops(self):
        for scenario, target in (('unknown_advance', repeated_page(4,433)), ('same_pixels', repeated_page(0,403))):
            pairs = [repeated_page(0,403), target, repeated_page(2,433)]
            for section in ('main', 'support'):
                with self.subTest(section=section, scenario=scenario), tempfile.TemporaryDirectory() as directory:
                    owner, saved, gestures, _ = owner_for(section,pairs,[pairs[1][1],pairs[2][1]],directory)
                    with self.assertRaisesRegex(RuntimeError,'reliable position solutions'):
                        FullShadowSession.scan(owner, section, pairs[0][1])
                    result = owner.sections[section]
                    self.assertFalse(result['complete'])
                    self.assertEqual(len(owner.inventories[section].items), 16)
                    self.assertEqual(result['recovery_attempts'],1)
                    self.assertEqual(len(gestures),2)
                    self.assertEqual(gestures[1][2]['recovery_plan']['target_rows'],1)
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



class SwipeCalibrationRecoveryTests(unittest.TestCase):
    def test_unique_four_and_three_row_overlap_are_measured_not_assumed(self):
        a,before=synthetic_page()
        for offset in (3,4):
            b,after=synthetic_page(offset,403)
            touch=offset*167
            decision=rows.classify_scan_transition(a,b,before,after,touch)
            self.assertEqual(decision['state'],'MOVED')
            feedback=rows.measured_forward(a['geometry'],b['geometry'],touch,decision['mapping'])
            self.assertAlmostEqual(feedback['observed_gain'],1.)
            self.assertEqual(feedback['actual_anchor_shift_px'],touch)
        for mapping in ({'row_offset':-1,'anchor_shift_px':-167,'geometric_shift_px':-167},
                        {'row_offset':0,'anchor_shift_px':0,'geometric_shift_px':0},
                        {'row_offset':4,'anchor_shift_px':668,'geometric_shift_px':650}):
            self.assertIsNone(rows.measured_forward(a['geometry'],a['geometry'],668,mapping))

    def test_unsafe_four_row_target_uses_short_overlap_then_returns_to_normal(self):
        page,_=synthetic_page(4,425)
        feedback={'observed_gain':598/630}
        plan=rows.next_swipe(page['geometry'],630,feedback)
        self.assertGreater(plan['four_row_suggested_touch_length_px'],700)
        self.assertEqual((plan['mode'],plan['target_rows']),('bridge',2))
        self.assertLess(plan['touch_length_px'],600)
        self.assertAlmostEqual(425+334-plan['touch_length_px']*feedback['observed_gain'],390,delta=1)
        settled,_=synthetic_page(6,390)
        normal=rows.next_swipe(settled['geometry'],plan['touch_length_px'],{'observed_gain':1.})
        self.assertEqual((normal['mode'],normal['target_rows']),('normal',4))
        self.assertTrue(600<=normal['touch_length_px']<=700)
        for c in rows.layout(settled['geometry'])['groups'][-1]:
            self.assertGreaterEqual(c['nameRect']['y']-normal['touch_length_px']*normal['gain_bounds'][1],278)

    def test_gain_can_exclude_alias_but_cannot_confirm_remaining_alias(self):
        a,before=repeated_page(0,403);b,after=repeated_page(4,433)
        raw=rows.classify_scan_transition(a,b,before,after,673)
        self.assertGreater(len(raw['diagnostics']['candidates']),1)
        decision=rows.classify_scan_transition(a,b,before,after,673,gain_bounds=(.94,1.02))
        self.assertEqual(decision['state'],'UNCERTAIN')
        self.assertEqual(len(decision['diagnostics']['candidates']),1)
        self.assertGreater(len(decision['diagnostics']['pixel_candidates']),1)
        self.assertTrue(decision['diagnostics']['gain_filter_cannot_confirm'])
        empty=rows.classify_scan_transition(a,b,before*0,after*0,673,gain_bounds=(.94,1.02))
        self.assertEqual(empty['state'],'UNCERTAIN')
        self.assertEqual(empty['diagnostics']['candidates'],[])

    def test_three_complete_recovery_rows_merge_atomically_and_keep_duplicates(self):
        a,before=synthetic_page();pending,after=synthetic_page(4,433)
        recovery,reverse=synthetic_page(2,433,(4,4,4))
        v=rows.PositionInventory('support');v.accept_initial(a);accepted=copy.deepcopy(v.items)
        result=v.recover_page(pending,recovery,before,after,reverse,332)
        self.assertEqual((result['resumed_base'],result['unresolved_base']),(2,4))
        self.assertEqual((len(v.items),v.next_row,v.last_base),(32,8,2))
        self.assertEqual(v.items[:16],accepted)
        self.assertEqual(set(v.positions),{(row,col) for row in range(8) for col in range(4)})
        self.assertTrue(all(item['name']=='天同' for item in v.items))
        mapping=result['unresolved_page']['position_mapping']
        feedback=rows.measured_forward(a['geometry'],pending['geometry'],638,mapping)
        self.assertAlmostEqual(feedback['observed_gain'],1.)
        summary={};plan=rows.plan_forward(summary,recovery['geometry'],694,feedback)
        self.assertEqual((plan['mode'],plan['target_rows']),('bridge',2))
        self.assertNotEqual(plan['touch_length_px'],694)
        self.assertEqual(summary['forward_gain_samples'],[1.])

    def test_ambiguous_three_row_recovery_stops_with_every_checkpoint_field_intact(self):
        a,before=repeated_page(0,403);pending,after=repeated_page(4,433)
        recovery,reverse=repeated_page(2,433)
        recovery['records']=[r for r in recovery['records'] if r['row']<3]
        recovery['geometry']['cards']=[r['card'] for r in recovery['records']]
        v=rows.PositionInventory('support');v.accept_initial(a);saved=checkpoint(v)
        with self.assertRaisesRegex(ValueError,'reliable position solutions'):
            v.recover_page(pending,recovery,before,after,reverse,332)
        self.assertEqual(checkpoint(v),saved)
        recovery['records'][0]['resolved']['name']=None
        with self.assertRaisesRegex(ValueError,'required OCR field'):
            v.recover_page(pending,recovery,before,after,reverse,332)
        self.assertEqual(checkpoint(v),saved)

    def test_two_row_recovery_and_trial_gap_do_not_change_checkpoint(self):
        a,before=synthetic_page();pending,after=synthetic_page(4,433)
        recovery,reverse=synthetic_page(2,433,(4,4))
        v=rows.PositionInventory('support');v.accept_initial(a);saved=checkpoint(v)
        with self.assertRaisesRegex(ValueError,'at least three'):
            v.recover_page(pending,recovery,before,after,reverse,332)
        self.assertEqual(checkpoint(v),saved)
        bad,_=synthetic_page(5,433,(4,4,4))
        with self.assertRaisesRegex(ValueError,'gap'):
            v._merge_at(bad,5,{'row_offset':5})
        self.assertEqual(checkpoint(v),saved)


if __name__ == "__main__":
    unittest.main()
