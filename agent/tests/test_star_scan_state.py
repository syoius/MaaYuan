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

from custom.action import star_full_local_ocr_shadow as action
from custom.action.star_full_local_ocr_shadow import FullShadowSession
from star_fixture_inputs import replay_owner
from star_fixture_inputs import synthetic_page
from utils import star_four_row_inventory as rows
from utils.star_four_row_inventory import PositionInventory


def checkpoint(inventory):
    return copy.deepcopy((inventory.items, inventory.positions, inventory.next_row,
                          inventory.last_page, inventory.last_base, inventory.observation_count,
                          inventory.tail_diagnostics))

class SharedMatchingTests(unittest.TestCase):

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

    def test_reliable_level_and_quality_conflicts_reject_a_partial_boundary(self):
        for field, value in (('level', 2), ('quality', '紫')):
            a, before = synthetic_page()
            b, after = synthetic_page(3, 433, (4,4,4,1))
            b['records'][0]['resolved'][field] = value
            if field == 'quality':
                a['records'][12]['quality'] = dict(quality='橙', confidence=.9)
                b['records'][0]['quality'] = dict(quality='紫', confidence=.9)
            decision=rows.classify_scan_transition(a,b,before,after,rows.initial_forward_plan(a['geometry']))
            self.assertEqual(decision['state'],'RECOVERY_REQUIRED')
            rejected=next(c for c in decision['position_evidence']['candidates'] if c['row_offset']==3)
            self.assertEqual(rejected['reason'],'overlap_field_conflict')


class RecoveryDataTests(unittest.TestCase):
    def test_forward_and_reverse_plans_keep_continuous_rows_for_normal_and_bridge(self):
        for base, forward_rows, reverse_rows in ((0,4,2), (0,2,2), (0,1,1), (4,1,2)):
            with self.subTest(base=base, forward_rows=forward_rows, reverse_rows=reverse_rows):
                inventory = PositionInventory('main')
                inventory.accept_initial(synthetic_page()[0])
                if base:
                    inventory.accept_moved(synthetic_page(base)[0],4)
                accepted = copy.deepcopy(inventory.items)
                pending_base = base + forward_rows
                recovered_base = pending_base - reverse_rows
                result = inventory.recover_page(synthetic_page(pending_base)[0], synthetic_page(recovered_base)[0],
                                                recovered_offset=forward_rows-reverse_rows, pending_offset=forward_rows)
                self.assertEqual((result['resumed_base'], result['unresolved_base']), (recovered_base, pending_base))
                self.assertEqual(inventory.last_base, recovered_base)
                self.assertEqual(inventory.items[:len(accepted)], accepted)
                self.assertEqual((result['new_count'], result['duplicate_count']),
                                 (forward_rows*4, 32-forward_rows*4))
                self.assertEqual(set(inventory.positions),
                                 {(row,col) for row in range(pending_base+4) for col in range(4)})

    def test_failed_recovery_preserves_every_part_of_the_checkpoint(self):
        initial, _ = synthetic_page()
        recovery, _ = synthetic_page(2,433)
        for mode in ('negative_row', 'conflict_after_trial_merge', 'unknown_field'):
            inventory = PositionInventory('support')
            inventory.accept_initial(initial)
            saved = checkpoint(inventory)
            pending, _ = synthetic_page(4,433)
            if mode == 'conflict_after_trial_merge':
                pending['records'][0]['resolved']['name'] = '巨门'
            if mode == 'unknown_field':
                pending['records'][0]['resolved']['name'] = None
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                inventory.recover_page(pending,recovery,recovered_offset=-1 if mode=='negative_row' else 2,pending_offset=4)
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
            plan = rows.recovery_swipe(page['geometry'], page['geometry'], target_rows=2)
            self.assertEqual(plan['row_pitch_px'], pitch)
            self.assertEqual(plan['touch_length_px'], round(2*pitch/rows.CALIBRATED_GAIN))
            self.assertLess(plan['swipe_begin'][1], plan['swipe_end'][1])
            self.assertLess(plan['swipe_end'][1], 1044)

class RecoveryLoopTests(unittest.TestCase):

    bridge = {'mode':'bridge','target_rows':2,'swipe_begin':[360,1000],
              'swipe_end':[360,655],'touch_length_px':345}

    def test_unproven_forward_step_recovers_with_the_actual_reverse_offset(self):
        initial = synthetic_page(0,403)
        pending = synthetic_page(3,433)
        recovered = synthetic_page(2,403)
        tail = synthetic_page(6,403)
        frames = [pending[1],recovered[1],tail[1],tail[1],tail[1]]
        with tempfile.TemporaryDirectory() as directory:
            owner,saved,gestures,_ = replay_owner([initial,pending,recovered,tail],frames,directory)
            with patch.object(action,'initial_forward_plan',return_value=self.bridge):
                result = FullShadowSession.scan(owner,'main',initial[1])
        self.assertTrue(result['complete'])
        self.assertEqual((result['final_count'],result['recovery_attempts']),(40,1))
        self.assertEqual([plan['target_rows'] for plan,_ in gestures[:3]],[2,2,4])
        self.assertEqual(saved[1]['scan_transition']['state'],'RECOVERY_REQUIRED')
        mapping = owner.position_recoveries[0]['verified_mapping']
        self.assertEqual((mapping['pending_offset'],mapping['reverse_rows']),(3,1))
        merged = saved[2]['acceptance']
        self.assertEqual((merged['resumed_base'],merged['unresolved_base']),(2,3))
        self.assertEqual(set(owner.inventories['main'].positions),{(row,col) for row in range(10) for col in range(4)})

    def test_failed_recovery_does_not_loop_or_merge_unresolved_instances(self):
        a,before = synthetic_page()
        b,after = synthetic_page(3,433)
        bad,reverse = synthetic_page(2,403)
        bad['records'][4]['resolved']['level'] = 2
        b['records'][0]['resolved']['level'] = 3
        with tempfile.TemporaryDirectory() as directory:
            owner,_,gestures,_ = replay_owner([(a,before),(b,after),(bad,reverse)],[after,reverse],directory)
            with patch.object(action,'initial_forward_plan',return_value=self.bridge):
                with self.assertRaisesRegex(rows.BoundaryPositionError,'0 verified row mappings'):
                    FullShadowSession.scan(owner,'main',before)
            self.assertEqual(len(gestures),2)
            self.assertEqual(len(owner.inventories['main'].items),16)
            self.assertEqual(owner.sections['main']['recovery_attempts'],1)
            self.assertFalse(owner.position_recoveries[0]['success'])

    def test_one_row_bridge_recovery_at_top_limits_reverse_to_one_row(self):
        initial = synthetic_page(0,403)
        pending = synthetic_page(1,433)
        forward = dict(self.bridge,target_rows=1,touch_length_px=160)
        with tempfile.TemporaryDirectory() as directory:
            owner,_,gestures,_ = replay_owner([initial,pending],[initial[1]],directory)
            owner.section='main'
            inventory=owner.inventories['main']
            inventory.accept_initial(initial[0])
            summary={'swipes':0,'recovery_attempts':0,'recovery_ms':0}
            owner.recover_position(summary,inventory,pending[0],initial[1],pending[1],forward)
        self.assertEqual(gestures[0][0]['target_rows'],1)
        self.assertEqual((inventory.last_base,inventory.next_row,len(inventory.items)),(0,5,20))

    def test_same_unresolved_frontier_cannot_trigger_another_reverse(self):
        a, image = synthetic_page()
        with tempfile.TemporaryDirectory() as directory:
            owner, _, gestures, _ = replay_owner([(a,image)], [], directory)
            owner.section = 'main'
            inventory = owner.inventories['main'];inventory.accept_initial(a)
            owner.recovery_checkpoints.add(('main',inventory.next_row))
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                owner.recover_position({},inventory,a,image,image,rows.initial_forward_plan(a['geometry']))
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
  def swipe(summary,plan,label):
   summary['swipes']+=1;gestures.append((plan,label));return next(frames),{}
  owner.recognize=recognize;owner.swipe_capture=swipe
  owner.save_page=lambda page,*args:saved.append(copy.deepcopy(page))
  owner.confirm_bottom=MethodType(FullShadowSession.confirm_bottom,owner)
  owner.compensate_name=MethodType(FullShadowSession.compensate_name,owner)
  return owner,saved,gestures

 def test_calibrated_planner_prefers_safe_four_rows_then_bridge(self):
  page,_=synthetic_page(first=420)
  result=rows.plan_forward(page['geometry'])
  self.assertEqual(result['calibrated_gain'],rows.CALIBRATED_GAIN)
  self.assertEqual((result['mode'],result['target_rows']),('normal',4))
  rows.validate_swipe(result['swipe_begin'],result['swipe_end'])
  self.assertEqual(result['first_row_phase_adjustment_px'],30)
  self.assertEqual(result['touch_length_px'],round((4*167+30)/rows.CALIBRATED_GAIN))
  for record in page['records']:
   dy=record['row']*9
   card=record['card']
   card['circle']['centerY']+=dy
   for rect in ('discRect','nameRect','levelRect'):
    card[rect]['y']+=dy
  bridge=rows.plan_forward(page['geometry'])
  self.assertEqual((bridge['mode'],bridge['target_rows']),('bridge',2))
  self.assertTrue(80<=bridge['touch_length_px']<600)

 def test_forward_plans_reuse_calibration_without_measured_feedback(self):
  pairs=[synthetic_page(base,first) for base,first in ((0,358),(4,400),(8,403))]
  start,first,last=[im for _,im in pairs]
  owner,saved,gestures=self.owner(pairs,[first,last,last,last])
  result=FullShadowSession.scan(owner,'main',start)
  self.assertEqual(result['final_count'],48)
  self.assertEqual([p['next_swipe_plan']['calibrated_gain'] for p in saved if 'next_swipe_plan' in p],
                   [rows.CALIBRATED_GAIN]*2)
  self.assertEqual(gestures[2][0],gestures[3][0])
  self.assertTrue(all('next_swipe_plan' not in p for p in saved[-2:]))

 def test_compensation_uses_calibrated_gain_and_commits_planned_rows_once(self):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  a,b,c=synthetic_page(first=425),synthetic_page(4,453),synthetic_page(4,418)
  clipped=(with_name_obstruction(b[0]),b[1]);owner,saved,gestures=self.owner([a,clipped,c],[b[1],c[1],c[1],c[1]])
  result=FullShadowSession.scan(owner,'main',a[1])
  self.assertTrue(result['complete']);self.assertEqual(result['final_count'],32)
  self.assertEqual(len(owner.inventories['main'].positions),32)
  self.assertEqual(sum(plan['mode']=='compensation' for plan,_ in gestures),1)
  plan,_=gestures[1]
  self.assertEqual(plan['swipe_begin'][1]-plan['swipe_end'][1],plan['touch_length_px'])
  self.assertEqual(plan['target_rows'],0)
  self.assertGreater(plan['target_content_shift_px'],10)
  self.assertEqual(saved[1]['status'],'name_compensation_required')
  self.assertEqual(saved[2]['name_compensation'],dict(first_row_center_shift_px=35.))
  self.assertEqual(saved[2]['acceptance']['new_count'],16)
  self.assertEqual(saved[2]['acceptance']['physical_base'],4)

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

 def test_compensation_requires_a_short_geometry_shift(self):
  a,b=synthetic_page(first=425),synthetic_page(4,453)
  clipped=(with_name_obstruction(b[0]),b[1])
  a2=synthetic_page(4,453)
  for record in a2[0]['records']:
   card=record['card']
   card['nameRect']['height']=min(card['nameRect']['height'],1044-card['nameRect']['y'])
  owner,_,gestures=self.owner([a,clipped,a2],[b[1],a2[1]])
  with self.assertRaisesRegex(RuntimeError,'geometry inconsistent'):
   FullShadowSession.scan(owner,'main',a[1])
  self.assertEqual(len(gestures),2);self.assertEqual(len(owner.inventories['main'].items),16)

 def test_confirmation_with_name_obstruction_stops_without_compensation(self):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  a,b=synthetic_page(first=425),synthetic_page(4,453)
  clipped=(with_name_obstruction(b[0]),b[1]);owner,_,gestures=self.owner([a,clipped],[a[1],b[1]])
  with self.assertRaisesRegex(RuntimeError,'incomplete name on confirmation'):
   FullShadowSession.scan(owner,'main',a[1])
  self.assertEqual([label for _,label in gestures],['page-001','bottom-confirmation'])
  self.assertEqual(len(owner.inventories['main'].items),16)

class LoopTests(unittest.TestCase):
 def test_full_overlap_then_full_advance_then_confirmed_bottom(self):
  from custom.action.star_full_local_ocr_shadow import FullShadowSession
  pages=[synthetic_page(base,403 if base!=1 else 433) for base in (0,1,5)]
  lookup={id(im):p for p,im in pages};frames=iter([pages[1][1],pages[2][1],pages[2][1],pages[2][1]]);saved=[]
  owner=SimpleNamespace(inventories={'main':PositionInventory('main')},sections={},progress=lambda stage:None,
   recognize=lambda im,sid:copy.deepcopy(lookup[id(im)]),save_page=lambda page,*args:saved.append(page),last_capture={'file':'unit.png'})
  def swipe(summary,plan,label):summary['swipes']+=1;return next(frames),{}
  owner.swipe_capture=swipe;owner.confirm_bottom=MethodType(FullShadowSession.confirm_bottom,owner)
  result=FullShadowSession.scan(owner,'main',pages[0][1])
  self.assertTrue(result['complete']);self.assertEqual(result['stop_reason'],'bottom_no_move_confirmed');self.assertEqual(result['final_count'],36);self.assertEqual(len(saved),5)
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


def owner_for(section, pairs, frames, directory):
    owner, saved, gestures, recognized = replay_owner(pairs, frames, directory)
    owner.inventories = {section: rows.PositionInventory(section)}
    return owner, saved, gestures, recognized

def repeated_page(base, first):
    """All rows share exactly the same observable name pixels, unlike unique textures."""
    page, image = synthetic_page(base, first)
    for record in page['records']:
        record['resolved']['name'] = '太阳'
    image[:] = 0
    for relative in range(-1 if base else 0, 4):
        for col in range(4):
            card = page['records'][col]['card']
            rect = card['nameRect']
            x, y, w, h = (rect[k] for k in ('x', 'y', 'width', 'height'))
            y += relative * 167
            texture = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
            top, bottom = max(272, y), min(1044, y+h)
            if bottom > top:
                image[top:bottom, x:x+w] = texture[top-y:bottom-y]
    return page, image

class ScanStateTests(unittest.TestCase):

    def test_boundary_no_move_and_regular_move_use_separate_evidence(self):
        a, before = synthetic_page()
        b, after = synthetic_page(4,403)
        plan = rows.initial_forward_plan(a['geometry'])
        decision = rows.classify_scan_transition(a,b,before,after,plan)
        landing = decision.pop('landing')
        self.assertEqual(decision, {'state':'MOVED','basis':'plan','row_offset':4})
        self.assertEqual(landing['first_row_center_px'],403)
        self.assertEqual(rows.classify_scan_transition(a,a,before,before,plan),
                         {'state':'NO_MOVE_CANDIDATE','basis':'boundary','row_offset':0})
        self.assertFalse(rows.confirmed_no_move(before,before,before))

    def test_changed_reading_is_not_silently_discarded_at_bottom(self):
        a, image = synthetic_page()
        b = copy.deepcopy(a)
        b['records'][0]['resolved']['level'] = 2
        plan = rows.initial_forward_plan(a['geometry'])
        with self.assertRaisesRegex(ValueError,'stationary boundary OCR disagrees'):
            rows.classify_scan_transition(a,b,image,image,plan)

    def test_confirmation_move_is_consumed_once_and_resumes_normal_scan_for_both_sections(self):
        pairs = [synthetic_page(), synthetic_page(4,403), synthetic_page(5,433,(4,4,4,1))]
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

    def test_periodic_page_landing_outside_prediction_uses_planned_four_rows(self):
        a, before = repeated_page(0,403)
        b, after = repeated_page(4,433)
        plan = rows.initial_forward_plan(a['geometry'])
        decision = rows.classify_scan_transition(a,b,before,after,plan)
        landing = decision.pop('landing')
        self.assertEqual(decision, {'state':'MOVED','basis':'plan','row_offset':4})
        self.assertGreater(landing['first_row_center_px'],landing['expected_center_interval_px'][1]+8)
        self.assertGreater(landing['error_from_expected_center_px'],0)

    def test_visible_preceding_name_rejects_a_five_row_skip_at_the_expected_phase(self):
        previous,before=synthetic_page(0,403)
        plan=rows.initial_forward_plan(previous['geometry'])
        exact,exact_frame=synthetic_page(4,390)
        skipped,skipped_frame=synthetic_page(5,390)
        decision=rows.classify_scan_transition(previous,exact,before,exact_frame,plan)
        self.assertEqual((decision['state'],decision['row_offset']),('MOVED',4))
        decision=rows.classify_scan_transition(previous,skipped,before,skipped_frame,plan)
        self.assertEqual(decision['state'],'RECOVERY_REQUIRED')
        self.assertEqual(decision['position_evidence']['candidates'][-1]['reason'],'name_anchor_mismatch')

    def test_bridge_landing_error_does_not_change_planned_rows(self):
        a,before = synthetic_page()
        for offset in (1,2):
            with self.subTest(offset=offset):
                b,after = synthetic_page(offset,433)
                touch = round((403+offset*167-rows.TARGET_FIRST_ROW_CENTER_PX)/rows.CALIBRATED_GAIN)
                plan = {'mode':'bridge','target_rows':offset,'touch_length_px':touch,
                        'landing_center_interval_px':[385,395]}
                decision = rows.classify_scan_transition(a,b,before,after,plan)
                self.assertEqual((decision['state'],decision['basis'],decision['row_offset']),('MOVED','boundary',offset))
                inventory = PositionInventory('main')
                inventory.accept_initial(a)
                merged = inventory.accept_moved(b,decision['row_offset'])
                self.assertEqual((merged['physical_base'],merged['new_count']),(offset,offset*4))

    def test_ambiguous_stationary_repeat_uses_two_sub_row_probes(self):
        page,image = repeated_page(4,403)
        with tempfile.TemporaryDirectory() as directory:
            owner,saved,gestures,_ = owner_for('support',[(page,image)],[image,image,image],directory)
            owner.bag={'complete':True,'currentCount':32,'capacity':320,'warnings':[]}
            owner.inventories['main']=PositionInventory('main')
            owner.inventories['main'].accept_initial(synthetic_page()[0])
            owner.sections['main']={'complete':True}
            summary=FullShadowSession.scan(owner,'support',image)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['final_count'],16)
        self.assertEqual(summary['terminal_selection']['selected_row_offset'],0)
        self.assertEqual(saved[1]['scan_transition']['row_offsets'],[0,1,2,3,4])
        self.assertEqual([plan['touch_length_px'] for plan,_ in gestures[1:]],[80,80])

    def test_periodic_probe_that_moves_requires_new_position_evidence(self):
        initial,before = repeated_page(4,403)
        moved,after = repeated_page(4,353)
        with tempfile.TemporaryDirectory() as directory:
            owner,_,gestures,_ = owner_for('main',[(initial,before),(moved,after)],[before,after,before],directory)
            with self.assertRaisesRegex(rows.BoundaryPositionError,'verified row mappings'):
                FullShadowSession.scan(owner,'main',before)
        self.assertFalse(owner.sections['main']['complete'])
        self.assertEqual(len(owner.inventories['main'].items),16)
        self.assertEqual([label for _,label in gestures],['page-001','bottom-probe','main-001-bottom-probe-recovery'])

    def test_two_small_shifts_cannot_accumulate_into_false_bottom(self):
        pairs = [synthetic_page(0,403), synthetic_page(0,402), synthetic_page(0,400)]
        with tempfile.TemporaryDirectory() as directory:
            owner, _, gestures, _ = owner_for('main', pairs, [pairs[1][1],pairs[2][1]], directory)
            with self.assertRaisesRegex(RuntimeError, 'endpoint position unconfirmed'):
                FullShadowSession.scan(owner, 'main', pairs[0][1])
            self.assertFalse(owner.sections['main']['complete'])
            self.assertEqual(len(gestures), 2)
            self.assertEqual(len(owner.inventories['main'].items), 16)


class AmbiguousTerminalTests(unittest.TestCase):
    def setUp(self):
        initial, self.before = synthetic_page()
        self.tail, self.after = synthetic_page(4,433,(4,4,4,2))
        self.initial = initial
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def visual_candidates(self,previous,current,before,after,touch,offsets,*,evidence=None):
        if before is self.before and after is self.after:
            return [(offset,shift,shift) for offset,shift in ((3,475.),(4,642.)) if offset in offsets]
        return self.match_boundary(previous,current,before,after,touch,offsets,evidence=evidence)

    def owner(self,bag_count,confirmation=None):
        confirmation,confirmed_image=confirmation if confirmation is not None else (self.tail,self.after)
        owner,saved,gestures,_=owner_for(
            'support',[(self.initial,self.before),(self.tail,self.after),(confirmation,confirmed_image)],
            [self.after,self.after,confirmed_image],self.directory)
        owner.bag={'complete':bag_count is not None,'currentCount':bag_count,'capacity':320,'warnings':[]}
        owner.inventories['main']=PositionInventory('main')
        owner.inventories['main'].accept_initial(self.initial)
        owner.sections['main']={'complete':True}
        return owner,saved,gestures

    def scan(self,owner,section='support'):
        self.match_boundary = rows._boundary_matches
        with patch.object(rows,'_boundary_matches',side_effect=self.visual_candidates):
            return FullShadowSession.scan(owner,section,self.before)

    def test_only_one_count_match_commits_once_after_existing_bottom_confirmation(self):
        owner,saved,gestures=self.owner(46)
        inventory=owner.inventories['support']
        with patch.object(inventory,'merge_terminal',wraps=inventory.merge_terminal) as merge:
            summary=self.scan(owner)
        merge.assert_called_once()
        self.assertEqual(merge.call_args.args[1],4)
        self.assertTrue(summary['complete'])
        self.assertEqual((summary['stop_reason'],summary['final_count']),('bottom_no_move_confirmed',30))
        self.assertEqual([label for _,label in gestures],['page-001','bottom-probe','bottom-confirmation'])
        self.assertEqual(saved[1]['scan_transition']['row_offsets'],[3,4])
        self.assertEqual(summary['terminal_confirmation']['probes'],2)
        self.assertTrue(summary['terminal_confirmation']['stationary'])
        self.assertEqual([c['expected_total'] for c in summary['terminal_selection']['candidates']],[42,46])

    def test_zero_matches_or_unconfirmed_bag_keeps_checkpoint(self):
        for count,error in ((45,'0 terminal candidates'),(None,'confirmed bag count')):
            with self.subTest(count=count):
                owner,_,_=self.owner(count)
                inventory=owner.inventories['support']
                with patch.object(inventory,'merge_terminal',wraps=inventory.merge_terminal) as merge:
                    with self.assertRaisesRegex(RuntimeError,error):
                        self.scan(owner)
                merge.assert_not_called()
                self.assertEqual((len(inventory.items),inventory.last_base,inventory.observation_count),(16,0,16))
                summary=owner.sections['support']
                self.assertFalse(summary['complete'])
                self.assertEqual([c['expected_total'] for c in summary['terminal_selection']['candidates']],[42,46])

    def test_candidates_ending_before_accepted_frontier_keep_checkpoint(self):
        owner,_,_=self.owner(64)
        owner.section='support'
        inventory=owner.inventories['support']
        inventory.accept_initial(self.initial)
        inventory.accept_moved(synthetic_page(4)[0],4)
        inventory.accept_moved(synthetic_page(8)[0],4)
        # The current view can lag a previously accepted frontier after recovery.
        inventory.last_page,inventory.last_base=self.initial,0
        saved=checkpoint(inventory)
        summary={}
        with self.assertRaisesRegex(RuntimeError,'0 terminal candidates'):
            FullShadowSession._select_terminal_offset(owner,summary,inventory,self.tail,[3,4])
        self.assertEqual(summary['terminal_selection']['candidates'],[
            {'row_offset':offset,'error':'terminal candidate ends before the accepted inventory frontier'}
            for offset in (3,4)])
        self.assertEqual(checkpoint(inventory),saved)

    def test_main_defers_committing_the_tail_until_support_is_counted(self):
        owner,_,gestures,_ = owner_for(
            'main',[(self.initial,self.before),(self.tail,self.after)],
            [self.after,self.after,self.after],self.directory)
        owner.bag={'complete':True,'currentCount':46,'capacity':320,'warnings':[]}
        summary=self.scan(owner,'main')
        inventory=owner.inventories['main']
        self.assertFalse(summary['complete'])
        self.assertTrue(summary['terminal_confirmation']['stationary'])
        self.assertEqual(len(inventory.items),16)
        self.assertEqual(len(gestures),3)
        owner.inventories['support']=PositionInventory('support')
        owner.inventories['support'].accept_initial(self.initial)
        owner.sections['support']={'complete':True}
        FullShadowSession.resolve_pending_terminals(owner)
        self.assertTrue(summary['complete'])
        self.assertIsNone(inventory.pending_terminal)
        self.assertEqual((summary['final_count'],summary['terminal_selection']['selected_row_offset']),(30,4))

    def test_two_pending_sections_commit_only_if_the_bag_matches_one_combination(self):
        for count in (60,56,55):
            with self.subTest(count=count):
                owner,_,_=self.owner(count)
                owner.sections.clear()
                for section in ('main','support'):
                    inventory=owner.inventories[section]
                    if not inventory.items:
                        inventory.accept_initial(self.initial)
                    summary=owner.sections[section]={'complete':False}
                    FullShadowSession._select_terminal_offset(owner,summary,inventory,self.tail,[3,4])
                if count==60:
                    FullShadowSession.resolve_pending_terminals(owner)
                    self.assertEqual([len(item.items) for item in owner.inventories.values()],[30,30])
                    self.assertTrue(all(summary['complete'] for summary in owner.sections.values()))
                else:
                    saved={section:checkpoint(inventory) for section,inventory in owner.inventories.items()}
                    with self.assertRaisesRegex(RuntimeError,'terminal combinations match'):
                        FullShadowSession.resolve_pending_terminals(owner)
                    self.assertEqual({section:checkpoint(inventory) for section,inventory in owner.inventories.items()},saved)

    def test_moving_confirmation_does_not_select_merge_or_finish(self):
        owner,_,gestures=self.owner(46,synthetic_page(4,426,(4,4,4,2)))
        inventory=owner.inventories['support']
        with patch.object(inventory,'merge_terminal',wraps=inventory.merge_terminal) as merge:
            with self.assertRaisesRegex(RuntimeError,'moved during bottom probe'):
                self.scan(owner)
        merge.assert_not_called()
        self.assertEqual([label for _,label in gestures],['page-001','bottom-probe','bottom-confirmation'])
        self.assertEqual(len(inventory.items),16)
        self.assertNotIn('terminal_selection',owner.sections['support'])
        self.assertFalse(owner.sections['support']['terminal_confirmation']['stationary'])
        self.assertFalse(owner.sections['support']['complete'])

    def test_probe_and_each_candidate_preview_leave_inventory_unchanged(self):
        owner,_,_=self.owner(46)
        inventory=owner.inventories['support']
        saved=None
        swipe=owner.swipe_capture
        preview=inventory.preview_terminal
        def check_swipe(summary,plan,label):
            nonlocal saved
            if saved is None:
                saved=checkpoint(inventory)
            self.assertEqual(checkpoint(inventory),saved)
            return swipe(summary,plan,label)
        def check_preview(page,offset):
            self.assertEqual(checkpoint(inventory),saved)
            result=preview(page,offset)
            self.assertEqual(checkpoint(inventory),saved)
            return result
        owner.swipe_capture=check_swipe
        with patch.object(inventory,'preview_terminal',side_effect=check_preview) as counted:
            self.scan(owner)
        self.assertEqual(counted.call_count,2)
        self.assertEqual(len(inventory.items),30)


if __name__ == "__main__":
    unittest.main()
