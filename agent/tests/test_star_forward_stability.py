"""Forward 600/600/0 settling and unchanged recovery; no emulator/cloud."""
import copy
import itertools
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent'))
import test_star_page_transitions as fixtures
from custom.action import star_full_local_ocr_shadow as production
from utils.star_frame_stability import name_motion
from utils.star_four_row_inventory import PositionInventory


class Clock:
    def __init__(self):self.value=100.;self.sleeps=[]
    def perf_counter(self):return self.value
    def sleep(self,seconds):self.sleeps.append(seconds);self.value+=seconds


def owner_for(offsets):
    page,image=fixtures.synthetic_page()
    values=sorted(set([0]+list(offsets)));images={v:np.roll(image,v,axis=0) for v in values}
    tags={id(im):v for v,im in images.items()};source=iter(offsets);clock=Clock();captures=[]
    def capture(label):
        value=next(source);clock.value+=.01;captures.append((label,clock.value))
        return images[value]
    def motion(a,b,*args):return {'reliable':True,'dx':0.,'dy':float(tags[id(b)]-tags[id(a)])}
    def stability(a,b):
        shift=tags[id(b)]-tags[id(a)]
        return {'stable':abs(shift)<=2,'ncc':1.,'shift_px':[0.,float(shift)],'response':1.}
    owner=SimpleNamespace(check=lambda:None,check_backpack=lambda *args:None,capture=capture,
        bridge=SimpleNamespace(geometry=lambda im:page['geometry']))
    return owner,clock,images[0],motion,stability,captures


class NameMotionTests(unittest.TestCase):
    def test_real_name_corners_measure_zero_and_motion(self):
        page,image=fixtures.real_page('003')
        for dy in (0,-12,3):
            result=name_motion(image,np.roll(image,dy,axis=0),page['geometry'])
            self.assertTrue(result['reliable']);self.assertAlmostEqual(result['dy'],dy,delta=.3)
    def test_only_names_move_while_public_discs_stay(self):
        page,image=fixtures.real_page('003');after=image.copy()
        for card in page['geometry']['cards']:
            r=card['nameRect'];x,y,w,h=[int(r[k]) for k in ('x','y','width','height')]
            block=image[y:y+h,x:x+w].copy()
            after[y-5:y+h,x:x+w]=np.median(block.reshape(-1,3),axis=0).astype(np.uint8)
            after[y-5:y+h-5,x:x+w]=block
        result=name_motion(image,after,page['geometry'])
        self.assertTrue(result['reliable']);self.assertAlmostEqual(result['dy'],-5,delta=.3)
    def test_no_name_evidence_is_not_stability_proof(self):
        page,image=fixtures.real_page('003')
        self.assertFalse(name_motion(image*0,image*0,page['geometry'])['reliable'])


class ForwardSettlingTests(unittest.TestCase):
    def settle(self,offsets,*,global_stable=False,geometry_delay=0):
        owner,clock,image,motion,stability,captures=owner_for(offsets)
        if geometry_delay:
            geometry=owner.bridge.geometry(image)
            owner.bridge.geometry=lambda im:clock.sleep(geometry_delay) or geometry
        if global_stable:stability=lambda *args:{'stable':True,'ncc':1.,'shift_px':[0.,0.],'response':1.}
        with patch.object(production,'time',clock),patch.object(production,'now_ms',lambda start:(clock.value-start)*1000),patch.object(production,'name_motion',side_effect=motion),patch.object(production,'frame_stability',side_effect=stability):
            result=production.FullShadowSession.stabilize_forward(owner,image,'unit','main',100.)
        return result,owner,clock,captures
    def test_two_comparisons_then_immediate_exit_without_full_window(self):
        result,owner,clock,captures=self.settle([0,0,0])
        self.assertEqual(len(captures),2);self.assertTrue(owner.current_forward_stability['complete'])
        self.assertLess(clock.value-100,.5)
    def test_first_frame_can_move_then_settle(self):
        _,owner,_,captures=self.settle([10,20,20,20])
        self.assertEqual(len(captures),4);self.assertFalse(owner.current_stability[0]['stable'])
    def test_more_than_six_comparisons_can_succeed_within_time_bound(self):
        _,owner,clock,captures=self.settle([3,6,9,12,15,18,21,21,21])
        self.assertEqual(len(captures),9);self.assertLess(clock.value-100,3.)
        self.assertTrue(owner.current_forward_stability['complete'])
    def test_tail_rebound_gets_more_checks_without_early_acceptance(self):
        _,owner,_,captures=self.settle([60,80,80,80])
        self.assertEqual(len(captures),4);self.assertEqual(owner.current_stability[-1]['direct_name_motion']['dy'],80)
    def test_public_disc_ncc_cannot_hide_name_motion(self):
        _,owner,_,captures=self.settle([4,8,8,8],global_stable=True)
        self.assertEqual(len(captures),4);self.assertTrue(all(x['stable'] for x in owner.current_stability))
        self.assertGreater(owner.current_stability[0]['name_motion']['dy'],2)
    def test_two_small_shifts_cannot_hide_cumulative_drift(self):
        _,owner,_,captures=self.settle([1.5,3,4.5,4.5,4.5],global_stable=True)
        self.assertEqual(len(captures),4)
        self.assertEqual(owner.current_stability[1]['stable_span_px'],3)
    def test_slow_initial_geometry_does_not_create_capture_burst(self):
        _,_,_,captures=self.settle([0,0],geometry_delay=.65)
        self.assertGreaterEqual(captures[1][1]-captures[0][1],.199)
    def test_timeout_retains_checks_and_never_returns_a_moving_frame(self):
        owner,clock,image,motion,stability,captures=owner_for(list(range(3,60,3)))
        with patch.object(production,'time',clock),patch.object(production,'now_ms',lambda start:(clock.value-start)*1000),patch.object(production,'name_motion',side_effect=motion),patch.object(production,'frame_stability',side_effect=stability):
            with self.assertRaisesRegex(RuntimeError,'before OCR'):
                production.FullShadowSession.stabilize_forward(owner,image,'unit','main',100.)
        self.assertFalse(owner.current_forward_stability['complete']);self.assertGreater(len(captures),6)
        self.assertLess(clock.value-100,3.1)


class SwipeIntegrationTests(unittest.TestCase):
    def owner(self,directory):
        page,image=fixtures.synthetic_page();clock=Clock();pipelines=[]
        detail=SimpleNamespace(status=SimpleNamespace(succeeded=True),nodes=[SimpleNamespace(action=SimpleNamespace(success=True,raw_detail={}))])
        owner=SimpleNamespace(check=lambda:None,section='main',out=Path(directory),events=[],check_backpack=lambda *args:None,
             capture=lambda label:image,current_stability=[],context=SimpleNamespace(run_task=lambda node,pipeline:pipelines.append(pipeline) or detail))
        def legacy(im,*args):owner.current_stability=[{'stable':True},{'stable':True}];return im
        def forward(im,*args):owner.current_stability=[{'stable':True},{'stable':True}];owner.current_forward_stability={'complete':True};return im
        owner.stabilize=Mock(side_effect=legacy);owner.stabilize_forward=Mock(side_effect=forward)
        return owner,clock,pipelines,image
    def test_forward_uses_600_600_zero_fixed_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            owner,clock,pipelines,_=self.owner(directory)
            with patch.object(production,'time',clock),patch.object(production,'now_ms',lambda start:(clock.value-start)*1000):
                _,event=production.FullShadowSession.swipe_capture(owner,{'swipes':0,'_deadline':200.},630,'unit')
            params=pipelines[0]['StarSwipeEndHoldCalibrationNative']['action']['param']
            self.assertEqual((params['duration'],params['end_hold'],params['begin'],params['end']),(600,600,[360,1000,1,1],[360,370,1,1]))
            self.assertEqual(clock.sleeps,[]);self.assertEqual(event['wait_ms'],0.)
            owner.stabilize_forward.assert_called_once();owner.stabilize.assert_not_called()
    def test_recovery_retains_original_parameters_wait_and_stabilizer(self):
        with tempfile.TemporaryDirectory() as directory:
            owner,clock,pipelines,_=self.owner(directory)
            plan={'swipe_begin':[360,397],'swipe_end':[360,729]}
            with patch.object(production,'time',clock),patch.object(production,'now_ms',lambda start:(clock.value-start)*1000):
                production.FullShadowSession.swipe_capture(owner,{'swipes':0,'_deadline':200.},332,'reverse',recovery_plan=plan)
            params=pipelines[0]['StarSwipeEndHoldCalibrationNative']['action']['param']
            self.assertEqual((params['duration'],params['end_hold']),(700,250));self.assertEqual(clock.sleeps,[1.])
            owner.stabilize.assert_called_once();owner.stabilize_forward.assert_not_called()
    def test_failed_settling_is_diagnosed_without_ocr_or_merging(self):
        with tempfile.TemporaryDirectory() as directory:
            owner,clock,_,image=self.owner(directory)
            page,_=fixtures.synthetic_page();owner.inventories={'main':PositionInventory('main')};owner.sections={};owner.progress=lambda *args:None
            owner.recognize=Mock(return_value=copy.deepcopy(page));owner.save_page=lambda *args:None
            owner.stabilize_forward.side_effect=RuntimeError('moving frame rejected before OCR')
            owner.swipe_capture=MethodType(production.FullShadowSession.swipe_capture,owner)
            with patch.object(production,'time',clock),patch.object(production,'now_ms',lambda start:(clock.value-start)*1000):
                with self.assertRaisesRegex(RuntimeError,'before OCR'):production.FullShadowSession.scan(owner,'main',image)
            self.assertEqual(owner.recognize.call_count,1) # Only the initial, already-stable page.
            self.assertEqual(len(owner.inventories['main'].items),16);self.assertFalse(owner.sections['main']['complete'])
            error=json.loads(next(Path(directory).glob('*stability-error.json')).read_text(encoding='utf-8'))
            self.assertFalse(error['settled']);self.assertIn('stability_checks',error)


if __name__=='__main__':unittest.main()
