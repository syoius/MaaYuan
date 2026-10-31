import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "utils/star_swipe_calibration.py"
if not PATH.exists(): PATH = Path(__file__).resolve().with_name("star_swipe_calibration.py")
SPEC = importlib.util.spec_from_file_location("calibration_test_core", PATH)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def raw():
    return {"group": "D", "trial": 1, "section": "support", "swipe_begin": [360, 930], "swipe_end": [360, 540],
            "manual_start_authorized": True, "manual_reset_confirmed": True, "backend_touch_hold_confirmed": True}


class Clock:
    def __init__(self): self.ns = 0; self.on_sleep = None
    def perf_counter_ns(self): return self.ns
    def time_ns(self): return 1793376000000000000 + self.ns
    def sleep(self, seconds):
        self.ns += max(1, round(seconds*1e9))
        if self.on_sleep: self.on_sleep()


class Controller:
    def __init__(self, clock, duration=20, fail_at=None):
        self.clock=clock; self.duration=duration; self.calls=0; self.fail_at=fail_at
        self.image=np.zeros((1280,720,3), np.uint8)
    def post_screencap(self):
        self.calls+=1
        self.clock.ns+=self.duration*1000000
        succeeded=self.calls!=self.fail_at
        def get():
            if not succeeded: raise AssertionError('failed screenshot cache was read')
            self.image[:]=self.calls
            return self.image
        return SimpleNamespace(wait=lambda:SimpleNamespace(succeeded=succeeded,get=get))


class Context:
    def __init__(self, clock, duration=20, fail_at=None):
        self.clock=clock
        self.tasker=SimpleNamespace(controller=Controller(clock,duration,fail_at),stopping=False)
        self.swipes=[]
    def run_task(self, entry, override):
        self.swipes.append((entry,override))
        p=override[entry]['action']['param']
        self.clock.ns+=(p['duration']+p['end_hold'])*1000000
        return SimpleNamespace(status=SimpleNamespace(succeeded=True),nodes=[SimpleNamespace(action=SimpleNamespace(success=True,raw_detail=p))])


class CalibrationTests(unittest.TestCase):
    def test_independent_F1_fixed_endpoint_and_one_2500ms_trial(self):
        p={**raw(),"group":"F","trial":1,"section":"main","swipe_end":[360,435]}
        params=CORE.parse_calibration_params(p)
        self.assertEqual(params["sample_offsets_ms"],[0,100,250,500,1000,2500])
        swipe=CORE.swipe_pipeline(params)[CORE.SWIPE_NODE]["action"]["param"]
        self.assertEqual((swipe["begin"],swipe["end"],swipe["duration"],swipe["end_hold"]),([360,930,1,1],[360,435,1,1],700,250))
        clock=Clock();context=Context(clock)
        with tempfile.TemporaryDirectory() as directory:
            m=CORE.collect_trial(context,params,Path(directory),clock=clock)
            self.assertEqual(m["status"],"captured")
            self.assertEqual(len(context.swipes),1)
            self.assertEqual(context.tasker.controller.calls,7)
            self.assertEqual(m["samples"][-1]["requested_offset_ms"],2500)
        for overrides in ({"trial":2},{"section":"support"},{"extended_observation":True},{"swipe_end":[360,480]}):
            with self.assertRaises(ValueError):CORE.parse_calibration_params({**p,**overrides})

    def test_extended_C2_and_independent_E1_keep_original_groups(self):
        self.assertEqual(CORE.GROUPS, {"A": (700,0), "B": (250,0), "C": (700,250), "D": (250,250)})
        for group, trial, end in (("C", 2, [360,540]), ("E", 1, [360,480])):
            p={**raw(), "group":group, "trial":trial, "swipe_end":end, "extended_observation":True}
            params=CORE.parse_calibration_params(p)
            self.assertEqual(params["sample_offsets_ms"], [0,100,250,500,1000,2500,3000,4000])
            self.assertEqual(params["post_release_settle_ms"],2500)
            self.assertEqual(params["tab_settle_ms"],2500)
            clock=Clock();context=Context(clock)
            with tempfile.TemporaryDirectory() as directory:
                manifest=CORE.collect_trial(context,params,Path(directory),clock=clock)
                self.assertEqual(manifest["status"],"captured")
                self.assertEqual(manifest["samples"][-1]["requested_offset_ms"],4000)
                self.assertEqual(context.tasker.controller.calls,9)
                self.assertTrue((Path(directory)/"after-4000ms.png").exists())
                self.assertEqual(len(context.swipes),1)
                self.assertEqual(context.swipes[0][1][CORE.SWIPE_NODE]["action"]["param"]["end"],end+[1,1])
        for overrides in ({"extended_observation":False},{"trial":2},{"swipe_end":[360,540]}):
            p={**raw(),"group":"E","trial":1,"swipe_end":[360,480],"extended_observation":True,**overrides}
            with self.assertRaises(ValueError):CORE.parse_calibration_params(p)

    def test_unknown_motion_and_missing_timepoints_do_not_claim_stability(self):
        tool_path=ROOT.parent/'tools/star_swipe_endhold.py'
        spec=importlib.util.spec_from_file_location('calibration_analysis_tool_test',tool_path)
        tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
        probe=tool.load_motion_algorithms(ROOT.parent)
        clock=Clock();context=Context(clock)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            manifest=CORE.collect_trial(context,CORE.parse_calibration_params(raw()),path,clock=clock)
            result=CORE.analyse_trial(path,probe)
            self.assertIsNone(result['stable_by_observed_ms'])
            self.assertIsNone(result['actual_content_shift_y_px'])
            ambiguous=SimpleNamespace(shift_y=630,confidence=1.,selected_hypothesis={'full_overlap_score':.95},
                motion_hypotheses=[{'shift_y':131,'full_overlap_score':.96,'full_overlap_valid':True}])
            with mock.patch.object(probe,'estimate_vertical_motion',return_value=ambiguous):
                unresolved=CORE.analyse_trial(path,probe)
            self.assertIsNone(unresolved['actual_content_shift_y_px'])
            self.assertEqual(len(unresolved['periodic_alias_candidates']),1)
            with self.assertRaises(FileExistsError):CORE.collect_trial(context,CORE.parse_calibration_params(raw()),path,clock=clock)
            self.assertEqual(len(context.swipes),1)
            manifest['samples'].pop()
            CORE.write_json(path/'trial.json',manifest)
            with self.assertRaises(ValueError):CORE.analyse_trial(path,probe)

    def test_probe_opt_in_dispatch_is_isolated_from_legacy_capture(self):
        agent_root=ROOT
        sys.path.insert(0,str(agent_root))
        path=agent_root/'custom/action/star_backpack_capture_probe.py'
        spec=importlib.util.spec_from_file_location('calibration_probe_dispatch_test',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            p={**raw(),'mode':'swipe_stability_probe','debug_dir':directory}
            import utils.star_swipe_calibration as installed_core
            with mock.patch.object(installed_core,'collect_trial',return_value={'status':'captured'}) as collect, mock.patch.object(module.StarBackpackCaptureProbe,'_capture',side_effect=AssertionError('legacy capture path entered')):
                result=module.StarBackpackCaptureProbe().run(SimpleNamespace(),SimpleNamespace(custom_action_param=p))
            self.assertTrue(result.success);self.assertEqual(collect.call_count,1)

    def test_all_groups_are_fixed_and_authorization_cannot_be_implicit(self):
        for group, expected in CORE.GROUPS.items():
            p=raw();p['group']=group
            params=CORE.parse_calibration_params(p)
            swipe=CORE.swipe_pipeline(params)[CORE.SWIPE_NODE]
            self.assertEqual((swipe['action']['param']['duration'],swipe['action']['param']['end_hold']),expected)
            self.assertEqual(swipe['action']['param']['begin'],[360,930,1,1])
            self.assertEqual(swipe['action']['param']['end'],[360,540,1,1])
            self.assertEqual(swipe['post_delay'],0)
        for flag in ('manual_start_authorized','manual_reset_confirmed','backend_touch_hold_confirmed'):
            p=raw();p[flag]=False
            with self.subTest(flag=flag),self.assertRaises(ValueError):CORE.parse_calibration_params(p)
        for field, value in [('trial',True),('swipe_duration_ms',True),('swipe_end',[360,541]),('post_release_settle_ms',0),('section','experience')]:
            p=raw();p[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):CORE.parse_calibration_params(p)

    def test_one_gesture_seven_owned_images_and_io_after_sampling(self):
        clock=Clock();context=Context(clock)
        actual_encode=cv2.imencode
        def checked_encode(*args,**kwargs):
            self.assertEqual(context.tasker.controller.calls,7)
            return actual_encode(*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(CORE.cv2,'imencode',side_effect=checked_encode) as encode:
                # Mutation of the shared screenshot buffer must not corrupt earlier frames.
                manifest=CORE.collect_trial(context,CORE.parse_calibration_params(raw()),Path(directory),clock=clock)
            self.assertEqual(manifest['status'],'captured')
            self.assertEqual(len(context.swipes),1)
            self.assertEqual([r['requested_offset_ms'] for r in manifest['samples']],[0,100,250,500,1000,2500])
            self.assertIsNone(manifest['release_physical_time'])
            self.assertFalse(manifest['release_reference']['is_physical_touch_up_timestamp'])
            before=CORE.read_image(Path(directory)/'before.png')
            after=CORE.read_image(Path(directory)/'after-2500ms.png')
            self.assertTrue(np.all(before==1));self.assertTrue(np.all(after==7))
            self.assertEqual(encode.call_count,7)

    def test_slow_screenshots_are_timestamped_late_not_relabelled(self):
        clock=Clock();context=Context(clock,duration=180)
        with tempfile.TemporaryDirectory() as directory:
            result=CORE.collect_trial(context,CORE.parse_calibration_params(raw()),Path(directory),clock=clock)
        self.assertEqual(result['samples'][0]['completion_offset_ms'],180)
        self.assertEqual(result['samples'][1]['request_offset_ms'],180)
        self.assertEqual(result['samples'][1]['lateness_ms'],80)

    def test_failed_screenshot_preserves_partial_evidence_and_rejects_cache(self):
        clock=Clock();context=Context(clock,fail_at=3)
        with tempfile.TemporaryDirectory() as directory:
            result=CORE.collect_trial(context,CORE.parse_calibration_params(raw()),Path(directory),clock=clock)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(len(result['samples']),1)
            self.assertTrue((Path(directory)/'before.png').exists())
            self.assertTrue((Path(directory)/'trial.json').exists())

    def test_cancel_during_wait_never_swipes_again(self):
        clock=Clock();context=Context(clock)
        clock.on_sleep=lambda:setattr(context.tasker,'stopping',True)
        with tempfile.TemporaryDirectory() as directory:
            result=CORE.collect_trial(context,CORE.parse_calibration_params(raw()),Path(directory),clock=clock)
        self.assertEqual(result['status'],'cancelled')
        self.assertEqual(len(context.swipes),1)
        self.assertEqual(len(result['samples']),1)

    def test_residual_measurement_handles_both_directions_and_rejects_blank(self):
        def prepare(img):return cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32),(3,3),0)
        def ncc(a,b):
            a=a.reshape(-1)-a.mean();b=b.reshape(-1)-b.mean();den=float(np.linalg.norm(a)*np.linalg.norm(b))
            return float(a@b/den) if den else 0.
        probe=SimpleNamespace(_prepare_compare_image=prepare,_normalized_correlation=ncc)
        rng=np.random.default_rng(905);first=rng.integers(0,255,(760,640,3),dtype=np.uint8)
        for shift in (-9,0,7):
            second=np.roll(first,-shift,axis=0)
            result=CORE.residual_motion(first,second,probe)
            self.assertTrue(result['measured']);self.assertEqual(result['signed_shift_y_px'],shift)
        self.assertFalse(CORE.residual_motion(np.zeros_like(first),np.zeros_like(first),probe)['measured'])


if __name__=='__main__': unittest.main()
