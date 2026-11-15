from pathlib import Path
import copy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent'))
from star_fixture_inputs import synthetic_page
from utils import star_four_row_inventory as rows


class FourRowPriorityTests(unittest.TestCase):
    def assert_safe_normal(self,page,plan):
        self.assertEqual((plan['mode'],plan['target_rows']),('normal',4))
        touch=plan['touch_length_px'];self.assertTrue(600<=touch<=700)
        geometry=page['geometry'];s=rows.layout(geometry);b=geometry['profile']['contentBounds']
        pitch=s['median_pitch_px']
        for gain in plan['gain_bounds']:
            for row in s['groups']:
                for c in row:
                    for key in ('discRect','nameRect','levelRect'):
                        r=c[key];y=r['y']+4*pitch-touch*gain
                        self.assertGreaterEqual(y,b['y']+4)
                        self.assertLessEqual(y+r['height'],b['y']+b['height']-4)
            for c in s['groups'][-1]:
                r=c['nameRect'];y=r['y']-touch*gain
                self.assertGreaterEqual(y,b['y']+6)
                self.assertLessEqual(y+r['height'],b['y']+b['height']-6)

    def test_nominal_above_700_uses_safe_four_row_interval(self):
        for first,gain,limits in [(401,.9661266568483063,(.9411266568483063,1.0076338639652678)),
                                   (414,.9666666666666667,(.9416666666666667,.9916666666666667))]:
            page,_=synthetic_page(4,first)
            plan=rows.next_swipe(page['geometry'],679,{'observed_gain':gain},gain_bounds=limits)
            self.assertGreater(plan['four_row_suggested_touch_length_px'],700)
            self.assert_safe_normal(page,plan)
            self.assertEqual(plan['reason'],'four_row_safe_landing_interval')

    def test_short_feedback_does_not_trigger_another_conservative_bridge(self):
        page,_=synthetic_page(4,401);summary={}
        normal={'current_touch_px':679,'observed_gain':.9661266568483063,'actual_anchor_shift_px':656}
        first=rows.plan_forward(summary,page['geometry'],679,normal)
        short={'current_touch_px':357,'observed_gain':333/357,'actual_anchor_shift_px':333}
        after=rows.plan_forward(summary,page['geometry'],357,short)
        self.assert_safe_normal(page,after)
        self.assertEqual(first['touch_length_px'],after['touch_length_px'])
        self.assertEqual(summary['forward_gain_samples'],[normal['observed_gain']])
        self.assertEqual(summary['bridge_gain_samples'],[short['observed_gain']])
        self.assertEqual(after['observed_gain'],normal['observed_gain'])
        self.assertEqual(after['measurement'],short)
        snapshot=copy.deepcopy((summary['forward_gain_samples'],summary['bridge_gain_samples']))
        rows.plan_forward(summary,page['geometry'],357)
        self.assertEqual((summary['forward_gain_samples'],summary['bridge_gain_samples']),snapshot)

    def test_recovered_full_view_recalculates_four_rows_from_current_phase(self):
        for first in (361,365):
            page,_=synthetic_page(2,first)
            plan=rows.plan_forward({},page['geometry'],700,
                {'current_touch_px':700,'observed_gain':.983})
            self.assert_safe_normal(page,plan)
            self.assertNotEqual(plan['touch_length_px'],700)

    def test_destination_visibility_does_not_override_boundary_name_visibility(self):
        page,_=synthetic_page(4,401)
        plan=rows.next_swipe(page['geometry'],679,{'observed_gain':.966},gain_bounds=(.941,1.008))
        self.assert_safe_normal(page,plan)
        self.assertLess(plan['touch_length_px'],700)
        # 700 can leave the next four rows readable while clipping the old
        # boundary names. It must not be selected solely by the touch ceiling.
        last=rows.layout(page['geometry'])['groups'][-1]
        self.assertTrue(any(c['nameRect']['y']-700*1.008<278 for c in last))

    def test_no_safe_four_row_interval_keeps_short_bridge_fallback(self):
        page,_=synthetic_page(first=358)
        plan=rows.next_swipe(page['geometry'],650,{'observed_gain':1.14},gain_bounds=(1.13,1.15),
                             bridge_feedback={'observed_gain':1.},bridge_gain_bounds=(.975,1.025))
        self.assertEqual((plan['mode'],plan['target_rows']),('bridge',2))
        self.assertLess(plan['touch_length_px'],600)
        self.assertEqual(plan['reason'],'no_safe_four_row_interval')
        self.assertEqual(plan['observed_gain'],1.)

    def test_three_complete_rows_only_allow_overlap_bridge(self):
        page,_=synthetic_page(2,433,(4,4,4))
        plan=rows.plan_forward({},page['geometry'],694,{'current_touch_px':694,'observed_gain':.98})
        self.assertEqual((plan['mode'],plan['target_rows']),('bridge',2))
        self.assertIsNone(plan['four_row_safe_touch_interval_px'])

    def test_no_safe_bridge_stops_instead_of_forcing_a_boundary_distance(self):
        page,_=synthetic_page(2,489,(4,4,4))
        for record in page['records']:
            dy=record['row']*18
            record['card']['circle']['centerY']+=dy
            for key in ('discRect','nameRect','levelRect'):record['card'][key]['y']+=dy
        with self.assertRaisesRegex(ValueError,'no safe overlap bridge'):
            rows.next_swipe(page['geometry'],700,{'observed_gain':1.},gain_bounds=(.85,1.15))


if __name__=='__main__':unittest.main()
