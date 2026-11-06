"""Single entry/reset remains separate from in-backpack Tab clicks."""
import json,sys,unittest
from pathlib import Path
from types import SimpleNamespace,MethodType
from unittest.mock import Mock
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'agent'))
from custom.action.star_full_local_ocr_shadow import FullShadowSession

def frame(section):
    im=np.zeros((1280,720,3),np.uint8)
    im[105:195,200:520]=np.random.default_rng(7).integers(0,255,(90,320,3),dtype=np.uint8)
    for x in (90,290,480):im[225:245,x:x+12,2]=30
    x={'main':90,'support':290,'experience':480}[section];im[225:245,x:x+12,2]=230
    return im

def owner(image):
    calls=[]
    def run(node,override):
        calls.append((node,override));return SimpleNamespace(status=SimpleNamespace(succeeded=True))
    result=SimpleNamespace(context=SimpleNamespace(tasker=SimpleNamespace(stopping=False),run_task=run),
        bag_ref=frame('main'),progress=Mock(),capture=Mock(return_value=image),
        stabilize=Mock(side_effect=lambda image,*args:image),prepare=Mock(side_effect=AssertionError('entry must not run on Tab switch')))
    result.check_backpack=MethodType(FullShadowSession.check_backpack,result)
    return result,calls

class TabSwitchTests(unittest.TestCase):
    def test_support_only_clicks_tab_then_keeps_list_stability(self):
        image=frame('support');s,calls=owner(image);self.assertIs(FullShadowSession.switch(s,'support'),image)
        self.assertEqual([n for n,_ in calls],['星石背包-点击辅星']);s.prepare.assert_not_called();s.capture.assert_called_once_with('selected-support');s.stabilize.assert_called_once()
        self.assertEqual(calls[0][1][calls[0][0]]['post_delay'],1500)
    def test_experience_one_identity_checked_capture_without_list_stabilizer(self):
        image=frame('experience');s,calls=owner(image);self.assertIs(FullShadowSession.switch(s,'experience'),image)
        self.assertEqual([n for n,_ in calls],['星石背包-点击经验星曜']);s.capture.assert_called_once_with('selected-experience');s.stabilize.assert_not_called();s.prepare.assert_not_called()
        self.assertEqual(calls[0][1][calls[0][0]]['post_delay'],1500)
    def test_wrong_selected_experience_tab_stops(self):
        s,_=owner(frame('support'))
        with self.assertRaisesRegex(RuntimeError,'identity or selected'):FullShadowSession.switch(s,'experience')
        s.capture.assert_called_once();s.stabilize.assert_not_called()
    def test_wrong_page_title_stops(self):
        image=frame('experience');image[105:195,200:520]=0;s,_=owner(image)
        with self.assertRaisesRegex(RuntimeError,'identity or selected'):FullShadowSession.switch(s,'experience')
        s.stabilize.assert_not_called()
    def test_both_resources_tabs_are_direct_clicks_without_navigation_chains(self):
        for bundle in ('base','zh_tw'):
            nodes=json.loads((ROOT/f'assets/resource/{bundle}/pipeline/star_local_ocr_navigation.json').read_text(encoding='utf-8'))
            for name in ('星石背包-点击辅星','星石背包-点击经验星曜'):
                node=nodes[name];self.assertEqual(node['recognition']['type'],'DirectHit');self.assertEqual(node['action']['type'],'Click');self.assertEqual(node.get('next',[]),[]);self.assertEqual(node.get('on_error',[]),['stop'])
    def test_runtime_tab_call_cannot_chain_next_or_error_navigation(self):
        s,calls=owner(frame('experience'));FullShadowSession.switch(s,'experience');override=calls[0][1][calls[0][0]]
        self.assertEqual(override['next'],[]);self.assertEqual(override['on_error'],[])

if __name__=='__main__':unittest.main()
