from pathlib import Path
from types import MethodType
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch
import json
import math
import numpy as np
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))

from custom.action import star_full_local_ocr_shadow as module
from custom.action import star_inventory_sync as sync
from custom.action.star_full_local_ocr_shadow import FullShadowSession
from star_ocr_golden import golden
from utils import star_backpack_navigation as nav
from utils import star_local_ocr_shadow as core
from utils.star_local_ocr_shadow import PACKAGED_RUNTIME
from utils.star_local_ocr_shadow import Recognizer


class TaskEntryTests(unittest.TestCase):
    def test_one_toolbox_entry_preserves_other_cases_and_connection_definition(self):
        current=json.loads((ROOT/'assets/interface.json').read_text(encoding='utf-8'))
        cases=current['option']['百宝箱-模式']['cases']
        self.assertEqual([c['name'] for c in cases],['剧情速读','密探赠礼薅五铢钱','使用兑换码','采集密探信息','自动识别背包','星石背包整理'])
        self.assertEqual(cases[-1]['option'],['连接码'])
        self.assertEqual(cases[-1]['pipeline_override'],{'百宝箱启动':{'next':['星石本地OCR-完整背包']}})
        self.assertIn('连接码',current['option'])
        self.assertNotIn('同步星石至YuanHub',current['option'])
        self.assertEqual(len([t for t in current['task'] if t.get('entry')=='百宝箱启动']),1)
        self.assertFalse(any(t.get('entry')=='星石本地OCR-完整背包' for t in current['task']))
    def test_formal_action_keeps_parameters_and_feedback_channels(self):
        resources = ROOT/'assets/resource'
        for name in ('star_local_ocr_task.json','star_local_ocr_navigation.json'):
            base = json.loads((resources/'base/pipeline'/name).read_text(encoding='utf-8'))
            traditional = json.loads((resources/'zh_tw/pipeline'/name).read_text(encoding='utf-8'))
            self.assertEqual(base, traditional)
        self.assertFalse((resources/'base/pipeline/star_backpack_capture_orchestration.json').exists())
        self.assertEqual({p.name for p in PACKAGED_RUNTIME.iterdir()},
                         {'star_catalog.json','ocr_aliases.json','README.md'})
        node=json.loads((resources/'base/pipeline/star_local_ocr_task.json').read_text(encoding='utf-8'))['星石本地OCR-完整背包']
        self.assertEqual(node['action']['param'],{'custom_action':'StarInventorySync','custom_action_param':{'shadow':True}})
        self.assertEqual(node['next'],[])
        self.assertEqual(node['focus']['Node.Action.Succeeded']['display'],['log','toast'])
        self.assertIn('重试原批次',node['focus']['Node.Action.Failed'])
    def test_success_reports_real_counts_and_unknown_capacity_is_not_zero(self):
        local={'sections':{'main':{'items':[{},{}]},'support':{'items':[{}]}},'bag':{'capacity':270}}
        self.assertEqual(sync.sync_success_message(local),'星石背包同步完成。\n主星2颗，辅星1颗，背包3/270。')
        for capacity in (None,0,True,2):
            local['bag']['capacity']=capacity
            self.assertIn('背包3颗，容量未知。',sync.sync_success_message(local))
            self.assertNotIn('3/0',sync.sync_success_message(local))
    def test_ocr_failure_never_reaches_sync_and_does_not_mark_pending(self):
        class Context:pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(sync,'REPO',root),patch.object(sync,'Context',Context),patch.object(sync.Library,'is_agent_server',return_value=True),patch.object(sync,'FullShadowSession') as scan,patch.object(sync,'sync_saved_inventory') as send,patch.object(sync.logger,'error') as log:
                scan.return_value.execute.return_value=False
                self.assertFalse(sync.StarInventorySync().run(Context(),SimpleNamespace(custom_action_param='{"output":".","shadow":true}')))
                send.assert_not_called();self.assertFalse((root/'latest.json').exists())
                self.assertIn('识别未完成',log.call_args.args[0])
    def test_success_status_only_after_sender_success(self):
        class Context:pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=root/'saved';batch.mkdir()
            (root/'latest.json').write_text(json.dumps({'directory':'saved','outcome':'cloud_sync_failed'}),encoding='utf-8')
            (batch/'inventory.json').write_text(json.dumps({'sections':{'main':{'items':[{}]},'support':{'items':[{}]}},'bag':{'capacity':270}}),encoding='utf-8')
            with patch.object(sync,'REPO',root),patch.object(sync,'Context',Context),patch.object(sync.Library,'is_agent_server',return_value=True),patch.object(sync,'FullShadowSession') as scan,patch.object(sync,'sync_saved_inventory',return_value={'success':True}),patch.object(sync.logger,'info') as log:
                self.assertTrue(sync.StarInventorySync().run(Context(),SimpleNamespace(custom_action_param='{"output":".","shadow":true}')))
                scan.assert_not_called();self.assertEqual(json.loads((root/'latest.json').read_text(encoding='utf-8'))['outcome'],'synced')
                self.assertIn('主星1颗，辅星1颗，背包2/270',log.call_args.args[0])

class Frame:
    def __init__(self,page):self.page=page
    def __getitem__(self,key):return SimpleNamespace(mean=lambda:255)

class Context:
    def __init__(self,wait_succeeds=True,render_after_wait=True):
        self.page='observatory';self.calls=[];self.tasker=SimpleNamespace(stopping=False)
        self.wait_succeeds=wait_succeeds;self.render_after_wait=render_after_wait
    def run_recognition(self,node,image):
        expected={'已在界面-观星':'observatory','星石本地OCR-已在背包':'backpack'}.get(node)
        return SimpleNamespace(hit=expected is not None and image.page==expected)
    def run_task(self,node,override):
        self.calls.append((node,override))
        if node=='星石本地OCR-点击背包入口':self.page='transition'
        elif node=='星石本地OCR-已在背包' and self.wait_succeeds and self.render_after_wait:self.page='backpack'
        return SimpleNamespace(status=SimpleNamespace(succeeded=node!='星石本地OCR-已在背包' or self.wait_succeeds))

class NavigationTest(unittest.TestCase):
    def run_entry(self,context):
        self.captures=[];self.records=[]
        def capture(label):self.captures.append(label);return Frame(context.page)
        return nav.prepare_star_backpack_entry(context,capture,self.records.append)
    def test_delayed_transition_waits_for_identity_without_second_click(self):
        context=Context();image,result=self.run_entry(context)
        self.assertEqual(image.page,'backpack');self.assertTrue(result['success'])
        self.assertEqual([x[0] for x in context.calls],['星石本地OCR-点击背包入口','星石本地OCR-已在背包'])
        settings=context.calls[1][1]['星石本地OCR-已在背包']
        self.assertEqual(settings['timeout'],6000);self.assertEqual(settings['post_delay'],0)
        self.assertEqual(settings['next'],[]);self.assertEqual(settings['on_error'],[])
    def test_identity_timeout_stops_before_accepting_a_backpack_frame(self):
        context=Context(wait_succeeds=False)
        with self.assertRaisesRegex(RuntimeError,'navigation failed'):self.run_entry(context)
        self.assertNotIn('entry-main-backpack',self.captures)
        self.assertEqual(len(context.calls),2);self.assertFalse(self.records[0]['success'])
    def test_final_image_still_requires_backpack_identity(self):
        context=Context(render_after_wait=False)
        with self.assertRaisesRegex(RuntimeError,'backpack identity missing'):self.run_entry(context)
        self.assertFalse(self.records[0]['success'])

class BackpackReturnContext(Context):
    def __init__(self,return_wait_succeeds=True,return_render=True):
        super().__init__();self.page='backpack';self.return_wait_succeeds=return_wait_succeeds;self.return_render=return_render
    def run_task(self,node,override):
        if node=='返回上级界面':self.calls.append((node,override));self.page='return_transition';return SimpleNamespace(status=SimpleNamespace(succeeded=True))
        if node=='已在界面-观星':
            self.calls.append((node,override))
            if self.return_wait_succeeds and self.return_render:self.page='observatory'
            return SimpleNamespace(status=SimpleNamespace(succeeded=self.return_wait_succeeds))
        return super().run_task(node,override)

class BackpackReturnTests(unittest.TestCase):
    run_entry=NavigationTest.run_entry
    # Dedicated cases use the same helper, but do not rerun inherited entry cases.
    def test_delayed_return_waits_for_observatory_before_new_screenshot(self):
        context=BackpackReturnContext();image,result=self.run_entry(context)
        self.assertTrue(result['success']);self.assertEqual(image.page,'backpack')
        self.assertEqual([x[0] for x in context.calls],['返回上级界面','已在界面-观星','星石本地OCR-点击背包入口','星石本地OCR-已在背包'])
        self.assertEqual(context.calls[1][1]['已在界面-观星']['timeout'],6000)
    def test_return_timeout_does_not_click_backpack_entry(self):
        context=BackpackReturnContext(return_wait_succeeds=False)
        with self.assertRaisesRegex(RuntimeError,'navigation failed'):self.run_entry(context)
        self.assertNotIn('entry-observatory',self.captures);self.assertNotIn('星石本地OCR-点击背包入口',[x[0] for x in context.calls])
    def test_return_wait_does_not_bypass_fresh_image_identity(self):
        context=BackpackReturnContext(return_render=False)
        with self.assertRaisesRegex(RuntimeError,'back did not reach observatory'):self.run_entry(context)
        self.assertFalse(self.records[0]['success'])

class HomeEntryTests(unittest.TestCase):
    def test_home_enters_observatory_then_backpack_once(self):
        class HomeContext(Context):
            def __init__(self):super().__init__();self.page='home'
            def run_recognition(self,node,image):
                if node=='已在界面-左半屏':return SimpleNamespace(hit=image.page=='home')
                return super().run_recognition(node,image)
            def run_task(self,node,override):
                if node=='进入界面-观星':
                    self.calls.append((node,override));self.page='observatory'
                    return SimpleNamespace(status=SimpleNamespace(succeeded=True))
                return super().run_task(node,override)
        context=HomeContext();image,result=nav.prepare_star_backpack_entry(context,lambda label:Frame(context.page),lambda result:None)
        self.assertEqual(image.page,'backpack');self.assertTrue(result['success'])
        self.assertEqual([x[0] for x in context.calls],['进入界面-观星','星石本地OCR-点击背包入口','星石本地OCR-已在背包'])
        self.assertNotIn('返回上级界面',[x[0] for x in context.calls])

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
            for name,roi in (('星石背包-点击辅星',[320,215,80,40]),('星石背包-点击经验星曜',[500,215,120,40])):
                node=nodes[name];self.assertEqual(node['recognition']['param']['roi'],roi);self.assertEqual(node['recognition']['type'],'DirectHit');self.assertEqual(node['action']['type'],'Click');self.assertEqual(node.get('next',[]),[]);self.assertEqual(node.get('on_error',[]),['stop'])
    def test_runtime_tab_call_cannot_chain_next_or_error_navigation(self):
        s,calls=owner(frame('experience'));FullShadowSession.switch(s,'experience');override=calls[0][1][calls[0][0]]
        self.assertEqual(override['next'],[]);self.assertEqual(override['on_error'],[])

class ShadowBoundaryTests(unittest.TestCase):
 def test_explicit_shadow_required_before_paths_or_resource_use(self):
  for params in ({},{'shadow':False},{'shadow':1}):
   with self.subTest(params=params),self.assertRaisesRegex(ValueError,'explicit local Shadow'):
    module.FullShadowSession(object(),params)
 def test_invalid_run_id_rejected_before_output_creation(self):
  for run_id in ('',' ','..','.','../outside','a/b',r'a\b',r'D:\outside',None):
   with self.subTest(run_id=run_id),self.assertRaisesRegex(ValueError,'single directory'):
    module.FullShadowSession(object(),{'shadow':True,'run_id':run_id})
 def test_debug_crop_default_remains_enabled_and_opt_out_is_local(self):
  class FieldBackend:
   def resolve(self,*args):return {'normalized':'天府','confidence':.99}
  def provider(image):return {'ocr_ms':1.,'all_results':[{'text':'天府','score':.99,'box':[0,0,9,9]}]}
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);im=np.full((12,12,3),100,np.uint8);rect={'x':1,'y':1,'width':3,'height':3};card={'rowIndex':0,'columnIndex':0}
   default=Recognizer(provider,FieldBackend(),root/'default');default.field(im,rect,'name','sample',card)
   self.assertTrue((root/'default/debug-crops/sample-r0c0-name.png').exists())
   quiet=Recognizer(provider,FieldBackend(),root/'quiet',save_debug_crops=False);quiet.field(im,rect,'name','sample',card)
   self.assertFalse((root/'quiet/debug-crops').exists())


class NativeScanTests(unittest.TestCase):
    def assert_geometry_parity(self, actual, expected):
        def without_timing(result):
            result = dict(result)
            for key in ('geometryMs', 'qualityMs'):
                elapsed = result.pop(key)
                self.assertTrue(math.isfinite(elapsed) and elapsed >= 0, (key, elapsed))
            result.pop('timingBreakdown', None)
            return result
        breakdown = actual['timingBreakdown']
        self.assertEqual(set(breakdown), {'profile_viewport_ms', 'content_bounds_ms', 'edge_map_ms',
                                         'ring_score_ms', 'proposal_generation_ms', 'row_cluster_and_rois_ms',
                                         'quality_ms', 'full_geometry_ms'})
        self.assertTrue(all(math.isfinite(elapsed) and elapsed >= 0 for elapsed in breakdown.values()))
        self.assertEqual(breakdown['quality_ms'], actual['qualityMs'])
        self.assertGreaterEqual(breakdown['full_geometry_ms'], actual['geometryMs'] + actual['qualityMs'])
        self.assertEqual(without_timing(actual), without_timing(expected))

    def test_synthetic_empty_geometry_quality(self):
        # An empty synthetic frame must not invent inventory cards.
        image = np.full((1280, 720, 3), 80, np.uint8)
        empty = core.NativeStarOcr.geometry(image)
        self.assert_geometry_parity(empty, golden('session', 'geometry', image))
        self.assertEqual(empty['cards'], [])
        with self.assertRaisesRegex(ValueError, '720x1280 uint8 BGR'):
            core.NativeStarOcr.geometry(image[:, :, 0])


if __name__ == "__main__":
    unittest.main()
