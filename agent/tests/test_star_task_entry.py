"""Toolbox wiring and task feedback only; no emulator or real cloud writes."""
import json,sys,unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent'))
from custom.action import star_inventory_sync as sync

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
        node=json.loads((ROOT/'assets/resource/base/pipeline/star_local_ocr_task.json').read_text(encoding='utf-8'))['星石本地OCR-完整背包']
        self.assertEqual(node['action']['param'],{'custom_action':'StarInventorySync','custom_action_param':{'shadow':True}})
        self.assertEqual(node['next'],[])
        self.assertEqual(node['focus']['Node.Action.Succeeded']['display'],['log','toast'])
        self.assertIn('重试原批次',node['focus']['Node.Action.Failed'])
    def test_shared_switch_nodes_are_unchanged(self):
        nodes=json.loads((ROOT/'assets/resource/base/pipeline/star_local_ocr_navigation.json').read_text(encoding='utf-8'))
        for key,roi in [('星石背包-点击辅星',[320,215,80,40]),('星石背包-点击经验星曜',[500,215,120,40])]:
            self.assertEqual(nodes[key],{'recognition':{'type':'DirectHit','param':{'roi':roi}},'action':{'type':'Click','param':{}},'on_error':['stop']})
    def test_formal_sender_keeps_structured_endpoint_and_instance_protocol(self):
        self.assertEqual(sync.SYNC_PATH,'/open-api/star/inventory-sync')
        local={'schemaVersion':2,'complete':True,'sections':{'main':{'complete':True,'count':1,'items':[{'section':'main','name':'天府','quality':'橙','level':1}]},'support':{'complete':True,'count':0,'items':[]}},'experience':{'complete':True,'structured':True,'orangeCount':0,'purpleCount':1,'whiteCount':2}}
        payload=sync.compact_inventory(local,'11111111-1111-4111-8111-111111111111')
        self.assertEqual(set(payload),{'format','version','scan_id','complete','counts','inventory','experience'})
        self.assertEqual(payload['inventory'],[{'kind':'main','name':'天府','quality':'orange','level':1}])
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

if __name__=='__main__':unittest.main()
