"""Focused experience integration checks. Bridge tests need the packaged desktop Node."""
import copy,importlib.util,json,re,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'agent'))
from utils.star_experience_ocr import recognize_experience
from utils.star_local_ocr_shadow import Bridge,PACKAGED_RUNTIME


class FakeBridge:
 def __init__(self,kinds=('white','orange','purple')):
  self.candidates=[{'quality':{'kind':k,'confidence':.99},'countRect':{'x':i*20,'y':0,'width':18,'height':18},'completeness':'complete'} for i,k in enumerate(kinds)]
 def experience_geometry(self,image):return {'tab':{'selected':True},'candidates':self.candidates,'geometryMs':0.}
 def experience_count(self,variants):
  valid=[(int(t['text']),t['score']) for v in variants for t in v['all_results'] if re.fullmatch(r'[0-9]{1,6}',t['text'])]
  best=max(valid,key=lambda z:z[1]) if valid else (None,0.)
  return {'count':best[0],'confidence':best[1]}


class ExperienceTests(unittest.TestCase):
 def image(self):return np.full((30,70,3),220,np.uint8)
 def provider(self,values,score=.99):
  values=iter(values)
  return lambda image:{'ocr_ms':1.,'all_results':[{'text':str(next(values)),'score':score,'box':[0,0,image.shape[1],image.shape[0]]}]}
 def test_three_counts_are_mapped_by_color_not_array_order(self):
  result=recognize_experience(self.image(),self.provider([91,1,525]),FakeBridge())
  self.assertTrue(result['complete']);self.assertEqual([result[k+'Count'] for k in ('orange','purple','white')],[1,525,91])
  self.assertEqual(result['timing']['ocr_calls'],3)
 def test_zero_is_confirmed_and_multidigit_preserved(self):
  result=recognize_experience(self.image(),self.provider([0,123456,42]),FakeBridge())
  self.assertTrue(result['complete']);self.assertEqual((result['whiteCount'],result['orangeCount']),(0,123456))
 def test_low_confidence_is_bounded_and_never_filled_with_zero(self):
  result=recognize_experience(self.image(),self.provider(['bad']*9,.2),FakeBridge())
  self.assertFalse(result['complete']);self.assertIsNone(result['orangeCount']);self.assertLessEqual(result['timing']['ocr_calls'],9)
 def test_bad_roi_and_partial_icon_are_not_recognized(self):
  b=FakeBridge();b.candidates[0]['countRect']['x']=-1;b.candidates[1]['completeness']='partial_bottom'
  result=recognize_experience(self.image(),self.provider([7]),b)
  self.assertFalse(result['complete']);self.assertEqual(result['timing']['ocr_calls'],1)
  self.assertIn('experience_invalid_count_roi',result['warnings']);self.assertIn('experience_icon_incomplete',result['warnings'])
 def test_duplicate_and_missing_quality_are_incomplete(self):
  result=recognize_experience(self.image(),self.provider([2,3,4]),FakeBridge(('orange','orange','white')))
  self.assertFalse(result['complete']);self.assertIsNone(result['orangeCount']);self.assertIsNone(result['purpleCount'])
 def test_unknown_quality_does_not_infer_from_position(self):
  b=FakeBridge();b.candidates[0]['quality']={'kind':None,'confidence':0.}
  result=recognize_experience(self.image(),self.provider([2,3]),b)
  self.assertFalse(result['complete']);self.assertIsNone(result['whiteCount'])
 def test_conflicting_fallback_stops_confirmation(self):
  # Initial valid .92 triggers retry; a different .99 result must remain a conflict.
  outputs=iter([('1',.92),('2',.99),('3',.99),('4',.99)])
  def provider(image):
   text,score=next(outputs);return {'ocr_ms':1.,'all_results':[{'text':text,'score':score}]}
  result=recognize_experience(self.image(),provider,FakeBridge())
  self.assertFalse(result['complete']);self.assertIsNone(result['whiteCount']);self.assertIn('experience_count_conflict',result['warnings'])


class PackagedBridgeTests(unittest.TestCase):
 def test_real_yuanstar_numeric_parser_and_packaged_modules(self):
  with tempfile.TemporaryDirectory() as name:
   b=Bridge(PACKAGED_RUNTIME,Path(name)/'bridge.log')
   try:
    for text,expected in [('0',0),('123456',123456),('-1',None),('1.5',None),('bad',None),('1000000',None)]:
     self.assertEqual(b.experience_count([{'variant':'unit','all_results':[{'text':text,'score':.99}]}])['count'],expected)
   finally:b.close()


spec=importlib.util.spec_from_file_location('experience_session_test',ROOT/'agent/custom/action/star_full_local_ocr_shadow.py')
session=importlib.util.module_from_spec(spec);spec.loader.exec_module(session)


class IntegrationTests(unittest.TestCase):
 def test_task_registration_and_automatic_sync_entry(self):
  interface=json.loads((ROOT/'assets/interface.json').read_text(encoding='utf-8'))
  tasks=[t for t in interface['task'] if t['name']=='星石背包整理（本地识别）'];self.assertEqual(len(tasks),1)
  task=tasks[0];self.assertFalse(task['default_check']);self.assertEqual(task['option'],['连接码'])
  pipeline=json.loads((ROOT/'assets/resource/base/pipeline/star_local_ocr_task.json').read_text(encoding='utf-8'))
  node=pipeline[task['entry']];self.assertEqual(node['action']['param']['custom_action'],'StarInventorySync')
  self.assertEqual(node['action']['param']['custom_action_param'],{'shadow':True});self.assertEqual(node['next'],[])
  self.assertEqual(set(node['focus']),{'Node.Action.Starting','Node.Action.Succeeded','Node.Action.Failed'})
  self.assertEqual(node['focus']['Node.Action.Succeeded']['display'],['log','toast'])
  self.assertIn('star_full_local_ocr_shadow',(ROOT/'agent/custom/action/__init__.py').read_text(encoding='utf-8'))
  self.assertIn('star_inventory_sync',(ROOT/'agent/custom/action/__init__.py').read_text(encoding='utf-8'))
 def test_outputs_are_separate_and_failure_preserves_instances(self):
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);runtime=root/'agent/runtime/star_ocr';runtime.mkdir(parents=True)
   with mock.patch.object(session,'REPO',root),mock.patch.object(session,'PACKAGED_RUNTIME',runtime):
    a=session.FullShadowSession(SimpleNamespace(),{'shadow':True});b=session.FullShadowSession(SimpleNamespace(),{'shadow':True})
    self.assertNotEqual(a.out,b.out)
    fakebridge=SimpleNamespace(resolve=lambda *args:None,close=lambda:None)
    def scan(section,image):
     a.sections[section]={'complete':True};a.inventories[section].items=[{'name':'test','section':section}]
    a.prepare=lambda:np.zeros((1280,720,3),np.uint8);a.scan=scan
    def switch(section):a.last_capture={'file':'evidence.png'};return a.prepare()
    a.switch=switch
    with mock.patch.object(session,'Bridge',return_value=fakebridge),mock.patch.object(session,'recognize_experience',return_value=session.empty_experience('ocr_failure')):
     self.assertFalse(a.execute())
    inv=json.loads((a.out/'inventory.json').read_text(encoding='utf-8'))
    self.assertEqual(inv['schemaVersion'],2);self.assertFalse(inv['complete'])
    self.assertNotIn('items',inv['experience']);self.assertEqual(inv['experience']['detailsFile'],'experience-results.json')
    self.assertEqual([inv['sections'][s]['count'] for s in ('main','support')],[1,1]);self.assertIsNone(inv['experience']['orangeCount'])
    for file in ('inventory.json','timing.json','main-results.json','support-results.json','experience-results.json','tail-merge.json','result.json','progress.json'):
     self.assertTrue((a.out/file).exists(),file);self.assertFalse((a.root/file).exists(),file)
    self.assertEqual(set(p.name for p in b.out.iterdir()),{'pages'})
 def test_formal_bridge_does_not_use_path_node_or_audit_runtime(self):
  from utils import star_local_ocr_shadow as core
  with tempfile.TemporaryDirectory() as name,mock.patch.object(core,'bundled_node',return_value=ROOT/'runtimes/star-ocr-node/node.exe'),mock.patch.object(core.subprocess,'Popen') as popen:
   b=core.Bridge(core.PACKAGED_RUNTIME,Path(name)/'stderr.log')
   try:
    argv=popen.call_args.args[0];self.assertNotEqual(argv[0],'node');self.assertIn('agent',argv[1]);self.assertNotIn('.ux',str(argv))
   finally:b.err.close()
 def test_desktop_builds_include_runtime_and_pin_target_arch(self):
  sys.path.insert(0,str(ROOT));from tools.ci.bundle_star_ocr_node import target,ARCHIVES
  self.assertEqual(len(ARCHIVES),6);self.assertEqual(target('win','aarch64'),('win','arm64'))
  for file in ('install.py','install4release.py'):
   text=(ROOT/file).read_text(encoding='utf-8');self.assertIn('bundle(install_path)',text);self.assertIn('working_dir / "agent"',text)
  workflow=(ROOT/'.github/workflows/install.yml').read_text(encoding='utf-8');self.assertEqual(workflow.count('STAR_OCR_NODE_ARCH:'),3)
  with self.assertRaises(RuntimeError):target('android','arm64')


if __name__=='__main__':unittest.main()
