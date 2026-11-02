"""No-device checks for Shadow boundaries and backward-compatible crop behavior."""
import ast,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'agent'))
from utils.star_local_ocr_shadow import Recognizer
path=ROOT/'agent/custom/action/star_full_local_ocr_shadow.py'
spec=importlib.util.spec_from_file_location('star_full_shadow_boundary_test',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)

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
  class Bridge:
   def resolve(self,*args):return {'normalized':'天府','confidence':.99}
  def provider(image):return {'ocr_ms':1.,'all_results':[{'text':'天府','score':.99,'box':[0,0,9,9]}]}
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);im=np.full((12,12,3),100,np.uint8);rect={'x':1,'y':1,'width':3,'height':3};card={'rowIndex':0,'columnIndex':0}
   default=Recognizer(provider,Bridge(),root/'default');default.field(im,rect,'name','sample',card)
   self.assertTrue((root/'default/debug-crops/sample-r0c0-name.png').exists())
   quiet=Recognizer(provider,Bridge(),root/'quiet',save_debug_crops=False);quiet.field(im,rect,'name','sample',card)
   self.assertFalse((root/'quiet/debug-crops').exists())
 def test_full_shadow_has_no_upload_resource_or_inventory_limit_constants(self):
  files=['agent/custom/action/star_full_local_ocr_shadow.py','agent/utils/star_backpack_navigation.py','agent/utils/star_four_row_inventory.py']
  forbidden={'upload_full_capture_batch','star_capture_transport','StarBackpackCaptureOrchestration','estimate_vertical_motion','motion_functions','match_pair'}
  for name in files:
   text=(ROOT/name).read_text(encoding='utf-8');tree=ast.parse(text)
   self.assertFalse(any(token in text for token in forbidden),name)
   self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'Resource','Tasker','AdbController'} for n in ast.walk(tree)),name)
   parents={id(child):parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
   def inventory_literal(node):
    if not isinstance(node,ast.Constant) or type(node.value) is not int or node.value not in {69,105,174}:return False
    parent=parents.get(id(node))
    # 105:195 is the existing title-image ROI, unrelated to stock limits.
    return not (node.value==105 and isinstance(parent,ast.Slice) and parent.lower is node and isinstance(parent.upper,ast.Constant) and parent.upper.value==195)
   self.assertFalse(any(inventory_literal(n) for n in ast.walk(tree)),name)
 def test_retired_upload_removed_and_shared_tab_navigation_preserved(self):
  self.assertFalse((ROOT/'assets/resource/base/pipeline/star_backpack_capture_orchestration.json').exists())
  self.assertFalse((ROOT/'agent/custom/action/star_backpack_capture_orchestration.py').exists())
  new_nodes=json.loads((ROOT/'assets/resource/base/pipeline/star_local_ocr_navigation.json').read_text(encoding='utf-8'))
  self.assertEqual(set(new_nodes),{'星石本地OCR-已在背包','星石本地OCR-点击背包入口','星石背包-点击辅星','星石背包-点击经验星曜'})
  self.assertNotIn('.ux/',json.dumps(new_nodes))

if __name__=='__main__':unittest.main()
