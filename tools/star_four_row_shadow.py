"""Dedicated one-shot four-row Agent Shadow host; never touches old Shadow locks or outputs."""
from pathlib import Path
import sys,os,argparse,json,time,subprocess,importlib.util
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'agent'))
from utils.star_local_ocr_shadow import write,now_ms
ROOT=REPO/'.ux/audits/star-four-row-shadow-20261031'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--agent',action='store_true');p.add_argument('--identifier')
 for name in ['manual-start-authorized','manual-reset-confirmed','backend-touch-hold-confirmed']:p.add_argument('--'+name,action='store_true')
 a=p.parse_args()
 if a.agent:
  from maa.agent.agent_server import AgentServer
  from maa.library import Library
  spec=importlib.util.spec_from_file_location('four_row_shadow_action',REPO/'agent/custom/action/star_four_row_shadow.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  write(ROOT/'agent-ready.json',{'pid':os.getpid(),'is_agent_server':Library.is_agent_server(),'action':'StarFourRowShadow'});assert AgentServer.start_up(a.identifier);AgentServer.join();AgentServer.shut_down();return
 if not(a.manual_start_authorized and a.manual_reset_confirmed and a.backend_touch_hold_confirmed):raise ValueError('manual confirmations required')
 if (ROOT/'host-started.json').exists() or (ROOT/'gesture-once.json').exists() or (ROOT/'live').exists():raise FileExistsError('one-shot run already attempted; no automatic retry')
 write(ROOT/'host-started.json',{'pid':os.getpid(),'time_ns':time.time_ns()})
 from maa.resource import Resource
 from maa.controller import AdbController
 from maa.toolkit import Toolkit
 from maa.tasker import Tasker
 from maa.agent_client import AgentClient
 from maa.library import Library
 Toolkit.init_option(ROOT/'host-native',{'logging':True,'save_on_error':False});start=time.perf_counter();resource=Resource();assert resource.use_cpu();t=time.perf_counter();assert resource.post_bundle(REPO/'assets/resource/base').wait().succeeded;load=now_ms(t)
 pipeline={'StarFourRowShadowEntry':{'recognition':'DirectHit','action':{'type':'Custom','param':{'custom_action':'StarFourRowShadow','custom_action_param':{'output':str(ROOT/'live'),'manual_start_authorized':True,'manual_reset_confirmed':True,'backend_touch_hold_confirmed':True}}},'next':[],'on_error':[],'pre_delay':0,'post_delay':0}}
 write(ROOT/'shadow-pipeline.json',pipeline);assert resource.post_pipeline(ROOT/'shadow-pipeline.json').wait().succeeded
 cfg=json.loads((REPO/'.ux/audits/star-swipe-endhold-20261030/controller-config.json').read_text(encoding='utf-8'))
 c=AdbController('D:/Program Files/Netease/MuMu Player 12/nx_main/adb.exe','127.0.0.1:16384',screencap_methods=64,input_methods=8,config=cfg);assert c.post_connection().wait().succeeded;assert c.set_screenshot_target_long_side(1280)
 tasker=Tasker();assert tasker.bind(resource,c) and tasker.inited;client=AgentClient.create_tcp(0);assert client.bind(resource);assert client.set_timeout(180000)
 env={**os.environ,'TEMP':str(ROOT/'temp'),'TMP':str(ROOT/'temp'),'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'};log=(ROOT/'agent-console.log').open('w',encoding='utf-8');proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--agent','--identifier',client.identifier],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 try:
  assert client.connect();assert 'StarFourRowShadow' in client.custom_action_list
  write(ROOT/'host-evidence.json',{'host_pid':os.getpid(),'agent_pid':proc.pid,'native_version':Library.version(),'resource_owned_count':1,'resource_hash':resource.hash,'resource_bundle':str(REPO/'assets/resource/base'),'resource_bundle_load_ms':load,'address':'127.0.0.1:16384','controller_uuid':c.uuid,'input_method':'MuMuPlayerExtras','screencap_method':'MuMuPlayerExtras','input_mask':8,'screencap_mask':64,'loopback_identifier':client.identifier})
  time.sleep(2.5);t=time.perf_counter();job=tasker.post_task('StarFourRowShadowEntry').wait();write(ROOT/'host-task-result.json',{'succeeded':job.succeeded,'task_wall_ms':now_ms(t),'host_total_wall_ms':now_ms(start)})
  if not job.succeeded:raise RuntimeError('Four-row Shadow stopped; preserve evidence; do not retry')
 finally:
  client.disconnect()
  try:proc.wait(timeout=8)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
  log.close()
if __name__=='__main__':main()
