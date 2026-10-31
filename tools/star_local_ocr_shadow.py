"""Opt-in real AgentServer Shadow host. One existing MaaYuan bundle/Resource, one live task."""
from pathlib import Path
import argparse,sys,subprocess,os,json,time,importlib.util,hashlib
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'agent'))
from utils.star_local_ocr_shadow import write,now_ms

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--agent',action='store_true');parser.add_argument('--identifier');parser.add_argument('--run-id',choices=['live','live-recovered'],default='live');parser.add_argument('--audit',type=Path,required=True)
    for name in ['manual-start-authorized','manual-reset-confirmed','backend-touch-hold-confirmed']:parser.add_argument('--'+name,action='store_true')
    a=parser.parse_args();audit=a.audit.resolve();tag='' if a.run_id=='live' else '-recovered'
    if audit!=(REPO/'.ux/audits/star-local-ocr-live-20261031').resolve():raise ValueError('fixed D audit root required')
    if a.agent:
        # Import AgentServer before any framework API, as in MaaYuan Agent startup.
        from maa.agent.agent_server import AgentServer
        from maa.library import Library
        path=REPO/'agent/custom/action/star_local_ocr_shadow.py';spec=importlib.util.spec_from_file_location('shadow_agent_action',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        write(audit/f'agent-ready{tag}.json',{'pid':os.getpid(),'is_agent_server':Library.is_agent_server(),'action':'StarLocalOcrShadow','identifier':a.identifier})
        assert AgentServer.start_up(a.identifier);AgentServer.join();AgentServer.shut_down();return
    if not(a.manual_start_authorized and a.manual_reset_confirmed and a.backend_touch_hold_confirmed):raise ValueError('manual start/reset/backend confirmations required')
    if (audit/'gesture-once.json').exists() or (audit/a.run_id).exists() or (audit/f'live-host-started{tag}.json').exists():raise FileExistsError('refuse repeat; preserve any failed evidence')
    write(audit/f'live-host-started{tag}.json',{'pid':os.getpid(),'time_ns':time.time_ns()})
    from maa.controller import AdbController
    from maa.resource import Resource
    from maa.tasker import Tasker
    from maa.toolkit import Toolkit
    from maa.agent_client import AgentClient
    from maa.library import Library
    Toolkit.init_option(audit/f'host-native{tag}',{'logging':True,'save_on_error':False})
    host_start=time.perf_counter();resource=Resource();assert resource.use_cpu();t=time.perf_counter()
    # The ONE task Resource loads MaaYuan's existing bundle, including its PP-OCRv6 model.
    assert resource.post_bundle(REPO/'assets/resource/base').wait().succeeded;bundle_ms=now_ms(t)
    pipeline={'StarLocalOcrShadowEntry':{'recognition':'DirectHit','action':{'type':'Custom','param':{'custom_action':'StarLocalOcrShadow','custom_action_param':{'output':str(audit/a.run_id),'manual_start_authorized':True,'manual_reset_confirmed':True,'backend_touch_hold_confirmed':True}}},'next':[],'on_error':[],'pre_delay':0,'post_delay':0}}
    write(audit/f'shadow-pipeline{tag}.json',pipeline);assert resource.post_pipeline(audit/f'shadow-pipeline{tag}.json').wait().succeeded
    cfg=json.loads((REPO/'.ux/audits/star-swipe-endhold-20261030/controller-config.json').read_text(encoding='utf-8'))
    controller=AdbController('D:/Program Files/Netease/MuMu Player 12/nx_main/adb.exe','127.0.0.1:16384',screencap_methods=64,input_methods=8,config=cfg)
    assert controller.post_connection().wait().succeeded;assert controller.set_screenshot_target_long_side(1280)
    tasker=Tasker();assert tasker.bind(resource,controller) and tasker.inited
    # Local loopback RPC avoids writing IPC socket files outside the D audit temp directory.
    client=AgentClient.create_tcp(0);assert client.bind(resource);assert client.set_timeout(180000)
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','TEMP':str(audit/'temp'),'TMP':str(audit/'temp')}
    log=(audit/f'agent-console{tag}.log').open('w',encoding='utf-8');proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--agent','--identifier',client.identifier,'--run-id',a.run_id,'--audit',str(audit)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        assert client.connect(),'AgentClient connect failed';assert 'StarLocalOcrShadow' in client.custom_action_list
        write(audit/f'host-evidence{tag}.json',{'host_pid':os.getpid(),'agent_pid':proc.pid,'native_version':Library.version(),'resource_owned_count':1,'bundle':str(REPO/'assets/resource/base'),'bundle_load_ms':bundle_ms,'host_resource_handle':str(resource._handle),'host_resource_hash':resource.hash,'controller_uuid':controller.uuid,'input_method':'MuMuPlayerExtras','screencap_method':'MuMuPlayerExtras','input_mask':8,'screencap_mask':64,'address':'127.0.0.1:16384','agent_actions':client.custom_action_list,'agent_connection_identifier':client.identifier,'no_independent_OCR_resource':True})
        time.sleep(2.5);t=time.perf_counter();job=tasker.post_task('StarLocalOcrShadowEntry').wait()
        write(audit/f'host-task-result{tag}.json',{'succeeded':job.succeeded,'task_wall_ms':now_ms(t),'host_total_wall_ms':now_ms(host_start),'bundle_load_ms':bundle_ms})
        if not job.succeeded:raise RuntimeError('Shadow task failed; inspect saved error; never retry automatically')
    finally:
        client.disconnect()
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
        log.close()
if __name__=='__main__':main()
