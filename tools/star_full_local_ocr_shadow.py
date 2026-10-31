"""Dedicated opt-in host for unified navigation and whole-inventory local OCR Shadow."""
from pathlib import Path
import sys,os,json,time,subprocess,importlib.util,argparse
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'agent'))
from utils.star_local_ocr_shadow import write,now_ms

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['launch','host','agent']);p.add_argument('--output',type=Path,required=True);p.add_argument('--identifier');a=p.parse_args()
    root=a.output.resolve()
    if not root.is_relative_to(REPO) or root.drive.upper()!='D:':raise ValueError('D-drive project audit root required')
    (root/'temp').mkdir(exist_ok=True,parents=True);os.environ.update(TEMP=str(root/'temp'),TMP=str(root/'temp'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    if a.mode=='agent':
        from maa.agent.agent_server import AgentServer
        from maa.library import Library
        spec=importlib.util.spec_from_file_location('star_full_local_ocr_shadow_action',REPO/'agent/custom/action/star_full_local_ocr_shadow.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        write(root/'agent-ready.json',{'pid':os.getpid(),'actual_agent_server':Library.is_agent_server()});assert AgentServer.start_up(a.identifier);AgentServer.join();AgentServer.shut_down();return
    if a.mode=='launch':
        log=(root/'host-console.log').open('x',encoding='utf-8');process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'host','--output',str(root)],cwd=REPO,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));write(root/'launch.json',{'pid':process.pid});return
    with (root/'host-once.json').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'time_ns':time.time_ns()},f)
    from maa.resource import Resource
    from maa.controller import AdbController
    from maa.toolkit import Toolkit
    from maa.tasker import Tasker
    from maa.agent_client import AgentClient
    from maa.library import Library
    started=time.perf_counter();Toolkit.init_option(root/'host-native',{'logging':True,'save_on_error':False});resource=Resource();assert resource.use_cpu();assert resource.post_bundle(REPO/'assets/resource/base').wait().succeeded
    cfg=json.loads((REPO/'.ux/audits/star-swipe-endhold-20261030/controller-config.json').read_text(encoding='utf-8'))
    controller=AdbController('D:/Program Files/Netease/MuMu Player 12/nx_main/adb.exe','127.0.0.1:16384',screencap_methods=64,input_methods=8,config=cfg);assert controller.post_connection().wait().succeeded;assert controller.set_screenshot_target_long_side(1280)
    tasker=Tasker();assert tasker.bind(resource,controller) and tasker.inited;client=AgentClient.create_tcp(0);assert client.bind(resource);assert client.set_timeout(660000)
    log=(root/'agent-console.log').open('x',encoding='utf-8');process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'agent','--output',str(root),'--identifier',client.identifier],cwd=REPO,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        assert client.connect();assert 'StarFullLocalOcrShadow' in client.custom_action_list
        write(root/'host-ready.json',{'host_pid':os.getpid(),'agent_pid':process.pid,'native_version':Library.version(),'owned_resources':1,'resource_hash':resource.hash,'input_mask':8,'screencap_mask':64,'controller_uuid':controller.uuid})
        deadline=time.perf_counter()+1200
        for index in range(3):
            path=root/f'command-{index:02d}.json'
            while not path.exists() and time.perf_counter()<deadline:time.sleep(.1)
            if not path.exists():break
            command=json.loads(path.read_text(encoding='utf-8'))
            if command.get('stop'):break
            params={'shadow':True,'output':str(root),'geometry_runtime':str(root/'_runtime'),**command}
            pipeline={'StarFullLocalShadowEntry':{'recognition':'DirectHit','action':{'type':'Custom','param':{'custom_action':'StarFullLocalOcrShadow','custom_action_param':params}},'next':[],'on_error':[],'pre_delay':0,'post_delay':0,'timeout':650000}}
            pipeline_file=root/f'command-{index:02d}-pipeline.json';write(pipeline_file,pipeline);assert resource.post_pipeline(pipeline_file).wait().succeeded
            t=time.perf_counter();job=tasker.post_task('StarFullLocalShadowEntry').wait();write(root/f'command-{index:02d}-done.json',{'succeeded':job.succeeded,'task_wall_ms':now_ms(t),'host_elapsed_ms':now_ms(started),'command':command})
            if not job.succeeded or not command.get('inspect_only'):break
    finally:
        client.disconnect()
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
        log.close();write(root/'host-stopped.json',{'time_ns':time.time_ns()})

if __name__=='__main__':main()
