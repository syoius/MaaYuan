"""Explicit Shadow action only. Real Agent Context OCR; one guarded native swipe; no upload."""
from __future__ import annotations
import json,os,time,traceback,hashlib
from pathlib import Path
import cv2,numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.library import Library
from utils.star_local_ocr_shadow import REPO,Bridge,Recognizer,write,read_image,save_image,match_pair,hash_image,now_ms,motion_functions
from utils.star_swipe_calibration import parse_calibration_params,swipe_pipeline,SWIPE_NODE

__all__ = ['StarLocalOcrShadow']

@AgentServer.custom_action('StarLocalOcrShadow')
class StarLocalOcrShadow(CustomAction):
    def run(self,context:Context,argv:CustomAction.RunArg):
        config=json.loads(argv.custom_action_param);out=Path(config['output']).resolve()
        root=(REPO/'.ux/audits/star-local-ocr-live-20261031').resolve()
        if out.parent!=root or out.name not in {'live','live-recovered'}:raise ValueError('Shadow output must be this audit/live on D drive')
        if any(config.get(k) is not True for k in ['manual_start_authorized','manual_reset_confirmed','backend_touch_hold_confirmed']):raise ValueError('explicit start/reset/touch-hold confirmations required')
        if not isinstance(context,Context) or not Library.is_agent_server():raise RuntimeError('real Agent Context required')
        out.mkdir(exist_ok=False);start=time.perf_counter();images=[];bridge=None;stage='start';ocr_calls=[]
        shared_resource=context.tasker.resource
        resource_handle=str(shared_resource._handle);resource_hash=shared_resource.hash
        def status(value):write(out/'status.json',{'stage':value,'agent_pid':os.getpid(),'resource_handle':resource_handle,'elapsed_ms':now_ms(start)})
        def check():
            if context.tasker.stopping:raise InterruptedError('Shadow stopped')
        def capture(name):
            check();t=time.perf_counter_ns();job=context.tasker.controller.post_screencap().wait();completed=time.perf_counter_ns()
            if not job.succeeded:raise RuntimeError('screencap failed; cached image rejected')
            image=job.get()
            if not isinstance(image,np.ndarray) or image.shape!=(1280,720,3):raise ValueError('real 720x1280 BGR required')
            image=np.ascontiguousarray(image).copy();image.flags.writeable=False;save_image(out/name,image)
            return image,{'fileName':name,'request_monotonic_ns':t,'completion_monotonic_ns':completed,'exposure_time':None,'pixel_sha256':hash_image(image)}
        def ocr(image):
            check();t=time.perf_counter();stamp=time.perf_counter_ns();height,width=image.shape[:2]
            node='StarLocalOcrShadowROI';detail=context.run_recognition(node,image,{node:{'recognition':{'type':'OCR','param':{'roi':[0,0,width,height],'only_rec':True,'threshold':0.0,'expected':''}},'next':[]}})
            duration=now_ms(t)
            if detail is None:raise RuntimeError('Context.run_recognition returned None')
            tokens=[]
            for z in detail.all_results:
                box=z.box;tokens.append({'text':str(z.text),'score':float(z.score),'box':[box.x,box.y,box.w,box.h] if hasattr(box,'x') else list(box)})
            result={'all_results':tokens,'ocr_ms':duration,'call_path':'actual maa.context.Context.run_recognition','reco_id':detail.reco_id,'hit':detail.hit,'raw_detail':detail.raw_detail,'input_shape':list(image.shape),'input_pixel_sha256':hash_image(image),'request_monotonic_ns':stamp,'completion_monotonic_ns':time.perf_counter_ns(),'resource_handle':resource_handle,'resource_hash':shared_resource.hash,'agent_pid':os.getpid()}
            if result['resource_hash']!=resource_hash:raise RuntimeError('bound Resource hash changed')
            ocr_calls.append(result)
            with (out/'context-ocr.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(result,ensure_ascii=False)+'\n')
            return result
        try:
            bridge=Bridge(root/'_runtime',out/'bridge.log');recognizer=Recognizer(ocr,bridge,out)
            stage='capture_before';status(stage);before,before_stamp=capture('before.png')
            t=time.perf_counter();geometry=bridge.geometry(before);geometry_wall=now_ms(t)
            baseline=read_image(REPO/'.ux/audits/star-swipe-endhold-20261030/runs/20261031-141612-main-F-1-ab1242cf/before.png');probe=motion_functions()
            from utils.star_swipe_calibration import residual_motion
            position=residual_motion(baseline[280:1040,40:680],before[280:1040,40:680],probe);write(out/'top-check.json',position)
            if not position['diagnostic_stable'] or len([c for c in geometry['cards'] if c['completeness']=='complete'])!=16:raise RuntimeError('initial main top differs from confirmed reference')
            # First actual Context invocation, then one same-ROI warm call, on immutable captured NumPy pixels.
            first=next(c for c in geometry['cards'] if c['completeness']=='complete');r=first['nameRect'];x,y,w,h=[r[k] for k in ['x','y','width','height']];warm_image=cv2.resize(before[y:y+h,x:x+w],None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
            stage='context_warmup';status(stage);cold=ocr(warm_image);warm=ocr(warm_image)
            if not cold['all_results'] or not warm['all_results']:raise RuntimeError('Context OCR warmup produced no text; no swipe authorized to proceed')
            stage='recognize_before';status(stage);images.append(recognizer.image(before,'live-before'));write(out/'ocr-results.json',{'mode':'actual Agent Context','images':images})
            stage='single_native_swipe';status(stage)
            params=parse_calibration_params({'group':'F','trial':1,'section':'main','swipe_begin':[360,930],'swipe_end':[360,435],'manual_start_authorized':True,'manual_reset_confirmed':True,'backend_touch_hold_confirmed':True})
            pipeline=swipe_pipeline(params);write(out/'native-swipe-pipeline.json',pipeline)
            with (root/'gesture-once.json').open('x',encoding='utf-8') as flag:json.dump({'parameters':params,'request_monotonic_ns':time.perf_counter_ns(),'output':str(out)},flag)
            write(out/'gesture-once.json',{'root_guard':str(root/'gesture-once.json')})
            check();swipe_started=time.perf_counter_ns();detail=context.run_task(SWIPE_NODE,pipeline);returned=time.perf_counter_ns()
            if not detail or not detail.status.succeeded:raise RuntimeError('single native Swipe task failed')
            actions=[n.action for n in detail.nodes if n.action]
            if len(actions)!=1 or not actions[0].success:raise RuntimeError('expected exactly one successful native action')
            write(out/'swipe-result.json',{'native_actions':[a.raw_detail for a in actions],'request_monotonic_ns':swipe_started,'return_monotonic_ns':returned,'physical_release_time':None})
            stage='settle_2500ms';status(stage)
            while time.perf_counter_ns()-returned<2500_000_000:check();time.sleep(.02)
            after,after_stamp=capture('after.png')
            stage='recognize_after';status(stage);images.append(recognizer.image(after,'live-after'))
            stage='overlap';status(stage);overlap=match_pair(*images,before,after)
            write(out/'overlap-analysis.json',overlap);write(out/'ocr-results.json',{'mode':'actual Agent Context via AgentClient/AgentServer','agent_pid':os.getpid(),'actual_context_type':str(type(context)),'Library_is_agent_server':Library.is_agent_server(),'shared_resource_handle':resource_handle,'shared_resource_hash':resource_hash,'images':images})
            write(out/'timing.json',{'context_first_call_ms':cold['ocr_ms'],'context_warm_call_ms':warm['ocr_ms'],'initial_page_guard_geometry_wall_ms':geometry_wall,'images':[i['timing'] for i in images],'total_ocr_calls_including_warmup':len(ocr_calls),'ocr_calls_total_ms':sum(c['ocr_ms'] for c in ocr_calls),'two_images_ocr_ms':sum(i['timing']['name_ocr_ms']+i['timing']['level_ocr_ms'] for i in images),'two_images_recognition_wall_ms':sum(i['timing']['page_wall_ms'] for i in images),'native_swipe_wall_ms':(returned-swipe_started)/1e6,'actual_post_return_to_capture_request_ms':(after_stamp['request_monotonic_ns']-returned)/1e6,'capture_records':[before_stamp,after_stamp],'short_flow_wall_ms':now_ms(start),'resource_initializations_inside_action':0,'synchronous_no_background_ocr':True})
            status('completed');return True
        except BaseException as exc:
            write(out/'error.json',{'stage':stage,'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),'images_completed':len(images),'gesture_started':(out/'gesture-once.json').exists()});status('failed');return False
        finally:
            if bridge:bridge.close()
