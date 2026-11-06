"""One independent four-row native gesture and synchronous real Context OCR diagnostics."""
from pathlib import Path
import os,time,json,traceback
import cv2,numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.library import Library
from utils.star_local_ocr_shadow import REPO,Bridge,Recognizer,read_image,save_image,write,hash_image,now_ms,motion_functions
from utils.star_swipe_calibration import swipe_pipeline,residual_motion
from utils.star_four_row_shadow import PARAMS,SAMPLE_OFFSETS_MS,claim_once,stable_pair,audit_candidate,summarize_geometry,motion_estimate,adaptive_suggestion
__all__=['StarFourRowShadow']

@AgentServer.custom_action('StarFourRowShadow')
class StarFourRowShadow(CustomAction):
    def run(self,context:Context,argv:CustomAction.RunArg):
        config=json.loads(argv.custom_action_param);root=(REPO/'.ux/audits/star-four-row-shadow-20261031').resolve();out=Path(config['output']).resolve()
        if out!=root/'live':raise ValueError('fixed independent four-row output root required')
        if not all(config.get(k) is True for k in ['manual_start_authorized','manual_reset_confirmed','backend_touch_hold_confirmed']):raise ValueError('manual start/reset/backend confirmations required')
        if not isinstance(context,Context) or not Library.is_agent_server():raise RuntimeError('real Agent Context required')
        out.mkdir(exist_ok=False);start=time.perf_counter();stage='start';bridge=None;buffers={};captures=[];calls=[];images=[];geometries={};resource=context.tasker.resource;resource_hash=resource.hash
        def check():
            if context.tasker.stopping:raise InterruptedError('task stopped')
        def status(value):write(out/'status.json',{'stage':value,'elapsed_ms':now_ms(start),'agent_pid':os.getpid()})
        def capture(filename,origin=None,offset=None):
            check();req=time.perf_counter_ns();utc=time.time_ns();job=context.tasker.controller.post_screencap().wait()
            if not job.succeeded:raise RuntimeError('failed screenshot; no cached image accepted')
            im=job.get()
            if not isinstance(im,np.ndarray) or im.shape!=(1280,720,3):raise ValueError('720x1280 BGR frame required')
            im=np.ascontiguousarray(im).copy();im.flags.writeable=False;complete=time.perf_counter_ns()
            record={'fileName':filename,'request_monotonic_ns':req,'completion_monotonic_ns':complete,'request_utc_ns':utc,'frame_exposure_time':None,'capture_wall_ms':(complete-req)/1e6}
            if origin is not None:record.update(requested_offset_ms=offset,request_offset_ms=(req-origin)/1e6,completion_offset_ms=(complete-origin)/1e6,lateness_ms=max(0,(req-origin)/1e6-offset))
            buffers[filename]=im;captures.append(record);return im,record
        def geometry(filename):
            if filename not in geometries:
                t=time.perf_counter();g=bridge.geometry(buffers[filename]);wall=now_ms(t);geometries[filename]={'geometry':g,'wall_ms':wall,'summary':summarize_geometry(g)}
            return geometries[filename]
        def ocr(im):
            check();t=time.perf_counter();stamp=time.perf_counter_ns();h,w=im.shape[:2];node='StarFourRowShadowROI'
            detail=context.run_recognition(node,im,{node:{'recognition':{'type':'OCR','param':{'roi':[0,0,w,h],'only_rec':True,'threshold':0.0,'expected':''}},'next':[]}})
            duration=now_ms(t)
            if detail is None:raise RuntimeError('actual Context OCR failed to start')
            tokens=[]
            for z in detail.all_results:
                box=z.box;tokens.append({'text':z.text,'score':float(z.score),'box':[box.x,box.y,box.w,box.h] if hasattr(box,'x') else list(box)})
            result={'ocr_ms':duration,'all_results':tokens,'raw_detail':detail.raw_detail,'reco_id':detail.reco_id,'resource_hash':resource.hash,'agent_pid':os.getpid(),'input_pixel_sha256':hash_image(im),'input_shape':list(im.shape),'request_monotonic_ns':stamp,'completion_monotonic_ns':time.perf_counter_ns(),'call_path':'actual Context.run_recognition'}
            if result['resource_hash']!=resource_hash:raise RuntimeError('resource hash changed')
            calls.append(result)
            with (out/'context-ocr.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
            return result
        def wait_until(origin,offset):
            deadline=origin+offset*1_000_000
            while time.perf_counter_ns()<deadline:check();time.sleep(min(.02,(deadline-time.perf_counter_ns())/1e9))
        def recognize(filename,source_id):
            cache=geometry(filename)
            if not cache['summary']['four_complete_rows']:raise RuntimeError('four complete rows/columns unavailable; no complete-instance padding')
            result=recognizer.image(buffers[filename],source_id,geometry=cache['geometry'],geometry_wall_ms=cache['wall_ms']);images.append(result)
            write(root/'ocr-results.json',{'mode':'real Agent Context; ordered observations without cross-image dedup','images':images,'ordered_records':[c for im in images for c in im['records']],'deduplication_performed':False})
            if any(c['resolved'][k] is None for c in result['records'] for k in ['name','level','quality']):raise RuntimeError('unresolved fields; stop after preserving raw OCR')
            return result
        try:
            bridge=Bridge(root/'_runtime',out/'bridge.log');recognizer=Recognizer(ocr,bridge,out);probe=motion_functions()
            stage='before_capture';status(stage);before,_=capture('before.png');initial=geometry('before.png')
            old=read_image(REPO/'.ux/audits/star-swipe-endhold-20261030/runs/20261031-141612-main-F-1-ab1242cf/before.png');top=residual_motion(old[280:1040,40:680],before[280:1040,40:680],probe);write(out/'top-check.json',top)
            if not top['diagnostic_stable'] or not initial['summary']['four_complete_rows']:raise RuntimeError('wrong initial main/top/geometry')
            c=next(c for c in initial['geometry']['cards'] if c['completeness']=='complete');r=c['nameRect'];x,y,w,h=[r[k] for k in ['x','y','width','height']];warm_im=cv2.resize(before[y:y+h,x:x+w],None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
            stage='warmup';status(stage);cold=ocr(warm_im);warm=ocr(warm_im)
            if not cold['all_results'] or not warm['all_results']:raise RuntimeError('Context warmup empty; no gesture')
            stage='before_ocr';status(stage);recognize('before.png','four-row-A')
            stage='single_native_swipe';status(stage);pipeline=swipe_pipeline(PARAMS);write(out/'native-swipe-pipeline.json',pipeline);claim_once(root)
            check();swipe_request=time.perf_counter_ns();detail=context.run_task('StarSwipeEndHoldCalibrationNative',pipeline);returned=time.perf_counter_ns()
            if not detail or not detail.status.succeeded:raise RuntimeError('native Swipe failed')
            actions=[n.action for n in detail.nodes if n.action]
            if len(actions)!=1 or not actions[0].success:raise RuntimeError('one native successful Swipe expected')
            stage='sampling';candidate=None;previous=None;streak=0;adjacent=[];skipped=[]
            for offset in SAMPLE_OFFSETS_MS:
                wait_until(returned,offset);filename=f'after-{offset:04d}ms.png';im,record=capture(filename,returned,offset)
                if previous is not None:
                    t=time.perf_counter();motion=residual_motion(previous[0][280:1040,40:680],im[280:1040,40:680],probe);cost=now_ms(t);ok=stable_pair(motion);streak=streak+1 if ok else 0
                    adjacent.append({'from':previous[1],'to':filename,'motion':motion,'candidate_condition':ok,'consecutive_passes':streak,'analysis_ms':cost})
                previous=(im,filename)
                if streak>=2:
                    cache=geometry(filename)
                    if not cache['summary']['four_complete_rows']:raise RuntimeError('candidate four-column geometry failed')
                    candidate=filename;candidate_record=record;break
            found_by_rule=candidate is not None
            early=found_by_rule and candidate_record['requested_offset_ms']<2500
            if candidate is None:
                candidate='after-2500ms.png';candidate_record=next(z for z in captures if z['fileName']==candidate)
            stage='candidate_ocr';status(stage);ocr_start=time.perf_counter_ns();page_b=recognize(candidate,'four-row-B');ocr_end=time.perf_counter_ns()
            # Synchronous OCR may delay the audit; never label a late capture as exactly2500ms.
            if candidate!='after-2500ms.png':
                wait_until(returned,2500);late,late_record=capture('audit-2500ms.png',returned,2500)
                skipped=[v for v in SAMPLE_OFFSETS_MS if candidate_record['requested_offset_ms']<v<2500]
                audit_filename='audit-2500ms.png'
            else:late=buffers[candidate];late_record=candidate_record;audit_filename=candidate
            stage='offline_checks';status(stage);t=time.perf_counter();late_motion=residual_motion(buffers[candidate][280:1040,40:680],late[280:1040,40:680],probe);late_analysis_ms=now_ms(t)
            audit_status=audit_candidate(late_motion) if early else 'not_an_early_candidate'
            audit_late=late_record['request_offset_ms']>2600
            late_geometry=geometry(audit_filename)
            measured=motion_estimate(before,late,initial['geometry'],late_geometry['geometry'],probe)
            suggestion=adaptive_suggestion(page_b['geometry'],measured['conditional_content_shift_px'])
            geometry_pass=late_geometry['summary']['four_complete_rows']
            stability={'sample_schedule_ms':list(SAMPLE_OFFSETS_MS),'samples':captures[1:],'adjacent_samples':adjacent,'candidate':candidate,'candidate_record':candidate_record,'candidate_was_early':early,'earliest_candidate_completion_ms':candidate_record['completion_offset_ms'] if found_by_rule else None,'candidate_rule_confirmed':found_by_rule,'criterion':'diagnostic only: two consecutive adjacent measured shifts<=2px, aligned NCC>=.97, same-position NCC>=.85; then four complete rows required; scroll ROI40,280,640,760','late_audit_file':audit_filename,'late_audit_record':late_record,'candidate_to_late_motion':late_motion,'candidate_to_late_analysis_ms':late_analysis_ms,'early_candidate_audit_status':audit_status,'late_audit_more_than_100ms_late':audit_late,'skipped_intermediate_offsets_during_synchronous_OCR':skipped,'early_stability_verified_at_observed_audit':early and audit_status=='passed' and not audit_late and geometry_pass,'physical_release_time':None,'exposure_times_known':False,'stop_after_one_gesture':True}
            write(root/'stability-analysis.json',stability);write(root/'geometry-analysis.json',{'frames':geometries,'motion':measured,'adaptive':suggestion,'old_fifth_row_expected_y':initial['geometry']['rowCenters'][0]+4*initial['summary']['median_pitch_px'],'observed_new_first_row_y':late_geometry['geometry']['rowCenters'][0],'first_row_goal_y':390,'no_cross_image_instance_matching':True})
            page_sum=sum(im['timing']['page_wall_ms'] for im in images);finish=time.perf_counter_ns()
            write(root/'timing.json',{'resource_context_first_call_ms':cold['ocr_ms'],'context_warm_call_ms':warm['ocr_ms'],'images':[i['timing'] for i in images],'context_calls_including_warmup':len(calls),'swipe_request_ns':swipe_request,'task_return_ns':returned,'native_swipe_wall_ms':(returned-swipe_request)/1e6,'nominal_end_hold_ms':250,'sample_count':len(captures)-1,'captures':captures,'candidate_OCR_start_after_return_ms':(ocr_start-returned)/1e6,'candidate_OCR_end_after_return_ms':(ocr_end-returned)/1e6,'post_return_total_observation_and_OCR_ms':(finish-returned)/1e6,'action_total_wall_ms':now_ms(start),'old_fixed_2500_reference_budget_after_return_ms':2500+page_b['timing']['page_wall_ms'],'reference_budget_note':'same-run page processing budget, not an observed old-mode A/B speedup','late_audit_and_analysis_included':True,'additional_fallback_OCR_calls':0,'synchronous_no_extra_OCR_resource':True})
            write(out/'swipe-result.json',{'native_actions':[a.raw_detail for a in actions],'request_ns':swipe_request,'return_ns':returned,'physical_release_time':None})
            write(out/'capture-manifest.json',{'captures':captures,'OCR_frame_A':'before.png','OCR_frame_B':candidate,'sample_count':len(captures)-1,'physical_release_time':None})
            self_state='completed'
            if measured['obvious_anomaly'] or not geometry_pass or (early and audit_status!='passed'):self_state='completed_with_diagnostic_failure_stop'
            if candidate_record['requested_offset_ms']==2500 and not early:self_state='completed_conservative_frame_stability_unconfirmed'
            status(self_state);return True
        except BaseException as exc:
            write(out/'error.json',{'stage':stage,'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),'gesture_claimed':(root/'gesture-once.json').exists()});write(out/'capture-manifest.json',{'captures':captures});status('failed_stop');return False
        finally:
            # Keep all actual frames, including B0 and partial evidence, on any exit.
            for filename,im in buffers.items():save_image(out/filename,im)
            if bridge:bridge.close()
