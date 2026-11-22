"""Opt-in complete local OCR Shadow; reusable navigation and positional tail merge.

No upload orchestration is imported or executed. Existing production entries stay unchanged.
"""
from __future__ import annotations
from pathlib import Path
import json,time,traceback,os,uuid
from datetime import datetime
import cv2,numpy as np
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.library import Library
from utils.star_local_ocr_shadow import REPO,PACKAGED_RUNTIME,NativeStarOcr,Recognizer,write,save_image,hash_image,now_ms
from utils.star_experience_ocr import recognize_experience,empty_experience,experience_summary
from utils.star_bag_ocr import recognize_bag,empty_bag,bag_summary
from utils import logger
from utils.star_swipe_calibration import swipe_pipeline
from utils.star_frame_stability import NameStability, glyph_motion
from utils.star_backpack_navigation import prepare_star_backpack_entry,identify_star_entry,run_navigation_node
from utils.star_four_row_inventory import (
    BoundaryPositionError, PageStore, PositionInventory, classify_boundary, classify_scan_transition, confirmed_no_move,
    frame_stability, initial_forward_plan, layout, name_compensation_candidate,
    name_compensation_swipe, plan_forward, recovery_swipe, validate_fields, validate_swipe, verify_recovery,
)
__all__=['StarFullLocalOcrShadow']
SCAN_SWIPE_TIMING = {'swipe_duration_ms': 600, 'swipe_end_hold_ms': 600}
FORWARD_SETTLE_MAX_S = 8
SYNC_DIAGNOSTICS_RETENTION_VERSION = 1


class FullShadowSession:
    def __init__(self,context,params):
        if params.get('shadow') is not True:raise ValueError('explicit local Shadow mode required')
        run_id=params.get('run_id',datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
        if not isinstance(run_id,str) or not run_id.strip() or run_id in {'.','..'} or any(c in run_id for c in ('/','\\',':')):
            raise ValueError('run_id must be a single directory name')
        self.context=context;self.params={**params,'run_id':run_id}
        self.root=(REPO/params.get('output','debug/star-local-ocr')).resolve()
        if not self.root.is_relative_to(REPO):raise ValueError('output must remain inside the current MaaYuan directory')
        if not PACKAGED_RUNTIME.is_dir():raise ValueError('local star OCR resources required')
        self.out=(self.root/run_id).resolve()
        if self.out.parent!=self.root:raise ValueError('run output must remain directly inside the output root')
        self.root.mkdir(parents=True,exist_ok=True);self.out.mkdir(exist_ok=False)
        self.started=time.perf_counter();self.deadline=self.started+600;self.section='main';self.stage='start'
        self.captures=[];self.events=[];self.page_timings=[];self.navigation=[];self.nav_total_ms=0.;self.nav_serial=0
        self.inventories={s:PositionInventory(s) for s in ('main','support')};self.sections={};self.call_count=0;self.ocr_backend=None;self.recognizer=None;self.pending=None;self.last_capture=None;self.bag_ref=None
        self.recovery_checkpoints=set();self.position_recoveries=[];self.compensation_checkpoints=set()
        self.settling_geometry=None;self.settled_frame=None
        self.store=PageStore(self.out/'pages');self.outcome='running';self.experience=empty_experience('experience_not_started');self.bag=empty_bag('bag_not_started')
    def check(self):
        if self.context.tasker.stopping:raise InterruptedError('Shadow stopped')
        if time.perf_counter()>self.deadline:raise TimeoutError('full Shadow total time limit')
    def progress(self,stage):
        self.stage=stage;write(self.out/'progress.json',{'run':self.params['run_id'],'stage':stage,'section':self.section,'elapsed_ms':now_ms(self.started),'saved_pages':len(self.store.index),'counts':{s:len(v.items) for s,v in self.inventories.items()}})
    def capture(self,label):
        self.check();name=f'{len(self.captures):03d}-{label}.png';started=time.perf_counter();job=self.context.tasker.controller.post_screencap().wait()
        if not job.succeeded:raise RuntimeError('screenshot failed')
        im=np.ascontiguousarray(job.get()).copy()
        if im.shape!=(1280,720,3):raise ValueError('720x1280 BGR required')
        im.flags.writeable=False;ms=now_ms(started);started=time.perf_counter();save_image(self.out/name,im)
        rec={'file':name,'capture_ms':ms,'png_save_ms':now_ms(started),'elapsed_ms':now_ms(self.started),'pixel_sha256':hash_image(im)}
        self.captures.append(rec);self.last_capture=rec;return im
    def record_navigation(self,result):
        self.navigation.append(result);self.nav_total_ms+=result['navigation_ms'];write(self.out/'navigation.json',{'routes':self.navigation,'navigation_ms':self.nav_total_ms,'reverse_list_swipes':0})
    def prepare(self):
        self.nav_serial+=1
        return prepare_star_backpack_entry(self.context,lambda label:self.capture(f'nav{self.nav_serial}-{label}'),self.record_navigation)[0]
    def read_bag(self,image):
        self.progress('bag_ocr')
        for attempt in range(3):
            self.bag=recognize_bag(image,self.ocr)
            self.bag['file']=self.last_capture['file']
            if self.bag['complete']:break
            if attempt<2:
                self.check();time.sleep(.3);image=self.capture('bag-stable-retry')
                self.check_backpack(image,'main')
        write(self.out/'bag-results.json',self.bag)
        return image
    def navigation_acceptance(self):
        self.progress('unified_navigation');image=self.prepare();covered={x['start'] for x in self.navigation if x['success']}
        if 'backpack' not in covered:image=self.prepare()
        if 'observatory' not in covered:
            run_navigation_node(self.context,'返回上级界面');image=self.capture('provision-observatory')
            if identify_star_entry(self.context,image)!='observatory':raise RuntimeError('cannot provision supported observatory navigation case')
            image=self.prepare()
        if 'home' not in covered:
            run_navigation_node(self.context,'返回主界面');image=self.capture('provision-home')
            if identify_star_entry(self.context,image)!='home':raise RuntimeError('cannot provision supported home navigation case')
            image=self.prepare()
        if {x['start'] for x in self.navigation if x['success']}!={'home','backpack','observatory'}:raise RuntimeError('three supported navigation routes not verified')
        self.bag_ref=image;return image
    def check_backpack(self,image,section):
        # Identity of title and selected Tab only. Never compares list contents to a top image.
        if self.bag_ref is None:raise RuntimeError('no confirmed backpack identity reference')
        a=cv2.cvtColor(self.bag_ref[105:195,200:520],cv2.COLOR_BGR2GRAY);b=cv2.cvtColor(image[105:195,200:520],cv2.COLOR_BGR2GRAY)
        title=float(cv2.matchTemplate(a,b,cv2.TM_CCOEFF_NORMED)[0,0]) if np.ptp(a)>0 and np.ptp(b)>0 else 0.;gold=[float(np.mean(image[225:245,x:x+12,2])) for x in (90,290,480)];selected=int(np.argmax(gold));expected={'main':0,'support':1,'experience':2}[section]
        if title<.95 or selected!=expected or gold[selected]-sorted(gold)[-2]<=40:raise RuntimeError('backpack identity or selected section changed')
        return {'title_ncc':title,'selected_tab':selected}
    def ocr(self,image):
        self.check();started=time.perf_counter();stamp=time.perf_counter_ns();h,w=image.shape[:2];node='StarFullLocalOcrROI'
        detail=self.context.run_recognition(node,image,{node:{'recognition':{'type':'OCR','param':{'roi':[0,0,w,h],'only_rec':True,'threshold':0.,'expected':''}},'next':[]}})
        if detail is None and self.section!='experience':
            raise RuntimeError('Context ROI OCR failed')
        self.call_count+=1
        # A missing experience OCR result belongs to this variant. Keep the
        # evidence and let the existing per-count confirmation handle it.
        rows=detail.all_results if detail is not None else []
        result={'call_index':self.call_count,'ocr_ms':now_ms(started),'all_results':[{'text':z.text,'score':float(z.score),'box':([z.box.x,z.box.y,z.box.w,z.box.h] if hasattr(z.box,'x') else list(z.box))} for z in rows],'raw_detail':detail.raw_detail if detail is not None else None,
                'reco_id':detail.reco_id if detail is not None else None,'resource_hash':self.context.tasker.resource.hash,'input_shape':list(image.shape),'input_pixel_sha256':hash_image(image),'request_ns':stamp,'completion_ns':time.perf_counter_ns(),'call_path':'actual Context.run_recognition'}
        # Page checkpoints already keep the raw ROI outputs. Avoid a second full
        # serialization unless explicitly collecting experimental diagnostics.
        if self.params.get('diagnostics',False):
            with (self.out/'context-ocr.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        return result
    def recognize(self,image,sid):
        self.pending={'sourceImageId':sid,'section':self.section,'status':'stopped','records':[],'timing':{},'capture':self.last_capture}
        check=self.check_backpack(image,self.section);started=time.perf_counter();geometry=self.ocr_backend.geometry(image);gm=now_ms(started);self.pending['geometry']=geometry
        self.settling_geometry=geometry
        page=self.recognizer.image(image,sid,geometry=geometry,geometry_wall_ms=gm);page['section']=self.section;page['capture']=self.last_capture;page['identity']=check;self.pending=page
        for r in page['records']:r['section']=self.section;r['businessCandidate']['page_type']=self.section
        validate_fields(page);return page
    def save_page(self,page,cycle,gesture=None):
        page['timing']['before_save_cycle_ms']=now_ms(cycle);entry=self.store.save(page)
        stat={'image':page['sourceImageId'],'section':self.section,'status':page['status'],**page['timing'],'capture_ms':page['capture']['capture_ms'],'png_save_ms':page['capture']['png_save_ms'],'save_ms':entry['save_ms'],'complete_cycle_ms':now_ms(cycle),'cumulative_wall_ms':now_ms(self.started)}
        if gesture:stat.update(native_swipe_ms=gesture['native_swipe_ms'],wait_ms=gesture['wait_ms'])
        self.page_timings.append(stat);self.pending=None
    def stabilize(self,image,label,section):
        checks=[];consecutive=0;self.current_stability=checks
        for attempt in range(6):
            self.check();time.sleep(.2);next_image=self.capture(label+f'-stable-{attempt}')
            self.check_backpack(next_image,section);evidence=frame_stability(image,next_image);checks.append(evidence)
            consecutive=consecutive+1 if evidence['stable'] else 0
            image=next_image
            if consecutive>=2:
                self.current_stability=checks;return image
        raise RuntimeError('page did not settle within bounded screenshot checks')
    def stabilize_forward(self,image,label,section,released):
        """Return immediately on reliable stability, within FORWARD_SETTLE_MAX_S."""
        deadline=released+FORWARD_SETTLE_MAX_S;checks=[];consecutive=0;self.current_stability=checks
        self.current_forward_stability={'window_ms':int(FORWARD_SETTLE_MAX_S*1000),'period_ms':200,'complete':False}
        started=time.perf_counter()
        tracker=NameStability(getattr(self,'settling_geometry',None))
        latest=tracker.prepare(image,getattr(self,'settled_frame',None))
        self.current_forward_stability['name_prepare_ms']=now_ms(started)
        older=None;previous_motion=None
        attempt=0;target=released+.2
        while target<deadline:
            self.check()
            remaining=target-time.perf_counter()
            if remaining>0:time.sleep(remaining)
            if time.perf_counter()>=deadline:break
            requested=time.perf_counter();target=max(target+.2,requested+.2)
            next_image=self.capture(label+f'-stable-{attempt}');attempt+=1
            self.check_backpack(next_image,section)
            if time.perf_counter()>=deadline:break
            started=time.perf_counter();current=tracker.prepare(next_image,latest)
            evidence=latest.compare(current);pair=glyph_motion(latest,current)
            reliable=pair.get('reliable') and max(abs(pair[k]) for k in ('dx','dy'))<=2
            consecutive=consecutive+1 if evidence['stable'] and reliable else 0
            span=None;direct=None
            if consecutive>=2:
                # Independently verify the same three frames, including the
                # accumulated displacement that two small steps could hide.
                direct=glyph_motion(older,current)
                if direct.get('reliable'):
                    span=max(max(abs(direct[k]),abs(pair[k]+previous_motion[k])) for k in ('dx','dy'))
            checks.append({**evidence,'name_motion':pair,'direct_name_motion':direct,
                           'stable_span_px':span,'check_ms':now_ms(started),
                           'after_release_ms':now_ms(released)})
            older,latest=latest,current;previous_motion=pair
            if consecutive>=2 and span is not None and span<=2 and time.perf_counter()<deadline:
                self.current_forward_stability.update(complete=True,comparisons=len(checks),settle_ms=now_ms(released))
                self.settled_frame=current
                return next_image
        self.current_forward_stability.update(comparisons=len(checks),settle_ms=now_ms(released))
        raise RuntimeError(f'forward page did not settle within {FORWARD_SETTLE_MAX_S} seconds; moving frame rejected before OCR')

    def swipe_capture(self,summary,plan,label):
        self.check()
        if summary['swipes']>=30:raise RuntimeError('section maximum 30 swipes reached before bottom')
        if time.perf_counter()>summary['_deadline']:raise TimeoutError('section time limit')
        kind=plan['mode'];touch=plan['touch_length_px'];rows=plan['target_rows']
        begin=plan['swipe_begin'];end=plan['swipe_end']
        if kind=='recovery':
            if (rows not in (1,2) or begin[0]!=360 or end[0]!=360
                    or not 280<=begin[1]<end[1]<=1040 or end[1]-begin[1]!=touch):
                raise ValueError('recovery swipe outside the verified content viewport')
        else:
            if begin!=[360,1000] or end!=[360,1000-touch]:
                raise ValueError('forward swipe does not match its plan')
            if kind=='normal' and rows==4:
                validate_swipe(begin,end)
            elif not (kind=='bridge' and rows in (1,2) and 80<=touch<600
                      or kind=='compensation' and rows==0 and 300<=end[1]<1000):
                raise ValueError('invalid forward or compensation swipe plan')
        summary['swipes']+=1
        params={'swipe_begin':begin,'swipe_end':end,**SCAN_SWIPE_TIMING}
        with (self.out/f'{self.section}-swipe-{summary["swipes"]:03d}-claim.json').open('x',encoding='utf-8') as f:json.dump(params,f)
        started=time.perf_counter();d=self.context.run_task('StarSwipeEndHoldCalibrationNative',swipe_pipeline(params));released=time.perf_counter();ms=now_ms(started)
        if not d or not d.status.succeeded:raise RuntimeError('native swipe failed')
        actions=[n.action for n in d.nodes if n.action]
        if len(actions)!=1 or not actions[0].success:raise RuntimeError('one successful native Swipe required')
        ev={'section':self.section,'swipe':summary['swipes'],'purpose':label,'params':params,'native_swipe_ms':ms,'raw_actions':[a.raw_detail for a in actions]}
        ev.update(kind=kind,plan=plan,target_rows=rows,wait_ms=0.,settled=False)
        self.events.append(ev);self.current_stability=[];self.current_forward_stability={}
        try:
            image=self.capture(self.section+'-'+label);self.check_backpack(image,self.section)
            image=self.stabilize_forward(image,self.section+'-'+label,self.section,released)
            ev['forward_stability']=self.current_forward_stability
            ev.update(settled=True,gesture_start_to_stable_ms=ms+now_ms(released))
            return image,ev
        except BaseException as exc:
            ev['stability_error']=str(exc);ev['stability_checks']=self.current_stability
            ev['forward_stability']=self.current_forward_stability
            write(self.out/f'{self.section}-swipe-{summary["swipes"]:03d}-stability-error.json',ev)
            raise
        finally:
            ev['stability_checks']=self.current_stability;ev['post_gesture_wall_ms']=now_ms(released)

    def process_transition(self,summary,inventory,previous,after,page,plan,cycle,event,*,extra_touch=0):
        decision=classify_scan_transition(inventory.last_page,page,previous,after,plan,extra_touch=extra_touch)
        page['scan_transition']=decision;event['scan_state']=decision['state']
        if decision['state']=='TERMINAL_CANDIDATES':
            page['status']='terminal_candidates'
            self.save_page(page,cycle,event)
            return after,page,plan,decision
        if decision['state']=='RECOVERY_REQUIRED':
            page['status']='recovery_required'
            self.save_page(page,cycle,event)
            image,recovered,next_plan=self.recover_position(
                summary,inventory,page,previous,after,plan,extra_touch=extra_touch)
            return image,recovered,next_plan,decision

        if decision['basis']=='boundary':
            transition=inventory.merge_terminal(page,decision['row_offset'])
        else:
            transition=inventory.accept_moved(page,decision['row_offset'])
        page['status']='accepted' if transition['new_count']==16 and transition['duplicate_count']==0 else 'position_merged'
        page['acceptance']=transition

        if decision['state']=='MOVED':
            groups=layout(page['geometry'])['groups']
            if len(groups)>=3 and all(len(row)==4 for row in groups):
                plan=plan_forward(page['geometry'])
                page['next_swipe_plan']=plan
        if page['status']=='accepted':summary['normal_pages']+=1
        self.save_page(page,cycle,event)
        return after,page,plan,decision

    def _observe_terminal_probe(self,previous,page,before,after,plan,cycle,event):
        """Observe a pending terminal page without touching the accepted inventory."""
        if name_compensation_candidate(page) is not None:
            raise RuntimeError('incomplete name on confirmation frame; checkpoint preserved')
        decision=classify_scan_transition(previous,page,before,after,plan)
        page.update(status='terminal_probe',scan_transition=decision)
        event['scan_state']=decision['state']
        self.save_page(page,cycle,event)
        event['stationary_confirmation']=decision['state']=='NO_MOVE_CANDIDATE'
        return decision

    def _trusted_bag_count(self):
        count=self.bag.get('currentCount')
        capacity=self.bag.get('capacity')
        if (self.bag.get('complete') is True and not self.bag.get('warnings')
                and type(count) is int and type(capacity) is int and 0<=count<=capacity and capacity>0):
            return count
        return None

    def _select_terminal_offset(self,summary,inventory,page,offsets):
        other_section='main' if inventory.section=='support' else 'support'
        other_count=None
        if self.sections.get(other_section,{}).get('complete') is True:
            other_count=len(self.inventories[other_section].items)
        count=FullShadowSession._trusted_bag_count(self)
        candidates=[]
        summary['terminal_selection']={'bag_count':self.bag.get('currentCount'),
            'bag_count_trusted':count is not None,'other_section_count':other_count,'candidates':candidates}
        for offset in offsets:
            try:
                candidate=inventory.preview_terminal(page,offset)
            except ValueError as exc:
                candidates.append({'row_offset':offset,'error':str(exc)})
                continue
            if other_count is not None:
                candidate['expected_total']=other_count+candidate['section_count']
                if count is not None:
                    candidate['count_difference']=candidate['expected_total']-count
            candidates.append(candidate)
        if other_count is None:
            if not any('section_count' in candidate for candidate in candidates):
                raise RuntimeError('no viable terminal candidates; checkpoint preserved')
            inventory.pending_terminal=page
            return None
        if count is None:
            raise RuntimeError('ambiguous terminal page requires a confirmed bag count; checkpoint preserved')
        matching=[candidate for candidate in candidates if candidate.get('expected_total')==count]
        if len(matching)!=1:
            raise RuntimeError(f'{len(matching)} terminal candidates match the bag count; checkpoint preserved')
        return matching[0]['row_offset']

    @staticmethod
    def _commit_terminal(summary,inventory,page,offset):
        result=inventory.merge_terminal(page,offset)
        summary['terminal_selection']['selected_row_offset']=offset
        summary.update(complete=True,final_count=len(inventory.items),
                       raw_observations=inventory.observation_count,
                       tail_duplicates=sum(item['duplicate_count'] for item in inventory.tail_diagnostics),
                       tail_new=sum(item['new_count'] for item in inventory.tail_diagnostics))
        inventory.pending_terminal=None
        return result

    def resolve_pending_terminals(self):
        inventories=[self.inventories[section] for section in ('main','support')]
        if not any(inventory.pending_terminal is not None for inventory in inventories):
            return
        summaries=[self.sections[inventory.section] for inventory in inventories]
        options=[]
        for inventory,summary in zip(inventories,summaries):
            if inventory.pending_terminal is not None:
                options.append([entry for entry in summary['terminal_selection']['candidates']
                                if 'section_count' in entry])
            elif summary['complete']:
                options.append([{'section_count':len(inventory.items)}])
            else:
                raise RuntimeError('other section did not reach a confirmed bottom; checkpoint preserved')
        count=FullShadowSession._trusted_bag_count(self)
        if count is None:
            raise RuntimeError('ambiguous terminal pages require a confirmed bag count; checkpoints preserved')
        matches=[(main,support) for main in options[0] for support in options[1]
                 if main['section_count']+support['section_count']==count]
        if len(matches)!=1:
            raise RuntimeError(f'{len(matches)} terminal combinations match the bag count; checkpoints preserved')
        for inventory,summary,choice,other in zip(inventories,summaries,matches[0],reversed(matches[0])):
            if inventory.pending_terminal is not None:
                selection=summary['terminal_selection']
                selection['other_section_count']=other['section_count']
                for candidate in selection['candidates']:
                    if 'section_count' in candidate:
                        candidate['expected_total']=candidate['section_count']+other['section_count']
                        candidate['count_difference']=candidate['expected_total']-count
                FullShadowSession._commit_terminal(summary,inventory,inventory.pending_terminal,choice['row_offset'])

    def _confirm_terminal_candidates(self,summary,previous,first,plan,terminal_page,decision):
        inventory=self.inventories[self.section]
        offsets=decision['row_offsets']
        confirmation={'page':terminal_page['sourceImageId'],'row_offsets':offsets,'probes':0,'stationary':False}
        summary['terminal_confirmation']=confirmation
        periodic=decision.get('stationary_alias',False)
        # A sub-row swipe cannot move an entire period between identical
        # views, so verify zero movement independently of the original swipe.
        probe_plan=plan if not periodic else {
            'mode':'bridge','target_rows':1,'swipe_begin':[360,1000],
            'swipe_end':[360,920],'touch_length_px':80,
        }
        before,page=first,terminal_page
        stationary=[]
        for label in ('bottom-probe','bottom-confirmation'):
            confirmation['probes']+=1
            if label=='bottom-confirmation':
                summary['bottom_attempts']+=1
            cycle=time.perf_counter()
            after,event=self.swipe_capture(summary,probe_plan,label)
            observed=self.recognize(after,page['sourceImageId']+'-'+('confirmation' if label=='bottom-confirmation' else label))
            outcome=FullShadowSession._observe_terminal_probe(
                self,page,observed,before,after,probe_plan,cycle,event)
            if outcome['state']!='NO_MOVE_CANDIDATE':
                if periodic:
                    summary['bottom_resumes']+=1
                    return self.recover_position(summary,inventory,observed,previous,after,plan,
                                                 extra_touch=confirmation['probes']*probe_plan['touch_length_px'])
                raise RuntimeError('ambiguous terminal page moved during bottom probe; checkpoint preserved')
            stationary.append(outcome)
            before,page=after,observed

        endpoint=classify_boundary(terminal_page,page,first,after,probe_plan)
        confirmed=confirmed_no_move(*stationary,endpoint)
        confirmation.update(stationary=confirmed,endpoint=endpoint)
        event.update(endpoint_position=endpoint,stationary_confirmation=confirmed)
        if not confirmed:
            summary['stop_reason']='bottom_endpoint_position_unconfirmed'
            raise RuntimeError('bottom endpoint position unconfirmed; no repeated confirmation without new evidence')
        offset=FullShadowSession._select_terminal_offset(self,summary,inventory,terminal_page,offsets)
        event['terminal_selection']=summary['terminal_selection']
        if offset is not None:
            event['terminal_merge']=FullShadowSession._commit_terminal(summary,inventory,terminal_page,offset)
        summary.update(stop_reason='bottom_no_move_confirmed',bottom_confirmation_file=self.last_capture['file'])
        return after,page,plan

    def confirm_bottom(self,summary,previous,first,plan,origin_page,first_page,first_decision):
        """Probe once more and compare both swipes and their endpoints."""
        if first_decision['state']=='TERMINAL_CANDIDATES':
            return FullShadowSession._confirm_terminal_candidates(self,summary,previous,first,plan,first_page,first_decision)
        summary['bottom_attempts']+=1;cycle=time.perf_counter()
        after,event=self.swipe_capture(summary,plan,'bottom-confirmation')
        page=self.recognize(after,first_page['sourceImageId']+'-confirmation')
        if name_compensation_candidate(page) is not None:
            raise RuntimeError('incomplete name on confirmation frame; checkpoint preserved')
        image,page,next_plan,decision=FullShadowSession.process_transition(
            self,summary,self.inventories[self.section],first,after,page,plan,cycle,event)
        if decision['state']!='NO_MOVE_CANDIDATE':
            summary['bottom_resumes']+=1
            event['stationary_confirmation']=False
            if decision['state']=='TERMINAL_CANDIDATES':
                return FullShadowSession._confirm_terminal_candidates(self,summary,first,image,next_plan,page,decision)
            return image,page,next_plan
        endpoint=classify_boundary(origin_page,page,previous,after,plan)
        event['endpoint_position']=endpoint
        stationary=confirmed_no_move(first_decision,decision,endpoint)
        event['stationary_confirmation']=stationary
        if not stationary:
            summary['stop_reason']='bottom_endpoint_position_unconfirmed'
            raise RuntimeError('bottom endpoint position unconfirmed; no repeated confirmation without new evidence')
        summary.update(complete=True,stop_reason='bottom_no_move_confirmed',bottom_confirmation_file=self.last_capture['file'])
        return image,page,next_plan

    def recover_position(self,summary,inventory,page,before,after,forward_plan,*,extra_touch=0):
        """Use one planned reverse at this accepted frontier."""
        checkpoint=(self.section,inventory.next_row)
        if checkpoint in self.recovery_checkpoints:
            raise RuntimeError('position recovery already attempted at this checkpoint')
        self.recovery_checkpoints.add(checkpoint);started=time.perf_counter()
        summary['recovery_attempts']+=1
        forward_rows=forward_plan['target_rows']
        reverse_rows=min(2,inventory.last_base+forward_rows)
        event={'section':self.section,'checkpoint_row':inventory.next_row,'unresolved_image':page['sourceImageId'],
               'forward_rows':forward_rows,'attempt':1,'success':False}
        self.position_recoveries.append(event);self.events.append(event)
        try:
            self.progress('position_recovery')
            plan=recovery_swipe(page['geometry'],inventory.last_page['geometry'],target_rows=reverse_rows)
            event['plan']=plan
            recovered,gesture=self.swipe_capture(summary,plan,page['sourceImageId']+'-recovery')
            candidate=self.recognize(recovered,page['sourceImageId']+'-recovery')
            next_plan=plan_forward(candidate['geometry'])
            mapping=verify_recovery(inventory.last_page,page,candidate,before,after,recovered,
                                    forward_touch=forward_plan['touch_length_px']+extra_touch,
                                    reverse_touch=plan['touch_length_px'])
            event['verified_mapping']=mapping
            result=inventory.recover_page(page,candidate,recovered_offset=mapping['recovered_offset'],
                                          pending_offset=mapping['pending_offset'])
            candidate.update(status='recovery_merged',acceptance=result,next_swipe_plan=next_plan)
            self.save_page(candidate,started,gesture)
            event.update(success=True,result=result,resumed_image=candidate['sourceImageId'])
            return recovered,candidate,next_plan
        except BaseException as exc:
            event['error']=str(exc)
            if isinstance(exc,BoundaryPositionError):
                event['position_evidence']=exc.evidence
                raise
            raise RuntimeError('bounded position recovery failed: '+str(exc)) from exc
        finally:
            event['wall_ms']=now_ms(started);summary['recovery_ms']+=event['wall_ms']
            write(self.out/'position-recovery.json',self.position_recoveries)

    def compensate_name(self,summary,inventory,previous,page,forward_plan,cycle,event,height):
        checkpoint=(self.section,inventory.next_row)
        if checkpoint in self.compensation_checkpoints:
            raise RuntimeError('name compensation already attempted at this checkpoint')
        self.compensation_checkpoints.add(checkpoint)
        plan=name_compensation_swipe(height)
        page.update(status='name_compensation_required',name_compensation=plan)
        self.save_page(page,cycle,event)
        corrected,gesture=self.swipe_capture(summary,plan,page['sourceImageId']+'-name-compensation')
        final=self.recognize(corrected,page['sourceImageId']+'-name-compensation')
        final_layout=layout(final['geometry'])
        if not final_layout['four_rows_valid']:
            raise RuntimeError('name compensation did not produce four complete rows')
        shift=layout(page['geometry'])['row_centers'][0]-final_layout['row_centers'][0]
        if not 2<shift<=plan['touch_length_px']*1.2+10:
            raise RuntimeError('name compensation geometry inconsistent with a short nudge; checkpoint preserved')
        final['name_compensation']={'first_row_center_shift_px':shift}
        return FullShadowSession.process_transition(
            self,summary,inventory,previous,corrected,final,forward_plan,cycle,gesture,
            extra_touch=plan['touch_length_px'])

    def scan(self,section,initial):
        self.section=section;self.progress('scan_'+section);started=time.perf_counter();cycle=time.perf_counter();v=self.inventories[section]
        summary={'section':section,'complete':False,'swipes':0,'normal_pages':0,'stop_reason':None,'_deadline':time.perf_counter()+180,'recovery_attempts':0,'recovery_ms':0.,'bottom_attempts':0,'bottom_resumes':0};self.sections[section]=summary
        page=self.recognize(initial,section+'-000');accepted=v.accept_initial(page);page['status']='accepted';page['acceptance']=accepted;self.save_page(page,cycle);summary['normal_pages']=int(len(page['records'])==16)
        previous=initial;plan=initial_forward_plan(page['geometry']);page_number=0
        while summary['swipes']<30:
            cycle=time.perf_counter();page_number+=1;self.progress(section+f'_page_{page_number}')
            after,event=self.swipe_capture(summary,plan,f'page-{page_number:03d}')
            origin_page=v.last_page
            page=self.recognize(after,f'{section}-{page_number:03d}')
            obstruction=name_compensation_candidate(page)
            if obstruction is not None:
                image,page,plan,decision=self.compensate_name(summary,v,previous,page,plan,cycle,event,obstruction)
            else:
                image,page,plan,decision=FullShadowSession.process_transition(
                    self,summary,v,previous,after,page,plan,cycle,event)
            if decision['state'] in ('NO_MOVE_CANDIDATE','TERMINAL_CANDIDATES'):
                image,page,plan=self.confirm_bottom(summary,previous,image,plan,origin_page,page,decision)
                if summary['complete'] or v.pending_terminal is not None:break
            previous=image
        if not summary['complete'] and v.pending_terminal is None:
            raise RuntimeError('section limit reached without confirmed bottom')
        summary.update(final_count=len(v.items),raw_observations=v.observation_count,tail_duplicates=sum(x['duplicate_count'] for x in v.tail_diagnostics),tail_new=sum(x['new_count'] for x in v.tail_diagnostics),section_wall_ms=now_ms(started));summary.pop('_deadline',None)
        return summary
    def switch(self,section):
        self.progress('switch_'+section);node={'support':'星石背包-点击辅星','experience':'星石背包-点击经验星曜'}[section]
        run_navigation_node(self.context,node,post_delay=1500);image=self.capture('selected-'+section);self.check_backpack(image,section)
        return image if section=='experience' else self.stabilize(image,'selected-'+section,section)
    def read_experience(self):
        self.section = 'experience'
        previous_capture = self.last_capture
        try:
            image = self.switch('experience')
            self.progress('experience_ocr')
            self.experience = {**recognize_experience(image, self.ocr), 'file': self.last_capture['file']}
        except (RuntimeError, ValueError, cv2.error) as exc:
            self.experience = empty_experience('experience_scan_failed')
            self.experience['failure'] = {'stage': self.stage, 'error': str(exc)}
            if self.last_capture is not previous_capture:
                self.experience.update(captured=True, file=self.last_capture['file'])
        self.check()

    def execute(self):
        try:
            self.ocr_backend=NativeStarOcr(PACKAGED_RUNTIME, lambda: self.section)
            owner=self
            class RecoverableRecognizer(Recognizer):
                def field(self,image,rect,kind,sid,card):
                    first=owner.call_count+1;result=super().field(image,rect,kind,sid,card)
                    with (owner.out/'completed-fields.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'image':sid,'row':card['rowIndex'],'column':card['columnIndex'],'field':kind,'source_rect':rect,'resolved':result['resolved'],'confidence':result['confidence'],'call_indices':list(range(first,owner.call_count+1))},ensure_ascii=False)+'\n')
                    return result
            self.recognizer=RecoverableRecognizer(self.ocr,self.ocr_backend,self.out,save_debug_crops=False)
            if self.params.get('inspect_only'):
                image=self.capture('current');kind=identify_star_entry(self.context,image);write(self.out/'page-identity.json',{'page':kind})
                if kind=='unknown':raise RuntimeError('unsupported current page')
            else:
                self.progress('unified_navigation')
                main=self.navigation_acceptance() if self.params.get('verify_navigation_routes',False) else self.prepare()
                self.bag_ref=main;main=self.stabilize(main,'initial-main','main');main=self.read_bag(main);self.scan('main',main);support=self.switch('support');self.scan('support',support)
                FullShadowSession.resolve_pending_terminals(self)
                self.read_experience()
            total=sum(len(v.items) for v in self.inventories.values())
            if not self.params.get('inspect_only') and not self.bag['complete']:
                raise RuntimeError('game backpack count/capacity unconfirmed; complete scan cannot upload')
            if self.bag['complete'] and self.bag['currentCount']!=total:
                self.bag['warnings'].append('bag_count_inventory_mismatch')
                write(self.out/'bag-results.json',self.bag)
                raise RuntimeError('game backpack count differs from complete scanned instance count')
            self.outcome='completed';self.progress('completed');return True
        except BaseException as exc:
            self.outcome='failed_stop'
            failure={'stage':self.stage,'section':self.section,'error':str(exc),'traceback':traceback.format_exc()}
            if isinstance(exc,BoundaryPositionError):
                failure['position_evidence']=exc.evidence
            write(self.out/'error.json',failure)
            if self.pending and not any(e['image']==self.pending['sourceImageId'] for e in self.store.index):
                self.pending.update(status='stopped',error=str(exc))
                if isinstance(exc,BoundaryPositionError):
                    self.pending['position_evidence']=exc.evidence
                self.store.save(self.pending)
            self.progress('failed_stop');return False
        finally:
            for summary in self.sections.values():summary.pop('_deadline',None)
            full=self.outcome=='completed' and all(self.sections.get(s,{}).get('complete',False) for s in ('main','support')) and 'bag_count_inventory_mismatch' not in self.bag['warnings']
            inventory={'schemaVersion':2,'mode':'local-shadow','complete':full,'sections':{s:{'complete':self.sections.get(s,{}).get('complete',False),'count':len(v.items),'items':v.items} for s,v in self.inventories.items()},'experience':experience_summary(self.experience),'bag':bag_summary(self.bag)}
            write(self.out/'inventory.json',inventory)
            write(self.out/'experience-results.json',self.experience)
            write(self.out/'tail-merge.json',{s:v.tail_diagnostics for s,v in self.inventories.items()})
            for s,v in self.inventories.items():
                write(self.out/(s+'-results.json'),{**self.sections.get(s,{'section':s,'complete':False}), 'final_count':len(v.items),'raw_observations':v.observation_count,'tail_duplicates':sum(t['duplicate_count'] for t in v.tail_diagnostics),'tail_new':sum(t['new_count'] for t in v.tail_diagnostics)})
            write(self.out/'result.json',{'run_id':self.params['run_id'],
                'diagnostics_retention_version':SYNC_DIAGNOSTICS_RETENTION_VERSION,'outcome':self.outcome,'sections':self.sections,'pages':self.store.index,'page_timings':self.page_timings,'captures':self.captures,'events':self.events,'context_calls':self.call_count,'action_wall_ms':now_ms(self.started),'experience':experience_summary(self.experience)})
            write(self.out/'timing.json',{'page_timings':self.page_timings,'navigation_ms':self.nav_total_ms,'sections':self.sections,'action_wall_ms':now_ms(self.started),'context_calls':self.call_count})
            write(self.root/'latest.json',{'run_id':self.params['run_id'],'directory':self.out.name,'outcome':self.outcome,'complete':full})
            logger.info(f'星石本地识别 {self.outcome}，结果：{self.out / "inventory.json"}')


@AgentServer.custom_action('StarFullLocalOcrShadow')
class StarFullLocalOcrShadow(CustomAction):
    def run(self,context:Context,argv:CustomAction.RunArg):
        if not isinstance(context,Context) or not Library.is_agent_server():raise RuntimeError('actual Agent Context required')
        from custom.action.star_diagnostics import start_star_diagnostics
        from custom.action.toolbox_diagnostics import call
        params=json.loads(argv.custom_action_param)
        session=FullShadowSession(context,params)
        diagnostics=start_star_diagnostics(context,params)
        try:
            return session.execute()
        finally:
            call(diagnostics,'finish_star',session.out,
                 stopped=getattr(getattr(context,'tasker',None),'stopping',False) or getattr(context,'stop',False) is True)
