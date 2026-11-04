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
from utils.star_local_ocr_shadow import REPO,PACKAGED_RUNTIME,Bridge,Recognizer,write,save_image,hash_image,now_ms
from utils.star_experience_ocr import recognize_experience,empty_experience,experience_summary
from utils.star_bag_ocr import recognize_bag,empty_bag,bag_summary
from utils import logger
from utils.star_swipe_calibration import swipe_pipeline
from utils.star_backpack_navigation import prepare_star_backpack_entry,identify_star_entry,run_navigation_node
from utils.star_four_row_inventory import PositionInventory,PageStore,layout,next_swipe,validate_swipe,same_position,terminal_reason,validate_fields,NO_MOVE_SIMILARITY,confirmed_no_move,frame_stability,PositionUncertain,recovery_swipe
__all__=['StarFullLocalOcrShadow']


class FullShadowSession:
    def __init__(self,context,params):
        if params.get('shadow') is not True:raise ValueError('explicit local Shadow mode required')
        run_id=params.get('run_id',datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
        if not isinstance(run_id,str) or not run_id.strip() or run_id in {'.','..'} or any(c in run_id for c in ('/','\\',':')):
            raise ValueError('run_id must be a single directory name')
        self.context=context;self.params={**params,'run_id':run_id}
        self.root=(REPO/params.get('output','debug/star-local-ocr')).resolve();self.runtime=(REPO/params.get('geometry_runtime',PACKAGED_RUNTIME)).resolve()
        if not self.root.is_relative_to(REPO):raise ValueError('output must remain inside the current MaaYuan directory')
        if not self.runtime.is_relative_to(REPO) or not self.runtime.is_dir():raise ValueError('local geometry runtime required')
        self.out=(self.root/run_id).resolve()
        if self.out.parent!=self.root:raise ValueError('run output must remain directly inside the output root')
        self.root.mkdir(parents=True,exist_ok=True);self.out.mkdir(exist_ok=False)
        self.started=time.perf_counter();self.deadline=self.started+600;self.section='main';self.stage='start'
        self.captures=[];self.events=[];self.page_timings=[];self.navigation=[];self.nav_total_ms=0.;self.nav_serial=0
        self.inventories={s:PositionInventory(s) for s in ('main','support')};self.sections={};self.call_count=0;self.bridge=None;self.recognizer=None;self.pending=None;self.last_capture=None;self.bag_ref=None
        self.recovery_checkpoints=set();self.position_recoveries=[]
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
        title=float(cv2.matchTemplate(a,b,cv2.TM_CCOEFF_NORMED)[0,0]);gold=[float(np.mean(image[225:245,x:x+12,2])) for x in (90,290,480)];selected=int(np.argmax(gold));expected={'main':0,'support':1,'experience':2}[section]
        if title<.95 or selected!=expected or gold[selected]-sorted(gold)[-2]<=40:raise RuntimeError('backpack identity or selected section changed')
        return {'title_ncc':title,'selected_tab':selected}
    def ocr(self,image):
        self.check();started=time.perf_counter();stamp=time.perf_counter_ns();h,w=image.shape[:2];node='StarFullLocalOcrROI'
        detail=self.context.run_recognition(node,image,{node:{'recognition':{'type':'OCR','param':{'roi':[0,0,w,h],'only_rec':True,'threshold':0.,'expected':''}},'next':[]}})
        if detail is None:raise RuntimeError('Context ROI OCR failed')
        self.call_count+=1
        result={'call_index':self.call_count,'ocr_ms':now_ms(started),'all_results':[{'text':z.text,'score':float(z.score),'box':([z.box.x,z.box.y,z.box.w,z.box.h] if hasattr(z.box,'x') else list(z.box))} for z in detail.all_results],'raw_detail':detail.raw_detail,'reco_id':detail.reco_id,'resource_hash':self.context.tasker.resource.hash,'input_shape':list(image.shape),'input_pixel_sha256':hash_image(image),'request_ns':stamp,'completion_ns':time.perf_counter_ns(),'call_path':'actual Context.run_recognition'}
        # Page checkpoints already keep the raw ROI outputs. Avoid a second full
        # serialization unless explicitly collecting experimental diagnostics.
        if self.params.get('diagnostics',False):
            with (self.out/'context-ocr.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        return result
    def recognize(self,image,sid):
        self.pending={'sourceImageId':sid,'section':self.section,'status':'stopped','records':[],'timing':{},'capture':self.last_capture}
        check=self.check_backpack(image,self.section);started=time.perf_counter();geometry=self.bridge.geometry(image);gm=now_ms(started);self.pending['geometry']=geometry
        page=self.recognizer.image(image,sid,geometry=geometry,geometry_wall_ms=gm);page['section']=self.section;page['capture']=self.last_capture;page['identity']=check;self.pending=page
        for r in page['records']:r['section']=self.section;r['businessCandidate']['page_type']=self.section
        validate_fields(page);return page
    def save_page(self,page,cycle,gesture=None):
        page['timing']['before_save_cycle_ms']=now_ms(cycle);entry=self.store.save(page)
        stat={'image':page['sourceImageId'],'section':self.section,'status':page['status'],**page['timing'],'capture_ms':page['capture']['capture_ms'],'png_save_ms':page['capture']['png_save_ms'],'save_ms':entry['save_ms'],'complete_cycle_ms':now_ms(cycle),'cumulative_wall_ms':now_ms(self.started)}
        if gesture:stat.update(native_swipe_ms=gesture['native_swipe_ms'],wait_ms=gesture['wait_ms'])
        self.page_timings.append(stat);self.pending=None
    def stabilize(self,image,label,section):
        checks=[];consecutive=0
        for attempt in range(6):
            self.check();time.sleep(.2);next_image=self.capture(label+f'-stable-{attempt}')
            self.check_backpack(next_image,section);evidence=frame_stability(image,next_image);checks.append(evidence)
            consecutive=consecutive+1 if evidence['stable'] else 0
            image=next_image
            if consecutive>=2:
                self.current_stability=checks;return image
        raise RuntimeError('page did not settle within bounded screenshot checks')
    def swipe_capture(self,summary,touch,label,*,recovery_plan=None):
        self.check()
        if summary['swipes']>=30:raise RuntimeError('section maximum 30 swipes reached before bottom')
        if time.perf_counter()>summary['_deadline']:raise TimeoutError('section time limit')
        if recovery_plan is None:
            begin=[360,1000];end=[360,1000-touch];validate_swipe(begin,end)
        else:
            begin=recovery_plan['swipe_begin'];end=recovery_plan['swipe_end']
            if begin[0]!=360 or end[0]!=360 or not 280<=begin[1]<end[1]<=1040 or end[1]-begin[1]!=touch:
                raise ValueError('recovery swipe outside the verified content viewport')
        summary['swipes']+=1
        params={'swipe_begin':begin,'swipe_end':end,'swipe_duration_ms':700,'swipe_end_hold_ms':250}
        with (self.out/f'{self.section}-swipe-{summary["swipes"]:03d}-claim.json').open('x',encoding='utf-8') as f:json.dump(params,f)
        started=time.perf_counter();d=self.context.run_task('StarSwipeEndHoldCalibrationNative',swipe_pipeline(params));ms=now_ms(started)
        if not d or not d.status.succeeded:raise RuntimeError('native swipe failed')
        actions=[n.action for n in d.nodes if n.action]
        if len(actions)!=1 or not actions[0].success:raise RuntimeError('one successful native Swipe required')
        ev={'section':self.section,'swipe':summary['swipes'],'purpose':label,'params':params,'native_swipe_ms':ms,'raw_actions':[a.raw_detail for a in actions]}
        if recovery_plan is not None:ev['recovery_plan']=recovery_plan
        started=time.perf_counter();time.sleep(1.0);ev['wait_ms']=now_ms(started)
        image=self.capture(self.section+'-'+label);self.check_backpack(image,self.section);image=self.stabilize(image,self.section+'-'+label,self.section);ev['stability_checks']=self.current_stability;self.events.append(ev);return image,ev
    def confirm_bottom(self,summary,previous,touch,first_candidate=None):
        after,ev=self.swipe_capture(summary,touch,'bottom-confirmation')
        stationary=confirmed_no_move(previous,first_candidate if first_candidate is not None else previous,after)
        ev['stationary_confirmation']=bool(stationary)
        if not stationary:raise RuntimeError('bottom confirmation moved; retained data needs review')
        summary.update(complete=True,stop_reason='bottom_no_move_confirmed',bottom_confirmation_file=self.last_capture['file'])
    def recover_position(self,summary,inventory,previous,page,image,touch,error):
        """One measured reverse gesture per unresolved accepted frontier."""
        checkpoint=(self.section,inventory.next_row)
        if checkpoint in self.recovery_checkpoints:raise RuntimeError('position recovery already attempted at this checkpoint; no new evidence')
        self.recovery_checkpoints.add(checkpoint);started=time.perf_counter()
        summary['recovery_attempts']+=1
        event={'section':self.section,'checkpoint_row':inventory.next_row,'unresolved_image':page['sourceImageId'],
               'attempt':1,'success':False,'initial_diagnostics':error.diagnostics}
        self.position_recoveries.append(event);self.events.append(event)
        try:
            self.progress('position_recovery')
            plan=recovery_swipe(page['geometry'],inventory.last_page['geometry']);event['plan']=plan
            recovered,gesture=self.swipe_capture(summary,plan['touch_length_px'],page['sourceImageId']+'-recovery',recovery_plan=plan)
            candidate=self.recognize(recovered,page['sourceImageId']+'-recovery')
            result=inventory.recover_page(page,candidate,previous,image,recovered,plan['touch_length_px'])
            event['positions_verified']=True
            candidate.update(status='recovery_merged',acceptance=result)
            candidate['timing']['recovery_mapping_ms']=result['mapping_ms']
            candidate['adaptive']=next_swipe(candidate['geometry'],touch)
            self.save_page(candidate,started,gesture)
            event.update(success=True,result=result,resumed_image=candidate['sourceImageId'])
            return recovered,candidate
        except BaseException as exc:
            event['error']=str(exc)
            raise RuntimeError('bounded two-row position recovery failed: '+str(exc)) from exc
        finally:
            event['wall_ms']=now_ms(started);summary['recovery_ms']+=event['wall_ms']
            write(self.out/'position-recovery.json',self.position_recoveries)

    def scan(self,section,initial):
        self.section=section;self.progress('scan_'+section);started=time.perf_counter();cycle=time.perf_counter();v=self.inventories[section]
        summary={'section':section,'complete':False,'swipes':0,'normal_pages':0,'stop_reason':None,'_deadline':time.perf_counter()+180,'position_checks':[],'recovery_attempts':0,'recovery_ms':0.};self.sections[section]=summary
        page=self.recognize(initial,section+'-000');accepted=v.accept_initial(page);page['status']='accepted';page['acceptance']=accepted;self.save_page(page,cycle);summary['normal_pages']=int(len(page['records'])==16)
        previous=initial;touch=630;page_number=0
        while summary['swipes']<30:
            cycle=time.perf_counter();page_number+=1;self.progress(section+f'_page_{page_number}')
            after,ev=self.swipe_capture(summary,touch,f'page-{page_number:03d}');correlation=same_position(previous,after)
            if correlation>=NO_MOVE_SIMILARITY:
                ev['no_move_candidate']=True;self.confirm_bottom(summary,previous,touch,after);break
            page=self.recognize(after,f'{section}-{page_number:03d}')
            page['geometry_hint']=terminal_reason(previous,after,page['geometry'])
            try:
                transition=v.accept_moved(page,previous,after,touch)
            except PositionUncertain as error:
                summary['position_checks'].append({'image':page['sourceImageId'],'status':'unresolved',**error.diagnostics})
                page.update(status='position_unconfirmed',position_diagnostics=error.diagnostics)
                self.save_page(page,cycle,ev)
                previous,recovered=self.recover_position(summary,v,previous,page,after,touch,error)
                touch=recovered['adaptive']['touch_length_px']
                continue
            mapping=transition['position_mapping']
            page['timing']['position_mapping_ms']=mapping['mapping_ms']
            summary['position_checks'].append({'image':page['sourceImageId'],'status':'confirmed','row_offset':mapping['row_offset'],
                'mapping_ms':mapping['mapping_ms'],'rejected_candidates':mapping['rejected_candidates']})
            page['status']='accepted' if transition['new_count']==16 and transition['duplicate_count']==0 else 'position_merged'
            page['acceptance']=transition
            if layout(page['geometry'])['four_rows_valid']:
                adaptive=next_swipe(page['geometry'],touch);page['adaptive']=adaptive;page['timing']['adaptive_ms']=adaptive['calculation_ms'];touch=adaptive['touch_length_px']
            if page['status']=='accepted':summary['normal_pages']+=1
            self.save_page(page,cycle,ev);previous=after
            # A moved page, full or partial, is not proof of bottom. The next
            # swipe either advances again or enters confirmed_no_move above.
        if not summary['complete']:raise RuntimeError('section limit reached without confirmed bottom')
        summary.update(final_count=len(v.items),raw_observations=v.observation_count,tail_duplicates=sum(x['duplicate_count'] for x in v.tail_diagnostics),tail_new=sum(x['new_count'] for x in v.tail_diagnostics),section_wall_ms=now_ms(started),
            normal_mapping_mean_ms=float(np.mean([x['mapping_ms'] for x in summary['position_checks'] if x['status']=='confirmed'])) if any(x['status']=='confirmed' for x in summary['position_checks']) else None);summary.pop('_deadline',None)
        return summary
    def switch(self,section):
        self.progress('switch_'+section);node={'support':'星石背包-点击辅星','experience':'星石背包-点击经验星曜'}[section]
        run_navigation_node(self.context,node,post_delay=1500);image=self.capture('selected-'+section);self.check_backpack(image,section);return self.stabilize(image,'selected-'+section,section)
    def execute(self):
        try:
            self.bridge=Bridge(self.runtime,self.out/'bridge.log');resolve=self.bridge.resolve;self.bridge.resolve=lambda kind,variants,unused='main':resolve(kind,variants,self.section)
            owner=self
            class RecoverableRecognizer(Recognizer):
                def field(self,image,rect,kind,sid,card):
                    first=owner.call_count+1;result=super().field(image,rect,kind,sid,card)
                    with (owner.out/'completed-fields.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'image':sid,'row':card['rowIndex'],'column':card['columnIndex'],'field':kind,'source_rect':rect,'resolved':result['resolved'],'confidence':result['confidence'],'call_indices':list(range(first,owner.call_count+1))},ensure_ascii=False)+'\n')
                    return result
            self.recognizer=RecoverableRecognizer(self.ocr,self.bridge,self.out,save_debug_crops=False)
            if self.params.get('inspect_only'):
                image=self.capture('current');kind=identify_star_entry(self.context,image);write(self.out/'page-identity.json',{'page':kind})
                if kind=='unknown':raise RuntimeError('unsupported current page')
            else:
                self.progress('unified_navigation')
                main=self.navigation_acceptance() if self.params.get('verify_navigation_routes',False) else self.prepare()
                self.bag_ref=main;main=self.stabilize(main,'initial-main','main');main=self.read_bag(main);self.scan('main',main);support=self.switch('support');self.scan('support',support)
                experience=self.switch('experience');self.section='experience';self.progress('experience_ocr')
                self.experience.update(captured=True,file=self.last_capture['file'])
                self.experience={**recognize_experience(experience,self.ocr,self.bridge),'file':self.last_capture['file']}
                if not self.experience['complete']:raise RuntimeError('experience OCR incomplete; main/support inventory preserved')
            total=sum(len(v.items) for v in self.inventories.values())
            if not self.params.get('inspect_only') and not self.bag['complete']:
                raise RuntimeError('game backpack count/capacity unconfirmed; complete scan cannot upload')
            if self.bag['complete'] and self.bag['currentCount']!=total:
                self.bag['warnings'].append('bag_count_inventory_mismatch')
                write(self.out/'bag-results.json',self.bag)
                raise RuntimeError('game backpack count differs from complete scanned instance count')
            self.outcome='completed';self.progress('completed');return True
        except BaseException as exc:
            self.outcome='failed_stop';write(self.out/'error.json',{'stage':self.stage,'section':self.section,'error':str(exc),'traceback':traceback.format_exc()})
            if self.pending and not any(e['image']==self.pending['sourceImageId'] for e in self.store.index):
                self.pending['status']='stopped';self.pending['error']=str(exc);self.store.save(self.pending)
            self.progress('failed_stop');return False
        finally:
            for summary in self.sections.values():summary.pop('_deadline',None)
            full=self.outcome=='completed' and all(self.sections.get(s,{}).get('complete',False) for s in ('main','support')) and self.experience['complete'] and 'bag_count_inventory_mismatch' not in self.bag['warnings']
            inventory={'schemaVersion':2,'mode':'local-shadow','complete':full,'sections':{s:{'complete':self.sections.get(s,{}).get('complete',False),'count':len(v.items),'items':v.items} for s,v in self.inventories.items()},'experience':experience_summary(self.experience),'bag':bag_summary(self.bag)}
            write(self.out/'inventory.json',inventory)
            write(self.out/'experience-results.json',self.experience)
            write(self.out/'tail-merge.json',{s:v.tail_diagnostics for s,v in self.inventories.items()})
            for s,v in self.inventories.items():
                write(self.out/(s+'-results.json'),{**self.sections.get(s,{'section':s,'complete':False}), 'final_count':len(v.items),'raw_observations':v.observation_count,'tail_duplicates':sum(t['duplicate_count'] for t in v.tail_diagnostics),'tail_new':sum(t['new_count'] for t in v.tail_diagnostics)})
            write(self.out/'result.json',{'outcome':self.outcome,'sections':self.sections,'pages':self.store.index,'page_timings':self.page_timings,'captures':self.captures,'events':self.events,'context_calls':self.call_count,'action_wall_ms':now_ms(self.started),'experience':experience_summary(self.experience)})
            write(self.out/'timing.json',{'page_timings':self.page_timings,'navigation_ms':self.nav_total_ms,'sections':self.sections,'action_wall_ms':now_ms(self.started),'context_calls':self.call_count})
            write(self.root/'latest.json',{'run_id':self.params['run_id'],'directory':self.out.name,'outcome':self.outcome,'complete':full})
            logger.info(f'星石本地识别 {self.outcome}，结果：{self.out / "inventory.json"}')
            if self.bridge:self.bridge.close()


@AgentServer.custom_action('StarFullLocalOcrShadow')
class StarFullLocalOcrShadow(CustomAction):
    def run(self,context:Context,argv:CustomAction.RunArg):
        if not isinstance(context,Context) or not Library.is_agent_server():raise RuntimeError('actual Agent Context required')
        return FullShadowSession(context,json.loads(argv.custom_action_param)).execute()
