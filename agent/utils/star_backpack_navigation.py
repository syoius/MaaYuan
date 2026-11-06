"""Unified supported entry: home / observatory / any star-backpack section.

Entering the backpack from the observatory resets to main/top by game rule.
No reverse swipe or top-image correlation is needed or performed here.
"""
from __future__ import annotations
import time

def _hit(context,node,image):
    detail=context.run_recognition(node,image)
    return bool(detail and detail.hit)

def identify_star_entry(context,image):
    if _hit(context,'星石本地OCR-已在背包',image):return 'backpack'
    if _hit(context,'已在界面-观星',image):return 'observatory'
    if _hit(context,'已在界面-左半屏',image) or _hit(context,'已在界面-右半屏',image):return 'home'
    return 'unknown'

def run_navigation_node(context,node,*,post_delay=1000):
    if context.tasker.stopping:raise InterruptedError('navigation stopped')
    override={node:{'next':[],'on_error':[],'pre_delay':0,'post_delay':post_delay,'timeout':6000}}
    detail=context.run_task(node,override)
    if not detail or not detail.status.succeeded:raise RuntimeError('navigation failed: '+node)
    return detail

def prepare_star_backpack_entry(context,capture,record):
    start=time.perf_counter();image=capture('entry-start');kind=identify_star_entry(context,image)
    result={'start':kind,'nodes':[],'success':False}
    try:
        if kind=='unknown':raise RuntimeError('unsupported current page')
        if kind=='backpack':
            run_navigation_node(context,'返回上级界面');result['nodes'].append('返回上级界面')
            run_navigation_node(context,'已在界面-观星',post_delay=0);result['nodes'].append('已在界面-观星')
            image=capture('entry-observatory')
            if not _hit(context,'已在界面-观星',image):raise RuntimeError('backpack back did not reach observatory')
        elif kind=='home':
            # Preserve the existing home left/right and OCR/TM navigation graph.
            detail=context.run_task('进入界面-观星',{'进入界面-观星':{'timeout':20000}})
            if not detail or not detail.status.succeeded:raise RuntimeError('home to observatory failed')
            result['nodes'].append('进入界面-观星');image=capture('entry-observatory')
            if not _hit(context,'已在界面-观星',image):raise RuntimeError('observatory not confirmed')
        run_navigation_node(context,'星石本地OCR-点击背包入口');result['nodes'].append('星石本地OCR-点击背包入口')
        # A successful click can precede the rendered page transition. Wait without another gesture.
        run_navigation_node(context,'星石本地OCR-已在背包',post_delay=0);result['nodes'].append('星石本地OCR-已在背包')
        image=capture('entry-main-backpack')
        if not _hit(context,'星石本地OCR-已在背包',image):raise RuntimeError('backpack identity missing after navigation')
        # Selected main Tab identity only; no top-reset experiment or reverse gesture.
        if float(image[225:245,90:102,2].mean())<150:raise RuntimeError('main Tab not selected after re-entry')
        result.update(success=True,main_top_by_game_rule=True,reverse_swipes=0)
        return image,result
    except BaseException as exc:
        result['error']=str(exc)
        raise
    finally:
        result['navigation_ms']=(time.perf_counter()-start)*1000;record(result)
