"""Independent four-row diagnostic rules; no OCR, cross-image item matching or device input."""
from pathlib import Path
import time,json
import numpy as np
SAMPLE_OFFSETS_MS=(0,200,400,600,1000,1500,2500)
PARAMS={'swipe_begin':[360,1000],'swipe_end':[360,370],'swipe_duration_ms':700,'swipe_end_hold_ms':250}

def claim_once(root):
    with (Path(root)/'gesture-once.json').open('x',encoding='utf-8') as f:json.dump({'parameters':PARAMS,'request_monotonic_ns':time.perf_counter_ns()},f)

def stable_pair(motion):
    # Diagnostic candidate only. Never changes production NCC thresholds.
    return bool(motion.get('measured') and abs(motion['signed_shift_y_px'])<=2 and motion['direct_score']>=.97 and motion['same_position_score']>=.85)

def audit_candidate(motion):
    if not motion.get('measured'):return 'unknown'
    return 'passed' if abs(motion['signed_shift_y_px'])<=2 and motion['direct_score']>=.97 else 'failed'

def summarize_geometry(g):
    t=time.perf_counter();rows={}
    for c in g['cards']:
        if c['completeness']=='complete':rows.setdefault(c['rowIndex'],[]).append(c)
    centers=list(g['rowCenters']);pitches=list(g['rowPitches'])
    complete_rows=[r for r,cards in rows.items() if len(cards)==4 and {c['columnIndex'] for c in cards}=={0,1,2,3}]
    return {'rowCenters':centers,'rowPitches':pitches,'median_pitch_px':float(np.median(pitches)) if pitches else None,'four_complete_rows':len(complete_rows)==4 and len([c for c in g['cards'] if c['completeness']=='complete'])==16,'complete_row_count':len(complete_rows),'completenessCounts':g['completenessCounts'],'contentBounds':g['profile']['contentBounds'],'metadata_summary_ms':(time.perf_counter()-t)*1000}

def motion_estimate(before,after,ga,gb,probe):
    # One aggregate image measurement; no names, individual anchors or semantic row proofs.
    t=time.perf_counter();estimate=probe.estimate_vertical_motion(before[280:1040,40:680],after[280:1040,40:680],(30,740),(30,740));duration=(time.perf_counter()-t)*1000
    pitches=ga['rowPitches'];pitch=float(np.median(pitches)) if pitches else None
    geometric=ga['rowCenters'][0]+4*pitch-gb['rowCenters'][0] if pitch is not None and gb['rowCenters'] else None
    # Conditional corroboration only: input-response prior 0.85..1.15 of the 630px touch.
    lo,hi=630*.85,630*1.15
    candidates=[c for c in estimate.motion_hypotheses if c.get('full_overlap_valid') and c.get('full_overlap_score',0)>=.85 and lo<=c['shift_y']<=hi and geometric is not None and abs(c['shift_y']-geometric)<=8]
    valid=len(candidates)==1
    chosen=candidates[0]['shift_y'] if valid else None
    return {'raw_selected_px':estimate.shift_y,'raw_confidence':estimate.confidence,'raw_hypotheses':estimate.motion_hypotheses,'geometric_four_row_phase_hypothesis_px':geometric,'conditional_content_shift_px':chosen,'credible_for_diagnostic_control':valid,'confidence_scope':'conditional on four-row phase and 0.85..1.15 touch-response prior; no OCR text used; raw periodic alternatives retained','input_response_range_px':[lo,hi],'motion_estimate_ms':duration,'obvious_anomaly':chosen is not None and abs(chosen-633)>60}

def adaptive_suggestion(geometry,content_shift):
    t=time.perf_counter();centers=geometry['rowCenters'];pitches=geometry['rowPitches'];pitch=float(np.median(pitches)) if pitches else None
    result={'executed':False,'max_touch_correction_px':30,'touch_start':[360,1000],'uses_OCR_names':False,'repeats_geometry_detection':False}
    if content_shift is None or not centers or pitch is None or not 0.85<=content_shift/630<=1.15:
        result.update(status='unreliable_measurement_no_parameter_recommendation',calculation_ms=(time.perf_counter()-t)*1000);return result
    desired=centers[0]+4*pitch-390;gain=content_shift/630;unbounded=desired/gain
    bounds=geometry['profile']['contentBounds'];safe_min=bounds['y']+28;safe_max=min(1000,bounds['y']+bounds['height']-20)
    target=min(660,max(600,unbounded));end=int(round(max(safe_min,min(safe_max,1000-target))))
    result.update(status='numerical_suggestion_only',current_first_row_y=centers[0],measured_row_pitch_px=pitch,desired_content_shift_px=desired,observed_gain=gain,unbounded_touch_length_px=unbounded,suggested_touch_length_px=1000-end,suggested_swipe_end=[360,end],duration_ms=700,end_hold_ms=250,allowed_endpoint_y=[safe_min,safe_max],endpoint_scope='inside the observed scroll-content ROI with margins; exact new endpoint has not been device-tested',calculation_ms=(time.perf_counter()-t)*1000)
    return result
