import collections,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def main():
    data=json.loads((ROOT/'research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/policies/b1/result.json').read_text(encoding='utf-8'))
    counts=collections.Counter();violations=collections.Counter();last=None;examples=[]
    for r in data['joint_round_logs']:
        d=r['decision_for_next_interval']
        if d['round']==last:continue
        last=d['round'];a=d['result']['audit'];original=a.get('original_sia_intent_audit')
        if original:
            counts['secondary_admission_rounds']+=1
            violations.update(original['violations'])
            if 'deepspeech2-159' in d['sia_fixed_gpu_configs'] and 'deepspeech2-159' not in d['result']['selected']:
                counts['DS159_GPU_INTENT_REJECTED_IN_SECOND_ADMISSION']+=1
                for dim in ['CPU','RAM','GPU']:counts['DS159_'+dim+'_VIOLATION_IN_ORIGINAL_INTENT']+=int(any(dim+' capacity' in v for v in original['violations']))
                if len(examples)<3:examples.append(dict(time=r['interval_end'],original_violations=original['violations'],rejected_config=d['sia_fixed_gpu_configs']['deepspeech2-159']))
    out=dict(counts=dict(counts),intent_violations=dict(violations),examples=examples,
        note='Violations concern the complete original proportional intent, not necessarily this job alone. Scheduler then picks a subset; CPU pressure and subset utility jointly determine the victim.')
    with (OUT/'historical_admission_causes.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
