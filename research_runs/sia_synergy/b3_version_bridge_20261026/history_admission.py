"""Classify historical intent/admission evidence; do not replay old training."""
import collections,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def main():
    result={};target='deepspeech2-159';old=ROOT/'research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/policies'
    for method in ['b1','b4','b3']:
        data=json.loads((old/method/'result.json').read_text(encoding='utf-8'));counts=collections.Counter();statuses=collections.Counter();targetrho=[];last=None;examples=[]
        for row in data['joint_round_logs']:
            d=row['decision_for_next_interval']
            if d['round']==last:continue
            last=d['round'];counts['decisions']+=1;statuses[d['result']['status']]+=1
            a=d['result']['audit'];counts['secondary_admission_audit']+=int('original_sia_intent_audit' in a)
            if target not in d['candidate_summary']:continue
            counts['target_active_rounds']+=1;c=d['result']['selected'].get(target);fixed=d['sia_fixed_gpu_configs'].get(target)
            counts['target_selected']+=int(c is not None)
            if c:
                targetrho.append(c['rho'])
                if len(examples)<2:examples.append(dict(time=row['interval_end'],config=c['config']['config_id'],cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu'],rho=c['rho']))
            if method!='b3':
                counts['target_missing_original_gpu_intent']+=int(fixed is None)
                counts['target_original_intent_but_not_admitted']+=int(fixed is not None and c is None)
        result[method]=dict(counts=dict(counts),statuses=dict(statuses),target_selected_rho_range=[min(targetrho),max(targetrho)] if targetrho else None,target_first_selected_examples=examples)
    result['interpretation']='Intent absence and auxiliary admission are different. Counts are decision rounds, NOT exact waiting seconds. Historical code lacks current partial-arrival timing/complete-lock guarantees.'
    with (OUT/'historical_admission_evidence.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
