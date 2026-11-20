"""Read-only historical context; no old source is executed or repaired."""
import hashlib,json,collections
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest().upper()
def main():
    old=ROOT/'research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000'
    evidence={str(f):sha(f) for f in [old/'frozen_config_6000.json',old/'workload.csv',old/'followup_preregistration.json',ROOT/'research_notes/synergy/B3_MEDIUM_NON_NEUTRAL_W10_LATE_FOLLOWUP.md']}
    results={}
    for m in ['b1','b4','b3']:
        f=old/'policies'/m/'result.json';evidence[str(f)]=sha(f);data=json.loads(f.read_text(encoding='utf-8'))
        s=data['summary'];results[m]={k:s[k] for k in ['all_jobs_completed','average_jct_completed_only','jcts','waiting_seconds','completion_times','rounds','max_rho','min_rho']}
    diffs={j:results['b3']['jcts'][j]-results['b4']['jcts'][j] for j in results['b3']['jcts']}
    net=sum(diffs.values());target='deepspeech2-159'
    p=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
    def rows(path):
        import csv
        with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    oldrows=rows(old/'workload.csv');newrows={k:rows(v) for k,v in p['datasets'].items()}
    code={}
    for tree in ['sia-joint-medium-frozen','sia-joint-version-bridge']:
        for filename,patterns in [('joint_prototype/joint_integration.py',['reference_rate =','cpu_scale =','arrival_time=','def _b3_result','def _restart_factor']),
            ('joint_prototype/joint_scheduler.py',['lower[idx] = 1.0','def _pareto_prune','def refresh_column','candidate kind not allowed']),
            ('joint_prototype/r0_liveness.py',['phi_c','FIXED_R0_FULL_55_DOMAIN'])]:
            f=ROOT/tree/'sia-simulator'/filename
            if not f.exists():continue
            code[str(f)]=dict(sha256=sha(f),lines=[dict(line=i,text=line.strip()) for i,line in enumerate(f.read_text(encoding='utf-8').splitlines(),1) if any(x in line for x in patterns)])
    value=dict(historical_results=results,b3_minus_b4=diffs,net_jct_delta=net,target=target,target_delta=diffs[target],target_net_share=diffs[target]/net,
        recorded_target_wait={m:s['waiting_seconds'][target] for m,s in results.items()},
        workload_composition={k:dict(collections.Counter(r['application'] for r in rs)) for k,rs in [('historical_w10_late',oldrows),*newrows.items()]},
        arrival_ranges={k:[min(float(r['time']) for r in rs),max(float(r['time']) for r in rs)] for k,rs in [('historical_w10_late',oldrows),*newrows.items()]},
        old_capacity=json.loads((old/'frozen_config_6000.json').read_text())['cluster']['capacity_scenarios'],
        new_capacity=json.loads(Path(p['config']).read_text())['cluster']['capacity_scenarios']['main'],
        warning='Old waiting counters predate partial-cycle correctness repair. Historical percentages are recorded observations, not safe exact causal counterfactuals.',
        code_evidence=code,hashes=evidence)
    with (OUT/'historical_context.json').open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False)
    print('HISTORICAL_TARGET_SHARE',value['target_net_share'],'WAIT',value['recorded_target_wait'])
if __name__=='__main__':main()
