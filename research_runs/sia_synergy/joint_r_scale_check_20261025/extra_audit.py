"""Explicit monotonicity and Neutral P/AUX identity, without touching prior audits."""
import gzip,json,math,collections
from pathlib import Path
OUT=Path(__file__).resolve().parent
checks={};mechanism={}
for n in [8,20]:
    if not (OUT/f'validation_{n}.json').exists():continue
    nseg=0
    for response in ['neutral','medium']:
        for method in ['sia_p','sia_aux','joint_r']:
            cid=f'n{n}_{response}_{method}';configs=collections.defaultdict(lambda:collections.defaultdict(float));filters=collections.defaultdict(collections.Counter);last_round=None
            with gzip.open(OUT/f'n{n}_{response}_{method}/rounds.jsonl.gz','rt',encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line)
                    for e in r['execution']:
                        assert e['progress_after']+1e-8>=e['progress_before']
                        if e['training_segments']:
                            assert e['training_segments'][0]['start_time']>=r['interval_start']+e['checkpoint_delay_consumed']-1e-8
                        a=e['resource_allocation']
                        if a:configs[e['job_id']][a['config_id']]+=e['training_seconds']
                        nseg+=1
                    d=r['decision_for_next_interval']
                    if d['round']!=last_round:
                        last_round=d['round']
                        for j,v in d['result']['audit'].get('adaptive_candidate_filters',{}).items():filters[j].update(v['rejection_counts'])
            mechanism[cid]=dict(training_by_job_config={j:dict(v) for j,v in configs.items()},candidate_rejection_counts={j:dict(v) for j,v in filters.items()})
    by={}
    for method in ['sia_p','sia_aux']:
        summary=json.loads((OUT/f'n{n}_neutral_{method}/summary.json').read_text())
        by[method]={j:{k:r[k] for k in ['jct','completion','wait','checkpoint','training','restarts','wasted_gpu_seconds']} for j,r in summary['jobs'].items()}
    checks[str(n)]=dict(monotonicity_segments=nseg,checkpoint_before_training=True,
                       neutral_per_job_p_aux_identical=by['sia_p']==by['sia_aux'])
with (OUT/'extra_audit.json').open('x',encoding='utf-8') as f:json.dump(checks,f,indent=2)
with (OUT/'mechanism_by_config.json').open('x',encoding='utf-8') as f:json.dump(mechanism,f,indent=2)
print(checks)
