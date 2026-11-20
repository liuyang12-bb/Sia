"""Compare already-complete v2 cells to final numerical-guard version."""
import json,gzip,itertools
from pathlib import Path
OUT=Path(__file__).resolve().parent
def signature(r):
    d=r['decision_for_next_interval']
    return dict(time=r['interval_end'],execution=[{k:e[k] for k in [
        'job_id','progress_before','progress_after','rho','checkpoint_delay_consumed','training_seconds','completion_time']} for e in r['execution']],
        allocation={j:dict(config=c['config']['config_id'],cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu'],placement=c['placement']) for j,c in d['result']['selected'].items()},
        end=r['job_end_state'])
results=[]
for cid in ['n8_neutral_joint_g','n8_medium_joint_g','n20_neutral_joint_g']:
    n=0;different=[]
    with gzip.open(OUT/'attempt2'/cid/'rounds.jsonl.gz','rt',encoding='utf-8') as a,gzip.open(OUT/'attempt3'/cid/'rounds.jsonl.gz','rt',encoding='utf-8') as b:
        for n,(left,right) in enumerate(itertools.zip_longest(a,b),1):
            if left is None or right is None or signature(json.loads(left))!=signature(json.loads(right)):different.append(n)
    results.append(dict(cell=cid,identical=not different,rounds=n,different_rounds=different))
with (OUT/'trajectory_regression.json').open('x',encoding='utf-8') as f:json.dump(results,f,indent=2)
print(results)
