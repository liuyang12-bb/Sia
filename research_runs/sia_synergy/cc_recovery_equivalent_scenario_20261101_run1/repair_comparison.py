"""Compare completed paired trajectories, not summary-only assertions."""
import gzip,json
from pathlib import Path
O=Path(__file__).resolve().parent;OLD=O.parent/'cc_persistent_opportunity_experiment_20261101_run1'
def rows(path):
 for l in gzip.open(path,'rt',encoding='utf8'):yield json.loads(l)
def state(r):
 return dict(selected={j:{k:v[k] for k in ('placement','cpu_per_gpu','ram_gb_per_gpu','rho')}|{'config_id':v['config']['config_id']} for j,v in r['decision_for_next_interval']['result']['selected'].items()},jobs=r['job_end_state'])
newstats=json.loads((O/'raw_audit.json').read_text());oldstats=json.loads((OLD/'raw_audit.json').read_text());newjobs=json.loads((O/'per_job.json').read_text());oldjobs=json.loads((OLD/'per_job.json').read_text())
result={}
for method in ('FO','FULL'):
 a='long_medium_'+method;b='repair_old_medium_'+method
 differences=[];identical=0
 for x,y in zip(rows(OLD/'cells'/a/'rounds.jsonl.gz'),rows(O/'cells'/b/'rounds.jsonl.gz')):
  assert x['interval_end']==y['interval_end']
  if state(x)!=state(y):differences.append(x['interval_end'])
  else:identical+=1
 result[method]=dict(old_mean=oldstats[a]['mean'],new_mean=newstats[b]['mean'],
  per_job_jct_saved={j:oldjobs[a][j]['jct']-newjobs[b][j]['jct'] for j in oldjobs[a]},
  first_trajectory_difference=min(differences) if differences else None,identical_rounds=identical,
  different_rounds_in_shared_time=len(differences),old_rounds=oldstats[a]['rounds'],new_rounds=newstats[b]['rounds'])
with (O/'repair_effect.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
