"""Independent raw-trajectory diagnostics, not a simulation or counterfactual JCT."""
import collections,gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_joint_queue_ram_tail_20261027_run3'
OLD=ROOT/'research_runs/sia_synergy/cc_joint_sustained_cpu_20job_20261027_run1'
def save(name,value):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def inspect(path):
 jobs=collections.defaultdict(lambda:dict(wait=0.,checkpoint=0.,training=0.,initial_wait=0.,later_wait=0.,first_allocation=None,first_training=None,completion=None,arrival=None,configs=collections.Counter(),restarts=0))
 durations=collections.Counter();decision_counts=collections.Counter();n=0
 with gzip.open(path,'rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);n+=1
   d=r['decision_for_next_interval'];res=d['result']
   decision_counts[res['status']]+=1
   for e in r['execution']:
    j=jobs[e['job_id']];j['arrival']=e['arrival_time'];j['completion']=e['completion_time']
    first=e['first_allocation_time'];j['first_allocation']=first;j['first_training']=e['first_productive_training_time']
    for dst,src in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:j[dst]+=e[src]
    if e['waiting_seconds']:
     j['initial_wait' if first is None or e['interval_start']<first else 'later_wait']+=e['waiting_seconds']
    a=e['resource_allocation']
    if a and e['training_seconds']:
     key=f"{a['config_id']}|CPU{a['cpu_per_gpu']}|RAM{a['ram_gb_per_gpu']}"
     j['configs'][key]+=e['training_seconds']
   durations['waiting_job_seconds']+=sum(e['waiting_seconds'] for e in r['execution'])
   durations['active_job_seconds']+=sum(e['observed_active_seconds'] for e in r['execution'])
 for name,j in jobs.items():
  j['jct']=j['completion']-j['arrival'] if j['completion'] is not None else None
  if j['jct'] is not None:assert math.isclose(j['wait']+j['checkpoint']+j['training'],j['jct'],abs_tol=1e-6),name
  j['configs']=dict(j['configs'])
 return dict(rounds=n,jobs=dict(jobs),queue_areas=dict(durations),status=dict(decision_counts),
  last_completion=max(((j['completion'],name) for name,j in jobs.items() if j['completion'] is not None),default=None))
def main():
 result={}
 for m in ['SIA_P','SIA_ECT_AUX','JOINT_CC_P','JOINT_CC']:
  result[m]=inspect(OLD/'cells/medium'/m/'rounds.jsonl.gz')
 a=result['SIA_P'];b=result['JOINT_CC']
 delta={name:{k:b['jobs'][name][k]-j[k] for k in ['jct','wait','initial_wait','later_wait','checkpoint','training']} for name,j in a['jobs'].items()}
 aggregate={k:sum(v[k] for v in delta.values()) for k in next(iter(delta.values()))}
 save('queue_tail_raw_diagnostics.json',dict(methods=result,per_job_CC_minus_Sia=delta,totals=aggregate,
  interpretation='Observed path decomposition, not an admission/placement causal ablation. Queue area integral equals cumulative waiting; active jobs area equals total JCT.',
  sha256={m:hashlib.sha256((OLD/'cells/medium'/m/'rounds.jsonl.gz').read_bytes()).hexdigest().upper() for m in result}))
 print(aggregate)
 print({m:r['last_completion'] for m,r in result.items()})
if __name__=='__main__':main()
