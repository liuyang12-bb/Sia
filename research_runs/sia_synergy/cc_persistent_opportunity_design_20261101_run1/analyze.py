"""Read existing logs only. No simulator or solver imports or calls."""
import csv,gzip,hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent;OLD=O.parent/'cc_competition_experiment_20261101_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
oldseal=json.loads((OLD/'artifact_hashes.json').read_text(encoding='utf8'))
assert all(sha(OLD/k)==v for k,v in oldseal.items())
states=[]
for method in ('FO','FULL'):
 for line in gzip.open(OLD/'cells'/('primary_'+method)/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(line);t=r['interval_end']
  if t not in (180,240,300,360,420,480,600,1320):continue
  d=r['decision_for_next_interval'];a=d['result']['audit'];selected=d['result']['selected']
  states.append(dict(method=method,time=t,required=a['required_jobs'],ready=a['ready'],events=a['geometry_event_reasons'],drain=a['drain'],
    selected={j:dict(config=c['config']['config_id'],gpu=c['config']['gpus'],cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu']) for j,c in selected.items()},
    locks={s['job_id']:s['checkpoint_locked'] for s in d['probe']['specs']},
    execution=[e for e in r['execution'] if e['job_id'] in ('bert-28','cifar10-2')]))
save('timeline.json',states)
for r in states:
 if r['time']==300:
  assert len(r['required'])==4
  assert sum(c['gpu'] for c in r['selected'].values())==4
  assert all(c['gpu']==1 for c in r['selected'].values())
rows=[]
for line in gzip.open(OLD/'gate_FO.jsonl.gz','rt',encoding='utf8'):
 r=json.loads(line)
 if r['time']!=240:continue
 for method,d in r['decisions'].items():
  c=d['result']['selected']['cifar10-2'];score=next(v for v in d['continuity_scores'] if v['column_id']==c['column_id'])
  aws2=[v for v in d['continuity_scores'] if v['job']=='cifar10-2' and v['column_id'].split('|')[1]=='aws:1:2' and v['estimated_completion'] is not None]
  rows.append(dict(method=method,selected=c,score=score,best_aws2_estimated_completion=min(v['estimated_completion'] for v in aws2)))
save('fixed_gpu_evidence.json',rows)
assert rows[0]['score']['estimated_completion']==rows[0]['best_aws2_estimated_completion']
(O/'inputs').mkdir()
with (OLD/'inputs/competition12.csv').open(encoding='utf8',newline='') as f:
 rd=csv.DictReader(f);fields=rd.fieldnames;work=list(rd)
new=[{**r,'time':str(int(r['time'])+900 if int(r['time'])>0 else 0)} for r in work]
for name,data in [('short_gap12.csv',work),('long_gap12.csv',new)]:
 with (O/'inputs'/name).open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
for a,b in zip(work,new):assert {k:v for k,v in a.items() if k!='time'}=={k:v for k,v in b.items() if k!='time'}
save('design_manifest.json',dict(status='DESIGN_ONLY_NO_NEW_TRAINING',base_source= 'cc-joint-competition-v1',base_execution_preregistration_sha256=sha(OLD/'preregistration.json'),
 source_snapshot=str((OLD/'source_snapshot.zip').resolve()),source_snapshot_sha256=sha(OLD/'source_snapshot.zip'),
 primary_workload='long_gap12.csv',arrival_times=[0,1200,1500,1800],
 rule='all nonzero arrivals delayed by fixed900s; relative gaps and all work/parameters preserved; no job deletion',
 window_design='warmup allowance300s + minimum target observation900s => next arrival1200s; not calibrated on JCT',
 capacity={'aws':[2,9,62.5],'dgx-ext':[2,3,187.5]},responses={'primary':'medium alpha.5','negative_control':'neutral alpha0'},methods=['FO','FULL'],
 planned_new_runs=4,old_control=str(OLD),
 precheck='first actual same-state selection advantage after complete eligibility; then >=6 actual consecutive intervals with distinct configurations and both incumbents legal, or earlier receiver completion reported separately; do not require a new geometry-switch event each round; later divergent prefixes are not same-state counterfactuals',
 failure='do not extend gap or modify algorithm to make it pass; report completion/model/geometry/admission stopping cause',
 round_limit=6000,interval=60,memory_gib=2,disk_gib=5,seed_profiles=[1,8],
 input_hashes={p.name:sha(p) for p in (O/'inputs').iterdir()},label='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
save('validation.json',dict(old_seal_unchanged=True,all12_jobs_preserved=True,work_and_batch_unchanged=True,new_training_runs=0,new_solver_calls=0,
 four_services_on_four_gpus_verified=True,fo_fixed_aws2_already_optimal_in_current_candidate_domain=True))
print('Offline evidence and design saved; no new training or solve')
