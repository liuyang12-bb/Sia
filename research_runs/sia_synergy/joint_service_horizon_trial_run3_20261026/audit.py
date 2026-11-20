"""Post-run raw completion/time/resource/score audit, without simulator steps."""
import collections,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;P=json.loads((OUT/'preregistration.json').read_text())
P.update(registry=P['scenario']['registry'],models=P['scenario']['models'],workload=str(OUT/'workload.csv'))
sys.path.insert(0,P['tree'])
from applications import APPLICATIONS
from sia import ZERO_ALLOC_GAIN
import simulator_config as sc
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.execution_audit import audit_execution
from run_stage2c_cell import independent_rho
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-5),(a,b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
apps={r.split(',')[0]:r.split(',')[2] for r in Path(P['workload']).read_text().splitlines()[1:]}
results=[]
for method in P['runs']:
 cell=dict(id=method,method=method,response='medium')
 directory=OUT/'cells'/method
 s=json.loads((directory/'summary.json').read_text());assert s['completed']==s['total']==6
 oracle=R0Oracle(P['registry'],P['models'],apps,cell['response'])
 time=collections.defaultdict(lambda:collections.Counter());progress=collections.defaultdict(float);complete={}
 counts=collections.Counter();last=None
 with gzip.open(directory/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);d=r['decision_for_next_interval']
   for e in r['execution']:
    assert audit_execution(e)['valid'];j=e['job_id'];close(progress[j],e['progress_before']);progress[j]=e['progress_after']
    assert e['progress_after']>=e['progress_before']-1e-9
    for name,key in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:time[j][name]+=e[key]
    if e['completion_time'] is not None:complete[j]=e['completion_time']
    a=e['resource_allocation']
    if a:
     rho=independent_rho(oracle,j,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu']);close(rho,e['rho'])
     if cell['response']=='neutral':assert rho==1.
     for seg in e['training_segments']:close(seg['effective_rate'],seg['raw_rate_before_rho']*rho)
     for name,factor in [('gpu_hours',1.),('cpu_core_hours',a['cpu_per_gpu']),('ram_gb_hours',a['ram_gb_per_gpu'])]:
      time[j][name]+=a['gpus']*factor*e['allocated_seconds']/3600
    counts['execution_records']+=1
   if last==d['round']:continue
   last=d['round'];counts['decisions']+=1
   selected=d['result']['selected'];nodes=collections.defaultdict(lambda:[0,0,0]);specs={v['job_id']:v for v in d['probe']['specs']}
   for j,c in selected.items():
    for node,g in c['placement'].items():
     nodes[node][0]+=g;nodes[node][1]+=g*c['cpu_per_gpu'];nodes[node][2]+=g*c['ram_gb_per_gpu']
   for node,used in nodes.items():
    typ=node.rsplit(':',1)[0];cap=P['scenario']['capacities'][typ]
    limit=(P['scenario']['gpus'][typ],cap['cpu_cores'],cap['ram_gb'])
    assert all(x<=y+1e-7 for x,y in zip(used,limit));counts['capacity_checks']+=1
   for j,v in specs.items():
    if v['checkpoint_locked'] and v['current_config_id']:
     c=selected[j];assert c['config']['config_id']==v['current_config_id'];assert c['placement']==dict(v['current_placement'])
     close(c['cpu_per_gpu'],v['current_cpu_per_gpu']);close(c['ram_gb_per_gpu'],v['current_ram_gb_per_gpu']);counts['lock_checks']+=1
   if method=='G111':
    a=d['result']['audit'];assert set(a['required_jobs'])<=set(selected)
    assert set(a['service_commitments'])==set(selected)
    assert not a['drain'] or not a['optional_selected']
    assert set(a['checkpoint_locks'])<=set(a['required_jobs'])
    counts['protected_admission_checks']+=1
   if cell['method'] in ['G11','G111','G112']:
    cols={c['column_id']:c for c in d['probe']['joint_columns']}
    events={e['job_id']:e for e in r['execution']}
    for score in d['net_progress_scores']:
     c=cols[score['column_id']];j=c['job_id'];v=specs[j]
     rho=independent_rho(oracle,j,c['config']['config_id'],c['query']['cpu_per_gpu'],c['query']['ram_gb_per_gpu'])
     rate=c['config']['sia_goodput']*rho
     same=c['config']['config_id']==v['current_config_id'] and c['placement']['node_gpus']==v['current_placement']
     delay=events[j]['checkpoint_remaining_before_decision'] if same else float(APPLICATIONS[c['config']['gpu_type']][apps[j]].rescale_time or 30.)
     close(delay,score['candidate_delay']);close(rate,score['normalized_rate_before_delay'])
     horizon=score['service_horizon'] if cell['method']=='G112' else 300.
     if cell['method']=='G112':
      assert 60.-1e-8<=horizon<=300.+1e-8
     expected=1.1-min(round(max(rate*max(horizon-delay,0)/horizon,ZERO_ALLOC_GAIN)**(-.5),2),sc.sia_goodput_clip_val)
     close(expected,score['utility']);counts['independent_score_checks']+=1
 for j,v in s['jobs'].items():
  close(v['completion'],complete[j]);close(v['jct'],complete[j]-v['arrival'])
  for key,value in time[j].items():close(v[key],value)
  close(v['jct'],sum(time[j][key] for key in ['wait','checkpoint','training']))
 close(s['mean_jct'],sum(complete[j]-v['arrival'] for j,v in s['jobs'].items())/6)
 results.append(dict(cell=cell,summary=s,audit_counts=dict(counts),valid=True,
    hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file()}))
 print('VALID',cell['id'],s['mean_jct'])
before=json.loads((OUT.parent/'joint_protected_admission_20261026/parent_hashes.json').read_text())
unchanged={r:sha(Path(P['tree']).parent/r)==h for r,h in before.items()}
assert {r.replace('\\','/') for r,valid in unchanged.items() if not valid}=={'sia-simulator/joint_prototype/adaptive_service.py'}
profiles={str(p):sha(p) for p in (Path(P['tree'])/'traces').rglob('*.csv')}
save(OUT/'supplemental_gpu_profile_hashes.json',profiles)
save(OUT/'comparison_audit.json',dict(valid=True,original_sources_unchanged=sum(unchanged.values()),profile_files=len(profiles),
    preregistration_sha256=sha(OUT/'preregistration.json'),results=results))
