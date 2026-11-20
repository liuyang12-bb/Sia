"""Reconstruct same-state candidate MIPs; independently check execution logs."""
import json,gzip,sys,math,hashlib,collections
from pathlib import Path
from dataclasses import fields
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'sia-joint-small-mechanism/sia-simulator'))
from joint_prototype.joint_profile import GpuConfig,JobSpec,NodeSpec,ProfileQuery
from joint_prototype.joint_scheduler import JointColumn,Placement
from joint_prototype.neutral_liveness import FactorScheduler
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.full160 import capacities
from joint_prototype.small_mechanism import fixed_set_solve
from joint_prototype.execution_audit import audit_execution
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def rounds(name):
 with gzip.open(OUT/name/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
  for l in f:yield json.loads(l)
def restore(c):
 d=dict(c);d['config']=GpuConfig(**d['config']);d['query']=ProfileQuery(**d['query']);d['placement']=Placement(tuple(tuple(a) for a in d['placement']['node_gpus']));d['node_usage']=tuple(tuple(a) for a in d['node_usage']);return JointColumn(**d)
def geo(c):return(c.job_id,c.config.config_id,c.placement.node_gpus,c.query.cpu_per_gpu,c.query.ram_gb_per_gpu)
def decode(d,p):
 probe=d['probe'];specs=[]
 for j in probe['specs']:
  j=dict(j);j['configs']=tuple(GpuConfig(**c) for c in j['configs']);j['current_placement']=tuple(tuple(a) for a in j['current_placement']);j['current_nodes']=tuple(j['current_nodes']);specs.append(JobSpec(**j))
 apps={j.job_id:j.job_id.split('-')[0] for j in specs};o=R0Oracle(p['registry'],p['models'],apps,'medium')
 cfg=json.loads(Path(p['config']).read_text());nodes=[NodeSpec(t+':0',t,g,g*3,g*62.5) for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]]
 s=FactorScheduler(nodes,specs,o,cfg['response']['aux_cpu_points'],cfg['response']['aux_ram_points'],allocation_bonus=0,verify_exact=False,minimum_cpu_per_gpu=1,minimum_ram_gb_per_gpu=20.83,candidate_policy='b3')
 cols=[restore(c) for c in probe['joint_columns']];regenerated=s.build_joint_columns();assert {geo(c) for c in cols}=={geo(c) for c in regenerated}
 for c in cols:
  q=o.query(s.job_map[c.job_id],c.config,c.query.cpu_per_gpu,c.query.ram_gb_per_gpu,(3,62.5));assert math.isclose(q.rho,c.query.rho,rel_tol=1e-12)
 return s,cols
def metrics(cols):
 return dict(utility=sum(c.scheduling_utility for c in cols),predicted_goodput=sum(c.query.effective_goodput for c in cols),cpu=sum(c.config.gpus*c.query.cpu_per_gpu for c in cols),ram=sum(c.config.gpus*c.query.ram_gb_per_gpu for c in cols),selected=[dict(job=c.job_id,config=c.config.config_id,cpu=c.query.cpu_per_gpu,ram=c.query.ram_gb_per_gpu,rho=c.query.rho,raw=c.query.raw_sia_goodput,effective=c.query.effective_goodput) for c in cols])
def main():
 p=json.loads((OUT/'preregistration.json').read_text());validation=[];summaries=[]
 for response,method in p['runs']:
  name=response+'_'+method;summ=json.loads((OUT/name/'summary.json').read_text());summaries.append(summ);checks=0;boost=0.;weighted=0.;train=0.;clip=0;nsel=0;rsum=collections.defaultdict(lambda:dict(wait=0.,checkpoint=0.,training=0.,gpu=0.,cpu=0.,ram=0.,completion=None))
  for r in rounds(name):
   d=r['decision_for_next_interval'];sel=d['result']['selected'];usage={t:[0.,0.,0.] for t in capacities('main')}
   for c in sel.values():
    t=c['config']['gpu_type'];g=c['config']['gpus'];usage[t][0]+=g;usage[t][1]+=g*c['cpu_per_gpu'];usage[t][2]+=g*c['ram_gb_per_gpu'];clip+=c['source_coordinates']['clipped'];nsel+=1
   for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]:assert usage[t][0]<=g and usage[t][1]<=3*g+1e-7 and usage[t][2]<=62.5*g+1e-7
   for e in r['execution']:
    assert audit_execution(e)['valid'];checks+=1;train+=e['training_seconds'];boost+=e['training_seconds'] if e['rho']>1.000001 else 0.;weighted+=e['rho']*e['training_seconds'];v=rsum[e['job_id']];v['wait']+=e['waiting_seconds'];v['checkpoint']+=e['checkpoint_delay_consumed'];v['training']+=e['training_seconds'];v['completion']=e['completion_time']
    a=e['resource_allocation']
    if a:
     dt=e['allocated_seconds']/3600;v['gpu']+=a['gpus']*dt;v['cpu']+=a['gpus']*a['cpu_per_gpu']*dt;v['ram']+=a['gpus']*a['ram_gb_per_gpu']*dt
    if e['checkpoint_remaining_before_decision']>0 and e['completion_time'] is None and a:
     new=d['result']['selected'].get(e['job_id']);assert new
     for k in ('cpu_per_gpu','ram_gb_per_gpu','placement'):assert new[k]==a[k]
     assert new['config']['config_id']==a['config_id']
   if method in ('sia_p','sia_aux'):
    for j,a in d['original_sia_allocations'].items():
     if a[0] is None or not a[1]:assert j not in sel
     else:assert j in sel and (sel[j]['config']['gpu_type'],sel[j]['config']['gpus'])==(a[0],len(a[1]))
  for j,v in rsum.items():
   row=summ['jobs'][j]
   for k in ('wait','checkpoint','training'):assert math.isclose(v[k],row[k],abs_tol=1e-7)
   for k,old in [('gpu','gpu_hours'),('cpu','cpu_core_hours'),('ram','ram_gb_hours')]:assert math.isclose(v[k],row[old],abs_tol=1e-7)
   assert v['completion']==row['completion']
  validation.append(dict(cell=name,valid=True,execution_checks=checks,boosted_training_seconds=boost,time_weighted_rho=weighted/train,selected_clip_fraction=clip/nsel))
 save('raw_validation.json',validation);save('comparison.json',summaries)
 selected=[];seen=set()
 for r in rounds('medium_sia_p'):
  d=r['decision_for_next_interval'];spec=d['probe']['specs'];conds={'first_three_active':len(spec)==3,'first_unlocked_after_training':r['interval_end']>60 and not any(j['checkpoint_locked'] for j in spec) and any(e['training_seconds']>0 for e in r['execution']),'first_reconfiguration_after_initial':r['interval_end']>60 and bool(r['rescale_events'] or r['migrate_events'])}
  for rule,ok in conds.items():
   if ok and rule not in seen and all(r['interval_end']!=x['interval_end'] for _,x in selected):seen.add(rule);selected.append((rule,r));break
  if len(selected)==3:break
 counter=[]
 for rule,r in selected:
  d=r['decision_for_next_interval'];s,cols=decode(d,p);required=set(d['result']['selected']);prop=[restore(c) for c in d['probe']['proportional_columns']]
  A=[next(c for c in prop if c.job_id==j and c.config.config_id==v['config']['config_id']) for j,v in d['result']['selected'].items()]
  B=fixed_set_solve(s,[c for c in cols if c.config.config_id==d['result']['selected'].get(c.job_id,{}).get('config',{}).get('config_id')],required)
  C=fixed_set_solve(s,cols,required)
  assert B.scheduling_objective>=sum(c.scheduling_utility for c in A)-1e-7 and C.scheduling_objective>=B.scheduling_objective-1e-7
  counter.append(dict(rule=rule,time=r['interval_end'],A=metrics(A),B=metrics(list(B.selected.values())),C=metrics(list(C.selected.values())),required=sorted(required),locks=[j.job_id for j in s.jobs if s.lock_active(j)],assumptions='Same Sia selected job set, real checkpoint locks, same p/lambda/restart, same parent normalization. B fixed GPU; C all GPU. No L2 added. Predictions only.'))
 save('same_state_ABC.json',counter)
 l2=[]
 for r in rounds('medium_joint_o'):
  d=r['decision_for_next_interval'];held=d['probe']['l2_before']['services']
  if not held:continue
  s,cols=decode(d,p);required=set(d['result']['selected']);geos={j:geo(restore(v['column'])) for j,v in held.items() if j in required};constrained=[c for c in cols if c.job_id not in geos or geo(c)==geos[c.job_id]]
  H=fixed_set_solve(s,constrained,required);F=fixed_set_solve(s,cols,required)
  l2.append(dict(time=r['interval_end'],held=list(geos),required=list(required),checkpoint_locks=[j.job_id for j in s.jobs if s.lock_active(j)],held_result=metrics(list(H.selected.values())),released_service_only=metrics(list(F.selected.values())),blocked_columns=len(cols)-len(constrained),note='Offline service geometry relaxation; real checkpoint locks retained; not a deployable liveness replacement'))
  if len(l2)==2:break
 save('l2_constraints.json',l2)
 print('VALIDATED',len(validation),'ABC',len(counter),'L2',len(l2))
if __name__=='__main__':main()
