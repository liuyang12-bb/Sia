"""Offline conditional mechanism analysis; no simulator or training advancement.
Three identical replicas of one observed CIFAR state are a mathematical test,
NOT the three real workload jobs and NOT a prediction of their JCT.
"""
import gzip,hashlib,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-mismatch-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.joint_scheduler import JointScheduler
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.completion_continuity import completion_priority
from residual_capacity import resource_points

def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
source=O.parent/'cc_fo_full_mismatch_20261101_run1/gate_rounds.jsonl.gz'
save('screen_plan.json',dict(states=[120,420],source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 assumption='three identical copies of the observed CIFAR0 model and remaining work; not real trace triplet',
 goal='same reciprocal current-ETA formula; all 3 served; all configs compared; no claim of actual admission or historical locks',
 startup=50,reason='observed CIFAR reconfiguration/startup50; common fresh-placement diagnostic',
 capacity={'aws':[2,9,62.5],'dgx-ext':[2,3,187.5]},no_training=True))
results=[]
for line in gzip.open(source,'rt',encoding='utf8'):
 r=json.loads(line)
 if r['time'] not in (120,420):continue
 d=r['decisions']['FULL'];sp=next(s for s in d['probe']['specs'] if s['job_id']=='cifar10-0')
 scores={}
 for v in d['continuity_scores']:
  if v['job']==sp['job_id'] and v['raw_rate'] is not None:scores[v['column_id'].split('|')[1]]=v
 names=['reference-copy-'+str(i) for i in range(3)]
 jobs=[JobSpec(n,sp['template_file'],tuple(GpuConfig(**c) for c in sp['configs'])) for n in names]
 oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{n:'cifar10' for n in names},'medium')
 nodes=[NodeSpec('aws:0','aws',2,9,62.5),NodeSpec('dgx-ext:0','dgx-ext',2,3,187.5)]
 rc=json.loads((T/'inputs/core80_config.json').read_text(encoding='utf8'))['response']
 s=JointScheduler(nodes,jobs,oracle,sorted(set(rc['aux_cpu_points'])|{8.}),rc['aux_ram_points'],enforce_proportional_progress_floor=False,minimum_cpu_per_gpu=1,minimum_ram_gb_per_gpu=20.83)
 s.resource_point_provider=resource_points
 joint=s.build_joint_columns();props=s.build_proportional_columns()
 def key(c):return (c.job_id,c.config.config_id,c.placement.node_gpus,c.query.cpu_per_gpu,c.query.ram_gb_per_gpu)
 cols=list({key(c):c for c in joint+props}.values())
 def score(c):
  v=scores[c.config.config_id]
  return completion_priority(v['remaining_work'],v['raw_rate'],c.query.rho,50.)
 scale=max(map(score,cols))
 def solve(domain):
  mat=[];lo=[];hi=[]
  for j in names:mat.append([int(c.job_id==j) for c in domain]);lo.append(1);hi.append(1)
  for n in nodes:
   for k,b in enumerate((n.gpus,n.cpu_cores,n.ram_gb)):
    mat.append([c.usage_for(n.node_id)[k] for c in domain]);lo.append(0);hi.append(b)
  weights=np.array([score(c)/scale for c in domain])
  result=milp(-weights,integrality=np.ones(len(domain)),bounds=Bounds(0,1),constraints=LinearConstraint(np.array(mat),lo,hi),options={'time_limit':30,'mip_rel_gap':0})
  assert result.status==0 and result.x is not None,result.message
  rounded=np.rint(result.x);assert max(abs(rounded-result.x))<1e-6
  plan=[c for c,x in zip(domain,rounded) if x]
  assert s.audit(plan)['valid'] and len(plan)==3
  return plan
 prop=solve(props);fixed={c.job_id:c.config.config_id for c in prop}
 serial=solve([c for c in cols if c.config.config_id==fixed[c.job_id]])
 full=solve(cols)
 def detail(plan):
  rows=[dict(job=c.job_id,config=c.config.config_id,cpu=c.query.cpu_per_gpu,ram=c.query.ram_gb_per_gpu,rho=c.query.rho,
     raw=scores[c.config.config_id]['raw_rate'],effective=scores[c.config.config_id]['raw_rate']*c.query.rho,
     eta=1/score(c),num_clip=c.query.source_coordinate_clipped,den_clip=c.query.baseline_coordinate_clipped) for c in plan]
  return dict(allocation=rows,reciprocal_eta_objective=sum(score(c) for c in plan),constant_rate_sum_eta=sum(1/score(c) for c in plan))
 a,b=detail(serial),detail(full)
 results.append(dict(state=r['time'],reference_remaining_work=next(iter(scores.values()))['remaining_work'],proposal=detail(prop),serial=a,joint=b,
 objective_gain_pct=100*(b['reciprocal_eta_objective']/a['reciprocal_eta_objective']-1),
 static_eta_reduction_pct=100*(1-b['constant_rate_sum_eta']/a['constant_rate_sum_eta']),
 disclaimer='conditional identical-state analytical example; not FO/FULL execution nor forecast of actual triplet'))
save('conditional_results.json',results)
for r in results:print(r['state'],'score gain',r['objective_gain_pct'],'static ETA reduction',r['static_eta_reduction_pct'],r['serial']['allocation'],r['joint']['allocation'])
