"""Finite enumeration verification, no Job.step and no future trace access."""
import hashlib,itertools,json,math,sys
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
P=json.loads((OUT/'preregistration.json').read_text());F=json.loads((OUT/'mechanism_cases.json').read_text())
sys.path.insert(0,P['tree'])
from joint_prototype.joint_profile import JointProfileOracle,NodeSpec,GpuConfig,JobSpec
from joint_prototype.neutral_liveness import FactorScheduler
from joint_prototype.exchange_netprogress import fair_utility,window_utility,configure_scheduler
from joint_prototype.small_mechanism import fixed_set_solve

class Analytic(JointProfileOracle):
 def __init__(self,curves):
  super().__init__(P['models'],'fixed_reference',neutral_response=True);self.curves=curves
 def query(self,j,c,cpu,ram,prop):
  kind=self.curves[(j.job_id,c.config_id)];axis=F['ram_grid'] if kind.startswith('ram') else F['cpu_grid']
  point=ram if kind.startswith('ram') else cpu
  if not min(axis)<=point<=max(axis):raise ValueError('ANALYTIC_OUT_OF_DOMAIN')
  rho=float(np.interp(point,axis,F['curves'][kind]))
  q=super().query(j,c,cpu,ram,prop)
  return replace(q,rho=rho,raw_rho=rho,response_raw=rho,response_baseline=1.,
   effective_goodput=c.sia_goodput*rho,template_id=kind,
   template_source=str(OUT/'mechanism_cases.json'),template_sha256=P['input_hashes'][str(OUT/'mechanism_cases.json')],
   source_coordinate_clipped=False,baseline_coordinate_clipped=False,evidence=F['evidence'])

results=[]
for case in F['cases']:
 curves={};jobs=[];nodes=[NodeSpec(t+':0',t,g,c,m) for t,g,c,m in case['nodes']]
 for index,(name,configs) in enumerate(case['jobs']):
  cs=[]
  for t,g,rate,kind in configs:
   c=GpuConfig(f'{t}:1:{g}',t,g,1,rate);cs.append(c);curves[(name,c.config_id)]=kind
  job=JobSpec(name,'bert_1.py',tuple(cs),source_row_index=index)
  cur=case.get('current',{}).get(name)
  if cur:
   c=next(c for c in cs if c.config_id==cur);n=next(n for n in nodes if n.gpu_type==c.gpu_type)
   job=replace(job,current_config_id=cur,current_nodes=(n.node_id,),current_placement=((n.node_id,c.gpus),),
       current_cpu_per_gpu=n.cpu_cores/n.gpus,current_ram_gb_per_gpu=n.ram_gb/n.gpus)
  jobs.append(job)
 # Independent GPU-only exhaustive intent, using shared fairness and GPU capacity.
 raw_choices=[]
 for combo in itertools.product(*(j.configs for j in jobs)):
  if any(sum(c.gpus for c in combo if c.gpu_type==n.gpu_type)>n.gpus for n in nodes):continue
  raw_choices.append((sum(fair_utility(c.sia_goodput) for c in combo),combo))
 _,intent=max(raw_choices,key=lambda v:v[0]);fixed={j.job_id:c.config_id for j,c in zip(jobs,intent)}
 for method in F['methods']:
  exchange=method in ['G10','G11'];net=method in ['G01','G11']
  s=FactorScheduler(nodes,jobs,Analytic(curves),F['cpu_grid'],F['ram_grid'],verify_exact=False,
     minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,candidate_policy='b3_prop' if method.endswith('prop') else 'b3')
  configure_scheduler(s,exchange)
  cs=s.build_proportional_columns() if method.endswith('prop') else s.build_joint_columns()
  if method.startswith('fixed'):cs=[c for c in cs if c.config.config_id==fixed[c.job_id]]
  def delay(c):
   j=next(j for j in jobs if j.job_id==c.job_id)
   return 0. if (j.current_config_id==c.config.config_id and j.current_placement==c.placement.node_gpus) else case.get('delay',50.)
  cs=[replace(c,scheduling_utility=(window_utility(c.query.effective_goodput,delay(c)) if net else fair_utility(c.query.effective_goodput))) for c in cs]
  r=fixed_set_solve(s,cs,[j.job_id for j in jobs])
  # Independent resource checks and exact objective over complete one-column/job products.
  best=-math.inf;feasible=0
  for combo in itertools.product(*([c for c in cs if c.job_id==j.job_id] for j in jobs)):
   if any(any(sum(v for c in combo for node,g in c.placement.node_gpus if node==n.node_id
        for v in [g*factor(c)])>cap+1e-7 for factor,cap in [(lambda c:1.,n.gpus),(lambda c:c.query.cpu_per_gpu,n.cpu_cores),(lambda c:c.query.ram_gb_per_gpu,n.ram_gb)]) for n in nodes):continue
   feasible+=1;best=max(best,sum(c.scheduling_utility for c in combo))
  assert abs(best-r.scheduling_objective)<1e-7
  row=dict(case=case['id'],method=method,fixed_gpu_only_intent=fixed,candidate_count=len(cs),
    exact_feasible_allocations=feasible,exact_objective=best,mip_objective=r.scheduling_objective,
    total_normalized_rate=sum(c.query.effective_goodput for c in r.selected.values()),
    window_progress=sum(max(300-delay(c),0)*c.query.effective_goodput for c in r.selected.values()),
    selected={j:asdict(c) for j,c in r.selected.items()},audit=r.audit)
  results.append(row)
  print(case['id'],method,round(row['total_normalized_rate'],4),round(best,4),flush=True)
with (OUT/'static_results.json').open('x',encoding='utf-8') as f:json.dump(dict(valid=True,states=7,runs=49,results=results),f,indent=2)
