"""Conditional resource-envelope proof, not a rollout or a new simulation.

Reads the first existing FO state with two AWS residents and usable 1/2-GPU
predictions. Keeps their observed predictions/work. Enumerates only a static
AWS subproblem under the proposed capacity; other GPU jobs are fixed witnesses.
This is NOT proof that the six-job online controller naturally chooses them.
"""
import csv,gzip,hashlib,json,sys,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.neutral_liveness import FactorScheduler
from residual_capacity import resource_points
from sia import SiaPolicy
from joint_prototype.physical_domain import restrict_single_node_domain

def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
cfg=json.loads((O/'inputs/proposed_cluster.json').read_text());cc=cfg['cluster'];rc=cfg['response']
policy=SiaPolicy(p_fairness=-.5,restart_penalty=30,lambda_a=.01,lambda_n=1.1,project_throughputs=False,share_max_replicas=False,sched_interval=60,enable_bsz_tuning=True)
policy.populate_valid_configs(cc['nnodes'],cc['ngpus_per_node'],cc['max_physical_nodes'])
restrict_single_node_domain(policy)
domain=[(str(t),int(n),int(g)) for t,n,g in policy.idx_to_config_map.values()]
assert set(domain)=={(t,1,g) for t in ('aws','rtx') for g in (1,2,4)}
path=O.parent/'cc_recovery_equivalent_scenario_20261101_run1/cells/balanced_medium_FO/rounds.jsonl.gz'
state=None;observed={}
for ln,line in enumerate(gzip.open(path,'rt',encoding='utf8'),1):
 row=json.loads(line);d=row['decision_for_next_interval'];scores=d['continuity_scores'];sel=d['result']['selected']
 candidates={}
 for name,c in sel.items():
  if c['config']['gpu_type']!='aws':continue
  configs={}
  for score in scores:
   if score['job']!=name or score['raw_rate'] is None:continue
   cid=score['column_id'].split('|')[1]
   if cid not in ('aws:1:1','aws:1:2'):continue
   entry={k:score[k] for k in ('raw_rate','remaining_work','delay','atomic_bsz','accum_steps','prediction_source')}
   if cid in configs:assert configs[cid]==entry
   configs[cid]=entry
  if len(configs)==2:candidates[name]=configs
 if len(candidates)>=2:
  names=sorted(candidates,key=lambda n:next(x['source_row_index'] for x in d['probe']['specs'] if x['job_id']==n))[:2]
  observed={n:candidates[n] for n in names};state=dict(path=str(path),line=ln,time=row['interval_end'],selection_rule='first logged FO state with two AWS residents and known positive AWS1/AWS2 predictions; source row order',jobs=observed);break
assert state is not None
save('observed_state.json',state)
nodes=[NodeSpec(t+':0',t,cc['ngpus_per_node'][t],v['cpu_cores'],v['ram_gb']) for t,v in cc['capacity_scenarios']['main'].items()]
oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{j:'cifar10' for j in observed},'medium')
jobs=[]
for name in observed:
 configs=[]
 for t,n,g in domain:
  cid=f'{t}:{n}:{g}';r=oracle.reference(name,cid)
  # Unit placeholder affects ONLY candidate construction, never a training
  # input/decision score. Actual static scores below use recorded raw rates.
  configs.append(GpuConfig(cid,t,g,n,1.,r['phi_c'],r['phi_m']))
 jobs.append(JobSpec(name,'alexnet_1.py',tuple(configs)))
s=FactorScheduler(nodes,jobs,oracle,rc['aux_cpu_points'],rc['aux_ram_points'],verify_exact=False,minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,candidate_policy='b3')
s.enforce_floor=False;s.resource_point_provider=resource_points
columns=s.build_joint_columns();coverage={}
for c in columns:
 if c.job_id!=jobs[0].job_id:continue
 v=coverage.setdefault(c.config.config_id,dict(columns=0,numerator_clipped=0,denominator_clipped=0,proportional=[]))
 v['columns']+=1;v['numerator_clipped']+=int(c.query.source_coordinate_clipped);v['denominator_clipped']+=int(c.query.baseline_coordinate_clipped)
 if abs(c.query.cpu_per_gpu-c.proportional_cpu_per_gpu)<1e-8 and abs(c.query.ram_gb_per_gpu-c.proportional_ram_per_gpu)<1e-8:v['proportional'].append(asdict(c.query))
assert len(coverage)==6
save('coverage.json',dict(domain=domain,coverage=coverage,missing=[],warning='raw=1 placeholder only for offline legal-domain enumeration; new AWS4/RTX4 online learned rates not inferred from R0'))

def pool(name,neutral=False):
 result=[]
 for c in columns:
  if c.job_id!=name or c.config.config_id not in observed[name]:continue
  x=observed[name][c.config.config_id];rho=1. if neutral else c.query.rho
  eta=x['delay']+x['remaining_work']/(x['raw_rate']*rho)
  result.append(dict(config=c.config.config_id,g=c.config.gpus,cpu=c.query.cpu_per_gpu,ram=c.query.ram_gb_per_gpu,rho=rho,
   raw=x['raw_rate'],delay=x['delay'],eta=eta,u=1./eta,clipped=c.query.source_coordinate_clipped or c.query.baseline_coordinate_clipped,
   prop=abs(c.query.cpu_per_gpu-c.proportional_cpu_per_gpu)<1e-8 and abs(c.query.ram_gb_per_gpu-c.proportional_ram_per_gpu)<1e-8))
 return result

def exact(a,b):
 # Exhaustive binary-product enumeration using vectorized feasibility.
 best=None;feasible=0;bt=np.array([[x['g'],x['g']*x['cpu'],x['g']*x['ram']] for x in b]);bu=np.array([x['u'] for x in b])
 for x in a:
  left=np.array([4-x['g'],18-x['g']*x['cpu'],125-x['g']*x['ram']])
  ids=np.flatnonzero(np.all(bt<=left+1e-8,axis=1));feasible+=len(ids)
  if len(ids)==0:continue
  k=ids[np.argmax(bu[ids])];y=b[int(k)];value=x['u']+y['u']
  if best is None or value>best['objective']+1e-15:
   best=dict(objective=value,sum_eta=x['eta']+y['eta'],selected=[x,y],usage=dict(gpu=x['g']+y['g'],cpu=x['g']*x['cpu']+y['g']*y['cpu'],ram=x['g']*x['ram']+y['g']*y['ram']))
 assert best is not None
 return {**best,'feasible_pairs':int(feasible)}
cases={}
for response in ('medium','neutral'):
 a,b=[pool(n,response=='neutral') for n in observed]
 proposal=exact([x for x in a if x['prop']],[x for x in b if x['prop']])
 fixed=[x['config'] for x in proposal['selected']]
 fo=exact([x for x in a if x['config']==fixed[0]],[x for x in b if x['config']==fixed[1]])
 full=exact(a,b)
 supported=exact([x for x in a if not x['clipped']],[x for x in b if not x['clipped']])
 cases[response]=dict(proportional_gpu_proposal=proposal,best_fixed_gpu_fo=fo,best_joint_full=full,best_joint_supported_witness=supported,
  relative_u_improvement=full['objective']/fo['objective']-1,sum_eta_improvement=1-full['sum_eta']/fo['sum_eta'])
save('conditional_envelope.json',dict(scope='two observed AWS-job resource subproblem; proposed AWS capacity4GPU18CPU125RAM; four RTX peer witnesses fixed at1GPU3CPU93.75GB; no full six-job/controller/cold-start/eligibility proof',
 jobs=list(observed),observed_state=state,cases=cases,method='exhaustive every current-grid feasible pair, including unequal CPU/RAM and E-legacy clipped FO candidates',
 exclusions=['no four-GPU point fits with second mandatory AWS task; not silently removed from global domain','no rollout, no Job.step, no actual JCT prediction','current source state transplanted into proposed resources; full real-decision gate still required']))
print(json.dumps({k:dict(fo_gpu=[x['config'] for x in v['best_fixed_gpu_fo']['selected']],fo=[(x['cpu'],x['ram']) for x in v['best_fixed_gpu_fo']['selected']],full_gpu=[x['config'] for x in v['best_joint_full']['selected']],full=[(x['cpu'],x['ram']) for x in v['best_joint_full']['selected']],u_improvement=v['relative_u_improvement'],sum_eta_improvement=v['sum_eta_improvement']) for k,v in cases.items()},indent=2))
