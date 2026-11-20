import gzip,json,sys
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-residual-capacity-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from residual_capacity import resource_points
from breakpoint_study import resource_points as old_points
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.joint_scheduler import JointScheduler
from joint_prototype.r0_liveness import R0Oracle

old=R/'research_runs/sia_synergy/cc_uncertainty_20261101_run1'
rows=[]
for line in gzip.open(old/'cells/NEW_FULL/rounds.jsonl.gz','rt',encoding='utf8'):
 r=json.loads(line)
 if r['interval_end'] not in (180,420,4800,5820):continue
 d=r['decision_for_next_interval'];specs=[]
 for x in d['probe']['specs']:
  z=dict(x);z['configs']=tuple(GpuConfig(**c) for c in z['configs'])
  for k in ('current_nodes','current_placement'):z[k]=tuple(tuple(v) if isinstance(v,list) else v for v in z[k])
  specs.append(JobSpec(**z))
 oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{s.job_id:('cifar10' if s.job_id.startswith('cifar10') else 'bert') for s in specs},'medium')
 nodes=[NodeSpec(t+':0',t,2,6.,125.) for t in ('aws','dgx-ext')]
 def scheduler(provider):
  s=JointScheduler(nodes,specs,oracle,(1.,2.,3.,4.,6.,8.),(20.83,41.66,62.5,125.),enforce_proportional_progress_floor=False,minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83)
  s.resource_point_provider=provider;return s
 before=scheduler(old_points);after=scheduler(resource_points)
 selected=d['result']['selected'];cs=after.build_joint_columns();chosen=[]
 # The real completion scorer also preserves proportional proposals, including
 # tail-recovery points Pareto-pruned from auxiliary-only generation.
 cs += [after.refresh_column(c) for c in after.build_proportional_columns()]
 for j,v in selected.items():
  matches=[c for c in cs if c.job_id==j and c.config.config_id==v['config']['config_id'] and dict(c.placement.node_gpus)==v['placement'] and c.query.cpu_per_gpu==v['cpu_per_gpu'] and c.query.ram_gb_per_gpu==v['ram_gb_per_gpu']]
  assert matches,j;chosen.append(matches[0])
 assert after.audit(chosen)['valid']
 opportunities=[]
 if r['interval_end']==420:
  for j in ('cifar10-0','cifar10-1'):
   current=next(c for c in chosen if c.job_id==j)
   candidate=next(c for c in cs if c.job_id==j and c.config.config_id==current.config.config_id and c.query.cpu_per_gpu==5. and c.query.ram_gb_per_gpu==current.query.ram_gb_per_gpu)
   bundle=[candidate if c.job_id==j else c for c in chosen]
   assert after.audit(bundle)['valid'];assert not before.audit(bundle)['valid']
   opportunities.append(dict(job=j,old_cpu=current.query.cpu_per_gpu,new_cpu=5.,old_rho=current.query.rho,new_rho=candidate.query.rho,speedup=candidate.query.rho/current.query.rho-1,old_grid_rejects=True,new_grid_accepts=True,peers_unchanged=True))
 rows.append(dict(time=r['interval_end'],incumbent_valid=True,locks=sum(s.checkpoint_locked for s in specs),opportunities=opportunities))
with (O/'same_state.json').open('x',encoding='utf8') as f:json.dump(rows,f,indent=2)
print(rows)
