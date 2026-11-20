"""Offline full physical config-domain response support, no training."""
import json,sys,hashlib
from dataclasses import asdict
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.neutral_liveness import FactorScheduler
from residual_capacity import resource_points
cfg=json.loads((O/'inputs/S4_balanced_gpu.json').read_text());nodes=[NodeSpec(t+':0',t,g,**{}) for t,g in []]
nodes=[NodeSpec(t+':0',t,g,cfg['cluster']['capacity_scenarios']['main'][t]['cpu_cores'],cfg['cluster']['capacity_scenarios']['main'][t]['ram_gb']) for t,g in cfg['cluster']['ngpus_per_node'].items()]
oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{'probe':'cifar10'},'medium');configs=[]
for n in nodes:
 for g in (1,2):
  r=oracle.reference('probe',f'{n.gpu_type}:1:{g}')
  configs.append(GpuConfig(r['config_id'],n.gpu_type,g,1,1.,r['phi_c'],r['phi_m']))
j=JobSpec('probe','alexnet_1.py',tuple(configs));rc=cfg['response']
s=FactorScheduler(nodes,[j],oracle,rc['aux_cpu_points'],rc['aux_ram_points'],verify_exact=False,
 minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,candidate_policy='b3')
s.enforce_floor=False;s.resource_point_provider=resource_points
cs=s.build_joint_columns();rows={}
for c in cs:
 v=rows.setdefault(c.config.config_id,dict(candidates=[],source=oracle.reference('probe',c.config.config_id)))
 v['candidates'].append(dict(cpu=c.query.cpu_per_gpu,ram=c.query.ram_gb_per_gpu,rho=c.query.rho,
  query=asdict(c.query),
  prop=(c.query.cpu_per_gpu==c.proportional_cpu_per_gpu and c.query.ram_gb_per_gpu==c.proportional_ram_per_gpu)))
assert set(rows)=={c.config_id for c in configs}
with (O/'offline_coverage.json').open('x',encoding='utf8') as f:json.dump(dict(configs=rows,raw_goodput='unit placeholder only for response enumeration; not a simulation input',missing=[],label='synthetic transfer'),f,indent=2)
print('OFFLINE coverage',len(rows),'configs',len(cs),'legal columns')
