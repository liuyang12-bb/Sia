"""Offline only. No Cluster, Job.step or scheduler decision is executed."""
import collections,csv,hashlib,itertools,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run1';SRC=ROOT/'cc-joint-clean-ab'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
 from breakpoint_study import resource_points
 from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
 from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
 from joint_prototype.joint_scheduler import Placement
 from joint_prototype.neutral_liveness import FactorScheduler
 from resource_study import ORIGINAL_CPU,ORIGINAL_RAM
 from run_stage2c_cell import independent_rho
 import pandas as pd
 workloads={n:pd.read_csv(OUT/'inputs'/f'{n}_workload.csv') for n in ('A20','COMPACT','SPREAD')}
 apps={a:a for w in workloads.values() for a in w.application};o=R0Oracle(SRC/'inputs/registry.json',SRC/'inputs/models',apps,'medium')
 rows=[];points=[];missing=[];refs=[];examples=[];pool={};sources={}
 for app in apps:
  for typ,g in itertools.product(('aws','dgx-ext'),(1,2)):
   cid=f'{typ}:1:{g}';r=o.reference(app,cid)
   refs.append(dict(application=app,config=cid,template=r['template'],phi_c=r['phi_c'],phi_m=r['phi_m'],reference_state=r['reference_state'],source_rows=r['source_rows']))
   for source in r['source_rows']:
    legacy=Path(source['file']);p=SRC/'sia-simulator/traces'/app/legacy.name
    sources[str(p)]=dict(exists=p.exists(),expected=source['sha256'],sha=sha(p) if p.exists() else None,legacy_path_exists=legacy.exists())
    if not p.exists() or sha(p)!=source['sha256']:missing.append(dict(app=app,config=cid,source=str(p),reason='SOURCE_ROW_HASH_UNRESOLVED'))
 for cap,(cpu,ram) in {'A':(6.,187.5),'B':(6.,125.)}.items():
  for app in apps:
   for typ,g in itertools.product(('aws','dgx-ext'),(1,2)):
    cid=f'{typ}:1:{g}';ref=o.reference(app,cid);cfg=GpuConfig(cid,typ,g,1,1.,ref['phi_c'],ref['phi_m'])
    j=JobSpec(app,APPLICATION_TEMPLATES[app],(cfg,),0.)
    s=FactorScheduler([NodeSpec(t+':0',t,2,cpu,ram) for t in ('aws','dgx-ext')],[j],o,ORIGINAL_CPU,ORIGINAL_RAM,verify_exact=False,
        minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,enforce_proportional_progress_floor=False)
    dims=resource_points(s,j,cfg,Placement(((typ+':0',g),)))
    base=o.query(j,cfg,cpu/2,ram/2,(cpu/2,ram/2))
    for method in ('P','CPU','MEM','FULL'):
     cp=(cpu/2,) if method in ('P','MEM') else sorted(set(dims[0])|{cpu/2})
     mm=(ram/2,) if method in ('P','CPU') else sorted(set(dims[1])|{ram/2})
     counts=collections.Counter();valid=[]
     for c,m in itertools.product(cp,mm):
      if c*g>cpu+1e-9 or m*g>ram+1e-9:continue
      q=o.query(j,cfg,c,m,(cpu/2,ram/2));assert math.isclose(q.rho,independent_rho(o,app,cid,c,m),rel_tol=1e-12)
      flags={}
      for tag,x,y in [('num',q.requested_source_cpu_coordinate,q.requested_source_ram_coordinate),('den',q.requested_baseline_source_cpu_coordinate,q.requested_baseline_source_ram_coordinate)]:
       flags[tag+'_cpu']=not 1.-1e-12<=x<=24.+1e-12;flags[tag+'_ram']=not 20.83-1e-12<=y<=250.+1e-12
      supported=not any(flags.values());counts['feasible']+=1;counts['supported']+=supported
      for k,v in flags.items():counts[k+'_out']+=v
      counts['faster_than_proportional']+=q.rho>base.rho+1e-10
      counts['slower_than_proportional']+=q.rho<base.rho-1e-10
      point=dict(capacity=cap,app=app,config=cid,method=method,cpu=c,ram=m,rho=q.rho,proportional_rho=base.rho,supported=supported,**flags)
      points.append(point);valid.append(point)
     if not valid:missing.append(dict(cap=cap,app=app,config=cid,method=method,reason='NO_PHYSICALLY_FEASIBLE_POINT'))
     rows.append(dict(capacity=cap,application=app,config=cid,method=method,**counts,
        maximum_rho=max((x['rho'] for x in valid),default=None),proportional_rho=base.rho,
        maximum_supported_rho=max((x['rho'] for x in valid if x['supported']),default=None)))
     if g==1 and method=='FULL':pool[cap,typ,app]=valid
 for (cap,typ,app),receiver in pool.items():
  cpu,ram={'A':(6.,187.5),'B':(6.,125.)}[cap]
  for peer in apps:
   if peer==app:continue
   donor=pool[cap,typ,peer]
   for strict in (True,False):
    best=None
    for a in receiver:
     if strict and not a['supported']:continue
     if a['rho']<=a['proportional_rho']+1e-10:continue
     for b in donor:
      if strict and not b['supported']:continue
      if a['cpu']+b['cpu']>cpu+1e-9 or a['ram']+b['ram']>ram+1e-9:continue
      delta=(a['rho']/a['proportional_rho']-1)+(b['rho']/b['proportional_rho']-1)
      if delta<=1e-10:continue
      if best is None or delta>best['sum_relative_rate_change']:
       best=dict(capacity=cap,gpu=typ,receiver=app,donor=peer,supported=strict,
          receiver_point=a,donor_point=b,sum_relative_rate_change=delta,
          donor_loss=max(0.,1-b['rho']/b['proportional_rho']))
    examples.append(best or dict(capacity=cap,gpu=typ,receiver=app,donor=peer,supported=strict,no_net_normalized_rate_gain_witness=True))
 matrices={}
 for app in apps:
  t=o.template(APPLICATION_TEMPLATES[app]);a=t.matrix
  matrices[app]=dict(template=t.template_id,sha=t.source_sha256,min=float(a.min()),max=float(a.max()),cpu_axis=[1,2,3,4,5,6,9,12,24],ram_axis=[20.83,62.5,125,187.5,250],
    cpu_decreasing_cells=int(np.sum(np.diff(a,axis=0)<-1e-12)),ram_decreasing_cells=int(np.sum(np.diff(a,axis=1)<-1e-12)),
    completely_flat=bool(np.max(a)-np.min(a)<1e-12),measured_on_target_hardware=False)
 save('C_coverage.json',dict(scope='OFFLINE_FULL_TOPOLOGY_APP_CONFIG_DOMAIN_NOT_SELECTED_TRAJECTORIES',rows=rows,missing=missing,
    matrices=matrices,source_resolution=sources,registry_references=refs,
    boundary='UNCHANGED_E_LEGACY; supported uses1e-12 endpoint analysis tolerance ONLY, no execution change',
    inference='rho response-only; dummy GPU rate1 is not a performance estimate or training input'))
 save('C_pair_witnesses.json',dict(witnesses=examples,warning='Static sum of normalized rates, NOT total training work/s, JCT optimality or actual co-residence; nonflat donor losses retained'))
 with (OUT/'C_points.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
 print('C_OFFLINE_COMPLETE',len(rows),'ROWS',len(points),'POINTS','MISSING',len(missing),flush=True)
if __name__=='__main__':main()
