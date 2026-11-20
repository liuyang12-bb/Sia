"""Offline response-only census and deterministic witnesses. No Job/Cluster."""
import csv,hashlib,itertools,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1'
SRC=OUT/'source';sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from breakpoint_study import resource_points
from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.joint_scheduler import Placement
from joint_prototype.neutral_liveness import FactorScheduler
from resource_study import ORIGINAL_CPU,ORIGINAL_RAM
from applications import APPLICATIONS
from simulator_config import SLOWDOWNS

def save(name,x):
 with (OUT/name).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def supported(q):return not(q.source_coordinate_clipped or q.baseline_coordinate_clipped)
def pack(q):return dict(cpu=q.cpu_per_gpu,ram=q.ram_gb_per_gpu,rho=q.rho,supported=supported(q),
 source_cpu=q.requested_source_cpu_coordinate,source_ram=q.requested_source_ram_coordinate,
 denominator_cpu=q.requested_baseline_source_cpu_coordinate,denominator_ram=q.requested_baseline_source_ram_coordinate)
def raw_profile(row,typ,g,phase):
 a=APPLICATIONS[typ][row['application']];b=int(row['batch_size']);local=math.ceil(b/g-1e-8)
 acc=math.ceil(local/a.get_max_local_bsz((g,))-1e-8)-1
 if g==1 and b>a.init_batch_size:acc=max(1,acc)
 atomic=min(math.ceil(local/(acc+1)-1e-8),int(a.max_batch_size/(g*(acc+1))))
 actual=g*atomic*(acc+1);epoch=min(a.max_epochs-1,int(a.max_epochs*phase))
 sq,var=a.get_grad_stats(actual,epoch);scale=actual/a.init_batch_size
 gain=(var+sq)/(var/scale+sq);step,sync=a.get_throughput((g,),atomic)
 rate=float(gain/(step+(step-sync)*acc)/SLOWDOWNS[row['application']])
 return dict(raw=rate,work=float(a.get_progress(a.max_epochs)),epoch=epoch,atomic_batch=atomic,
             accumulation=acc,declared_batch=b,phase=phase)

def main():
 t=time.time();cohorts={m:list(csv.DictReader((OUT/'inputs'/f'{m}_workload.csv').open(encoding='utf8'))) for m in ['M1','M2','M3']}
 unique={r['name']:r for rows in cohorts.values() for r in rows};apps={n:r['application'] for n,r in unique.items()}
 oracle=R0Oracle(SRC/'inputs/registry.json',SRC/'inputs/models',apps,'medium')
 specs={};profiles={};coverage=[];options={};prop={};point_cache={}
 for n,r in unique.items():
  cfgs=[]
  for typ,g in itertools.product(['aws','dgx-ext'],[1,2]):
   cid=f'{typ}:1:{g}';ref=oracle.reference(n,cid)
   cfgs.append(GpuConfig(cid,typ,g,1,1.,ref['phi_c'],ref['phi_m']))
   profiles[n,cid]=[raw_profile(r,typ,g,p) for p in [0.,.25,.75]]
  specs[n]=JobSpec(n,APPLICATION_TEMPLATES[r['application']],tuple(cfgs),float(r['time']))
 for cap,cpu in [('R11',6.),('R01',12.)]:
  for n,j in specs.items():
   for cfg in j.configs:
    node=NodeSpec(f'{cfg.gpu_type}:0',cfg.gpu_type,2,cpu,187.5)
    nodes=[NodeSpec(f'{typ}:0',typ,2,cpu,187.5) for typ in ['aws','dgx-ext']]
    s=FactorScheduler(nodes,[j],oracle,ORIGINAL_CPU,ORIGINAL_RAM,verify_exact=False,
       minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,enforce_proportional_progress_floor=False)
    placement=Placement(((node.node_id,cfg.gpus),));bp=oracle.query(j,cfg,cpu/2,93.75,(cpu/2,93.75));prop[cap,n,cfg.config_id]=bp
    dims=resource_points(s,j,cfg,placement);counts={}
    for factor in ['CPU','MEM','FULL']:
     cp=dims[0] if factor!='MEM' else (cpu/2,)
     mm=dims[1] if factor!='CPU' else (93.75,)
     pairs=[(c,m) for c,m in itertools.product(sorted(set(cp+(cpu/2,))),sorted(set(mm+(93.75,)))) if c*cfg.gpus<=cpu+1e-9 and m*cfg.gpus<=187.5+1e-9]
     cachekey=(j.template_file,cfg.config_id,cap,factor)
     if cachekey not in point_cache:point_cache[cachekey]=[oracle.query(j,cfg,c,m,(cpu/2,93.75)) for c,m in pairs]
     qs=point_cache[cachekey];options[cap,n,cfg.config_id,factor]=qs
     counts[factor]=dict(feasible=len(qs),strict_supported=sum(supported(q) for q in qs),numerator_clipped=sum(q.source_coordinate_clipped for q in qs),denominator_clipped=sum(q.baseline_coordinate_clipped for q in qs))
    coverage.append(dict(capacity=cap,job=n,application=apps[n],config=cfg.config_id,proportional=pack(bp),counts=counts,profiles=profiles[n,cfg.config_id]))
 save('offline_coverage.json',coverage)
 # Whole input source order, fixed capacity/type priority; never select by JCT.
 ordered=sorted(unique.values(),key=lambda r:int(r['original_source_row_index']))
 witnesses={};census=[]
 for factor in ['CPU','MEM']:
  for ra,rb in itertools.combinations(ordered,2):
   if ra['application']==rb['application']:continue
   na,nb=ra['name'],rb['name']
   for typ in ['aws','dgx-ext']:
    cid=f'{typ}:1:1'
    for cap,cpu in [('R11',6.),('R01',12.)]:
     base=[prop[cap,n,cid] for n in [na,nb]]
     if not all(map(supported,base)):continue
     counts=0;first=None;best=None
     qa_list=[q for q in options[cap,na,cid,factor] if supported(q)]
     qb_list=[q for q in options[cap,nb,cid,factor] if supported(q)]
     for qa,qb in itertools.product(qa_list,qb_list):
      if qa.cpu_per_gpu+qb.cpu_per_gpu>cpu+1e-9 or qa.ram_gb_per_gpu+qb.ram_gb_per_gpu>187.5+1e-9:continue
      phase=[]
      for k,p in enumerate([0.,.25,.75]):
       v=[profiles[n,cid][k] for n in [na,nb]]
       deltas=[v0['work']*(1-p)/v0['raw']*(1/q.rho-1/b.rho) for v0,q,b in zip(v,[qa,qb],base)]
       rate=sum(v0['raw']/v0['work']*(q.rho-b.rho) for v0,q,b in zip(v,[qa,qb],base))
       phase.append(dict(phase=p,individual_eta_delta=deltas,total_eta_delta=sum(deltas),net_fraction_rate=rate))
      if not all(x['total_eta_delta'] < -1e-7 and x['net_fraction_rate']>1e-12 for x in phase):continue
      counts+=1
      record=dict(factor=factor,jobs=[na,nb],capacity=cap,gpu=typ,config=cid,proportional=list(map(pack,base)),alternative=list(map(pack,[qa,qb])),phases=phase,source='OFFLINE_FIXED_DECLARED_BATCH_EPOCH_SNAPSHOTS_NOT_INJECTED_INTO_ONLINE_POLICY')
      if first is None:first=record
      if best is None or phase[0]['total_eta_delta']<best['phases'][0]['total_eta_delta']:best=record
     census.append(dict(factor=factor,jobs=[na,nb],capacity=cap,gpu=typ,positive_eta_and_rate_points=counts,first=first))
     if first and factor not in witnesses:witnesses[factor]=first
 save('offline_pair_census.json',census);save('mechanism_witnesses.json',witnesses)
 result=dict(offline_only=True,training_runs=0,scheduler_solves=0,coverage_rows=len(coverage),unique_jobs=len(unique),elapsed_seconds=time.time()-t,
   strict_CPU_and_MEM_opportunities=set(witnesses)=={'CPU','MEM'},missing_GPU_configs=[],response_changed=False,
   parent_inputs={p.name:sha(p) for p in (OUT/'inputs').glob('*.csv')},witnesses=witnesses)
 save('offline_validation.json',result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
