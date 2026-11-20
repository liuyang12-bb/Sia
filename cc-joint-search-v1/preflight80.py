"""Offline full-domain response/competition audit. Never calls Cluster.step."""
import collections,csv,dataclasses,gzip,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from core80 import ROOT,GPUS,build,saturation_per_gpu
from scale_storage import storage_context
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.joint_scheduler import Placement
from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
from joint_prototype.neutral_liveness import FactorScheduler
from breakpoint_study import resource_points

def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
def supported(q):return not(q.source_coordinate_clipped or q.baseline_coordinate_clipped)
def main(out):
 out=Path(out);out.mkdir(exist_ok=True);apps=['cifar10','deepspeech2','bert']
 registry=json.loads((ROOT/'inputs/registry.json').read_text());groups={};coverage=[];counts=collections.Counter();neutral=True;props=True
 oracle=R0Oracle(ROOT/'inputs/registry.json',ROOT/'inputs/models',{a:a for a in apps},'medium')
 nodes=[NodeSpec(t+':0',t,g,3.*g,62.5*g) for t,g in GPUS.items()]
 with storage_context(oracle),gzip.open(out/'full_domain_points.csv.gz','wt',newline='',encoding='utf8') as f:
  fields=['application','config','cpu','ram','rho','supported','numerator_cpu_clip','numerator_ram_clip','denominator_cpu_clip','denominator_ram_clip'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in registry:
   if r['application'] not in apps:continue
   app=r['application'];typ,nn,g=r['config_id'].split(':');g=int(g)
   if int(nn)!=1 or g>GPUS[typ]:continue
   config=GpuConfig(r['config_id'],typ,g,1,1.,r['phi_c'],r['phi_m']);job=JobSpec(app,r['template'],(config,))
   s=FactorScheduler(nodes,[job],oracle,[1.,2.,3.,4.,6.,8.,9.,12.,24.],[20.83,62.5,125.,187.5,250.],verify_exact=False,
      minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,candidate_policy='b3');s.resource_point_provider=resource_points
   pl=Placement(((typ+':0',g),));cp,mp=resource_points(s,job,config,pl)
   points=[];counter=collections.Counter()
   for cpu,ram in itertools.product(sorted(set(cp+(3.,))),sorted(set(mp+(62.5,)))):
    if cpu*g>3*GPUS[typ]+1e-9 or ram*g>62.5*GPUS[typ]+1e-9:continue
    q=oracle.query(job,config,cpu,ram,(3.,62.5));row=dict(application=app,config=config.config_id,cpu=cpu,ram=ram,rho=q.rho,supported=supported(q),
       numerator_cpu_clip=not 1<=q.requested_source_cpu_coordinate<=24,numerator_ram_clip=not 20.83<=q.requested_source_ram_coordinate<=250,
       denominator_cpu_clip=not 1<=q.requested_baseline_source_cpu_coordinate<=24,denominator_ram_clip=not 20.83<=q.requested_baseline_source_ram_coordinate<=250)
    w.writerow(row);points.append(row);counter['points']+=1
    for k in fields[6:]:counter[k]+=row[k]
    counter['fully_supported']+=row['supported']
   groups[app,config.config_id]=points;counts.update(counter)
   p=oracle.query(job,config,3.,62.5,(3.,62.5));neutral &= p.rho==1.;props &= g*3<=3*GPUS[typ] and g*62.5<=62.5*GPUS[typ]
   coverage.append(dict(application=app,config=config.config_id,registry_key=r['key'],template=r['template'],phi_c=r['phi_c'],phi_m=r['phi_m'],proportional_rho=p.rho,
       proportional_supported=supported(p),counts=dict(counter),max_rho=max(x['rho'] for x in points),max_cpu_only_rho=max(x['rho'] for x in points if x['ram']==62.5),
       max_ram_only_rho=max(x['rho'] for x in points if x['cpu']==3.)))
 witnesses=[]
 for typ in GPUS:
  cid=f'{typ}:1:1'
  for a,b in [('cifar10','bert'),('cifar10','deepspeech2')]:
   left=[r for r in groups[a,cid] if r['supported'] and r['cpu']<=5 and r['ram']<=104.17]
   right=[r for r in groups[b,cid] if r['supported'] and r['cpu']<=5 and r['ram']<=104.17]
   best=None
   for x in left:
    for y in right:
     if x['cpu']+y['cpu']>6+1e-9 or x['ram']+y['ram']>125+1e-9:continue
     gain=x['rho']+y['rho']-2
     if best is None or gain>best['sum_relative_rate_gain']:
      best=dict(gpu_type=typ,jobs=[a,b],receiver=x,peer=y,sum_relative_rate_gain=gain,
       both_no_slowdown=min(x['rho'],y['rho'])>=1-1e-9,cpu_sum=x['cpu']+y['cpu'],ram_sum=x['ram']+y['ram'],
       baseline=[3.,62.5,1.],interpretation='Sum of per-job relative speeds, not an absolute cross-application Goodput sum or JCT claim.')
   if best:witnesses.append(best)
 sat,evidence=saturation_per_gpu();save(out/'optional_source_saturation.json',dict(capacities_per_gpu=sat,evidence=evidence))
 save(out/'coverage.json',dict(configurations=coverage,all_points_counts=dict(counts),complete_registry_domain=len(coverage)==33,proportional_all_rho1=neutral,proportional_all_fits=props,
     warning='Raw matrices official; mapping, R0, interpolation/E-legacy and node specifications synthetic. Full domain preserved, clipping disclosed, no extrapolation change.'))
 save(out/'pair_opportunities.json',dict(witnesses=witnesses,proof='Two1GPU jobs use <=6CPU125GB. Repeating two AWS, four DGX and four RTX pairs fills20GPU within60CPU1250GB if supported pair witnesses exist for every type.',
      full20gpu_supported_pareto_possible=all(any(x['gpu_type']==t and x['both_no_slowdown'] and x['sum_relative_rate_gain']>1e-8 for x in witnesses) for t in GPUS)))
 import pandas as pd
 df=pd.read_csv(ROOT/'inputs/core80_workload.csv');first=df[df.time==0]
 save(out/'competition.json',dict(total_jobs=len(df),applications=dict(df.application.value_counts().items()),arrival_counts={str(k):int(v) for k,v in df.time.value_counts().items()},
      first_arrival_jobs=len(first),gpus=20,minimum_gang_gpu_sum_at_start=len(first),minimum_waiters_if_all_jobs_at_least_one_gpu=max(0,len(first)-20),
      demand_sum_first_arrival=int(first.num_replicas.sum()),source_fields_kept=True,training_work_shortened=False,workload_sha256=hashlib.sha256((ROOT/'inputs/core80_workload.csv').read_bytes()).hexdigest(),
      conclusion='Initial40 jobs guarantee contention at20GPU, but sustained waiting and realized co-residency require round logs; static opportunities alone do not establish JCT gain.'))
 print('OFFLINE_PREFLIGHT_COMPLETE',len(coverage),dict(counts),flush=True)

if __name__=='__main__':main(sys.argv[1])
