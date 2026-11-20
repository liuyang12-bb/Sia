"""Shared scenario construction; no future duration supplied to policies."""
import copy,json
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent
def make(method):
 from joint_prototype.full160 import make_cluster
 from joint_prototype.small_mechanism import ProbeSia,ProbeAux
 from joint_prototype.exchange_netprogress import install as old_install
 from joint_prototype.protected_admission import install
 from joint_prototype.physical_domain import restrict_single_node_domain
 p=json.loads((OUT/'scenario.json').read_text(encoding='utf-8'))
 cfg=json.loads(Path(p['base_config']).read_text(encoding='utf-8'))
 cfg['cluster']['ngpus_per_node']=p['gpus']
 c=make_cluster(cfg,pd.read_csv(OUT/'workload.csv'),p['registry'],p['models'],
    'sia_p' if method.startswith('sia') else 'joint_o','medium','half',True)
 c.policy.node_capacities=p['capacities']
 restrict_single_node_domain(c.policy)
 if method=='sia_p':c.policy.__class__=ProbeSia
 elif method=='sia_aux':c.policy.__class__=ProbeAux;c.policy.mode='b4'
 elif method=='G00':old_install(c,False,False)
 elif method=='G11':old_install(c,True,True)
 elif method=='G111':install(c,True,True)
 else:raise ValueError(method)
 return c

def opportunity(columns,caps,gpus):
 """Same job/config/placement alternatives; discrete per-node witnesses."""
 by={}
 for c in columns:by.setdefault((c.job_id,c.config.config_id,c.placement.node_gpus),[]).append(c)
 proportional={}
 for key,values in by.items():
  typ=values[0].config.gpu_type;pc=caps[typ]['cpu_cores']/gpus[typ];pm=caps[typ]['ram_gb']/gpus[typ]
  prop=next((v for v in values if abs(v.query.cpu_per_gpu-pc)<1e-8 and abs(v.query.ram_gb_per_gpu-pm)<1e-8),None)
  if prop:proportional[key]=prop
 flips=[];exchanges=[]
 for ka,a in proportional.items():
  for kb,b in proportional.items():
   if ka[0]==kb[0] and a.config.gpus==b.config.gpus and a.config.gpu_type!=b.config.gpu_type:
    if a.config.sia_goodput>b.config.sia_goodput+1e-8 and a.query.effective_goodput+1e-8<b.query.effective_goodput:
     flips.append((a,b))
   if ka[0]==kb[0] or a.placement.node_ids!=b.placement.node_ids:continue
   typ=a.config.gpu_type
   if a.config.gpus+b.config.gpus>gpus[typ]:continue
   for aa in by[ka]:
    if aa.query.effective_goodput<a.query.effective_goodput*1.05:continue
    for bb in by[kb]:
     if bb.query.effective_goodput+1e-8<b.query.effective_goodput*.9:continue
     cpu=aa.config.gpus*aa.query.cpu_per_gpu+bb.config.gpus*bb.query.cpu_per_gpu
     ram=aa.config.gpus*aa.query.ram_gb_per_gpu+bb.config.gpus*bb.query.ram_gb_per_gpu
     released=(bb.query.cpu_per_gpu<b.query.cpu_per_gpu-1e-8 or bb.query.ram_gb_per_gpu<b.query.ram_gb_per_gpu-1e-8)
     if released and cpu<=caps[typ]['cpu_cores']+1e-8 and ram<=caps[typ]['ram_gb']+1e-8 and aa.query.effective_goodput+bb.query.effective_goodput>a.query.effective_goodput+b.query.effective_goodput+1e-8:
      exchanges.append((a,b,aa,bb));break
    if exchanges:break
 return proportional,flips,exchanges
