"""One three-job real-Sia integration cell, isolated process and durable logs."""
import argparse,collections,dataclasses,gzip,hashlib,json,math,sys,time,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent
P=json.loads((OUT/'preregistration.json').read_text());sys.path.insert(0,P['tree'])
import pandas as pd
import psutil
from joint_prototype.full160 import make_cluster
from joint_prototype.small_mechanism import ProbeSia,ProbeAux
from joint_prototype.exchange_netprogress import install,window_utility
from joint_prototype.execution_audit import audit_execution
from run_stage2c_cell import Accumulator,independent_rho

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def default(v):
 if dataclasses.is_dataclass(v):return dataclasses.asdict(v)
 if hasattr(v,'item'):return v.item()
 raise TypeError(type(v).__name__)
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,default=default,indent=2)

class Metrics(Accumulator):
 def __init__(self,c):
  super().__init__(c);self.clipping=collections.Counter();self.selected_count=0;self.below_floor=0
 def consume(self,c,r):
  d=r['decision_for_next_interval'];res=d['result'];fresh=d.get('round')!=getattr(self,'last_decision_round',None)
  if fresh:
   self.last_decision_round=d['round']
   assert res['audit']['valid']
   self.status[res['status']]+=1;self.solve.append(res['solve_seconds']);self.events.update(e['event'] for e in d.get('liveness_events',[]))
   totals={t+':0':[0.,0.,0.] for t in c.ngpus_per_node}
   specs={j['job_id']:j for j in d['probe']['specs']}
   for j,col in res['selected'].items():
    cfg=col['config'];typ=cfg['gpu_type'];cap=c.policy.node_capacities[typ];ng=c.ngpus_per_node[typ]
    rho=independent_rho(c.policy.oracle,j,cfg['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu'])
    assert math.isclose(rho,col['rho'],rel_tol=1e-12)
    base=independent_rho(c.policy.oracle,j,cfg['config_id'],cap['cpu_cores']/ng,cap['ram_gb']/ng)
    self.below_floor+=rho+1e-9<base
    if not getattr(c.policy,'allow_exchange',False):assert rho+1e-9>=base
    self.selected_count+=1;rr=c.policy.oracle.reference(j,cfg['config_id'])
    for prefix,cpu,ram in [('numerator',col['cpu_per_gpu'],col['ram_gb_per_gpu']),('denominator',3.,62.5)]:
     self.clipping[prefix+'_cpu']+=not 1<=cpu/rr['phi_c']<=24
     self.clipping[prefix+'_ram']+=not 20.83<=ram/rr['phi_m']<=250
    for node,g in col['placement'].items():
     totals[node][0]+=g;totals[node][1]+=g*col['cpu_per_gpu'];totals[node][2]+=g*col['ram_gb_per_gpu']
    spec=specs[j]
    if spec['checkpoint_locked'] and spec['current_config_id']:
     assert cfg['config_id']==spec['current_config_id']
     assert dict(spec['current_placement'])==col['placement']
     assert spec['current_cpu_per_gpu']==col['cpu_per_gpu'] and spec['current_ram_gb_per_gpu']==col['ram_gb_per_gpu']
   for node,v in totals.items():
    typ=node.rsplit(':',1)[0];cap=c.policy.node_capacities[typ];ng=c.ngpus_per_node[typ]
    assert v[0]<=ng+1e-8 and v[1]<=cap['cpu_cores']+1e-8 and v[2]<=cap['ram_gb']+1e-8
    self.idle_saturated+=v[1]>=cap['cpu_cores']-1e-8 and v[0]<ng
    for key,x in zip(['gpu','cpu','ram'],v):self.node_peaks[node][key]=max(self.node_peaks[node][key],x)
   if getattr(c.policy,'net_progress',False):
    for score in d['net_progress_scores']:
     assert math.isclose(score['utility'],window_utility(score['normalized_rate_before_delay'],score['candidate_delay']),abs_tol=1e-10)
  for e in r['execution']:assert audit_execution(e)['valid']
  # Parent handles only execution integration here. Its old floor-specific
  # decision audit is replaced by the explicit policy-dependent checks above.
  super().consume(c,r)
 def summary(self,c,p):
  v=super().summary(c,p);v.update(clipping_counts=dict(self.clipping),selected_columns=self.selected_count,
       below_proportional_floor_selected=self.below_floor)
  return v

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True);a=ap.parse_args()
 cell=next(c for c in P['runs'] if c['id']==a.cell)
 if psutil.virtual_memory().available<2*2**30:raise RuntimeError('MEMORY_GATE_2GiB')
 if psutil.disk_usage(str(OUT)).free<5*2**30:raise RuntimeError('DISK_GATE_5GiB')
 for path,h in P['input_hashes'].items():assert sha(path)==h,path
 for path,h in P['source_hashes'].items():assert sha(Path(P['tree'])/path)==h,path
 for path,h in P['runner_hashes'].items():assert sha(OUT/path)==h,path
 directory=OUT/'real_cells'/cell['id'];directory.mkdir(parents=True,exist_ok=False)
 save(directory/'start.json',dict(cell=cell,preregistration_sha256=sha(OUT/'preregistration.json'),memory_gib=psutil.virtual_memory().available/2**30))
 start=time.perf_counter();config=json.loads(Path(P['config']).read_text())
 c=make_cluster(config,pd.read_csv(P['workload']),P['registry'],P['models'],
    'sia_p' if cell['method'].startswith('sia') else 'joint_o',cell['response'],'half',True)
 if cell['method']=='sia_p':c.policy.__class__=ProbeSia
 elif cell['method']=='sia_aux':c.policy.__class__=ProbeAux;c.policy.mode='b4'
 else:install(c,cell['method'][1]=='1',cell['method'][2]=='1')
 m=Metrics(c)
 try:
  with gzip.open(directory/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
   for i in range(P['max_rounds']):
    if c.all_complete():break
    c.step(60);r=c.joint_round_logs[-1]
    f.write(json.dumps(r,default=default,separators=(',',':'))+'\n');f.flush();m.consume(c,r)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
  summary=m.summary(c,dict(rmst_tau_seconds=18000));summary.update(cell=cell,total_wall_seconds=time.perf_counter()-start)
  save(directory/'summary.json',summary)
  print('RESULT',cell['id'],summary['completed'],summary['mean_jct'],flush=True)
 except Exception as exc:
  save(directory/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=c.current_time,last_decision=c.policy.last_decision));raise

if __name__=='__main__':main()
