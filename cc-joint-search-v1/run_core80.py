"""One preregistered full-work cell. Independent process, streamed raw evidence."""
import argparse,collections,dataclasses,gzip,hashlib,importlib.metadata,json,math,os,platform,sys,time,traceback
from pathlib import Path
import numpy as np
import psutil
from core80 import ROOT,build,GPUS
from run_stage2c_cell import Accumulator,independent_rho
from joint_prototype.execution_audit import audit_execution

def encode(v):
 if dataclasses.is_dataclass(v):return dataclasses.asdict(v)
 if isinstance(v,set):return sorted(v)
 if hasattr(v,'tolist'):return v.tolist()
 if hasattr(v,'item'):return v.item()
 raise TypeError(type(v).__name__)
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,default=encode)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def gate(path):
 d=Path(path)
 while not d.exists():d=d.parent
 r=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(d)).free/2**30)
 if r['memory_gib']<2 or r['disk_gib']<5:raise RuntimeError('SAFETY_GATE '+str(r))
 return r

class Metrics(Accumulator):
 def __init__(self,c):
  super().__init__(c);self.clip=collections.Counter();self.selected_count=0;self.fallback=collections.Counter();self.lock_checks=0
  self.waiting_rounds=0;self.max_queue=0;self.co_resident_opportunities=0
 def consume(self,c,r):
  d=r['decision_for_next_interval'];res=d['result'];fresh=d['round']!=getattr(self,'last_decision_round',None)
  if fresh:
   self.last_decision_round=d['round'];assert res['audit']['valid'];self.status[res['status']]+=1;self.solve.append(res['solve_seconds'])
   self.events.update(e['event'] for e in d.get('liveness_events',[]));self.max_queue=max(self.max_queue,len(res['unallocated_jobs']))
   self.waiting_rounds+=bool(res['unallocated_jobs'])
   if res['status'].startswith('FALLBACK'):self.fallback[res['status']]+=1
   specs={s['job_id']:s for s in d['probe']['specs']};totals={t+':0':[0.,0.,0.] for t in c.ngpus_per_node}
   for j,col in res['selected'].items():
    cfg=col['config'];rho=independent_rho(c.policy.oracle,j,cfg['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu'])
    assert math.isclose(rho,col['rho'],rel_tol=1e-12);self.selected_count+=1
    if c.policy.oracle.response_exponent==0:assert rho==1.
    ref=c.policy.oracle.reference(j,cfg['config_id'])
    assert (cfg['cpu_demand_scale'],cfg['memory_demand_scale'])==(ref['phi_c'],ref['phi_m'])
    for label,x,lo,hi in [('num_cpu',col['source_coordinates']['requested']['cpu'],1,24),('num_ram',col['source_coordinates']['requested']['ram_gb'],20.83,250),('den_cpu',col['baseline_source_coordinates']['requested']['cpu'],1,24),('den_ram',col['baseline_source_coordinates']['requested']['ram_gb'],20.83,250)]:self.clip[label]+=not lo<=x<=hi
    s=specs[j]
    if s['checkpoint_locked'] and s['current_config_id']:
     assert cfg['config_id']==s['current_config_id'] and col['placement']==dict(s['current_placement'])
     assert (col['cpu_per_gpu'],col['ram_gb_per_gpu'])==(s['current_cpu_per_gpu'],s['current_ram_gb_per_gpu']);self.lock_checks+=1
    for n,g in col['placement'].items():
     for i,x in enumerate((g,g*col['cpu_per_gpu'],g*col['ram_gb_per_gpu'])):totals[n][i]+=x
   for n,used in totals.items():
    typ=n.rsplit(':',1)[0];caps=c.policy.node_capacities[typ]
    assert all(x<=y+1e-7 for x,y in zip(used,(c.ngpus_per_node[typ],caps['cpu_cores'],caps['ram_gb'])))
    self.idle_saturated+=used[1]>=caps['cpu_cores']-1e-7 and used[0]<c.ngpus_per_node[typ]
    for k,x in zip(('gpu','cpu','ram'),used):self.node_peaks[n][k]=max(self.node_peaks[n][k],x)
   if d.get('strict_full_intent'):
    assert c.policy.canonical(c.allocations,c.allocations)==c.policy.canonical(d['original_sia_allocations'],c.allocations)
  for e in r['execution']:assert audit_execution(e)['valid']
  super().consume(c,r)
 def summary(self,c):
  r=super().summary(c,dict(rmst_tau_seconds=360000))
  for j in r['jobs'].values():
   j['first_wait']=(j['first_allocation']-j['arrival'] if j['first_allocation'] is not None else j['censor_time'])
   j['later_wait']=j['wait']-j['first_wait'];assert j['later_wait']>=-1e-7
  r.update(clipping_counts=dict(self.clip),selected_columns=self.selected_count,checkpoint_lock_checks=self.lock_checks,
      fallback_status=dict(self.fallback),fallback_count=sum(self.fallback.values()),waiting_decision_rounds=self.waiting_rounds,max_waiting_jobs=self.max_queue)
  return r

def compact_decision(d):
 # Keep raw selected queries/execution, state/specs, constraints and service
 # ledgers. Candidate scores are a reproducible catalog digest/count, not1M
 # duplicate dictionaries per round. No decision consumes this log reduction.
 remove=('continuity_scores','completion_scores','net_progress_scores','factor_candidate_keys')
 for k in remove:
  if k in d:
   seq=d.pop(k);d[k+'_count']=len(seq)
   h=hashlib.sha256()
   for value in seq:h.update(json.dumps(value,default=encode,sort_keys=True,separators=(',',':')).encode())
   d[k+'_sha256']=h.hexdigest().upper()
 if 'probe' in d:
  for k in ('joint_columns','proportional_columns','floor_rejected'):
   seq=d['probe'].pop(k,None)
   if seq is not None:d['probe'][k+'_count']=len(seq)
 return d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--cell',required=True);a=ap.parse_args()
 p=json.loads(Path(a.manifest).read_text(encoding='utf8'));cell=next(x for x in p['cells'] if x['id']==a.cell)
 for package,version in p['environment']['packages'].items():
  if importlib.metadata.version(package)!=version:raise RuntimeError('DEPENDENCY_CHANGED '+package)
 for path,h in p['source_hashes'].items():
  if sha(ROOT/path)!=h:raise RuntimeError('SOURCE_HASH '+path)
 for path,h in p['input_hashes'].items():
  if sha(path)!=h:raise RuntimeError('INPUT_HASH '+path)
 out=Path(p['output'])/'cells'/a.cell;resources=gate(out);out.mkdir(exist_ok=False)
 start=time.time();save(out/'start.json',dict(cell=cell,command=sys.argv,source_manifest_sha256=sha(a.manifest),resources=resources,start_unix=start,pid=os.getpid(),python=sys.version))
 c=None;m=None
 try:
  c=build(cell['method'],cell['response'],cell['scenario']);m=Metrics(c)
  with gzip.open(out/'rounds.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
   for i in range(6000):
    if c.all_complete():break
    t=time.perf_counter();c.step(60);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-t
    r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time,progress=float(j.progress),epoch=int(j.epoch)) for j in c.jobs}
    compact_decision(r['decision_for_next_interval'])
    f.write(json.dumps(r,default=encode,separators=(',',':'))+'\n');f.flush();m.consume(c,r)
    if (i+1)%10==0 or i<3:
     save(out/f'progress_{i+1:04}.json',dict(round=i+1,simulated_seconds=c.current_time,completed=sum(j.completion_time is not None for j in c.jobs),round_wall_seconds=r['round_wall_seconds'],elapsed_wall_seconds=time.time()-start))
     print('PROGRESS',a.cell,i+1,sum(j.completion_time is not None for j in c.jobs),time.time()-start,flush=True)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
    for name in ('exchange_columns','exchange_scores','continuity_scores','completion_scores','actual_completion_columns'):
     if hasattr(c.policy,name):setattr(c.policy,name,[])
  s=m.summary(c);s.update(cell=cell,termination='ALL_COMPLETE' if c.all_complete() else 'ROUND_LIMIT',total_wall_seconds=time.time()-start)
  save(out/'summary.json',s);save(out/'output_hashes.json',{x.name:sha(x) for x in out.iterdir() if x.is_file()});print('RESULT',a.cell,s['termination'],s['completed'],s['mean_jct'],flush=True)
 except Exception as e:
  save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),rounds=m.rounds if m else 0,time=c.current_time if c else None,last_decision=compact_decision(c.policy.last_decision) if c else None))
  if getattr(e,'solver_diagnostics',None) is not None:
   with gzip.open(out/'failure_solver_diagnostics.json.gz','xt',encoding='utf8') as f:json.dump(e.solver_diagnostics,f,default=encode)
  raise

if __name__=='__main__':main()
