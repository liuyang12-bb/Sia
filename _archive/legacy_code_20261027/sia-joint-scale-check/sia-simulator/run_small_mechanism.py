"""Pre-registered 3-job Neutral/Medium probe, independent process per cell."""
import argparse,copy,gzip,json,time,traceback,dataclasses,hashlib
from pathlib import Path
import pandas as pd
import psutil
from joint_prototype.full160 import make_cluster
from joint_prototype.small_mechanism import ProbeSia,ProbeAux,ProbeJoint
from run_full160_cell import FullMetrics
def default(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,default=default,indent=2)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--method',choices=['sia_p','sia_aux','joint_o'],required=True);ap.add_argument('--response',choices=['neutral','medium'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=json.loads(a.manifest.read_text(encoding='utf-8'))
 if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('MEMORY_GATE_2GiB')
 for path,h in p['hashes'].items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()!=h:raise RuntimeError('FROZEN_INPUT_OR_SOURCE_CHANGED:'+path)
 a.output.mkdir(exist_ok=False);tick=time.perf_counter()
 cfg=json.loads(Path(p['config']).read_text());wl=pd.read_csv(p['workload'])
 c=make_cluster(cfg,wl,p['registry'],p['models'],'joint_o' if a.method=='joint_o' else 'sia_p',a.response,'main',True)
 c.policy.__class__={'sia_p':ProbeSia,'sia_aux':ProbeAux,'joint_o':ProbeJoint}[a.method]
 if a.method=='sia_aux':c.policy.mode='b4'
 m=FullMetrics(c);save(a.output/'start.json',dict(command=__import__('sys').argv,manifest_hash=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),memory_gib=psutil.virtual_memory().available/1024**3))
 try:
  with gzip.open(a.output/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
   for i in range(p['max_rounds']):
    if c.all_complete():break
    t=time.perf_counter();c.step(60);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-t;r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
    f.write(json.dumps(r,default=default)+'\n');f.flush();m.consume(c,r)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
  result=m.summary(c,dict(rmst_tau_seconds=p['max_rounds']*60-max(wl.time)));result['method']=a.method;result['response']=a.response;result['total_wall_seconds']=time.perf_counter()-tick;save(a.output/'summary.json',result)
  print(json.dumps({k:v for k,v in result.items() if k not in ['jobs','node_resource_hours','node_peaks']},default=default))
 except Exception as e:save(a.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time,last_decision=c.policy.last_decision));raise
if __name__=='__main__':main()
