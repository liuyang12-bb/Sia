"""One frozen scale cell; independent process, no historical output writes."""
import argparse, gzip, hashlib, json, sys, time, traceback
from pathlib import Path
import pandas as pd
import psutil
from joint_prototype.full160 import make_cluster
from joint_prototype.small_mechanism import ProbeSia, ProbeAux
from joint_prototype.scaled_adaptive import install
from run_small_mechanism import save, default
from run_full160_cell import FullMetrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--size',type=int,choices=[8,20],required=True)
    ap.add_argument('--response',choices=['neutral','medium'],required=True)
    ap.add_argument('--method',choices=['sia_p','sia_aux','joint_r'],required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    p=json.loads(a.manifest.read_text(encoding='utf-8'))
    if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('MEMORY_GATE_2GiB')
    if psutil.disk_usage(str(a.output.parent)).free<5*1024**3:raise RuntimeError('DISK_GATE_5GiB')
    for path,h in p['hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()!=h:raise RuntimeError('FROZEN_HASH_CHANGED:'+path)
    a.output.mkdir(exist_ok=False);start=time.perf_counter()
    wl=pd.read_csv(p['workloads'][str(a.size)])
    c=make_cluster(json.loads(Path(p['config']).read_text()),wl,p['registry'],p['models'],
                   'joint_o' if a.method=='joint_r' else 'sia_p',a.response,'main',True)
    if a.method=='joint_r':install(c)
    else:
        c.policy.__class__=ProbeSia if a.method=='sia_p' else ProbeAux
        if a.method=='sia_aux':c.policy.mode='b4'
    m=FullMetrics(c)
    save(a.output/'start.json',dict(command=sys.argv,memory_gib=psutil.virtual_memory().available/1024**3,
        manifest_hash=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        actual_delay_costs=getattr(getattr(c.policy,'liveness',None),'delay_costs',None)))
    try:
        with gzip.open(a.output/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
            for i in range(p['max_rounds']):
                if c.all_complete():break
                tick=time.perf_counter();c.step(60);r=c.joint_round_logs[-1]
                r['round_wall_seconds']=time.perf_counter()-tick
                r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,
                    wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
                f.write(json.dumps(r,default=default)+'\n');f.flush();m.consume(c,r)
                c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
                if (i+1)%50==0:print('PROGRESS',i+1,sum(j.completion_time is not None for j in c.jobs),flush=True)
        result=m.summary(c,dict(rmst_tau_seconds=p['max_rounds']*60))
        result.update(method=a.method,response=a.response,size=a.size,total_wall_seconds=time.perf_counter()-start)
        save(a.output/'summary.json',result)
        print('RESULT',result['completed'],result['mean_jct'],flush=True)
    except Exception as e:
        save(a.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time,
            last_decision=c.policy.last_decision));raise
if __name__=='__main__':main()
