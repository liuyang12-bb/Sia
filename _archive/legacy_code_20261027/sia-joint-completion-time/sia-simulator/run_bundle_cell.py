"""One frozen JOINT-G cell; same inputs/execution/window, no parameter search."""
import argparse,gzip,json,sys,time,traceback,hashlib
from pathlib import Path
import pandas as pd
import psutil
from joint_prototype.full160 import make_cluster
from joint_prototype.bundle_adaptation import install
from run_full160_cell import FullMetrics
from run_small_mechanism import save,default

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--size',type=int,choices=[8,20],required=True);ap.add_argument('--response',choices=['neutral','medium'],required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p=json.loads(a.manifest.read_text())
    memory=psutil.virtual_memory().available/1024**3
    if memory<2:raise RuntimeError(f'MEMORY_GATE_2GiB: {memory:.6f}')
    if psutil.disk_usage(str(a.output.parent)).free<5*1024**3:raise RuntimeError('DISK_GATE_5GiB')
    for path,h in p['hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()!=h:raise RuntimeError('FROZEN_HASH_CHANGED:'+path)
    a.output.mkdir(exist_ok=False);start=time.perf_counter()
    c=install(make_cluster(json.loads(Path(p['config']).read_text()),pd.read_csv(p['workloads'][str(a.size)]),
        p['registry'],p['models'],'joint_o',a.response,'main',True));m=FullMetrics(c)
    save(a.output/'start.json',dict(command=sys.argv,manifest_hash=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        memory_gib=memory,actual_delay_costs=c.policy.liveness.delay_costs))
    try:
        with gzip.open(a.output/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
            for i in range(p['max_rounds']):
                if c.all_complete():break
                tick=time.perf_counter();c.step(60);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-tick
                r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,
                    wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
                f.write(json.dumps(r,default=default)+'\n');f.flush();m.consume(c,r)
                c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
                if (i+1)%50==0:print('PROGRESS',i+1,sum(j.completion_time is not None for j in c.jobs),flush=True)
        result=m.summary(c,dict(rmst_tau_seconds=p['max_rounds']*60))
        result.update(method='joint_g',response=a.response,size=a.size,total_wall_seconds=time.perf_counter()-start)
        save(a.output/'summary.json',result);print('RESULT',result['completed'],result['mean_jct'],flush=True)
    except Exception as exc:
        save(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=c.current_time,
            last_decision=c.policy.last_decision,solver_diagnostics=getattr(exc,'solver_diagnostics',None)));raise
if __name__=='__main__':main()
