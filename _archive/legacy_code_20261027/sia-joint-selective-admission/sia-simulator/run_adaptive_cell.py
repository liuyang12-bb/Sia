"""Fixed short adaptive-service cell; no full-trace batch entry."""
import argparse,gzip,json,time,traceback,hashlib
from pathlib import Path
import pandas as pd
import psutil
from joint_prototype.full160 import make_cluster
from joint_prototype.adaptive_probe import install
from run_small_mechanism import default,save
from run_full160_cell import FullMetrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--response',required=True,choices=['neutral','medium']);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    p=json.loads(a.manifest.read_text(encoding='utf-8'))
    if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('MEMORY_GATE_2GiB')
    for path,h in p['hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()!=h:raise RuntimeError('FROZEN_SOURCE_OR_INPUT:'+path)
    a.output.mkdir(exist_ok=False);start=time.perf_counter()
    c=install(make_cluster(json.loads(Path(p['config']).read_text()),pd.read_csv(p['workload']),p['registry'],p['models'],'joint_o',a.response,'main',True));m=FullMetrics(c)
    save(a.output/'start.json',dict(command=__import__('sys').argv,actual_delay_costs=c.policy.liveness.delay_costs,memory_gib=psutil.virtual_memory().available/1024**3))
    try:
        with gzip.open(a.output/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
            for i in range(p['max_rounds']):
                if c.all_complete():break
                tick=time.perf_counter();c.step(60);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-tick;r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
                f.write(json.dumps(r,default=default)+'\n');f.flush();m.consume(c,r)
                c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
        result=m.summary(c,dict(rmst_tau_seconds=p['max_rounds']*60));result.update(method='joint_r',response=a.response,total_wall_seconds=time.perf_counter()-start);save(a.output/'summary.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['jobs','node_resource_hours','node_peaks']},default=default))
    except Exception as e:
        save(a.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time,last_decision=c.policy.last_decision,services=c.policy.liveness.services));raise
if __name__=='__main__':main()
