"""One pre-registered bounded cell; no automatic parameter changes."""
import argparse,gzip,hashlib,json,sys,time,traceback
from pathlib import Path
import pandas as pd
import psutil
from joint_prototype.full160 import make_cluster
from joint_prototype.selective_admission import install
from joint_prototype.small_mechanism import ProbeSia,ProbeAux
from run_full160_cell import FullMetrics
from run_small_mechanism import save,default

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--cell',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    p=json.loads(a.manifest.read_text());cell=next(c for c in p['runs'] if c['id']==a.cell)
    memory=psutil.virtual_memory().available/1024**3
    if memory<2:raise RuntimeError('MEMORY_GATE_2GiB')
    if psutil.disk_usage(str(a.output.parent)).free<5*1024**3:raise RuntimeError('DISK_GATE_5GiB')
    for path,h in p['hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()!=h:raise RuntimeError('FROZEN_HASH_CHANGED:'+path)
    a.output.mkdir(exist_ok=False);start=time.perf_counter();method=cell['method']
    c=make_cluster(json.loads(Path(p['config']).read_text()),pd.read_csv(p['datasets'][cell['dataset']]),
        p['registry'],p['models'],'joint_o' if method=='joint_s' else 'sia_p',cell['response'],'main',True)
    if method=='joint_s':install(c)
    else:
        c.policy.__class__=ProbeSia if method=='sia_p' else ProbeAux
        if method=='sia_aux':c.policy.mode='b4'
    costs={j.name:{t:float(app.rescale_time or 30) for t,app in j.applications.items()} for j in c.jobs}
    m=FullMetrics(c)
    save(a.output/'start.json',dict(command=sys.argv,manifest_hash=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        cell=cell,memory_gib=memory,actual_delay_costs=costs))
    try:
        with gzip.open(a.output/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
            for i in range(p['max_rounds']):
                if c.all_complete():break
                tick=time.perf_counter();c.step(p['interval']);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-tick
                r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,
                    wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
                f.write(json.dumps(r,default=default)+'\n');f.flush();m.consume(c,r)
                c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
                if (i+1)%50==0:print('PROGRESS',i+1,sum(j.completion_time is not None for j in c.jobs),flush=True)
        result=m.summary(c,dict(rmst_tau_seconds=p['max_rounds']*p['interval']))
        result.update(method=method,response=cell['response'],size=len(c.jobs),dataset=cell['dataset'],total_wall_seconds=time.perf_counter()-start)
        save(a.output/'summary.json',result);print('RESULT',result['completed'],result['mean_jct'],flush=True)
    except Exception as exc:
        save(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=c.current_time,
            last_decision=c.policy.last_decision,solver_diagnostics=getattr(exc,'solver_diagnostics',None)));raise
if __name__=='__main__':main()
