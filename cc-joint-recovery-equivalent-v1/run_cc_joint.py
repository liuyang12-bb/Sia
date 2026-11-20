"""One explicit CC-Joint/baseline run; never automatically launches a matrix."""
import argparse,dataclasses,gzip,json,platform,sys,time,traceback
from pathlib import Path
from cc_factory import ROOT,METHODS,RESPONSES,construct,sha,verify

def encode(value):
    if dataclasses.is_dataclass(value):return dataclasses.asdict(value)
    if hasattr(value,'tolist'):return value.tolist()
    if hasattr(value,'item'):return value.item()
    raise TypeError(type(value).__name__)

def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,default=encode)

def resources(output):
    import psutil
    d=output.parent
    while not d.exists():d=d.parent
    r=dict(available_memory_gib=psutil.virtual_memory().available/2**30,free_disk_gib=psutil.disk_usage(str(d)).free/2**30)
    if r['available_memory_gib']<2 or r['free_disk_gib']<5:raise RuntimeError('SAFETY_GATE: '+str(r))
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,default='JOINT_CC')
    p.add_argument('--response',choices=RESPONSES,default='medium')
    p.add_argument('--workload',type=Path,default=ROOT/'inputs/workload.csv')
    p.add_argument('--check',action='store_true',help='Construct and verify only; no training/time advancement')
    p.add_argument('--output',type=Path,help='Required new output directory for an explicit run')
    p.add_argument('--max-rounds',type=int,default=6000)
    a=p.parse_args();manifest=verify()
    if not a.check and a.output is None:p.error('--output is required; no training starts implicitly')
    if not 1<=a.max_rounds<=6000:p.error('max-rounds must be in1..6000; no automatic horizon extension')
    if a.check:
        c=construct(a.method,a.response,a.workload)
        print(json.dumps(dict(valid=True,training_run=False,method=a.method,response=a.response,jobs=len(c.jobs),
            gpu_configs=list(c.policy.config_to_idx_map),capacities=c.policy.node_capacities,
            simulated_seconds=c.current_time,all_job_progress_zero=all(j.progress==0 for j in c.jobs),
            all_job_completion_none=all(j.completion_time is None for j in c.jobs),
            original_source_fingerprint=manifest['original_full_source_fingerprint']),indent=2,default=encode))
        return
    output=a.output.resolve()
    if output.exists():raise FileExistsError('Refusing existing output directory: '+str(output))
    gate=resources(output);output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    save(output/'manifest.json',dict(method=a.method,response=a.response,workload=str(a.workload.resolve()),
      workload_sha256=sha(a.workload),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
      entry_hashes={n:sha(ROOT/n) for n in ['cc_factory.py','cc_metrics.py','run_cc_joint.py']},
      interval=60,max_rounds=a.max_rounds,seed_profiles=[1,8],batch_adaptation=True,resources=gate,
      python=sys.version,platform=platform.platform(),start_unix=started,
      warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
    c=None
    try:
      c=construct(a.method,a.response,a.workload)
      from cc_metrics import metrics_class
      m=metrics_class()(c)
      with gzip.open(output/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
        for i in range(a.max_rounds):
          if c.all_complete():break
          tick=time.perf_counter();c.step(60);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-tick
          r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,
              wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
          f.write(json.dumps(r,default=encode,separators=(',',':'))+'\n');f.flush();m.consume(c,r)
          c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
          if (i+1)%100==0:print('PROGRESS',i+1,sum(j.completion_time is not None for j in c.jobs),flush=True)
      s=m.summary(c,dict(rmst_tau_seconds=360000));s.update(method=a.method,response=a.response,
          termination='ALL_COMPLETE' if c.all_complete() else 'ROUND_LIMIT',total_wall_seconds=time.time()-started)
      save(output/'summary.json',s)
      print('RESULT',s['termination'],s['completed'],'/',s['total'],'mean_jct',s['mean_jct'],flush=True)
    except Exception as e:
      save(output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),
          time=c.current_time if c else None,last_decision=c.policy.last_decision if c else None))
      raise

if __name__=='__main__':main()
