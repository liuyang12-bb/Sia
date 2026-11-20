import argparse,gzip,hashlib,importlib.metadata,json,os,sys,time,traceback
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-uncertainty-v4'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from core80 import build
from run_core80 import Metrics,gate,compact_decision,encode
from joint_prototype.uncertainty import install
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,default=encode)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True);a=ap.parse_args()
 p=json.loads((O/'preregistration.json').read_text());cell=next(x for x in p['cells'] if x['id']==a.cell)
 for path,h in p['source_hashes'].items():assert sha(T/path)==h,path
 for path,h in p['input_hashes'].items():assert sha(path)==h,path
 for package,v in p['environment']['packages'].items():assert importlib.metadata.version(package)==v,package
 out=O/'cells'/a.cell;resources=gate(O);out.mkdir(parents=True,exist_ok=False)
 start=time.time();save(out/'start.json',dict(cell=cell,command=sys.argv,resources=resources,start_unix=start,preregistration_sha256=sha(O/'preregistration.json'),pid=os.getpid()))
 c=None;m=None
 try:
  c=build(cell['method'],workload_path=O/'inputs/workload.csv',topology=p['topology'])
  if cell['new_evaluation']:install(c)
  m=Metrics(c)
  with gzip.open(out/'rounds.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
   for i in range(p['max_rounds']):
    if c.all_complete():break
    tick=time.perf_counter();c.step(p['interval']);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-tick
    r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time,progress=float(j.progress),epoch=int(j.epoch)) for j in c.jobs}
    compact_decision(r['decision_for_next_interval']);f.write(json.dumps(r,default=encode,separators=(',',':'))+'\n');f.flush();m.consume(c,r)
    if (i+1)%20==0 or i<3:
     v=dict(round=i+1,completed=sum(j.completion_time is not None for j in c.jobs),time=c.current_time,wall=time.time()-start)
     save(out/f'progress_{i+1:04}.json',v);print('PROGRESS',a.cell,v,flush=True)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
    for key in ('exchange_columns','exchange_scores','continuity_scores','completion_scores','actual_completion_columns'):
     if hasattr(c.policy,key):setattr(c.policy,key,[])
  result=m.summary(c);result.update(cell=cell,total_wall_seconds=time.time()-start,termination='ALL_COMPLETE' if c.all_complete() else 'ROUND_LIMIT')
  
  if hasattr(c,'_residual_bank'):
   with gzip.open(out/'calibration_records.json.gz','xt',encoding='utf8') as f:json.dump(c._residual_bank.rows,f)
  save(out/'summary.json',result);save(out/'output_hashes.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
  print('RESULT',a.cell,result['completed'],result['mean_jct'],flush=True)
 except Exception as e:
  save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),rounds=m.rounds if m else 0,time=c.current_time if c else None))
  if getattr(e,'solver_diagnostics',None):
   with gzip.open(out/'solver_failure.json.gz','xt',encoding='utf8') as f:json.dump(e.solver_diagnostics,f,default=encode)
  raise
if __name__=='__main__':main()
