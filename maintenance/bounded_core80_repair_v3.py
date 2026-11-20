"""Fixed-prefix real80 correctness test, not a JCT experiment."""
import contextlib,gzip,hashlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TREE=ROOT/'cc-joint-core80-repair-v3';OUT=ROOT/'research_runs/sia_synergy/cc_core80_repair_20261031_run3'
sys.path[:0]=[str(TREE),str(TREE/'sia-simulator')]
from core80 import build
from run_core80 import compact_decision,encode,gate

def main():
 method=sys.argv[1];limit=int(sys.argv[2]);d=OUT/'tests'/('real80_'+method+'_'+str(limit)+('_'+sys.argv[3] if len(sys.argv)>3 else ''));gate(OUT);d.mkdir(exist_ok=False)
 files={str(p.relative_to(TREE)):hashlib.sha256(p.read_bytes()).hexdigest().upper() for p in TREE.rglob('*.py') if 'tests' not in p.parts and '__pycache__' not in p.parts}
 (d/'runtime_source_hashes.json').write_text(json.dumps(files,indent=2),encoding='utf8')
 start=time.time();last=0
 with (d/'console.log').open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
  c=build(method,'neutral' if method.startswith('A') else 'medium')
  with gzip.open(d/'prefix.jsonl.gz','xt',encoding='utf8') as f:
   try:
    for i in range(limit):
     assert all(hashlib.sha256((TREE/k).read_bytes()).hexdigest().upper()==h for k,h in files.items()),'SOURCE_CHANGED_DURING_BOUND'
     t=time.time();c.step(60);r=c.joint_round_logs[-1];r['job_end_state']={j.name:dict(progress=j.progress,epoch=j.epoch,restarts=j.num_restarts,completion=j.completion_time) for j in c.jobs}
     compact_decision(r['decision_for_next_interval']);f.write(json.dumps(r,default=encode)+'\n');f.flush();last=i+1
     print('BOUNDARY',method,last,c.current_time,round(time.time()-t,3),r['decision_for_next_interval']['result']['audit'].get('engineering_solver'),flush=True)
     c.joint_round_logs.clear();c.logs.clear()
     for name in ('exchange_columns','exchange_scores','continuity_scores','completion_scores','actual_completion_columns'):
      if hasattr(c.policy,name):setattr(c.policy,name,[])
    result=dict(status='PASS',method=method,boundaries=last,wall_seconds=time.time()-start,scope='Fixed prefix/all80 declarations and work unchanged; no JCT claim.')
   except Exception as e:
    result=dict(status='FAIL',method=method,boundaries=last,time=c.current_time,error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.time()-start)
    if getattr(e,'solver_diagnostics',None) is not None:
     with gzip.open(d/'solver_failure.json.gz','xt',encoding='utf8') as z:json.dump(e.solver_diagnostics,z,default=encode)
 with (d/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
