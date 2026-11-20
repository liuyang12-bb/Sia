"""Optional bounded failure reproduction AFTER the fixed batch; no JCT claim.

Exactly five original60sec boundaries, all80 declarations/full work intact.
Same frozen v3 source/parameters. Only catch and preserve exception diagnostics
that the formal runner's generic error record did not serialize. Never repair.
"""
import gzip,hashlib,json,math,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TREE=ROOT/'cc-joint-core80-8h-v3';OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'
sys.path[:0]=[str(TREE),str(TREE/'sia-simulator')]
from core80 import build
from run_core80 import encode,compact_decision,gate,sha,save
def finite(v):
 if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [finite(x) for x in v]
 if hasattr(v,'tolist'):return finite(v.tolist())
 if isinstance(v,float) and not math.isfinite(v):return {'nonfinite':str(v)}
 return v
def main():
 p=json.loads((OUT/'preregistration.json').read_text());out=OUT/'diagnostics_full_five_boundaries';resources=gate(out)
 if time.time()+720>p['formal_stop_unix']:raise RuntimeError('NO_DIAGNOSTIC_WALL_BUDGET')
 for path,h in p['source_hashes'].items():assert sha(TREE/path)==h,path
 out.mkdir(exist_ok=False);save(out/'preregistered_debug.json',dict(method='FULL',response='medium',scenario='dual',boundaries=5,interval=60,wall_limit_seconds=720,resources=resources,
     rule='After main batch, serial, current source unchanged; no JCT evaluation/no extension if failure does not repeat.',frozen_manifest_sha256=sha(OUT/'preregistration.json')))
 c=build('FULL');start=time.time();last=0
 with gzip.open(out/'prefix.jsonl.gz','xt',encoding='utf8') as f:
  try:
   for i in range(5):
    c.step(60);r=c.joint_round_logs[-1];compact_decision(r['decision_for_next_interval']);f.write(json.dumps(r,default=encode)+'\n');f.flush();last=i+1
    c.joint_round_logs.clear();c.logs.clear()
    print('DEBUG_BOUNDARY',last,c.current_time,flush=True)
   save(out/'result.json',dict(status='NOT_REPRODUCED_IN_FIVE_BOUNDARIES',completed_boundaries=last,simtime=c.current_time,wall=time.time()-start))
  except Exception as e:
   details=getattr(e,'solver_diagnostics',None)
   save(out/'result.json',dict(status='REPRODUCED',error=repr(e),completed_boundaries=last,simtime=c.current_time,wall=time.time()-start,diagnostics_available=details is not None,traceback=traceback.format_exc()))
   if details is not None:
    save(out/'solver_brief.json',finite({k:v for k,v in details.items() if k not in ('model','original_x','retry_x','canonical_x')}))
    with gzip.open(out/'solver_full_diagnostics.json.gz','xt',encoding='utf8') as full:json.dump(finite(details),full,default=encode,allow_nan=False)
   print('DEBUG_FAILURE_CAPTURED',repr(e),flush=True)
if __name__=='__main__':main()
