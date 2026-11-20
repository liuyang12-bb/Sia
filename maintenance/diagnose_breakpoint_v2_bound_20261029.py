"""Read failed traceback locals in fixed16-round replay; no solver changes."""
import dataclasses,gzip,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt2';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from breakpoint_study import build
def enc(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(str(type(x)))
def main():
 dest=OUT/'bounded_failure_diagnosis';dest.mkdir(exist_ok=False)
 c=build('CC_CPU','R11','medium',OUT/'inputs/M3_workload.csv')
 try:
  for i in range(16):c.step(60);c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
 except Exception as exc:
  frames=[];tb=exc.__traceback__
  while tb:
   f=tb.tb_frame
   if f.f_code.co_name=='_decide' and 'comparison' in f.f_locals:
    loc=f.f_locals;res=loc['result'];frames.append(dict(file=f.f_code.co_filename,line=tb.tb_lineno,comparison=loc['comparison'],result=res.to_dict(),baseline=[dataclasses.asdict(v) for v in loc['baseline']],columns=[dataclasses.asdict(v) for v in loc['columns']]))
   tb=tb.tb_next
  with gzip.open(dest/'traceback_state.json.gz','xt',encoding='utf8') as f:json.dump(dict(error=repr(exc),time=c.current_time,round=i+1,frames=frames),f,default=enc)
  with (dest/'summary.json').open('x',encoding='utf8') as f:json.dump(dict(error=repr(exc),time=c.current_time,round=i+1,comparisons=[v['comparison'] for v in frames]),f,indent=2)
  print('BOUND_FAILURE',c.current_time,[v['comparison'] for v in frames])
 else:raise RuntimeError('FAILURE_NOT_REPRODUCED_WITHIN16')
if __name__=='__main__':main()
