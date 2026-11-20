"""Bounded exact failed-cell replay: capture diagnostics without changing solve."""
import dataclasses,gzip,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from breakpoint_study import build
from joint_prototype.completion_continuity import NumericalAdmission
def enc(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(str(type(x)))
def main():
 dest=OUT/'numeric_diagnosis';dest.mkdir(exist_ok=False)
 with (dest/'BOUNDS.json').open('x',encoding='utf8') as f:json.dump(dict(cell='M3_R11_medium_CC_CPU',round_limit=16,seconds=60,purpose='Same frozen-source failure reproduction only, no performance evaluation, no solver/model/target changes'),f,indent=2)
 old=NumericalAdmission._milp
 def capture(self,cost,columns,constraints):
  try:return old(self,cost,columns,constraints)
  except Exception as exc:
   d=getattr(exc,'solver_diagnostics',None)
   if d:
    with gzip.open(dest/'solver_diagnostics.json.gz','xt',encoding='utf8') as f:json.dump(d,f,default=enc)
    x=np.rint(d['retry_x']);violations=[]
    for ci,v in enumerate(constraints):
     value=np.asarray(v.A@x).reshape(-1);lo=np.broadcast_to(v.lb,value.shape);hi=np.broadcast_to(v.ub,value.shape)
     narrow=np.isfinite(lo)&np.isfinite(hi)&(hi-lo<1e-6);tol=np.where(narrow,1e-10,1e-7)
     delta=np.maximum(lo-value,value-hi)
     for row in np.flatnonzero(delta>tol):violations.append(dict(constraint=ci,row=int(row),value=float(value[row]),lb=float(lo[row]),ub=float(hi[row]),violation=float(delta[row]),tolerance=float(tol[row]),narrow=bool(narrow[row])))
    with (dest/'diagnostic_summary.json').open('x',encoding='utf8') as f:json.dump(dict(columns=len(columns),constraints=len(constraints),rounded_row_violations=violations,original_validation=d['original_validation'],retry_validation=d['retry_validation']),f,indent=2)
   raise
 NumericalAdmission._milp=capture
 c=build('CC_CPU','R11','medium',OUT/'inputs/M3_workload.csv')
 try:
  for i in range(16):
   c.step(60);c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
 except Exception as exc:
  with (dest/'replay_outcome.json').open('x',encoding='utf8') as f:json.dump(dict(error=repr(exc),time=c.current_time,round=i+1,traceback=traceback.format_exc(),captured=(dest/'solver_diagnostics.json.gz').exists()),f,indent=2)
  print('FAILURE_REPRODUCED',c.current_time,i+1,repr(exc),flush=True)
 else:raise RuntimeError('FROZEN_FAILURE_NOT_REPRODUCED_WITHIN_FIXED_BOUND')
if __name__=='__main__':main()
