"""One predetermined algebraic conditioning probe, unchanged solver options."""
import gzip,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';SRC=OUT/'source'
sys.path.insert(0,str(SRC/'sia-simulator'))
from joint_prototype.completion_continuity import canonical_integral_solution
def conditioned_constraints(constraints):
 out=[];exponents=[]
 for v in constraints:
  a=np.atleast_2d(np.asarray(v.A,dtype=float));lo=np.broadcast_to(v.lb,(a.shape[0],)).copy();hi=np.broadcast_to(v.ub,(a.shape[0],)).copy()
  exps=np.zeros(a.shape[0],dtype=int)
  for i in range(a.shape[0]):
   width=hi[i]-lo[i]
   if np.isfinite(width) and 0<width<1e-6:exps[i]=math.ceil(-math.log2(width))
  aa=np.ldexp(a,exps[:,None]);ll=np.ldexp(lo,exps);hh=np.ldexp(hi,exps)
  assert np.array_equal(np.ldexp(aa,-exps[:,None]),a)
  assert np.array_equal(np.ldexp(ll,-exps),lo) and np.array_equal(np.ldexp(hh,-exps),hi)
  out.append(LinearConstraint(aa,ll,hh));exponents.append(exps.tolist())
 return out,exponents
def main():
 dest=OUT/'numeric_diagnosis';d=json.loads(gzip.open(dest/'solver_diagnostics.json.gz','rt',encoding='utf8').read());m=d['model']
 constraints=[LinearConstraint(v['A'],v['lb'],v['ub']) for v in m['constraints']]
 conditioned,exps=conditioned_constraints(constraints);start=time.time()
 result=milp(c=m['c'],integrality=np.ones(len(m['c'])),bounds=Bounds(0,1),constraints=conditioned,options=d['retry_options'])
 row=dict(status=int(result.status),message=str(result.message),unchanged_options=d['retry_options'],exponents=exps,power_of_two_reversible=True,wall_seconds=time.time()-start)
 if result.status==0 and result.x is not None:
  try:
   rounded=canonical_integral_solution(result.x,constraints)
   row.update(valid_in_original_unscaled_rows=True,secondary_cost=float(np.dot(m['c'],rounded)),selected_column_ids=[v['column_id'] for v,x in zip(m['columns'],rounded) if x>.5],x=rounded.tolist())
  except ValueError as exc:row.update(valid_in_original_unscaled_rows=False,error=repr(exc))
 with (dest/'conditioning_probe.json').open('x',encoding='utf8') as f:json.dump(row,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:v for k,v in row.items() if k not in ['x','selected_column_ids']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
