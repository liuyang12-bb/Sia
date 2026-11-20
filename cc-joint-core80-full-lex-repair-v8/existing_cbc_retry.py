"""Independently solve the SAME failed model with already-installed CBC.

Only an explicit opt-in scope may call this after two undecided native solves.
No gap, capacity or primary/lex row is relaxed; fractional/timeout output is
never a successful result. The caller validates rounded original rows again.
"""
import time
import numpy as np
import cvxpy as cp
from scipy.optimize import OptimizeResult
def solve(cost,constraints):
 tick=time.perf_counter();x=cp.Variable(len(cost),boolean=True);rules=[]
 for c in constraints:
  lo=np.broadcast_to(c.lb,(c.A.shape[0],));hi=np.broadcast_to(c.ub,(c.A.shape[0],))
  good=np.isfinite(lo)
  if np.any(good):rules.append(c.A[good]@x>=lo[good])
  good=np.isfinite(hi)
  if np.any(good):rules.append(c.A[good]@x<=hi[good])
 problem=cp.Problem(cp.Minimize(np.asarray(cost)@x),rules)
 problem.solve(solver=cp.CBC,maximumSeconds=30.,allowableFractionGap=0.,verbose=False)
 okay=problem.status==cp.OPTIMAL
 return OptimizeResult(status=0 if okay else 1,success=okay,x=np.asarray(x.value) if x.value is not None else None,fun=problem.value,message='EXISTING_CBC_'+str(problem.status),elapsed=time.perf_counter()-tick)
