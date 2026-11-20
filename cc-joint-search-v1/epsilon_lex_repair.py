"""Exact startup minimization by monotone epsilon-constraint certificates.

For original primary band [L,U] and integral startup d with gcd g, a currently
valid point of startup V is optimal iff no original point with d*x <= V-g
reaches L. Maximize the ORIGINAL primary on that budget, without its lower
band. Valid primary-band witnesses decrease V. A global upper bound below L
proves V optimal. Unknown/time-limit output never proves infeasibility.

Only the numerical representation changes. All execution/model rows, lex
bands and the final incumbent are independently checked on the originals.
No training, response data or policy parameters are accessed here.
"""
import math,time
from decimal import Decimal
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds,LinearConstraint,OptimizeResult
from joint_prototype.completion_continuity import canonical_integral_solution,objective_solver_units
from primary_band_presolve import dot_error,upper_float,UP,DOWN,D

def lower_primary_cutoff(lower,primary,n):
 return DOWN.subtract(DOWN.subtract(D(float(lower)),D(1e-10)),dot_error(primary,n))

def outer_rows(rows,n):
 """Proof-only superset of the ORIGINAL binary validator domain."""
 result=[]
 for c in rows:
  A=sparse.csr_matrix(c.A);lo=np.broadcast_to(c.lb,(A.shape[0],)).copy();hi=np.broadcast_to(c.ub,(A.shape[0],)).copy()
  for i in range(A.shape[0]):
   row=A.getrow(i);narrow=np.isfinite(lo[i]) and np.isfinite(hi[i]) and hi[i]-lo[i]<1e-6;tol=1e-10 if narrow else 1e-7
   error=UP.add(D(tol),dot_error(row.data,n))
   if np.isfinite(lo[i]):lo[i]=-upper_float(UP.add(D(float(lo[i])).copy_negate(),error))
   if np.isfinite(hi[i]):hi[i]=upper_float(UP.add(D(float(hi[i])),error))
   # Integral job/GPU rows have an exact lattice: the expanded real interval
   # can be inward rounded WITHOUT excluding any accepted binary point.
   if np.array_equal(row.data,np.rint(row.data)) and sum(abs(int(x)) for x in row.data)<=2**52:
    if np.isfinite(lo[i]):lo[i]=math.ceil(lo[i])
    if np.isfinite(hi[i]):hi[i]=math.floor(hi[i])
  result.append(LinearConstraint(A.tocsc(),lo,hi))
 return result

def solve(self,cost,columns,constraints,incumbent,keep):
 from gpu_group_formulation import prepare
 if not getattr(self,'_repair_scope',False) or incumbent is None or len(constraints)<2:return None
 target=constraints[1];lo=np.asarray(target.lb).reshape(-1);hi=np.asarray(target.ub).reshape(-1)
 if len(lo)!=1 or len(hi)!=1 or not np.isfinite(lo[0]) or not np.isfinite(hi[0]) or not 0<=hi[0]-lo[0]<1e-6:return None
 c=np.asarray(cost,dtype=float)
 if np.any(c<0) or not np.array_equal(c,np.rint(c)) or np.max(c,initial=0.)>2**40 or np.sum(c)>2**52:return None
 g=math.gcd(*(int(x) for x in c if x)) if np.any(c) else 0
 if not g:return None
 best=canonical_integral_solution(incumbent,constraints);n=len(c);keep=np.arange(n) if keep is None else keep
 if np.any(best[np.setdiff1d(np.arange(n),keep)]):raise RuntimeError('EPSILON_DOMAIN_LOST_VALID_INCUMBENT')
 p=sparse.csr_matrix(constraints[1].A).toarray().reshape(-1);L=float(np.asarray(constraints[1].lb).item());U=float(np.asarray(constraints[1].ub).item())
 cutoff=lower_primary_cutoff(L,p,n)
 from capacity_cover import strengthen
 resource=strengthen(self,columns,[constraints[0]])[0]
 # Keep EVERY already-fixed startup/geometry/rank row; only the primary
 # band is removed from this proof-only maximum-primary query.
 outer=outer_rows([resource,*constraints[2:]],n)
 from joint_prototype import adaptive_service
 if not hasattr(self,'_repair_epsilon_trace'):self._repair_epsilon_trace=[]
 while True:
  value=int(c@best);budget=value-g
  if budget<0:return OptimizeResult(x=best,status=0,success=True,fun=float(value),message='EXACT_NONNEGATIVE_INTEGER_LEX_LOWER_BOUND')
  query=outer+[LinearConstraint(c.reshape(1,-1),-np.inf,float(budget))]
  query=[LinearConstraint(r.A[:,keep],r.lb,r.ub) for r in query]
  small_columns=[columns[i] for i in keep]
  # Group-link construction is independent of the current rank objective.
  # Its returned zero objective is discarded; the original primary stays
  # on x. Nonuniform rank/budget rows also stay on x, never averaged on y.
  extended=prepare(self,np.zeros(len(keep)),small_columns,query,True)
  if extended is None:E=None;backend_rows=query;objective=-p[keep]
  else:
   _,backend_rows,E=extended;objective=np.concatenate([-p[keep],np.zeros(E.shape[0])])
  backend_cost,scale=objective_solver_units(objective)
  t=time.perf_counter();r=adaptive_service.milp(c=backend_cost,integrality=np.ones(len(backend_cost)),bounds=Bounds(0.,1.),constraints=backend_rows,options={'time_limit':30.,'mip_rel_gap':0.})
  record=dict(budget=budget,incumbent_integer_cost=value,lex_stage='STARTUP' if len(constraints)==2 else 'INTEGER_TIE_RANK',retained_prior_constraint_groups=len(constraints)-2,gcd=g,status=int(r.status),message=str(r.message),seconds=time.perf_counter()-t,variables=len(backend_cost),original_variables=n,proof='EXACT_INTEGER_LEX_EPSILON_CONSTRAINT',options={'time_limit':30.,'mip_rel_gap':0.})
  self._repair_epsilon_trace.append(record)
  # A witness only improves an UPPER bound; it need not be globally optimal.
  query_valid=False
  if r.x is not None:
   try:
    integral=canonical_integral_solution(r.x,backend_rows)
    query_valid=True
   except ValueError:integral=None;record['invalid_query_point']=True
  if query_valid:
   full=np.zeros(n);full[keep]=integral[:len(keep)]
   try:
    witness=canonical_integral_solution(full,constraints)
    if c@witness<=budget:
     best=witness;record['valid_improved_integer_cost']=int(c@best)
     if c@best==0.:record.update(proved=True,proof='EXACT_NONNEGATIVE_INTEGER_LEX_ATTAINED_ZERO',objective=0)
     continue
   except ValueError:record['query_point_outside_original_band']=True
  bound=getattr(r,'mip_dual_bound',None)
  if query_valid and r.status==0 and r.success and bound is not None and np.isfinite(bound):
   # The solver supplies a global MIP bound (ordinary zero-gap acceptance).
   # Directed conversion plus original row dot/tolerance guards is stricter
   # than comparing the primal optimum with the nominal lower band.
   upper=UP.add(D(float(-bound/scale)),dot_error(p,n));record['primary_global_upper']=str(upper);record['original_primary_lower_guard']=str(cutoff)
   if upper<cutoff:
    canonical_integral_solution(best,constraints)
    record['proved']=True
    return OptimizeResult(x=best,status=0,success=True,fun=float(c@best),message='EXACT_INTEGER_LEX_BUDGET_GLOBAL_PRIMARY_BOUND_BELOW_ORIGINAL_BAND')
  record['proved']=False
  return None
