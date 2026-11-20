"""Canonical-first validation on original rows; existing solver settings only."""
import hashlib,time
import numpy as np
from scipy.optimize import Bounds,OptimizeResult
from scale_storage import matrix_bytes,matrix_payload

def fingerprint(cost,constraints):
 h=hashlib.sha256(np.asarray(cost,dtype=float).tobytes())
 for c in constraints:
  for x in (c.A,c.lb,c.ub):h.update(matrix_bytes(x))
 return h.hexdigest()

def canonical_bundle(self,cost,columns,constraints):
 from joint_prototype.completion_continuity import canonical_integral_solution,condition_constraint_bands
 from joint_prototype.bundle_adaptation import BundleAdmission
 from joint_prototype import adaptive_service
 previous=getattr(self,'_repair_last_x',None)
 valid=None
 if getattr(self,'_repair_last_columns',None) is columns and previous is not None:
  try:
   valid=canonical_integral_solution(previous,constraints);mins={}
   required=set()
   if getattr(self,'_repair_scope',False) and getattr(self,'_repair_job_order',None):
    base=constraints[0];lo=np.broadcast_to(base.lb,(base.A.shape[0],))
    required={j for i,j in enumerate(self._repair_job_order) if lo[i]==1.}
   for c,v in zip(columns,cost):mins[c.job_id]=min(mins.get(c.job_id,float('inf') if c.job_id in required else 0.),float(v))
   # Ignore capacities and all cross-job bands, retaining the actual mandatory
   # one-choice job rows. Required jobs cannot take the optional zero cost.
   # This strictly strengthens the same relaxation without changing any goal.
   # Ignoring capacities gives a lower bound (at most one column/job). A
   # current feasible vector attaining it is a proof, never a heuristic.
   if float(np.dot(cost,valid))==sum(mins.values()):
    self._repair_bound_certificates=getattr(self,'_repair_bound_certificates',0)+1
    return OptimizeResult(x=valid,status=0,success=True,fun=float(np.dot(cost,valid)),message='CERTIFIED_ATTAINED_JOBWISE_LOWER_BOUND; all current original rows validated')
  except ValueError:pass
 before=fingerprint(cost,constraints)
 from integer_row_lattice import tighten_integer_rows
 solver_constraints,lattice=tighten_integer_rows(constraints)
 from capacity_cover import strengthen
 solver_constraints=strengthen(self,columns,solver_constraints)
 self._repair_integer_rows_tightened=getattr(self,'_repair_integer_rows_tightened',0)+len(lattice)
 from primary_band_presolve import allowed,certificate_on_domain
 primary_keep=allowed(self,columns,solver_constraints,valid,cache_token=constraints[:3])
 if valid is not None and getattr(self,'_repair_scope',False):
  from dual_certificate import refresh_incumbent,public_proof
  proof=certificate_on_domain(cost,constraints,valid,primary_keep)
  if proof is not None:
   self._repair_lp_calls=getattr(self,'_repair_lp_calls',0)+1
   self._repair_lp_seconds=getattr(self,'_repair_lp_seconds',0.)+proof['seconds']
   if not proof['proved'] and proof.get('_lp_support'):
    support=np.asarray(sorted(set(proof['_lp_support'])|set(np.flatnonzero(valid))),dtype=int)
    if len(support)<len(columns):
     from scipy.optimize import LinearConstraint
     rows=[LinearConstraint(c.A[:,support],c.lb,c.ub) for c in constraints]
     tick=time.perf_counter();hint=adaptive_service.milp(c=np.asarray(cost)[support],integrality=np.ones(len(support)),bounds=Bounds(np.zeros(len(support)),np.ones(len(support))),constraints=rows,options={'time_limit':30.,'mip_rel_gap':0.})
     elapsed=time.perf_counter()-tick
     self._repair_native_calls=getattr(self,'_repair_native_calls',0)+1;self._repair_native_seconds=getattr(self,'_repair_native_seconds',0.)+elapsed
     if not hasattr(self,'_repair_support_trace'):self._repair_support_trace=[]
     trace=dict(status=int(hint.status),message=str(hint.message),variables=len(support),seconds=elapsed,used_as_upper_bound=False)
     self._repair_support_trace.append(trace)
     if hint.status==0 and hint.success and hint.x is not None:
      hx=np.zeros(len(columns));hx[support]=hint.x
      try:
       hx=canonical_integral_solution(hx,constraints)
       improved=refresh_incumbent(proof,cost,hx)
       from decimal import Decimal
       if not improved['contradiction'] and Decimal(improved['objective'])<Decimal(proof['objective']):valid=hx;proof=improved;trace['used_as_upper_bound']=True
      except ValueError:trace['rejected_original_rows']=True
   if proof['proved']:
    self._repair_bound_certificates=getattr(self,'_repair_bound_certificates',0)+1
    self._repair_last_x=valid.copy();self._repair_last_columns=columns
    return OptimizeResult(x=valid,status=0,success=True,fun=float(np.dot(cost,valid)),message='CERTIFIED_DUAL_LATTICE_LOWER_BOUND; '+str({k:v for k,v in public_proof(proof).items() if k!='fix_zero'}))
 # Exact budget proof avoids minimizing startup in a hairline primary strip.
 # It returns only a current original-row valid, globally certified optimum.
 from epsilon_lex_repair import solve as startup_certificate
 certified=startup_certificate(self,cost,columns,constraints,valid,primary_keep)
 if certified is not None:
  self._repair_last_x=certified.x.copy();self._repair_last_columns=columns
  return certified
 # Keep the literal first solve. The preserved first-conditioned technical
 # variant timed out on a later lex phase; use the existing reversible row
 # representation only when numerical validation/infeasibility needs it.
 narrow=any(np.any(np.isfinite(c.lb)&np.isfinite(c.ub)&(np.asarray(c.ub)-np.asarray(c.lb)>0)&(np.asarray(c.ub)-np.asarray(c.lb)<1e-6)) for c in constraints)
 from exact_grid_presolve import reduced
 small_cost,small_columns,small_rows,kept=reduced(self,cost,columns,solver_constraints)
 if primary_keep is not None and len(primary_keep)<len(columns):
  kept=primary_keep if kept is None else np.intersect1d(kept,primary_keep)
  from scipy.optimize import LinearConstraint
  small_cost=np.asarray(cost)[kept];small_columns=[columns[i] for i in kept]
  small_rows=[LinearConstraint(c.A[:,kept],c.lb,c.ub) for c in solver_constraints]
 if valid is not None and getattr(self,'_repair_scope',False) and proof is not None and proof.get('fix_zero'):
  from scipy.optimize import LinearConstraint
  allowed=np.ones(len(columns),dtype=bool);allowed[proof['fix_zero']]=False
  kept=np.flatnonzero(allowed) if kept is None else kept[allowed[kept]]
  small_cost=np.asarray(cost)[kept];small_columns=[columns[i] for i in kept]
  small_rows=[LinearConstraint(c.A[:,kept],c.lb,c.ub) for c in solver_constraints]
  self._repair_lp_variables_removed=max(getattr(self,'_repair_lp_variables_removed',0),len(proof['fix_zero']))
 def lift(r):
  if kept is not None and r.x is not None:
   x=np.zeros(len(columns));x[kept]=r.x;r.x=x
  return r
 tick=time.perf_counter();result=lift(super(BundleAdmission,self)._milp(small_cost,small_columns,small_rows))
 self._repair_native_seconds=getattr(self,'_repair_native_seconds',0.)+time.perf_counter()-tick;self._repair_native_calls=getattr(self,'_repair_native_calls',0)+1
 if fingerprint(cost,constraints)!=before:raise RuntimeError('SOLVER_MUTATED_INPUT_MODEL')
 def accept(r):
  if r.status!=0 or not r.success or r.x is None:return None
  try:v=canonical_integral_solution(r.x,constraints)
  except ValueError:return None
  delta=float(np.max(abs(np.asarray(r.x)-v)))
  if delta:self.numerical_solver_retries.append(dict(reformulation='CANONICAL_BINARY_BEFORE_RETRY',max_integrality_error=delta,original_rows_validated=True,solver_status=int(r.status)))
  self._repair_last_x=v.copy();self._repair_last_columns=columns
  return OptimizeResult(x=v,status=0,success=True,fun=float(np.dot(cost,v)),message=str(r.message)+'; canonical first, all original rows validated')
 approved=accept(result)
 if approved is not None:return approved
 needs_retry=(result.status==0 or result.status==2 and narrow or result.status==1 and getattr(self,'_repair_scope',False))
 if not needs_retry:
  if result.status!=2 and getattr(self,'_repair_scope',False):
   self._repair_last_failure=dict(status=int(result.status),message=str(result.message),x=np.asarray(result.x).tolist() if result.x is not None else None,
    model=dict(c=np.asarray(cost).tolist(),columns=[dict(job_id=c.job_id,column_id=c.column_id) for c in columns],constraints=[dict(A=matrix_payload(c.A),lb=np.asarray(c.lb).tolist(),ub=np.asarray(c.ub).tolist()) for c in constraints]))
  self._repair_last_x=None;return result
 conditioned,powers=condition_constraint_bands(solver_constraints) if narrow else (solver_constraints,[])
 retry_rows=conditioned if narrow else solver_constraints
 if kept is not None:
  from scipy.optimize import LinearConstraint
  retry_rows=[LinearConstraint(c.A[:,kept],c.lb,c.ub) for c in retry_rows]
 tick=time.perf_counter();retry=lift(adaptive_service.milp(c=small_cost,integrality=np.ones(len(small_columns)),bounds=Bounds(np.zeros(len(small_columns)),np.ones(len(small_columns))),constraints=retry_rows,options={'time_limit':30.,'mip_rel_gap':0.,'presolve':False}))
 self._repair_native_seconds+=time.perf_counter()-tick;self._repair_native_calls+=1
 if fingerprint(cost,constraints)!=before:raise RuntimeError('SOLVER_MUTATED_INPUT_MODEL')
 record=dict(reformulation='FIXED_EXISTING_NO_PRESOLVE_RETRY',original_status=int(result.status),retry_status=int(retry.status),row_exponents=powers,solver_options={'time_limit':30.,'mip_rel_gap':0.,'presolve':False})
 self.numerical_solver_retries.append(record);approved=accept(retry)
 if approved is not None:return approved
 if result.status==1 and retry.status==1 and getattr(self,'_repair_allow_cbc',False):
  from existing_cbc_retry import solve as independent_solve
  tick=time.perf_counter();safety=lift(independent_solve(small_cost,retry_rows))
  self._repair_native_seconds+=time.perf_counter()-tick;self._repair_native_calls+=1
  record['independent_cbc']=dict(status=int(safety.status),success=bool(safety.success),message=str(safety.message),options=dict(maximumSeconds=30.,allowableFractionGap=0.))
  self.numerical_solver_retries.append(dict(independent_backend='EXISTING_CBC',**record['independent_cbc']))
  approved=accept(safety)
  if approved is not None:return approved
  record['independent_cbc_x']=np.asarray(safety.x).tolist() if safety.x is not None else None
 if retry.status==2 and result.status==2:
  if previous is not None and getattr(self,'_repair_last_columns',None) is columns:
   try:canonical_integral_solution(previous,constraints)
   except ValueError:return retry
  else:return retry
 error=RuntimeError('REPAIRED_SOLVER_NO_VERIFIED_OPTIMUM')
 error.solver_diagnostics=dict(**record,original_message=str(result.message),retry_message=str(retry.message),original_x=np.asarray(result.x).tolist() if result.x is not None else None,retry_x=np.asarray(retry.x).tolist() if retry.x is not None else None,last_valid_x=valid.tolist() if valid is not None else None,dual_proof=public_proof(proof) if valid is not None and getattr(self,'_repair_scope',False) and proof is not None else None,
  model=dict(c=np.asarray(cost).tolist(),columns=[dict(job_id=c.job_id,column_id=c.column_id) for c in columns],constraints=[dict(A=matrix_payload(c.A),lb=np.asarray(c.lb).tolist(),ub=np.asarray(c.ub).tolist()) for c in constraints]))
 raise error
