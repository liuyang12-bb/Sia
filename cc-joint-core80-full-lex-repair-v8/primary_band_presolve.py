"""Safe reduced-cost elimination implied by the already fixed primary band.

For min c*x on [0,1], any y<=0 for A*x<=b proves
 B=b*y+sum(min(0,c-A.T*y)). With x_i=1 the bound is
 B+max(c_i-A_i.T*y,0). Here c=-primary and its upper cutoff is
 minus the ORIGINAL primary lower bound. A variable is removed only if
 that conditional lower bound exceeds the cutoff. This is feasibility
 presolve, not a new objective or approximate search domain.

Directed Decimal bounds include original validation tolerance and a generous
 floating dot-product error bound. No LP solution is submitted to training.
"""
import time,hashlib
from decimal import Context,Decimal,ROUND_CEILING,ROUND_FLOOR
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from dual_certificate import dual_interval

UP=Context(prec=80,rounding=ROUND_CEILING)
DOWN=Context(prec=80,rounding=ROUND_FLOOR)
D=Decimal.from_float

def dot_error(row,n):
 data=np.asarray(row,dtype=float)
 if np.array_equal(data,np.rint(data)) and sum(abs(int(x)) for x in data)<=2**52:return Decimal(0)
 total=Decimal(0)
 for v in data:total=UP.add(total,D(abs(float(v))))
 unit=D(2.**-53)
 nu=UP.multiply(Decimal(n),unit)
 if nu>=1:raise ValueError('DOT_ERROR_BOUND_SCOPE')
 gamma=UP.divide(nu,DOWN.subtract(Decimal(1),nu))
 return UP.multiply(gamma,total)

def upper_float(x):
 v=float(x)
 return np.nextafter(v,np.inf) if D(v)<x else v

def expanded_inequalities(rows,n):
 matrices=[];bounds=[]
 for c in rows:
  a=sparse.csr_matrix(c.A);lo=np.broadcast_to(c.lb,(a.shape[0],));hi=np.broadcast_to(c.ub,(a.shape[0],))
  for i in range(a.shape[0]):
   v=a.getrow(i);l=float(lo[i]);h=float(hi[i]);tol=1e-10 if np.isfinite(l) and np.isfinite(h) and h-l<1e-6 else 1e-7
   error=UP.add(D(tol),dot_error(v.data,n))
   if np.isfinite(h):matrices.append(v);bounds.append(upper_float(UP.add(D(h),error)))
   if np.isfinite(l):matrices.append(-v);bounds.append(upper_float(UP.add(D(l).copy_negate(),error)))
 return sparse.vstack(matrices).tocsc(),np.asarray(bounds)

def primary_certificate(constraints,n):
 if len(constraints)<2:return None
 if any(sparse.issparse(c.A) and not getattr(c.A,'has_canonical_format',False) for c in constraints[:3]):return None
 target=constraints[1];a=sparse.csr_matrix(target.A)
 if a.shape!=(1,n):return None
 lo=float(np.asarray(target.lb).reshape(-1)[0]);hi=float(np.asarray(target.ub).reshape(-1)[0])
 if not np.isfinite(lo) or not np.isfinite(hi) or not 0<=hi-lo<1e-6 or not a.nnz:return None
 # Resource/job and the fixed primary band exist before startup minimization.
 # Include the startup optimum only AFTER it has actually been fixed.
 # Dropping later geometry masks is a conservative relaxation;
 # the certificate remains valid when those masks are subsequently appended.
 other=[constraints[0]]+([constraints[2]] if len(constraints)>=3 else [])
 cost=-a.toarray().reshape(-1);A,b=expanded_inequalities(other,n)
 cutoff=UP.add(UP.add(D(lo).copy_negate(),D(1e-10)),dot_error(a.data,n))
 start=time.perf_counter();lp=linprog(cost,A_ub=A,b_ub=b,bounds=(0.,1.),method='highs',options={'time_limit':30.})
 record=dict(variables=n,seconds=time.perf_counter()-start,lp_status=int(lp.status),removed=0,
  proof='PRIMARY_BAND_DIRECTED_WEAK_DUALITY_CONDITIONAL_BOUND',cutoff=str(cutoff))
 y=getattr(getattr(lp,'ineqlin',None),'marginals',None)
 if lp.status!=0 or y is None or not np.all(np.isfinite(y)):return np.arange(n),record
 bound,residual=dual_interval(cost,A,b,y)
 eliminated=[i for i,r in enumerate(residual) if DOWN.add(bound,max(r,Decimal(0)))>cutoff]
 keep=np.ones(n,dtype=bool);keep[eliminated]=False
 record.update(lower_bound=str(bound),removed=len(eliminated),original_primary_band=[lo,hi],tolerance_and_roundoff_included=True)
 return np.flatnonzero(keep),record

def allowed(self,columns,constraints,incumbent=None,cache_token=None):
 if not getattr(self,'_repair_scope',False) or len(constraints)<2:return None
 token=tuple(constraints[:3] if cache_token is None else cache_token);old=getattr(self,'_repair_primary_domain_cache',None)
 from scale_storage import matrix_bytes
 h=hashlib.sha256()
 for c in token:
  for part in (c.A,c.lb,c.ub):h.update(matrix_bytes(part))
 signature=h.hexdigest()
 if old is None or old[0] is not columns or any(a is not b for a,b in zip(old[1],token)) or old[2]!=signature:
  proof=primary_certificate(constraints,len(columns))
  if proof is None:return None
  keep,record=proof
  self._repair_primary_domain_cache=(columns,token,signature,keep)
  record['model_prefix_sha256']=signature
  self._repair_primary_band_lp_calls=getattr(self,'_repair_primary_band_lp_calls',0)+1
  self._repair_primary_band_lp_seconds=getattr(self,'_repair_primary_band_lp_seconds',0.)+record['seconds']
  self._repair_primary_band_variables_removed=max(getattr(self,'_repair_primary_band_variables_removed',0),record['removed'])
  if not hasattr(self,'_repair_primary_certificates'):self._repair_primary_certificates=[]
  if len(self._repair_primary_certificates)<16:self._repair_primary_certificates.append(record)
 else:keep=old[3]
 if incumbent is not None:
  lost=np.ones(len(columns),dtype=bool);lost[keep]=False
  if np.any(np.asarray(incumbent)[lost]!=0.):raise RuntimeError('PRIMARY_PRESOLVE_CONTRADICTS_CURRENT_VALID_INCUMBENT')
 return keep

def certificate_on_domain(cost,constraints,incumbent,keep):
 """Run the existing objective proof on the already certified feasible domain.

 Lift reduced-cost/support indices back to the ORIGINAL variable numbering.
 Removed variables are proved zero at this primary band, so padding their
 residuals by zero and using the remaining cost lattice is valid on the full
 feasible set. The LP primal is never an execution allocation.
 """
 from dual_certificate import certificate
 from scipy.optimize import LinearConstraint
 if keep is None or len(keep)==len(cost):return certificate(cost,constraints,incumbent)
 keep=np.asarray(keep,dtype=int);mask=np.ones(len(cost),dtype=bool);mask[keep]=False
 if np.any(np.asarray(incumbent)[mask]!=0.):raise ValueError('INCUMBENT_OUTSIDE_CERTIFIED_DOMAIN')
 rows=[LinearConstraint(c.A[:,keep] if sparse.issparse(c.A) else np.atleast_2d(c.A)[:,keep],c.lb,c.ub) for c in constraints]
 proof=certificate(np.asarray(cost)[keep],rows,np.asarray(incumbent)[keep])
 if proof is None:return None
 proof=dict(proof,certified_domain_variables=len(keep),original_variables=len(cost))
 if '_residuals' in proof:
  values=[Decimal(0)]*len(cost)
  for i,v in zip(keep,proof['_residuals']):values[int(i)]=v
  proof['_residuals']=values
  proof['_lp_support']=[int(keep[i]) for i in proof.get('_lp_support',[])]
  proof['fix_zero']=[int(keep[i]) for i in proof.get('fix_zero',[])]
 return proof
