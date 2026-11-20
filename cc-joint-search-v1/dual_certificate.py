"""Exact weak-duality/lattice proof for an already valid integer incumbent.

No LP primal is submitted. Any finite nonpositive inequality multipliers yield
L = b*y + sum(min(0,c-A.T*y)) on the unit box. Directed Decimal intervals
compute a LOWER bound on L from exact binary input floats. If c is integral
with gcd g and L > incumbent-g, the original incumbent is globally optimal.
Original rows are checked separately; proof rows are conservatively expanded
by existing validation tolerances, never used as a new execution domain.
"""
import math,time
from decimal import Context,Decimal,ROUND_CEILING,ROUND_FLOOR
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

def dual_interval(cost,A,b,y):
 up=Context(prec=80,rounding=ROUND_CEILING);down=Context(prec=80,rounding=ROUND_FLOOR)
 D=Decimal.from_float;y=[D(float(min(0.,v))) for v in y]
 base=Decimal(0)
 for v,w in zip(b,y):base=down.add(base,down.multiply(D(float(v)),w))
 A=A.tocsc();A.sum_duplicates();total=base;residuals=[]
 for j,c in enumerate(cost):
  dot=Decimal(0)
  for k in range(A.indptr[j],A.indptr[j+1]):
   v=y[A.indices[k]]
   if v:dot=up.add(dot,up.multiply(D(float(A.data[k])),v))
  residual=down.subtract(D(float(c)),dot)
  residuals.append(residual)
  if residual<0:total=down.add(total,residual)
 return total,residuals

def lower_bound(cost,A,b,y):return dual_interval(cost,A,b,y)[0]

def refresh_incumbent(proof,cost,incumbent):
 bound=proof['_bound'];residuals=proof['_residuals'];g=proof['gcd']
 up=Context(prec=80,rounding=ROUND_CEILING);down=Context(prec=80,rounding=ROUND_FLOOR);objective=Decimal(0)
 for i,v in enumerate(incumbent):
  if v==1.:objective=up.add(objective,Decimal.from_float(float(cost[i])))
 removed=[i for i,r in enumerate(residuals) if down.add(bound,max(r,Decimal(0)))>objective]
 contradiction=bound>objective or any(incumbent[i]!=0. for i in removed)
 return {**proof,'proved':not contradiction and bool(g) and bound>objective-Decimal(g),'objective':str(objective),'fix_zero':[] if contradiction else removed,'contradiction':contradiction}

def public_proof(proof):return {k:v for k,v in proof.items() if not k.startswith('_')}

def certificate(cost,constraints,incumbent):
 c=np.asarray(cost,dtype=float)
 if not np.all(np.isfinite(c)) or np.max(np.abs(c),initial=0)>2**40 or not np.any(c):return None
 integral=np.array_equal(c,np.rint(c));values={int(x) for x in c if x} if integral else set();g=math.gcd(*values) if values else 0
 A=[];b=[]
 for row in constraints:
  a=sparse.csr_matrix(np.atleast_2d(row.A)) if not sparse.issparse(row.A) else row.A.tocsr()
  lo=np.broadcast_to(row.lb,(a.shape[0],));hi=np.broadcast_to(row.ub,(a.shape[0],))
  narrow=np.isfinite(lo)&np.isfinite(hi)&(hi-lo>0)&(hi-lo<1e-6);tol=np.where(narrow,1e-10,1e-7)
  upper=np.isfinite(hi);lower=np.isfinite(lo)
  A.extend([a[upper],-a[lower]])
  b.extend([np.nextafter(hi[upper]+tol[upper],np.inf),np.nextafter(-lo[lower]+tol[lower],np.inf)])
 A=sparse.vstack(A).tocsc();b=np.concatenate(b);tick=time.perf_counter()
 lp=linprog(c,A_ub=A,b_ub=b,bounds=(0.,1.),method='highs',options={'time_limit':30.})
 if lp.status!=0 or lp.ineqlin.marginals is None or not np.all(np.isfinite(lp.ineqlin.marginals)):return dict(proved=False,lp_status=int(lp.status),seconds=time.perf_counter()-tick)
 bound,residuals=dual_interval(c,A,b,lp.ineqlin.marginals)
 # With x_i=1, the same box relaxation gains max(residual_i,0).
 # Strictly worse than a currently feasible objective => x_i=0 in every
 # minimizer. Apply only to this phase, never carry into a later lex cost.
 proof=dict(_bound=bound,_residuals=residuals,_lp_support=np.flatnonzero(lp.x).tolist(),lower_bound=str(bound),gcd=g,lp_status=int(lp.status),seconds=time.perf_counter()-tick,proof='DIRECTED_DECIMAL_WEAK_DUALITY_LATTICE_AND_REDUCED_COST')
 return refresh_incumbent(proof,c,incumbent)
