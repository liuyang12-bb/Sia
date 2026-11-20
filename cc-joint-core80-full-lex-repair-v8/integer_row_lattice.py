"""Exact integer-row bounds, preserving both nominal and audit-feasible points.

For an integral row and binary x, a*x is an exact multiple of gcd(a).
Replace a bound by the nearest inward lattice bound, only if the original
validator's tolerance accepts exactly the same lattice values. Fraction uses
the actual binary input floats; no decimal rounding, grid/weight changes or
new tolerance. Fractional rows remain untouched. Original rows are always
revalidated before a solver answer is accepted.
"""
import math
from fractions import Fraction
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint

def tighten_integer_rows(constraints):
 result=[];records=[]
 for ci,c in enumerate(constraints):
  if sparse.issparse(c.A) and not getattr(c.A,'has_canonical_format',False):
   result.append(c);continue  # Do not reinterpret duplicate fractional entries.
  A=c.A.tocsr(copy=True) if sparse.issparse(c.A) else np.atleast_2d(np.asarray(c.A,dtype=float))
  if sparse.issparse(A):A.sum_duplicates();A.eliminate_zeros()
  lo=np.broadcast_to(c.lb,(A.shape[0],)).copy();hi=np.broadcast_to(c.ub,(A.shape[0],)).copy()
  changed=False
  for i in range(A.shape[0]):
   row=A.data[A.indptr[i]:A.indptr[i+1]] if sparse.issparse(A) else A[i]
   row=row[row!=0.]
   if (not len(row) or not np.all(np.isfinite(row)) or
       not np.array_equal(row,np.rint(row)) or sum(abs(int(v)) for v in row)>2**52):continue
   g=math.gcd(*(int(v) for v in row))
   lower=float(lo[i]);upper=float(hi[i])
   if math.isnan(lower) or math.isnan(upper):continue
   # Exactly the frozen original-row validator's tolerance, not a new option.
   tol=1e-10 if math.isfinite(lower) and math.isfinite(upper) and upper-lower<1e-6 else 1e-7
   def lower_value(v):return math.ceil(Fraction.from_float(v)/g)*g
   def upper_value(v):return math.floor(Fraction.from_float(v)/g)*g
   l=lower_value(lower) if math.isfinite(lower) else lower
   u=upper_value(upper) if math.isfinite(upper) else upper
   # If audit tolerance changes which integer is allowed, leave the row alone.
   if math.isfinite(lower) and l!=lower_value(lower-tol):continue
   if math.isfinite(upper) and u!=upper_value(upper+tol):continue
   if l>u:continue  # Keep existing infeasibility handling, never forge a point.
   if l==lower and u==upper:continue
   if (math.isfinite(l) and abs(l)>2**52) or (math.isfinite(u) and abs(u)>2**52):continue
   lo[i]=l;hi[i]=u;changed=True
   records.append(dict(constraint=ci,row=i,gcd=g,old_lower=lower,old_upper=upper,new_lower=l,new_upper=u,
    proof='EXACT_BINARY_INTEGER_ROW_LATTICE',original_audit_tolerance=tol,audit_domain_unchanged=True))
  # Do not mutate either bounds or coefficient storage in the input model.
  result.append(LinearConstraint(c.A,lo,hi) if changed else c)
 return result,records
