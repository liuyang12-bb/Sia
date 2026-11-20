"""Exact threshold-cover inequalities; no binary feasible point is removed.

If k+1 selected allocations each use more than C/(k+1), their total
exceeds capacity C. Account conservatively for the existing validator's
tolerance and floating summation error. All k are derived from the physical
GPU capacity and active-job count, not fitted to an observed result.
"""
import hashlib
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint
from primary_band_presolve import UP,D,dot_error,upper_float
from decimal import Decimal
from scale_storage import matrix_bytes

def strengthen(self,columns,constraints):
 if not getattr(self,'_repair_scope',False) or not constraints:return constraints
 jobs=getattr(self,'_repair_job_order',None)
 if not jobs:return constraints
 base=constraints[0];A=sparse.csr_matrix(base.A);n=len(jobs);extra=A.shape[0]-n
 if extra<=0 or extra%3:return constraints
 lo=np.broadcast_to(base.lb,(A.shape[0],));hi=np.broadcast_to(base.ub,(A.shape[0],))
 h=hashlib.sha256(matrix_bytes(A[n:]));h.update(matrix_bytes(hi[n:]));h.update(str(n).encode());signature=h.hexdigest()
 old=getattr(self,'_repair_capacity_cover_cache',None)
 if old is None or old[0] is not columns or old[1]!=signature:
  rr=[];cc=[];values=[];upper=[];proofs=[]
  for start in range(n,A.shape[0],3):
   g=float(hi[start])
   if not np.isfinite(g) or g<1 or g!=round(g):continue
   for row in (start+1,start+2):
    v=A.getrow(row);cap=float(hi[row])
    if not np.isfinite(cap) or cap<=0 or np.any(v.data<0) or not np.all(np.isfinite(v.data)):continue
    tol=1e-10 if np.isfinite(lo[row]) and cap-lo[row]<1e-6 else 1e-7
    bound=UP.add(UP.add(D(cap),D(tol)),dot_error(v.data,len(columns)));previous=None
    for k in range(1,min(int(g),n)):
     threshold=upper_float(UP.divide(bound,Decimal(k+1)));indices=v.indices[v.data>threshold]
     if len(indices)<=k or previous is not None and np.array_equal(indices,previous):continue
     if len({columns[int(i)].job_id for i in indices})<=k:continue
     previous=indices.copy();idx=len(upper);rr.extend([idx]*len(indices));cc.extend(indices.tolist());values.extend([1.]*len(indices));upper.append(float(k))
     if len(proofs)<12:proofs.append(dict(original_row=row,k=k,threshold=threshold,capacity=cap,columns=len(indices),proof='K_PLUS_ONE_STRICTLY_EXCEEDS_ORIGINAL_VALIDATOR_CAPACITY'))
  cuts=sparse.csc_matrix((values,(rr,cc)),shape=(len(upper),len(columns)));upper=np.asarray(upper)
  self._repair_capacity_cover_cache=(columns,signature,cuts,upper,proofs)
 else:cuts,upper,proofs=old[2:]
 if not len(upper):return constraints
 self._repair_capacity_cover_peak=max(getattr(self,'_repair_capacity_cover_peak',0),len(upper));self._repair_capacity_cover_proofs=proofs
 first=LinearConstraint(sparse.vstack([A,cuts]).tocsc(),np.concatenate([lo,np.full(len(upper),-np.inf)]),np.concatenate([hi,upper]))
 return [first,*constraints[1:]]
