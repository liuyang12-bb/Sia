"""Bijection-preserving GPU-geometry binaries for the startup MIP.

The flat model has thousands of auxiliary columns with the SAME job/GPU/
placement startup cost. Introduce y_g=sum(x_i for i in g), binary because
the original job row already permits at most one column. Move uniform rows
(job/GPU capacity) and uniform startup coefficients onto y. CPU/RAM and
nonuniform performance rows stay on x. No new allocation is expressible;
no old binary allocation is removed; objective identity is exact.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint
from joint_prototype.completion_continuity import canonical_integral_solution

def secondary_phase(cost,constraints):
 if len(constraints)!=2 or np.any(np.asarray(cost)<0):return False
 row=constraints[1];lo=np.asarray(row.lb).reshape(-1);hi=np.asarray(row.ub).reshape(-1)
 return len(lo)==len(hi)==1 and np.isfinite(lo[0]) and np.isfinite(hi[0]) and 0<=hi[0]-lo[0]<1e-6

def prepare(self,cost,columns,constraints,enabled):
 if not enabled or not getattr(self,'_repair_scope',False):return None
 order=getattr(self,'_repair_job_order',None)
 if not order:return None
 jobs={j:i for i,j in enumerate(order)};base=sparse.csr_matrix(constraints[0].A);upper=np.broadcast_to(constraints[0].ub,(base.shape[0],));groups={};ids=[];first=[]
 for i,c in enumerate(columns):
  parts=c.column_id.split('|')
  if c.job_id not in jobs:return None
  # LOCK/unknown identifiers are SINGLETONS, never a grant of lock legality.
  # Candidate/lock validity still comes from the current original constraints.
  key=(c.job_id,parts[1],parts[4]) if len(parts)==5 else (c.job_id,'SINGLETON',i)
  if key not in groups:groups[key]=len(groups);first.append(i)
  ids.append(groups[key])
 ids=np.asarray(ids,dtype=int);first=np.asarray(first,dtype=int);n=len(columns);g=len(first)
 if g==n or not g:return None
 c=np.asarray(cost,dtype=float);cg=c[first]
 if not np.array_equal(c,cg[ids]):return None
 # Prove group sums are binary from CURRENT original job rows, not labels.
 for name,row in jobs.items():
  own=np.asarray([i for i,v in enumerate(columns) if v.job_id==name],dtype=int)
  if not len(own):continue
  a=base.getrow(row)
  if upper[row]>1. or np.any(a.data<0) or not np.all(a[:,own].toarray()==1.):return None
 transformed=[]
 for row in constraints:
  A=sparse.csr_matrix(row.A);Y=A[:,first];diff=A-Y[:,ids];diff.eliminate_zeros();uniform=np.diff(diff.indptr)==0
  X=A.multiply((~uniform)[:,None]);Y=Y.multiply(uniform[:,None])
  transformed.append(LinearConstraint(sparse.hstack([X,Y]).tocsc(),row.lb,row.ub))
 E=sparse.csc_matrix((np.ones(n),(ids,np.arange(n))),shape=(g,n))
 transformed.append(LinearConstraint(sparse.hstack([E,-sparse.eye(g)]).tocsc(),np.zeros(g),np.zeros(g)))
 return np.concatenate([np.zeros(n),cg]),transformed,E

def solve(self,cost,columns,constraints,backend,enabled):
 model=prepare(self,cost,columns,constraints,enabled)
 if model is None:return backend(cost,constraints)
 c,rows,E=model;result=backend(c,rows);n=len(columns)
 self._repair_gpu_group_calls=getattr(self,'_repair_gpu_group_calls',0)+1;self._repair_gpu_group_peak=max(getattr(self,'_repair_gpu_group_peak',0),E.shape[0])
 if result.x is not None:
  original=np.asarray(result.x)[:n]
  if result.status==0 and result.success:
   try:
    whole=canonical_integral_solution(result.x,rows);point=canonical_integral_solution(whole[:n],constraints)
    if not np.array_equal(E@point,whole[n:]) or not np.array_equal(E.T@c[n:],np.asarray(cost)):raise ValueError('EXTENDED_OBJECTIVE_OR_LINK_MISMATCH')
    original=point
   except ValueError:
    result.x=None;result.success=False;result.message=str(result.message)+'; EXTENDED_OPTIMUM_FAILED_ORIGINAL_CHECKS';return result
  result.x=original;result.fun=float(np.asarray(cost)@original)
 result.message=str(result.message)+f'; EXACT_GPU_GROUP_BINARIES={E.shape[0]}'
 return result
