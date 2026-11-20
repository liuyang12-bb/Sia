"""Exact binary complement around a CURRENT validated feasible incumbent.

x = anchor + (1-2*anchor)*v. Exact dyadic row offsets are decomposed into
representable doubles on fixed-one variables, so original bounds/objective
are not rounded or widened. v=0 is a feasible origin (after fixed-one vars).
This is a bijection, not a policy fallback or an accepted unproved solution.
"""
from fractions import Fraction
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint
from joint_prototype.completion_continuity import canonical_integral_solution

def components(value):
 result=[]
 while value:
  v=float(value)
  if not np.isfinite(v) or v==0.:raise ValueError('UNREPRESENTABLE_DYADIC_OFFSET')
  result.append(v);value-=Fraction.from_float(v)
 return result

def prepare(cost,rows,anchor):
 anchor=canonical_integral_solution(anchor,rows);n=len(anchor);selected=np.flatnonzero(anchor);sign=1.-2.*anchor;parts=[];width=0
 for c in rows:
  B=sparse.csr_matrix(c.A)[:,selected];bias=[]
  for i in range(B.shape[0]):
   val=sum((Fraction.from_float(float(x)) for x in B.data[B.indptr[i]:B.indptr[i+1]]),Fraction(0));p=components(val);bias.append(p);width=max(width,len(p))
  parts.append(bias)
 cp=components(sum((Fraction.from_float(float(cost[i])) for i in selected),Fraction(0)));width=max(width,len(cp),1);answer=[]
 for c,bias in zip(rows,parts):
  C=np.zeros((len(bias),width))
  for i,v in enumerate(bias):C[i,:len(v)]=v
  A=sparse.csr_matrix(c.A).multiply(sign[None,:]);answer.append(LinearConstraint(sparse.hstack([A,sparse.csc_matrix(C)]).tocsc(),c.lb,c.ub))
 answer.append(LinearConstraint(sparse.hstack([sparse.csc_matrix((width,n)),sparse.eye(width)]).tocsc(),np.ones(width),np.ones(width)))
 c=np.zeros(n+width);c[:n]=np.asarray(cost)*sign;c[n:n+len(cp)]=cp
 return c,answer,anchor,sign

def solve(cost,rows,anchor,backend):
 c,transformed,a,sign=prepare(cost,rows,anchor);r=backend(c,transformed);n=len(a)
 if r.x is not None:
  point=a+sign*np.asarray(r.x)[:n]
  if r.status==0 and r.success:
   try:
    v=canonical_integral_solution(r.x,transformed);point=canonical_integral_solution(a+sign*v[:n],rows)
   except ValueError:
    r.x=None;r.success=False;r.message=str(r.message)+'; COMPLEMENT_FAILED_ORIGINAL_ROWS';return r
  r.x=point;r.fun=float(np.asarray(cost)@point)
 r.message=str(r.message)+'; EXACT_CURRENT_INCUMBENT_COMPLEMENT'
 return r
