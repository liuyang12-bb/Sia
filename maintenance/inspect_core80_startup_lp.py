"""One offline relaxation of a preserved failed model, never training."""
import gzip,json,time,math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
R=Path(__file__).resolve().parents[1];O=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run2'
def main():
 d=json.loads(gzip.decompress((O/'tests/real80_FULL_12_required_bound/solver_failure.json.gz').read_bytes()));m=d['model'];As=[];bs=[]
 for t in m['constraints']:
  a=t['A'];A=sparse.coo_matrix((a['data'],(a['row'],a['col'])),shape=a['shape']).tocsr() if isinstance(a,dict) else sparse.csr_matrix(np.atleast_2d(a));lo=np.broadcast_to(t['lb'],(A.shape[0],));hi=np.broadcast_to(t['ub'],(A.shape[0],))
  u=np.isfinite(hi);l=np.isfinite(lo);As.extend([A[u],-A[l]]);bs.extend([hi[u],-lo[l]])
 A=sparse.vstack(As).tocsc();b=np.concatenate(bs);start=time.time();r=linprog(m['c'],A_ub=A,b_ub=b,bounds=(0,1),method='highs',options={'time_limit':30.})
 report=dict(status=int(r.status),message=r.message,fun=r.fun,seconds=time.time()-start,columns=len(m['c']),shape=A.shape,max_fractionality=float(np.max(abs(r.x-np.rint(r.x)))) if r.x is not None else None,rounded_objective=float(np.dot(m['c'],np.rint(r.x))) if r.x is not None else None)
 if r.x is not None:report['rounded_max_excess']=float(np.max(A@np.rint(r.x)-b))
 with (O/'tests/captured_step10_lp.json').open('x') as f:json.dump(report,f,indent=2)
 with gzip.open(O/'tests/captured_step10_lp_vectors.json.gz','xt') as f:json.dump(dict(x=r.x.tolist() if r.x is not None else None,dual=r.ineqlin.marginals.tolist() if r.ineqlin.marginals is not None else None),f)
 print(report)
if __name__=='__main__':main()
