"""One equivalent static startup MIP replay, no Job/Cluster/training."""
import collections,gzip,json,sys,time
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
from scipy import sparse
from scipy.optimize import milp,Bounds,LinearConstraint
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-repair-v3';O=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run3'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from exact_grid_presolve import indices
from scale_storage import sparse_condition
def main():
 d=json.loads(gzip.decompress((O/'tests/real80_FULL_12_primal_dual/solver_failure.json.gz').read_bytes()));m=d['model'];rows=[]
 for t in m['constraints']:
  a=t['A'];a=sparse.coo_matrix((a['data'],(a['row'],a['col'])),shape=a['shape']).tocsc() if isinstance(a,dict) else sparse.csc_matrix(np.atleast_2d(a));rows.append(LinearConstraint(a,t['lb'],t['ub']))
 base=rows[0].A;N=base.shape[0]-9;g=np.asarray(base[[N,N+3,N+6]].sum(axis=0)).ravel();cpu=np.asarray(base[[N+1,N+4,N+7]].sum(axis=0)).ravel();ram=np.asarray(base[[N+2,N+5,N+8]].sum(axis=0)).ravel();u=np.asarray(rows[1].A.toarray()).ravel()
 cols=[S(job_id=c['job_id'],config=S(config_id=c['column_id'].split('|')[1]),placement=S(node_gpus=c['column_id'].split('|')[-1]),query=S(cpu_per_gpu=cpu[i],ram_gb_per_gpu=ram[i]),scheduling_utility=u[i]) for i,c in enumerate(m['columns'])]
 kept=indices(cols);cost=np.asarray(m['c']);groups=collections.defaultdict(list)
 for local,i in enumerate(kept):groups[cols[i].job_id,cols[i].config.config_id,cols[i].placement.node_gpus].append(local)

 import cvxpy as cp
 cr,_=sparse_condition(rows);x=cp.Variable(len(kept),boolean=True);rules=[]
 for c in cr:
  lo=np.broadcast_to(c.lb,(c.A.shape[0],));hi=np.broadcast_to(c.ub,(c.A.shape[0],));good=np.isfinite(lo);rules.append(c.A[good][:,kept]@x>=lo[good]);good=np.isfinite(hi);rules.append(c.A[good][:,kept]@x<=hi[good])
 problem=cp.Problem(cp.Minimize(cost[kept]@x),rules);start=time.time();problem.solve(solver=cp.CBC,maximumSeconds=30.,allowableFractionGap=0.,verbose=False)
 record=dict(status=problem.status,objective=problem.value,wall_seconds=time.time()-start,variables=len(kept),original_variables=len(cost))
 if x.value is not None:
  full=np.zeros(len(cost));full[kept]=np.rint(x.value);record['rounded_objective']=float(cost@full);record['max_integrality']=float(np.max(abs(x.value-np.rint(x.value))));record['max_original_row_excess']=max(float(np.max(np.maximum(c.lb-c.A@full,c.A@full-c.ub))) for c in rows)
 with (O/'tests/static_existing_cbc.json').open('x') as f:json.dump(record,f,indent=2)
 print(record)
if __name__=='__main__':main()
