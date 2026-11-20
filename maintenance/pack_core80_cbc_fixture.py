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

 import hashlib
 result=dict(source_failure_sha256=hashlib.sha256((O/'tests/real80_FULL_12_primal_dual/solver_failure.json.gz').read_bytes()).hexdigest(),original_variables=len(cost),proof='Same job/config/placement Pareto variable substitution only',expected_startup=295.,cost=cost[kept].tolist(),constraints=[])
 for c in rows:
  a=c.A[:,kept].tocoo();result['constraints'].append(dict(A=dict(data=a.data.tolist(),row=a.row.tolist(),col=a.col.tolist(),shape=a.shape),lb=np.broadcast_to(c.lb,(a.shape[0],)).tolist(),ub=np.broadcast_to(c.ub,(a.shape[0],)).tolist()))
 with gzip.open(T/'sia-simulator/tests/fixtures/core80_failed_startup_reduced.json.gz','xt',encoding='utf8') as f:json.dump(result,f)
 print('Captured CBC regression fixture',len(kept))
if __name__=='__main__':main()
