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
 n=len(kept);rr=[];cc=[];vv=[];z=[]
 for k,ix in enumerate(groups.values()):
  assert len({cost[kept[i]] for i in ix})==1
  z.append(cost[kept[ix[0]]]);rr.extend([k]*(len(ix)+1));cc.extend(ix+[n+k]);vv.extend([1.]*len(ix)+[-1.])
 link=sparse.csc_matrix((vv,(rr,cc)),shape=(len(groups),n+len(groups)));cr,_=sparse_condition(rows)
 ext=[LinearConstraint(sparse.hstack([c.A[:,kept],sparse.csc_matrix((c.A.shape[0],len(groups)))]).tocsc(),c.lb,c.ub) for c in cr];ext.append(LinearConstraint(link,0.,0.))
 start=time.time();res=milp(c=np.r_[np.zeros(n),z],integrality=np.ones(n+len(groups)),bounds=Bounds(np.zeros(n+len(groups)),np.ones(n+len(groups))),constraints=ext,options={'time_limit':30.,'mip_rel_gap':0.})
 record=dict(status=int(res.status),success=bool(res.success),message=res.message,objective=float(res.fun) if res.fun is not None else None,wall_seconds=time.time()-start,original_variables=len(cost),nondominated_variables=n,selector_variables=len(groups))
 if res.x is not None:
  x=np.zeros(len(cost));x[kept]=np.rint(res.x[:n]);record['rounded_objective']=float(cost@x);record['max_original_row_excess']=max(float(np.max(np.maximum(c.lb-c.A@x,c.A@x-c.ub))) for c in rows)
 with (O/'tests/static_selector_extension.json').open('x') as f:json.dump(record,f,indent=2)
 print(record)
if __name__=='__main__':main()
