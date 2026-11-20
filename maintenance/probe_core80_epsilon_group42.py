"""Offline budget decision with exact geometry binaries; no training."""
import gzip,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint,Bounds,milp
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-full-lex-repair-v7';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run4';sys.path[:0]=[str(T),str(T/'sia-simulator')]
from primary_band_presolve import primary_certificate
from capacity_cover import strengthen
from gpu_group_formulation import prepare
from joint_prototype.completion_continuity import canonical_integral_solution,objective_solver_units
def main():
 d=json.loads(gzip.decompress((T/'sia-simulator/tests/fixtures/core80_full_startup_step42.json.gz').read_bytes()));cost=np.asarray(d['model']['c']);rows=[]
 for r in d['model']['constraints']:
  a=r['A'];A=sparse.coo_matrix((a['data'],(a['row'],a['col'])),shape=a['shape']).tocsc() if isinstance(a,dict) else np.atleast_2d(a);rows.append(LinearConstraint(A,r['lb'],r['ub']))
 cols=[SimpleNamespace(**q) for q in d['model']['columns']];B=rows[0].A.tocsr();n=B.shape[0]-9;names=[cols[B.indices[B.indptr[i]]].job_id for i in range(n)];ctrl=SimpleNamespace(_repair_scope=True,_repair_job_order=names)
 start=time.perf_counter();strength=strengthen(ctrl,cols,rows);keep,proof=primary_certificate(strength,len(cost));p=sparse.csr_matrix(rows[1].A).toarray().ravel();q=[strength[0],LinearConstraint(cost.reshape(1,-1),-np.inf,220.)];q=[LinearConstraint(r.A[:,keep],r.lb,r.ub) for r in q];cg,rg,E=prepare(ctrl,cost[keep],[cols[i] for i in keep],q,True);c,scale=objective_solver_units(np.concatenate([-p[keep],np.zeros(E.shape[0])]))
 results=[]
 for preset in ({'time_limit':30.,'mip_rel_gap':0.},{'time_limit':30.,'mip_rel_gap':0.,'presolve':False}):
  tick=time.perf_counter();r=milp(c=c,integrality=np.ones(len(c)),bounds=Bounds(0.,1.),constraints=rg,options=preset);o=dict(status=int(r.status),message=str(r.message),seconds=time.perf_counter()-tick,options=preset)
  for k in ('mip_gap','mip_dual_bound','fun','mip_node_count'):
   if getattr(r,k,None) is not None:o[k]=float(getattr(r,k))
  if r.x is not None:
   x=np.zeros(len(cost));x[keep]=r.x[:len(keep)]
   try:
    x=canonical_integral_solution(x,[rows[0],LinearConstraint(cost.reshape(1,-1),-np.inf,220.)]);o.update(primary=float(p@x),startup=float(cost@x),query_point_valid=True)
    try:canonical_integral_solution(x,rows);o['original_rows_valid']=True
    except ValueError:o['original_rows_valid']=False
   except ValueError as e:o['query_error']=repr(e)
  results.append(o)
  if r.status==0:break
 out=dict(variables=len(cost),kept=len(keep),groups=E.shape[0],budget=220.,primary_band=[float(rows[1].lb[0]),float(rows[1].ub[0])],primary_proof=proof,results=results,total_seconds=time.perf_counter()-start)
 with (O/'probe_case42_epsilon_group.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
