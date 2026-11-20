"""Offline exact lex startup budget query, no simulation or parameter search."""
import gzip,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint,Bounds,milp
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-full-lex-repair-v7';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run4';sys.path[:0]=[str(T),str(T/'sia-simulator')]
from primary_band_presolve import primary_certificate
from joint_prototype.completion_continuity import canonical_integral_solution,objective_solver_units
def main():
 d=json.loads(gzip.decompress((T/'sia-simulator/tests/fixtures/core80_full_startup_step42.json.gz').read_bytes()));cost=np.asarray(d['model']['c']);rows=[]
 for r in d['model']['constraints']:
  a=r['A'];A=sparse.coo_matrix((a['data'],(a['row'],a['col'])),shape=a['shape']).tocsc() if isinstance(a,dict) else np.atleast_2d(a);rows.append(LinearConstraint(A,r['lb'],r['ub']))
 start=time.perf_counter();keep,proof=primary_certificate(rows,len(cost));p=sparse.csr_matrix(rows[1].A).toarray().ravel();query=[rows[0],LinearConstraint(rows[1].A,-np.inf,rows[1].ub),LinearConstraint(cost.reshape(1,-1),-np.inf,220.)];query=[LinearConstraint(r.A[:,keep],r.lb,r.ub) for r in query];c,scale=objective_solver_units(-p[keep]);tick=time.perf_counter();r=milp(c=c,integrality=np.ones(len(keep)),bounds=Bounds(0.,1.),constraints=query,options={'time_limit':30.,'mip_rel_gap':0.})
 out=dict(variables=len(cost),kept=len(keep),budget=220.,primary_band=[float(rows[1].lb[0]),float(rows[1].ub[0])],primary_proof=proof,status=int(r.status),message=str(r.message),native_seconds=time.perf_counter()-tick,total_seconds=time.perf_counter()-start)
 for k in ('mip_gap','mip_dual_bound','fun','mip_node_count'):
  if getattr(r,k,None) is not None:out[k]=float(getattr(r,k))
 if r.x is not None:
  x=np.zeros(len(cost));x[keep]=r.x
  try:
   x=canonical_integral_solution(x,[rows[0],LinearConstraint(cost.reshape(1,-1),-np.inf,220.)]);out.update(primary=float(p@x),startup=float(cost@x),query_point_valid=True)
   try:canonical_integral_solution(x,rows);out['original_rows_valid']=True
   except ValueError:out['original_rows_valid']=False
  except ValueError as e:out['query_error']=repr(e)
 with (O/'probe_case42_epsilon.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
