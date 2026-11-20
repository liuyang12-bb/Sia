"""One offline captured-model cover-cut check; no simulation or tuning."""
import gzip,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint,Bounds,milp
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-full-lex-repair-v7';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run4';sys.path[:0]=[str(T),str(T/'sia-simulator')]
from capacity_cover import strengthen
from gpu_group_formulation import prepare
from primary_band_presolve import primary_certificate
from integer_row_lattice import tighten_integer_rows
from joint_prototype.completion_continuity import canonical_integral_solution
from scale_storage import sparse_condition
from joint_prototype import completion_continuity as cc
def main():
 d=json.loads(gzip.decompress((T/'sia-simulator/tests/fixtures/core80_full_startup_step42.json.gz').read_bytes()));cost=np.asarray(d['model']['c']);rows=[]
 for r in d['model']['constraints']:
  a=r['A'];A=sparse.coo_matrix((a['data'],(a['row'],a['col'])),shape=a['shape']).tocsc() if isinstance(a,dict) else np.atleast_2d(a);rows.append(LinearConstraint(A,r['lb'],r['ub']))
 cols=[SimpleNamespace(**q) for q in d['model']['columns']];B=rows[0].A.tocsr();n=B.shape[0]-9;names=[cols[B.indices[B.indptr[i]]].job_id for i in range(n)];ctrl=SimpleNamespace(_repair_scope=True,_repair_job_order=names)
 start=time.perf_counter();rs,_=tighten_integer_rows(rows);rs=strengthen(ctrl,cols,rs);keep,proof=primary_certificate(rs,len(cost));cc._core80_original_condition=cc.condition_constraint_bands;rs,_=sparse_condition(rs);rs=[LinearConstraint(r.A[:,keep],r.lb,r.ub) for r in rs]
 tick=time.perf_counter();group=prepare(ctrl,cost[keep],[cols[i] for i in keep],rs,True);cg,rg,E=group;r=milp(c=cg,integrality=np.ones(len(cg)),bounds=Bounds(0.,1.),constraints=rg,options={'time_limit':30.,'mip_rel_gap':0.});original_r=r.x[:len(keep)] if r.x is not None else None
 out=dict(variables=len(cost),kept=len(keep),gpu_groups=E.shape[0],cover_rows=ctrl._repair_capacity_cover_peak,cover_proofs=ctrl._repair_capacity_cover_proofs,primary_proof=proof,status=int(r.status),success=bool(r.success),message=str(r.message),native_seconds=time.perf_counter()-tick,total_seconds=time.perf_counter()-start)
 if r.x is not None:
  x=np.zeros(len(cost));x[keep]=original_r
  try:x=canonical_integral_solution(x,rows);out.update(objective=float(cost@x),original_rows_valid=True,selected=np.flatnonzero(x).tolist())
  except ValueError as e:out['original_rows_error']=repr(e)
 with (O/'probe_case42_group.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
