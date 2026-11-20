"""Offline secondary-stage probe, no training simulation."""
import gzip,json,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint,Bounds,milp
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-full-lex-repair-v6';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run3';sys.path[:0]=[str(T),str(T/'sia-simulator')]
from primary_band_presolve import primary_certificate
from integer_row_lattice import tighten_integer_rows
from joint_prototype.completion_continuity import canonical_integral_solution
from scale_storage import sparse_condition
from joint_prototype import completion_continuity as cc
def main():
 d=json.loads(gzip.decompress((T/'sia-simulator/tests/fixtures/core80_full_startup_step37.json.gz').read_bytes()));c=np.asarray(d['model']['c']);rows=[]
 for v in d['model']['constraints']:
  a=v['A'];A=sparse.coo_matrix((a['data'],(a['row'],a['col'])),shape=a['shape']).tocsc() if isinstance(a,dict) else np.atleast_2d(a);rows.append(LinearConstraint(A,v['lb'],v['ub']))
 assert len(rows)==2;start=time.perf_counter()
 # The dummy zero row contributes no constraint; this is precisely the
 # proposed LP presolve with no already-fixed startup optimum.
 keep,proof=primary_certificate(rows+[LinearConstraint(sparse.csc_matrix((1,len(c))),0.,0.)],len(c))
 rs,_=tighten_integer_rows(rows);cc._core80_original_condition=cc.condition_constraint_bands;rs,_=sparse_condition(rs)
 rs=[LinearConstraint(r.A[:,keep],r.lb,r.ub) for r in rs]
 tick=time.perf_counter();x=milp(c=c[keep],integrality=np.ones(len(keep)),bounds=Bounds(0.,1.),constraints=rs,options={'time_limit':30.,'mip_rel_gap':0.});ans=dict(variables=len(c),kept=len(keep),proof=proof,status=int(x.status),success=bool(x.success),message=str(x.message),native_seconds=time.perf_counter()-tick,total_seconds=time.perf_counter()-start)
 if x.x is not None:
  vector=np.zeros(len(c));vector[keep]=x.x
  try:valid=canonical_integral_solution(vector,rows);ans.update(original_rows_valid=True,objective=float(c@valid),selected=np.flatnonzero(valid).tolist())
  except ValueError as e:ans['validation_error']=repr(e)
 with (O/'probe_secondary_primary_domain.json').open('x',encoding='utf8') as f:json.dump(ans,f,ensure_ascii=False,indent=2)
 print(json.dumps(ans),flush=True)
if __name__=='__main__':main()
