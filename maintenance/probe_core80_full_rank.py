"""Offline replay only; no Job/Cluster simulation. Preserve every probe."""
import argparse,gzip,json,sys,time,hashlib,math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint,Bounds,milp
R=Path(__file__).resolve().parents[1];O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run1'
def main():
 p=argparse.ArgumentParser();p.add_argument('variant',choices=['old','lattice','primary','repair']);a=p.parse_args()
 T=R/('cc-joint-core80-repair-v3' if a.variant=='old' else 'cc-joint-core80-full-lex-repair-v4');sys.path[:0]=[str(T),str(T/'sia-simulator')]
 src=R/'research_runs/sia_synergy/cc_core80_repair_full_20261031_run1/cells/dual_medium_FULL/failure_solver_diagnostics.json.gz';d=json.loads(gzip.decompress(src.read_bytes()));cost=np.array(d['model']['c']);cs=[]
 for row in d['model']['constraints']:
  v=row['A'];A=sparse.coo_matrix((v['data'],(v['row'],v['col'])),shape=v['shape']).tocsc() if isinstance(v,dict) else np.atleast_2d(v);cs.append(LinearConstraint(A,row['lb'],row['ub']))
 out=dict(variant=a.variant,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest().upper(),variables=len(cost),started=time.time());start=time.perf_counter()
 from joint_prototype.completion_continuity import canonical_integral_solution
 if a.variant in ('old','repair'):
  from types import SimpleNamespace
  from engineering_repair import repair_context
  from joint_prototype.completion_continuity import NumericalAdmission
  controller=NumericalAdmission(300,250,60);controller.numerical_solver_retries=[];controller._repair_scope=True;controller._repair_allow_cbc=True
  cols=[SimpleNamespace(**q) for q in d['model']['columns']];controller._repair_last_columns=cols;controller._repair_last_x=np.rint(d['last_valid_x'])
  B=cs[0].A.tocsr();njobs=B.shape[0]-9
  controller._repair_job_order=[cols[B.indices[B.indptr[i]]].job_id for i in range(njobs)]
  try:
   with repair_context(None):ans=controller._milp(cost,cols,cs)
   out.update(status=int(ans.status),success=bool(ans.success),message=str(ans.message),objective=float(cost@canonical_integral_solution(ans.x,cs)) if ans.x is not None else None,
    primary_band_variables_removed=getattr(controller,'_repair_primary_band_variables_removed',0),primary_certificates=getattr(controller,'_repair_primary_certificates',[]))
  except Exception as e:out.update(error=repr(e),expected_red=True)
 else:
  from integer_row_lattice import tighten_integer_rows
  rows,proof=tighten_integer_rows(cs);out['lattice_rows']=proof
  if a.variant=='primary':
   from dual_certificate import certificate
   cp=-np.asarray(cs[1].A.toarray() if sparse.issparse(cs[1].A) else cs[1].A).reshape(-1);pr=certificate(cp,[rows[0],rows[2]],np.rint(d['last_valid_x']))
   from decimal import Decimal
   cutoff=Decimal.from_float(float(-np.asarray(cs[1].lb).item()+1e-10))
   eliminate=[i for i,v in enumerate(pr['_residuals']) if pr['_bound']+max(v,Decimal(0))>cutoff]
   keep=np.setdiff1d(np.arange(len(cost)),eliminate);out['primary_row_eliminated']=len(eliminate);out['primary_bound']=str(pr['_bound']);out['primary_cutoff']=str(cutoff)
  else:keep=np.arange(len(cost))
  from scale_storage import sparse_condition
  from joint_prototype import completion_continuity as cc
  cc._core80_original_condition=cc.condition_constraint_bands
  rows,_=sparse_condition(rows);rows=[LinearConstraint(c.A[:,keep],c.lb,c.ub) for c in rows]
  ans=milp(c=cost[keep],integrality=np.ones(len(keep)),bounds=Bounds(0,1),constraints=rows,options={'time_limit':30.,'mip_rel_gap':0.})
  out.update(status=int(ans.status),success=bool(ans.success),message=str(ans.message),solver_dual_bound=getattr(ans,'mip_dual_bound',None))
  if ans.x is not None:
   x=np.zeros(len(cost));x[keep]=ans.x
   try:x=canonical_integral_solution(x,cs);out.update(objective=float(cost@x),original_rows_valid=True,selected=np.flatnonzero(x).tolist())
   except ValueError as e:out['original_rows_error']=repr(e)
 out['seconds']=time.perf_counter()-start
 with (O/('probe_'+a.variant+'.json')).open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
