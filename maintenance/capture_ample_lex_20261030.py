import gzip,json,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from joint_prototype.completion_continuity import NumericalAdmission,canonical_integral_solution
from breakpoint_study import build
from resource_study import CAPACITIES
parent=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1'
cap=json.loads((parent/'preregistration.json').read_text())['capacities']['AMPLE'];CAPACITIES['AMPLE']=tuple(cap)
original=NumericalAdmission._milp;last={};method=sys.argv[1];suffix='_'+sys.argv[2] if len(sys.argv)>2 else ''
def wrapper(self,cost,columns,constraints):
 result=original(self,cost,columns,constraints)
 if result.status==2:
  aliases={};identities=[]
  for i,c in enumerate(columns):aliases.setdefault(id(c),i);identities.append(aliases[id(c)])
  model=dict(c=np.asarray(cost).tolist(),columns=[dict(job=c.job_id,column_id=c.column_id,object_alias=identities[i]) for i,c in enumerate(columns)],constraints=[dict(A=np.asarray(v.A).tolist(),lb=np.asarray(v.lb).tolist(),ub=np.asarray(v.ub).tolist()) for v in constraints])
  witness=last.get(id(self));valid=False;reason=None
  if witness is not None:
   try:canonical_integral_solution(witness['x'],constraints);valid=True
   except ValueError as e:reason=str(e)
  if len(constraints)>1:
   with gzip.open(OUT/f'{method}{suffix}_failed_model.json.gz','xt',encoding='utf8') as f:json.dump(dict(status=int(result.status),message=str(result.message),model=model,previous=witness,previous_feasible_for_failed_rows=valid,witness_error=reason),f)
 elif result.status==0 and result.x is not None:last[id(self)]=dict(x=result.x.tolist(),fun=float(np.dot(cost,result.x)),constraints=len(constraints))
 return result
NumericalAdmission._milp=wrapper
c=build(method,'AMPLE','medium',OUT/'inputs/BASE_workload.csv')
try:
 for i in range(4):c.step(60);c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
except Exception as e:
 with (OUT/f'{method}{suffix}_red.json').open('x',encoding='utf8') as f:json.dump(dict(error=str(e),time=c.current_time,traceback=traceback.format_exc()),f,indent=2)
 print('RED_CAPTURED',method,str(e),c.current_time,flush=True)
else:raise RuntimeError('EXPECTED_RED_NOT_REPRODUCED')
