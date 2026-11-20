import gzip,json,sys,collections
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import LinearConstraint
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1'
sys.path.insert(0,str(OUT/'source/sia-simulator'))
from joint_prototype.completion_continuity import NumericalAdmission,canonical_integral_solution
proof={}
for method in ['CC_CPU','CC_MEM']:
 d=json.load(gzip.open(OUT/f'{method}_failed_model.json.gz','rt'));m=d['model'];rows=[LinearConstraint(v['A'],v['lb'],v['ub']) for v in m['constraints']];cols=[SimpleNamespace(job_id=c['job'],column_id=c['column_id']) for c in m['columns']]
 a=NumericalAdmission(300.,250.,60.);a.numerical_solver_retries=[];a._diagnostic_job_order=[]
 old=a._milp(np.asarray(m['c']),cols,rows);assert old.status==2
 # Directly minimize the original secondary cost, keeping original resource
 # and primary optimum rows. No future or performance data enter this proof.
 sec=np.asarray(rows[2].A).reshape(-1);opt=a._milp(sec,cols,rows[:2]);assert opt.status==0
 canonical_integral_solution(opt.x,rows[:2]);best=float(sec@opt.x)
 fixed=rows[:2]+[LinearConstraint(sec,best-1e-9,best+1e-9)]+rows[3:]
 new=a._milp(np.asarray(m['c']),cols,fixed);assert new.status==0;canonical_integral_solution(new.x,fixed)
 proof[method]=dict(columns=len(cols),old_status=int(old.status),wrong_secondary_lower=float(np.asarray(rows[2].lb).reshape(-1)[0]),independent_minimum_startup=best,repaired_status=int(new.status),resource_and_primary_rows_unchanged=True,solver_settings_unchanged=True)
with (OUT/'independent_model_proof.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2)
print(proof)
