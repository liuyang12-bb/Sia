"""Validate a known better incumbent and one exact common-objective rescaling."""
import gzip,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';OUT=BASE/'attempt2/bounded_failure_diagnosis'
def main():
 d=json.loads(gzip.open(BASE/'numeric_diagnosis/solver_diagnostics.json.gz','rt',encoding='utf8').read())
 state=json.loads(gzip.open(OUT/'traceback_state.json.gz','rt',encoding='utf8').read())
 frame=state['frames'][0];model=d['model'];ids=[c['column_id'] for c in model['columns']]
 selected={c['column_id'] for c in frame['baseline']};assert selected<=set(ids)
 incumbent=np.array([float(x in selected) for x in ids]);row=model['constraints'][0]
 a=np.asarray(row['A']);lo=np.broadcast_to(row['lb'],(len(a),));hi=np.broadcast_to(row['ub'],(len(a),));values=a@incumbent
 assert np.all(values>=lo-1e-10) and np.all(values<=hi+1e-10)
 weights=np.asarray(model['constraints'][1]['A']).reshape(-1)
 baseline=float(weights@incumbent);assert abs(baseline-frame['comparison']['baseline_objective'])<1e-12
 scale=2.**math.ceil(-math.log2(1e-9));cost=-weights*scale
 assert np.array_equal(cost/scale,-weights)
 tick=time.time();result=milp(c=cost,integrality=np.ones(len(ids)),bounds=Bounds(0,1),constraints=LinearConstraint(a,lo,hi),options=d['retry_options'])
 assert result.success and result.status==0
 x=np.rint(result.x);v=a@x;assert np.all(v>=lo-1e-10) and np.all(v<=hi+1e-10)
 value=float(weights@x);assert value>=baseline-1e-10
 data=dict(valid=True,known_feasible_baseline=baseline,old_reported_optimum=frame['comparison']['chosen_objective'],
    false_optimality_gap=baseline-frame['comparison']['chosen_objective'],
    known_incumbent_obeys_original_admission_and_resources=True,cost_scale=scale,scale_source='2^ceil(-log2(existing primary half-band1e-9)); common positive power-of-two, relative weights and ordering unchanged',
    cost_roundtrip_bit_exact=True,scaled_primary_objective=value,improvement_over_known_incumbent=value-baseline,
    unchanged_solver_options=d['retry_options'],wall_seconds=time.time()-tick,
    selected_column_ids=[v for v,k in zip(ids,x) if k>.5])
 with (OUT/'primary_precision_proof.json').open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
 print(json.dumps(data,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
