import collections,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1';SRC=OUT/'source';sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from breakpoint_study import build
from resource_study import CAPACITIES
from joint_prototype.execution_audit import audit_execution
from run_stage2c_cell import independent_rho
CAPACITIES['AMPLE']=tuple(json.loads((ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1/preregistration.json').read_text())['capacities']['AMPLE'])
result={}
for method in ['CC_CPU','CC_MEM']:
 c=build(method,'AMPLE','medium',OUT/'inputs/BASE_workload.csv');statuses=collections.Counter()
 with gzip.open(OUT/f'{method}_bounded_green.jsonl.gz','xt',encoding='utf8') as f:
  for i in range(8):
   c.step(60);r=c.joint_round_logs[-1];d=r['decision_for_next_interval'];selected=d['result']['selected'];assert d['result']['audit']['valid']
   specs={q['job_id']:q for q in d['probe']['specs']};assert {j for j,q in specs.items() if q['checkpoint_locked']}<=set(selected)
   for e in r['execution']:assert audit_execution(e)['valid']
   for j,col in selected.items():assert abs(col['rho']-independent_rho(c.policy.oracle,j,col['config']['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu']))<1e-10
   statuses[d['result']['status']]+=1;f.write(json.dumps(r,default=lambda v:v.tolist() if hasattr(v,'tolist') else v.__dict__)+'\n')
   c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
 result[method]=dict(rounds=8,time=c.current_time,passed_original_failure_time=True,statuses=dict(statuses),progress={j.name:j.progress for j in c.jobs})
with (OUT/'bounded_green.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print('REAL_PATH8_ROUNDS_PASS',result,flush=True)
