"""Same M3 t960 path, fixed16 rounds, real optimize and unchanged parameters."""
import dataclasses,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt2';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from breakpoint_study import build
from joint_prototype.execution_audit import audit_execution
def enc(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(str(type(x)))
def main():
 c=build('CC_CPU','R11','medium',OUT/'inputs/M3_workload.csv');conditioning=0;start=time.time()
 with (OUT/'bounded_16round_green.jsonl').open('x',encoding='utf8') as f:
  for i in range(16):
   c.step(60);r=c.joint_round_logs[-1];d=r['decision_for_next_interval'];assert d['result']['audit']['valid']
   for e in r['execution']:assert audit_execution(e)['valid']
   for record in d['result']['audit'].get('numerical_solver_retries',[]):
    if record.get('reformulation')=='REVERSIBLE_POWER_OF_TWO_ROW_SCALING':
     assert record['unscaled_original_rows_validated'];conditioning+=1
   f.write(json.dumps(r,default=enc,separators=(',',':'))+'\n')
   c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
 assert conditioning>0,'RECORDED_FAILURE_REPAIR_PATH_NOT_EXERCISED'
 with (OUT/'bounded_16round_summary.json').open('x',encoding='utf8') as f:json.dump(dict(valid=True,rounds=16,simulation_time=c.current_time,conditioning_recoveries=conditioning,all_original_rows_valid=True,wall_seconds=time.time()-start,full_cohort_JCT_not_computed=True),f,indent=2)
 print('BOUNDED_FAILURE_PATH_GREEN',conditioning)
if __name__=='__main__':main()
