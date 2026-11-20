import json,sys
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/sys.argv[1]
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from core80 import build
from run_core80 import compact_decision,encode
c=build('FULL',workload_path=R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1/inputs/workload.csv',topology={'aws':2,'dgx-ext':2});records=[]
for _ in range(3):
 c.step(60);r=c.joint_round_logs[-1];d=r['decision_for_next_interval'];compact_decision(d)
 records.append(dict(time=c.current_time,execution=r['execution'],selected=d['result']['selected'],status=d['result']['status'],
  jobs={j.name:dict(progress=j.progress,epoch=j.epoch,restarts=j.num_restarts,rescale=j.rescale_time,completion=j.completion_time) for j in c.jobs}))
print('FULL_PROBE='+json.dumps(records,default=encode).replace(T.name,'SOURCE_ROOT'))
