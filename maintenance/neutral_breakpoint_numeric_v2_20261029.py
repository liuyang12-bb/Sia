import contextlib,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt2';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from breakpoint_study import build
from resource_study import METHODS
def main():
 traces={};tick=time.time()
 with (OUT/'neutral_acceptance.log').open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
  for method in METHODS:
   c=build(method,'R11','neutral',OUT/'small_gates/fixtures/MEM_three.csv');rows=[]
   for _ in range(5):
    c.step(60)
    assert c.policy.last_decision['result']['audit']['valid']
    assert all(v['rho']==1. for v in c.policy.last_decision['result']['selected'].values())
    rows.append(dict(allocations=c.allocations,states=[(j.name,j.progress,j.num_restarts,j.completion_time) for j in c.jobs]))
    c.joint_round_logs.clear();c.logs.clear()
   traces[method]=rows
  equal={m:traces[m]==traces['CC_P'] for m in METHODS if m.startswith('CC')}
  equal['SIA_AUX_vs_SIA_P']=traces['SIA_P']==traces['SIA_AUX']
  assert all(equal.values()),equal
 with (OUT/'neutral_acceptance.json').open('x',encoding='utf8') as f:json.dump(dict(valid=True,rounds=5,methods=6,all_rho_exact_one=True,trajectory_equal=equal,elapsed_seconds=time.time()-tick),f,indent=2)
 print('NEUTRAL_ACCEPTANCE_PASS')
if __name__=='__main__':main()
