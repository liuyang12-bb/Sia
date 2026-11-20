import copy,gzip,json,sys,time
from pathlib import Path
from unittest.mock import patch
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-policy-rollout-v3';P=R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from core80 import build
from run_core80 import compact_decision,encode,gate
from joint_prototype.policy_rollout import PolicyRolloutMixin,install,CURRENT
from joint_prototype.release_evaluation import ReleaseEvaluationMixin,evaluate_reconfig_plan
from job import Job
from applications import Application

def main():
 gate(O);c=build('FULL',workload_path=P/'inputs/workload.csv',topology={'aws':2,'dgx-ext':2});install(c)
 original=c.policy.liveness.__class__;results=[];started=time.time()
 with gzip.open(P/'cells/NEW_FULL/rounds.jsonl.gz','rt',encoding='utf8') as f:history={r['interval_end']:r for r in map(json.loads,f) if r['interval_end']<=5820}
 class GateAdmission(original):
  def evaluate_completion_plan(self,*args):
   if getattr(self,'_legacy_gate',False):return evaluate_reconfig_plan(*args)
   return super().evaluate_completion_plan(*args)
  def refine_completion_bundle(self,s,cols,legacy,now):
   if now not in (4800,5820):return legacy,None
   fingerprint={j.name:(j.progress,j.execution_time,j.rescale_time) for j in c.jobs}
   # Hard forbid future ground truth access INSIDE forecast, not profile replay.
   with patch.object(Job,'step',side_effect=AssertionError('FORECAST_CALLED_JOB_STEP')),patch.object(Application,'get_throughput',side_effect=AssertionError('FUTURE_THROUGHPUT')),patch.object(Application,'get_grad_stats',side_effect=AssertionError('FUTURE_GRADIENT')):
    new,ev=PolicyRolloutMixin.refine_completion_bundle(self,s,cols,legacy,now)
   assert ev['status']=='OK'
   assert fingerprint=={j.name:(j.progress,j.execution_time,j.rescale_time) for j in c.jobs}
   results.append(dict(time=now,evaluation=copy.deepcopy(ev),real_state_unchanged=True,no_true_training_queries=True))
   print('STATE',now,ev['selected_plan'],[(p['name'],round(p['evaluation']['sum_completion'],3)) for p in ev['plans']],flush=True)
   # Restore the HISTORICAL V2 initial decision only to reach the second
   # recorded state. No new forecast decision changes the replayed prefix.
   self._legacy_gate=True
   try:chosen,old=ReleaseEvaluationMixin.refine_completion_bundle(self,s,cols,legacy,now)
   finally:self._legacy_gate=False
   assert {z.job_id:z.column_id for z in chosen}=={j:v['column_id'] for j,v in history[now]['decision_for_next_interval']['result']['selected'].items()}
   return chosen,old
 c.policy.liveness.__class__=GateAdmission
 for i in range(97):
  c.step(60);r=c.joint_round_logs[-1];d=r['decision_for_next_interval'];compact_decision(d);h=history[r['interval_end']]
  assert json.loads(json.dumps(r['execution'],default=encode).replace('cc-joint-policy-rollout-v3','cc-joint-release-reconfig-v2'))==h['execution'],('EXACT_PREFIX_EXECUTION',r['interval_end'])
  assert json.loads(json.dumps(d['probe']['specs'],default=encode))==h['decision_for_next_interval']['probe']['specs']
  assert json.loads(json.dumps(d['result']['selected'],default=encode).replace('cc-joint-policy-rollout-v3','cc-joint-release-reconfig-v2'))==h['decision_for_next_interval']['result']['selected']
  c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
  for key in ('exchange_columns','exchange_scores','continuity_scores','completion_scores','actual_completion_columns'):
   if hasattr(c.policy,key):setattr(c.policy,key,[])
 with (O/'same_state.json').open('x',encoding='utf8') as f:json.dump(dict(gate=len(results)==2,results=results,prefix_rounds_exact=97,wall=time.time()-started),f,indent=2,default=encode)
if __name__=='__main__':main()
