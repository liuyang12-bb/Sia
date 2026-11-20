import collections,gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TREE=ROOT/'cc-joint-core80-8h-v3';OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'
sys.path[:0]=[str(TREE),str(TREE/'sia-simulator')]
from core80 import build
from run_core80 import Metrics,compact_decision,encode,save
start=time.perf_counter();c=build('FULL');m=Metrics(c);stats=[]
with gzip.open(OUT/'tests/real80_full_bounded_rounds.jsonl.gz','xt',encoding='utf8') as f:
 for i in range(3):
  tick=time.perf_counter();c.step(60);r=c.joint_round_logs[-1];dt=time.perf_counter()-tick;m.consume(c,r)
  n=len(c.policy.exchange_columns);compact_decision(r['decision_for_next_interval']);f.write(json.dumps(r,default=encode)+'\n');f.flush()
  stats.append(dict(round=i+1,seconds=dt,actual_scored_columns=n,seen=len(c.policy.liveness.seen),ready=len(c.policy.liveness.ready),services=len(c.policy.liveness.services),capacity_audit=c.policy.last_decision['result']['audit']['valid'],positive_training_jobs=sum(j.execution_time>0 for j in c.jobs)))
  print('REAL80_FULL_BOUNDARY',stats[-1],flush=True)
  c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
  for name in ('exchange_columns','exchange_scores','continuity_scores','completion_scores'):
   if hasattr(c.policy,name):setattr(c.policy,name,[])
save(OUT/'tests/real80_full_bounded.json',dict(status='PASS',cases=stats,wall=time.perf_counter()-start,scope='Three60sec boundaries; not full completion or JCT comparison, all80 declarations intact.'))
