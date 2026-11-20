"""Exactly16 real rounds across the previous time900 numerical failure."""
import sys,time
from experiment import OUT,LIVE,gate,construct,save
sys.path.insert(0,str(LIVE));resources=gate();c=construct('SCORE_ONLY');start=time.time()
records=[];canonical=0
for i in range(16):
    c.step(60);d=c.policy.last_decision
    assert d['result']['audit']['valid']
    canonical+=sum(v.get('canonical_binary_accepted',False) for v in d['result']['audit'].get('numerical_solver_retries',[]))
    records.append(dict(time=c.current_time,status=d['result']['status'],audit=d['result']['audit']))
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
assert c.current_time==960 and canonical>0
save(OUT/'bounded_solver_repair.json',dict(valid=True,round_limit=16,canonical_checks=canonical,
    resources=resources,elapsed=time.time()-start,records=records))
print('BOUNDED_NUMERICAL_REPAIR_PASS',canonical,flush=True)
