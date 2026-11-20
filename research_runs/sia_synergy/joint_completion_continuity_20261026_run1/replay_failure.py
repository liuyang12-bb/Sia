"""Bounded 15-round exact-state reproduction of SCORE_ONLY solver exception."""
import json,sys,time,traceback
from experiment import OUT,load,gate,construct,save
load();resources=gate();c=construct('SCORE_ONLY');started=time.time()
try:
    for i in range(15):
        c.step(60)
        c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
except Exception as e:
    save(OUT/'solver_failure_reproduction.json',dict(exception=repr(e),traceback=traceback.format_exc(),
        time=c.current_time,round_limit=15,resources=resources,elapsed=time.time()-started,
        solver_diagnostics=getattr(e,'solver_diagnostics',None)))
    print('REPRODUCED',c.current_time,repr(e),flush=True)
else:
    raise RuntimeError('Expected recorded failure not reproduced; no further rounds authorized')
