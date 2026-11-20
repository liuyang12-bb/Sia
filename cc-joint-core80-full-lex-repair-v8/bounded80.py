"""Pre-formal bounded interface tests; full declarations, no JCT claims."""
import contextlib,copy,io,json,tempfile,time
from pathlib import Path
import pandas as pd
from core80 import ROOT,build
from run_core80 import Metrics,compact_decision,save

def main(out):
 out=Path(out);data=pd.read_csv(ROOT/'inputs/core80_workload.csv').iloc[[0,3,4]].copy();data.time=0
 evidence=[]
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'three.csv';data.to_csv(path,index=False)
  for method in ('SIA_P','SIA_AUX','PROP','CPU','MEM','FULL','FO','A00','A01','A10','A11'):
   with contextlib.redirect_stdout(io.StringIO()):
    c=build(method,'neutral' if method.startswith('A') else 'medium',workload_path=path,topology={'aws':2,'dgx-ext':2})
    m=Metrics(c);start=time.perf_counter()
    for _ in range(8):
     c.step(60);r=c.joint_round_logs[-1];compact_decision(r['decision_for_next_interval']);m.consume(c,r)
     c.joint_round_logs.clear();c.logs.clear()
    s=m.summary(c);evidence.append(dict(method=method,audits=s['execution_audits'],rounds=8,completed=s['completed'],positive_progress=sum(j.progress>0 for j in c.jobs),
        capacity_violations=s['violations'],locks=s['checkpoint_lock_checks'],wall=time.perf_counter()-start))
   print('BOUNDED_OK',evidence[-1],flush=True)
 save(out/'bounded_interfaces.json',dict(status='PASS',cases=evidence,meaning='8 real optimize_policy/Job.step boundaries each, original work retained; not a full JCT experiment.'))

if __name__=='__main__':
 import sys;main(sys.argv[1])
