"""One allocation call, no training/time advancement. Reference mode is explicit."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1'
sys.dont_write_bytecode=True;sys.path[:0]=[str(T),str(T/'sia-simulator')]
from mismatch_adapter import build
from run_core80 import gate,encode
from job import Job
from cluster import Cluster
def forbidden(*a,**k):raise AssertionError('TRAINING_FORBIDDEN')
Job.step=forbidden;Cluster.step=forbidden
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,default=encode)
ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['FO','FULL'],required=True);ap.add_argument('--state',choices=['cold','reference_E0'],required=True);a=ap.parse_args()
stem='oneshot_v2_'+a.state+'_'+a.method
try:
 p=json.loads((O/'preregistration.json').read_text(encoding='utf8'))
 for n,h in p['source_hashes'].items():assert hashlib.sha256((T/n).read_bytes()).hexdigest().upper()==h,n
 gate(O);c=build(a.method,O/'inputs/config.json',O/'inputs/workload.csv','medium')
 receipt=[]
 for j in c.jobs:
  # Empty observed segment ledger for a direct allocation call at t=0.
  # Normal simulation creates this in Job.step; no training is performed here.
  j.joint_step_segments=[]
  assert j.grad_params is None
  if a.state=='reference_E0' and j.submission_time<=0:
   j.grad_params=tuple(float(x) for x in j.reference_app.get_grad_stats(j.reference_app.init_batch_size,0))
   receipt.append(dict(job=j.name,gradient=j.grad_params,epoch=0,batch=j.reference_app.init_batch_size,
       source='OFFLINE_REFERENCE_ASSUMPTION_NOT_ONLINE_OBSERVATION'))
 before={j.name:j.progress for j in c.jobs};start=time.perf_counter();c.optimize_policy();d=c.policy.last_decision
 assert c.current_time==0 and {j.name:j.progress for j in c.jobs}==before
 r=d['result'];audit=r['audit']
 save(stem+'.json',dict(method=a.method,state=a.state,elapsed=time.perf_counter()-start,reference_assumptions=receipt,
     actual_policy_allocation_only=True,training_steps=0,time_advanced=0,selected=r['selected'],unallocated=r['unallocated_jobs'],
     objective=r['scheduling_objective'],valid=audit['valid'],violations=audit['violations'],node_totals=audit['node_totals'],
     serial_proposals=audit.get('serial_proposals',[]),required=audit.get('required_jobs'),optional=audit.get('optional_jobs'),
     ready=audit.get('ready'),contracts=d['shared_contract'],declared_work=d['declared_work'],
     warning='This allocates only arrived firstwave. No next round, production or persistence verified; reference_E0 deliberately adds offline gradients only to this disposable diagnostic object.'))
 print(stem,'selected',list(r['selected']),'valid',audit['valid'],'wall',time.perf_counter()-start,flush=True)
except Exception as e:
 save(stem+'_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
