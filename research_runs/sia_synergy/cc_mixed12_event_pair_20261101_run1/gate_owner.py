import argparse,copy,gzip,hashlib,json,sys,time,traceback
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[2];T=ROOT/'cc-joint-recovery-equivalent-v1'
sys.dont_write_bytecode=True;sys.path[:0]=[str(T),str(T/'sia-simulator')]
from mismatch_adapter import build
from run_core80 import gate,encode,Metrics,compact_decision
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,default=encode)
ap=argparse.ArgumentParser();ap.add_argument('--owner',required=True,choices=['FO','FULL']);a=ap.parse_args()
p=json.loads((O/'preregistration.json').read_text(encoding='utf8'));start=time.time();c=None
try:
 for n,h in p['source_hashes'].items():assert hashlib.sha256((T/n).read_bytes()).hexdigest().upper()==h,n
 gate(O);c=build(a.owner,O/'inputs/config.json',O/'inputs/workload.csv','medium');base=c.__class__;m=Metrics(c)
 class Capture(base):
  def optimize_policy(self):
   self.before=copy.deepcopy(self);self.before.__class__=base
   return super().optimize_policy()
 c.__class__=Capture
 with gzip.open(O/f'gate_{a.owner}.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
  for i in range(p['gate_rounds']):
   if c.all_complete():break
   if hasattr(c,'before'):del c.before
   c.step(60);s=c.before;del c.before;r=c.joint_round_logs[-1];actual=c.policy.last_decision;ds={}
   reasons=actual['result']['audit'].get('geometry_event_reasons') or []
   if set(reasons)&set(p['gate_event_types']):
    for method in ('FO','FULL'):
     q=copy.deepcopy(s);q.policy.liveness.clean_gpu='fixed' if method=='FO' else 'joint';q.policy.liveness.resource_factor=method;q.optimize_policy()
     d=copy.deepcopy(q.policy.last_decision);assert d['result']['audit']['valid'];compact_decision(d);ds[method]=d
   compact_decision(actual);m.consume(c,r)
   f.write(json.dumps(dict(time=c.current_time,actual=actual,decisions=ds,execution=r['execution'],geometry_events=reasons),default=encode,separators=(',',':'))+'\n');f.flush()
   print('GATE',a.owner,i+1,'time',c.current_time,'completed',sum(j.completion_time is not None for j in c.jobs),'wall',time.time()-start,flush=True)
   c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
   for k in ('exchange_columns','exchange_scores','continuity_scores','completion_scores','actual_completion_columns'):
    if hasattr(c.policy,k):setattr(c.policy,k,[])
 save(f'gate_{a.owner}_summary.json',dict(rounds=i+1,wall=time.time()-start,completed=sum(j.completion_time is not None for j in c.jobs),metrics=m.summary(c)))
except Exception as e:
 save(f'gate_{a.owner}_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=getattr(c,'current_time',None)));raise
