import copy,csv,gzip,json,sys,time,traceback
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-mismatch-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from mismatch_adapter import build
from run_core80 import gate,encode,Metrics,compact_decision

def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,default=encode)

with (O/'inputs/compact12.csv').open(encoding='utf8') as f:
 rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
with (O/'inputs/gate2.csv').open('x',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows[:2])
save('gate_plan.json',dict(rounds=30,workload='first two records of COMPACT12 unchanged',scenario='S3_crossed',response='medium',
 real_state='capture before actual optimize_policy, clone same current state, call real optimize_policy FO vs FULL; no seeded gradients or fake rates',
 pass_rule='a CIFAR allocation contracts GPU count relative to FO, effective predicted rate increases, no resource/lock violation; actual objectives unchanged',limits='current learned-rate prediction; no JCT inference'))
gate(O);c=build('FULL',O/'inputs/S3_crossed.json',O/'inputs/gate2.csv')
base=c.__class__
class Capture(base):
 def optimize_policy(self):
  snap=copy.deepcopy(self);snap.__class__=base
  self.before=snap
  return super().optimize_policy()
c.__class__=Capture
events=[];trigger=[];m=Metrics(c)
with gzip.open(O/'gate_rounds.jsonl.gz','xt',encoding='utf8') as log:
 for i in range(30):
  if c.all_complete():break
  if hasattr(c,'before'):del c.before
  c.step(60);state=c.before;del c.before
  rr=c.joint_round_logs[-1];m.consume(c,rr)
  comp={}
  for method in ('FO','FULL'):
   q=copy.deepcopy(state);q.policy.liveness.clean_gpu='fixed' if method=='FO' else 'joint'
   q.policy.optimize # full real path, not a stand-alone helper
   q.optimize_policy();d=q.policy.last_decision
   assert d['result']['audit']['valid']
   comp[method]=copy.deepcopy(d)
  row=dict(time=c.current_time,decisions=comp)
  fs=comp['FO']['result']['selected'];js=comp['FULL']['result']['selected']
  for j in set(fs)&set(js):
   a,b=fs[j],js[j]
   if j.startswith('cifar10') and b['config']['gpus']<a['config']['gpus'] and b['effective_goodput']>a['effective_goodput']:
    trigger.append(dict(time=c.current_time,job=j,fo=a,full=b))
  events.append(dict(time=c.current_time,fo={j:(v['config']['config_id'],v['rho']) for j,v in fs.items()},full={j:(v['config']['config_id'],v['rho']) for j,v in js.items()}))
  log.write(json.dumps(row,default=encode)+'\n');log.flush()
  c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
save('gate_result.json',dict(passed=bool(trigger),triggers=trigger,states=events,rounds=len(events),completed=sum(j.completion_time is not None for j in c.jobs),reason='GPU contraction triggered' if trigger else 'No real-policy contraction in preregistered 30-round window; stop before matrix'))
print('GATE',bool(trigger),'triggers',len(trigger),flush=True)
