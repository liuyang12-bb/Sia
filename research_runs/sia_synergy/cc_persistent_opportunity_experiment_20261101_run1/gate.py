import copy,gzip,json,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-persistent-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from mismatch_adapter import build
from run_core80 import gate,Metrics,encode
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,default=encode)
save('gate_plan.json',dict(rounds=30,interval=60,workload='all long_gap12 records',prefixes=['FO','FULL'],pass_rule='at least3 CIFAR served, FO naturally uses AWS, same selected set; all selected ETA known; FULL sum utility higher and sum ETA lower after actual filters/costs; valid complete atomic audit',no_training_shortening=True))
events=[];triggers=[];start=time.time()
for owner in ('FO','FULL'):
 gate(O);c=build(owner,O/'inputs/S3_crossed.json',O/'inputs/long_gap12.csv');base=c.__class__
 class Capture(base):
  def optimize_policy(self):
   snap=copy.deepcopy(self);snap.__class__=base;self.before=snap
   return super().optimize_policy()
 c.__class__=Capture;m=Metrics(c)
 with gzip.open(O/('gate_'+owner+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
  for i in range(30):
   if c.all_complete():break
   if hasattr(c,'before'):del c.before
   c.step(60);s=c.before;del c.before;m.consume(c,c.joint_round_logs[-1])
   comps={}
   for method in ('FO','FULL'):
    q=copy.deepcopy(s);q.policy.liveness.clean_gpu='fixed' if method=='FO' else 'joint';q.optimize_policy()
    d=copy.deepcopy(q.policy.last_decision);assert d['result']['audit']['valid'];comps[method]=d
   f.write(json.dumps(dict(time=c.current_time,decisions=comps,actual=c.policy.last_decision,execution=c.joint_round_logs[-1]['execution']),default=encode)+'\n');f.flush()
   fs=comps['FO']['result']['selected'];js=comps['FULL']['result']['selected']
   def eta(d):
    scores={x['column_id']:x for x in d['continuity_scores']};vals=[scores[x['column_id']]['estimated_completion'] for x in d['result']['selected'].values()]
    return sum(vals) if all(v is not None for v in vals) else None
   a,b=eta(comps['FO']),eta(comps['FULL'])
   triple=sum(j.startswith('cifar10') for j in fs)>=3
   aws=any(j.startswith('cifar10') and x['config']['gpu_type']=='aws' for j,x in fs.items())
   ua=comps['FO']['result']['scheduling_objective'];ub=comps['FULL']['result']['scheduling_objective']
   e=dict(prefix=owner,time=c.current_time,three_cifar=triple,fo_aws=aws,same_set=set(fs)==set(js),fo_eta=a,full_eta=b,fo_utility=ua,full_utility=ub)
   if triple and aws and set(fs)==set(js) and a is not None and b is not None and b<a-1e-6 and ub>ua+1e-9:triggers.append(e)
   events.append(e)
   c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
 save('gate_progress_'+owner+'.json',dict(rounds=i+1,completed=sum(j.completion_time is not None for j in c.jobs),triggers=len(triggers),wall=time.time()-start))
save('gate_result.json',dict(passed=bool(triggers),triggers=triggers,states=events,wall=time.time()-start))
print('GATE',bool(triggers),'TRIGGERS',len(triggers),flush=True)

# Separate, predeclared persistence test of actual prefix allocations.
actual={}
for owner in ('FO','FULL'):
 actual[owner]={}
 for line in gzip.open(O/('gate_'+owner+'.jsonl.gz'),'rt',encoding='utf8'):
  r=json.loads(line);actual[owner][r['time']]=r['actual']['result']['selected']
rows=[];streak=0;maximum=0
for t in sorted(set(actual['FO'])&set(actual['FULL'])):
 a=actual['FO'][t];b=actual['FULL'][t];j='cifar10-2'
 active=sum(n.startswith('cifar10') for n in a)>=3 and set(a)==set(b)
 different=active and j in a and j in b and a[j]['config']['config_id']!=b[j]['config']['config_id']
 good=different and a[j]['config']['gpu_type']=='aws' and b[j]['config']['gpu_type']=='aws' and a[j]['config']['gpus']>b[j]['config']['gpus']
 streak=streak+1 if good else 0;maximum=max(maximum,streak)
 rows.append(dict(time=t,persistent=good,streak=streak,fo=a.get(j),full=b.get(j)))
save('persistence_gate.json',dict(passed=bool(triggers) and maximum>=6,initial_opportunity=bool(triggers),max_consecutive_intervals=maximum,required_intervals=6,states=rows))
print('PERSISTENCE',maximum,maximum>=6,flush=True)
