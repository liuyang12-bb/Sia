"""Fixed 60-round real-decision gate; no parameter search or JCT selection."""
import copy,gzip,json,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from mismatch_adapter import build
from run_core80 import gate,encode

def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,default=encode)

plan=json.loads((O/'design_manifest.json').read_text());events=[];actual={};start=time.time()
for owner in ('FO','FULL'):
 gate(O);c=build(owner,O/'inputs/S4_balanced_gpu.json',O/'inputs/grouped6.csv');base=c.__class__
 class Capture(base):
  def optimize_policy(self):
   snap=copy.deepcopy(self);snap.__class__=base;self.before=snap
   return super().optimize_policy()
 c.__class__=Capture;actual[owner]={}
 with gzip.open(O/('gate_'+owner+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
  for i in range(plan['gate']['rounds']):
   if c.all_complete():break
   if hasattr(c,'before'):del c.before
   c.step(60);s=c.before;del c.before;comps={}
   for method in ('FO','FULL'):
    q=copy.deepcopy(s);q.policy.liveness.clean_gpu='fixed' if method=='FO' else 'joint';q.optimize_policy()
    d=copy.deepcopy(q.policy.last_decision);assert d['result']['audit']['valid'];comps[method]=d
   row=dict(time=c.current_time,decisions=comps,actual=c.policy.last_decision,execution=c.joint_round_logs[-1]['execution'],
    job_end_state={j.name:dict(progress=float(j.progress),completion=j.completion_time) for j in c.jobs})
   f.write(json.dumps(row,default=encode)+'\n');f.flush()
   actual[owner][c.current_time]=copy.deepcopy({k:v for k,v in row.items() if k!='decisions'})
   fs=comps['FO']['result']['selected'];js=comps['FULL']['result']['selected']
   def eta(d):
    scores={x['column_id']:x for x in d['continuity_scores']};values=[scores[x['column_id']]['estimated_completion'] for x in d['result']['selected'].values()]
    return sum(values) if values and all(v is not None for v in values) else None
   a,b=eta(comps['FO']),eta(comps['FULL']);ua=comps['FO']['result']['scheduling_objective'];ub=comps['FULL']['result']['scheduling_objective']
   changed=[j for j in set(fs)&set(js) if fs[j]['config']['config_id']!=js[j]['config']['config_id']]
   events.append(dict(prefix=owner,time=c.current_time,selected=list(fs),same_set=set(fs)==set(js),fo_eta=a,full_eta=b,fo_utility=ua,full_utility=ub,changed_gpu=changed,
    opportunity=len(fs)>=3 and set(fs)==set(js) and bool(changed) and a is not None and b is not None and b<a-1e-6 and ub>ua+1e-9))
   c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
   if (i+1)%10==0:print('GATE',owner,i+1,sum(j.completion_time is not None for j in c.jobs),flush=True)
save('gate_events.json',events)
episodes=[];cur=None;total_progress={j:0. for j in actual['FO'][60]['job_end_state']}
for t in sorted(set(actual['FO'])&set(actual['FULL'])):
 a,b=actual['FO'][t],actual['FULL'][t];sa=a['actual']['result']['selected'];sb=b['actual']['result']['selected']
 changed=[j for j in set(sa)&set(sb) if sa[j]['config']['config_id']!=sb[j]['config']['config_id']]
 good=len(sa)>=3 and set(sa)==set(sb) and bool(changed)
 if good:
  if cur is None:cur=dict(start=t,end=t+60,intervals=0,jobs=set(),progress_gain={},full_progress={},fo_progress={},receiver_completed=False)
  cur['end']=t+60;cur['intervals']+=1;cur['jobs'].update(changed)
  # Decision at t executes over [t,t+60); use next persisted interval.
  if t+60 in actual['FO'] and t+60 in actual['FULL']:
   ee={m:{e['job_id']:e for e in actual[m][t+60]['execution']} for m in ('FO','FULL')}
   for j in set(ee['FO'])&set(ee['FULL']):
    x,y=ee['FO'][j],ee['FULL'][j]
    cur['progress_gain'][j]=cur['progress_gain'].get(j,0.)+y['progress_delta']-x['progress_delta']
    cur['full_progress'][j]=cur['full_progress'].get(j,0.)+y['progress_delta'];cur['fo_progress'][j]=cur['fo_progress'].get(j,0.)+x['progress_delta']
    if j in changed and y['completion_time'] is not None:cur['receiver_completed']=True
 else:
  if cur is not None:cur['jobs']=sorted(cur['jobs']);episodes.append(cur);cur=None
if cur is not None:cur['jobs']=sorted(cur['jobs']);episodes.append(cur)
# Total work is not guessed: derive from observed completion, else mark unknown.
completed_work={}
for m,series in actual.items():
 for row in series.values():
  for j,v in row['job_end_state'].items():
   if v['completion'] is not None:completed_work[j]=v['progress']
for e in episodes:
 e['net_progress_gain']=sum(e['progress_gain'].values())
 e['receiver_work_shares']={j:e['full_progress'].get(j,0.)/completed_work[j] for j in e['jobs'] if completed_work.get(j,0)>0}
 e['substantial']=e['receiver_completed'] or any(x>=.25 for x in e['receiver_work_shares'].values())
 e['qualifies']=e['intervals']>=6 and e['net_progress_gain']>0 and e['substantial']
passed=any(x['opportunity'] for x in events) and any(e['qualifies'] for e in episodes)
save('gate_result.json',dict(passed=passed,episodes=episodes,opportunity_events=[x for x in events if x['opportunity']],
 repeated_episodes=sum(e['qualifies'] for e in episodes),work_share_definition='conservative full observed total work; missing total not assumed',
 wall=time.time()-start,scope='same-state predictions separate from diverged-prefix actual execution; no JCT gate'))
print('GATE_RESULT',passed,'episodes',len(episodes),flush=True)
