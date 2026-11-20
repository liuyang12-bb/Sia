import argparse,copy,gzip,hashlib,json,sys,time,traceback
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from mismatch_adapter import build
from run_core80 import gate,encode
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,default=encode)
ap=argparse.ArgumentParser();ap.add_argument('--owner',choices=['FO','FULL'],required=True);a=ap.parse_args()
p=json.loads((O/'preregistration.json').read_text());start=time.time();events=[];c=None
try:
 for n,h in p['source_hashes'].items():assert hashlib.sha256((T/n).read_bytes()).hexdigest().upper()==h,n
 gate(O);c=build(a.owner,O/'inputs/proposed_cluster.json',O/'inputs/proposed_6job.csv');base=c.__class__
 class Capture(base):
  def optimize_policy(self):
   snap=copy.deepcopy(self);snap.__class__=base;self.before=snap
   return super().optimize_policy()
 c.__class__=Capture
 with gzip.open(O/f'gate_{a.owner}.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
  for i in range(p['gate_rounds']):
   if c.all_complete():break
   if hasattr(c,'before'):del c.before
   c.step(60);s=c.before;del c.before;ds={}
   for method in ('FO','FULL'):
    q=copy.deepcopy(s);q.policy.liveness.clean_gpu='fixed' if method=='FO' else 'joint';q.optimize_policy()
    ds[method]=copy.deepcopy(q.policy.last_decision);assert ds[method]['result']['audit']['valid']
   x,y=[ds[m]['result']['selected'] for m in ('FO','FULL')]
   common=set(x)&set(y); changed=[j for j in common if x[j]['config']['config_id']!=y[j]['config']['config_id']]
   modified=[j for j in common if x[j]['column_id']!=y[j]['column_id']]
   clipped=[(j,m) for j in modified for m,z in [('FO',x),('FULL',y)] if z[j]['source_coordinates']['clipped'] or z[j]['baseline_source_coordinates']['clipped']]
   ux,uy=[ds[m]['result']['scheduling_objective'] for m in ('FO','FULL')]
   event=dict(time=c.current_time,same_set=set(x)==set(y),changed=changed,modified=modified,clipped_changed=clipped,fo_objective=ux,full_objective=uy,
     structural=len(x)>=2 and set(x)==set(y) and bool(changed) and uy>ux+1e-9)
   event['supported']=event['structural'] and not clipped;events.append(event)
   r=dict(time=c.current_time,decisions=ds,actual=c.policy.last_decision,execution=c.joint_round_logs[-1]['execution'],event=event,
     job_end_state={j.name:dict(progress=float(j.progress),completion=j.completion_time) for j in c.jobs})
   f.write(json.dumps(r,default=encode,separators=(',',':'))+'\n');f.flush()
   if (i+1)%5==0: print('GATE',a.owner,i+1,'complete',sum(j.completion_time is not None for j in c.jobs),'structural',sum(x['structural'] for x in events),'supported',sum(x['supported'] for x in events),'wall',time.time()-start,flush=True)
   c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
 def longest(k):
  best=cur=0
  for e in events:cur=cur+1 if e[k] else 0;best=max(best,cur)
  return best
 save(f'gate_{a.owner}_summary.json',dict(owner=a.owner,rounds=len(events),events=events,structural_longest=longest('structural'),supported_longest=longest('supported'),passed=longest('supported')>=p['gate_consecutive'],wall=time.time()-start))
except Exception as e:
 save(f'gate_{a.owner}_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=getattr(c,'current_time',None),events=events,wall=time.time()-start));raise
