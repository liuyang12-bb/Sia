import csv,gzip,json,sys
from pathlib import Path
import raw_audit_lib as lib
O=Path(__file__).resolve().parent
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
p=json.loads((O/'preregistration.json').read_text());done=json.loads((O/'pipeline_complete.json').read_text());stats={};jobs={};pairs={};recoveries={}
for cell in p['cells']:
 name=cell['id']
 if name not in done['completed']:continue
 cfg=json.loads((O/f"inputs/{cell['capacity']}.json").read_text());pp={**p,'capacities':cfg['cluster']['capacity_scenarios']['main'],'topology':cfg['cluster']['ngpus_per_node']}
 with (O/f"inputs/{cell['workload']}.csv").open(newline='') as f:work={r['name']:r for r in csv.DictReader(f)}
 stats[name],jobs[name],_=lib.audit(cell,pp,work);assert not stats[name]['errors'],stats[name]['errors']
 recoveries[name]=[]
 for l in gzip.open(O/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(l);d=r['decision_for_next_interval'];a=d['result']['audit'];sel=d['result']['selected']
  for sp in d['probe']['specs']:
   if sp['checkpoint_locked'] and sp['current_config_id']:assert sp['job_id'] in sel
  for z in a.get('serial_proposals',[]):assert z['fixed']=={v['job']:v['config'] for v in z['selected']}
  for x in a.get('tail_recoveries',[]):
   assert x['column_id']==sel[x['job']]['column_id'];assert x['cumulative_delay']<=250+x['extra_budget']+1e-9
   recoveries[name].append(x)
for i in range(0,len(p['cells']),2):
 a,b=[c['id'] for c in p['cells'][i:i+2]]
 if a not in stats or b not in stats:continue
 if not(stats[a]['complete'] and stats[b]['complete']):pairs[a]={'status':'CENSORED'};continue
 saved={j:jobs[a][j]['jct']-jobs[b][j]['jct'] for j in jobs[a]};total=sum(saved.values());best=max(saved,key=saved.get)
 pairs[a]=dict(full=b,mean_improvement_pct=100*(1-stats[b]['mean']/stats[a]['mean']),per_job_saved=saved,
  better=sum(v>0 for v in saved.values()),worse=sum(v<0 for v in saved.values()),equal=sum(v==0 for v in saved.values()),
  total_jct_saved=total,top_job=best,top_saved=saved[best],top_share=saved[best]/total if total else None,
  without_top_mean_saved=(total-saved[best])/(len(saved)-1),
  time_saved={k:stats[a]['totals'][k]-stats[b]['totals'][k] for k in ('wait','training','checkpoint')})
save('raw_audit.json',stats);save('per_job.json',jobs);save('comparison_summary.json',pairs);save('recovery_events.json',recoveries)
with (O/'per_job_utf8.csv').open('x',newline='',encoding='utf-8-sig') as f:
 fields=['cell','job']+list(next(iter(next(iter(jobs.values())).values())))
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for cell,rows in jobs.items():
  for j,v in rows.items():w.writerow(dict(cell=cell,job=j,**v))
print(json.dumps(dict(stats=stats,pairs=pairs),ensure_ascii=False,indent=2))
