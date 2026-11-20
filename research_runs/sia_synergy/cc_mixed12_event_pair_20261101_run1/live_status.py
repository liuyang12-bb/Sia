"""Read complete persisted records; never repair/append running evidence."""
import gzip,json
from pathlib import Path
O=Path(__file__).resolve().parent
for owner in ('FO','FULL'):
 p=O/f'gate_{owner}.jsonl.gz'
 if not p.exists():continue
 n=0;events=[];last=None
 try:
  with gzip.open(p,'rt',encoding='utf8') as f:
   for l in f:
    r=json.loads(l);n+=1;last=r['time'];d=r['decisions']
    if not d:continue
    x,y=[d[m]['result'] for m in ('FO','FULL')];a,b=x['selected'],y['selected']
    changed=[j for j in a.keys()&b.keys() if a[j]['config']['config_id']!=b[j]['config']['config_id']]
    events.append(dict(time=r['time'],same_set=set(a)==set(b),changed=changed,objective_gap=y['scheduling_objective']-x['scheduling_objective']))
 except (EOFError,json.JSONDecodeError):pass
 print(owner,n,last,events,flush=True)
