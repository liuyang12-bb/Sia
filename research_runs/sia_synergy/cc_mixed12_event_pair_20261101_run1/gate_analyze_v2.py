import gzip,json
from pathlib import Path
O=Path(__file__).resolve().parent;p=json.loads((O/'preregistration.json').read_text(encoding='utf8'))
rows={m:[json.loads(l) for l in gzip.open(O/f'gate_{m}.jsonl.gz','rt',encoding='utf8')] for m in ('FO','FULL')}
lookup={m:{r['time']:r for r in rs} for m,rs in rows.items()};witnesses=[];events=[]
for owner,rs in rows.items():
 for r in rs:
  ds=r['decisions']
  if not ds:continue
  a,b=[ds[m]['result']['selected'] for m in ('FO','FULL')];ua,ub=[ds[m]['result']['scheduling_objective'] for m in ('FO','FULL')]
  changed=sorted(j for j in set(a)&set(b) if a[j]['config']['config_id']!=b[j]['config']['config_id'])
  structural=set(a)==set(b) and len(a)>=2 and bool(changed) and ub>ua+1e-9
  ev=dict(owner=owner,time=r['time'],same_set=set(a)==set(b),changed=changed,fo_objective=ua,full_objective=ub,structural=structural,geometry_events=r['geometry_events'])
  events.append(ev)
  if not structural:continue
  for j in changed:
   span=[];productive=0;delta={m:0. for m in ('FO','FULL')}
   for t in range(int(r['time']),int(r['time'])+p['gate_rounds']*60,60):
    rr={m:lookup[m].get(t) for m in ('FO','FULL')};nn={m:lookup[m].get(t+60) for m in ('FO','FULL')}
    if not all(rr.values()) or not all(nn.values()):break
    sels={m:rr[m]['actual']['result']['selected'] for m in ('FO','FULL')}
    if any(j not in sels[m] or sels[m][j]['config']['config_id']!=ds[m]['result']['selected'][j]['config']['config_id'] for m in ('FO','FULL')):break
    span.append(t)
    for m in delta:delta[m]+=sum(e['progress_delta'] for e in nn[m]['execution'] if e['job_id']==j)
    productive+=any(e['job_id']==j and e['progress_delta']>0 for e in nn['FULL']['execution'])
   if len(span)>=p['gate_consecutive'] and productive>=p['gate_productive']:
    witnesses.append(dict(**ev,key=j,fo_config=a[j]['config']['config_id'],full_config=b[j]['config']['config_id'],intervals=len(span),productive_intervals=productive,end=span[-1]+60,key_progress=delta,
      clipping={m:{k:ds[m]['result']['selected'][j][k]['clipped'] for k in ('source_coordinates','baseline_source_coordinates')} for m in ('FO','FULL')}))
with (O/'gate_result_v2.json').open('x',encoding='utf8') as f:json.dump(dict(passed=bool(witnesses),witnesses=witnesses,events=events,scope='same-state same-service EVENT selection plus actual diverged paths; subsequent service-set changes allowed by frozen definition; no JCT inferred'),f,indent=2,ensure_ascii=False)
print('GATE_RESULT',bool(witnesses),'events',len(events),'witnesses',len(witnesses),flush=True)
