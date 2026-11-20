"""Post-hoc diagnostic only; does not change the preregistered gate verdict."""
import gzip,json
from pathlib import Path
O=Path(__file__).resolve().parent
read=lambda n:json.loads((O/n).read_text(encoding='utf8'))
d=read('gate_diagnosis.json');look={m:{r['time']:r for r in (json.loads(l) for l in gzip.open(O/f'gate_{m}.jsonl.gz','rt',encoding='utf8'))} for m in ('FO','FULL')};out=[]
for x in d['all_records']:
 span=[];productive=0;first_mismatch=None;deltas={m:0. for m in look}
 for t in range(int(x['event_time']),2400,60):
  sel={m:look[m][t]['actual']['result']['selected'] for m in look}
  if any(x['job'] not in sel[m] or sel[m][x['job']]['config']['config_id']!=x['configs'][m] for m in look):break
  if set(sel['FO'])!=set(sel['FULL']) and first_mismatch is None:first_mismatch=t
  span.append(t)
  for m in look:deltas[m]+=sum(e['progress_delta'] for e in look[m][t+60]['execution'] if e['job_id']==x['job'])
  productive+=any(e['job_id']==x['job'] and e['progress_delta']>0 for e in look['FULL'][t+60]['execution'])
 out.append(dict(owner=x['owner'],job=x['job'],event_time=x['event_time'],configs=x['configs'],geometry_intervals=len(span),productive_intervals=productive,
  end=span[-1]+60 if span else x['event_time'],first_service_set_difference=first_mismatch,key_progress=deltas))
out.sort(key=lambda x:(-x['geometry_intervals'],x['event_time']))
with (O/'geometry_duration_diagnostic.json').open('x',encoding='utf8') as f:json.dump(dict(posthoc_not_gate=True,preregistered_gate_passed=False,rows=out),f,ensure_ascii=False,indent=2)
print(json.dumps(out[:4],ensure_ascii=False,indent=2))
for m,s in d['prefix'].items():print('PREFIX',m,s['completed'],s['normalized_work_done'],{j:(v['progress'],v['waiting'],v['completion']) for j,v in s['jobs'].items() if j.startswith('deepspeech')})
