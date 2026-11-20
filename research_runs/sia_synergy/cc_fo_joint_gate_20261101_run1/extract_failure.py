import gzip,json,hashlib
from pathlib import Path
R=Path(r'D:/科研启航/Sia');O=Path(__file__).resolve().parent;OLD=R/'research_runs/sia_synergy/cc_bounded_correction_20261031_run1'
d=json.loads((OLD/'cells/dual_medium_FO/failure.json').read_text());active=set(d['last_decision']['result']['selected'])
history={j:[] for j in active};last=[];printed=False
with gzip.open(OLD/'cells/dual_medium_FO/rounds.jsonl.gz','rt',encoding='utf8') as f:
 for line in f:
  r=json.loads(line);last=(last+[r])[-2:]
  for e in r['execution']:
   if e['job_id'] in active:
    if e['training_segments']:
     history[e['job_id']].append(dict(time=r['interval_end'],cluster=e['cluster'],placement=e['placement'],segments=e['training_segments']))
     if not printed:print('SEGMENT',history[e['job_id']][-1],flush=True);printed=True
payload=dict(active=sorted(active),failure={k:v for k,v in d.items() if k!='last_decision'},last=last,observations=history)
with (O/'failure_observations.json').open('x',encoding='utf8') as f:json.dump(payload,f,ensure_ascii=False)
print('EXTRACTED',len(active),{j:len(v) for j,v in history.items()},flush=True)
