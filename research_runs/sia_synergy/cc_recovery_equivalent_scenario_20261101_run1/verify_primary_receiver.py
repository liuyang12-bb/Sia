"""Independent analysis of completed gate logs, no optimization or simulation."""
import json,gzip
from pathlib import Path
O=Path(__file__).resolve().parent;rows={}
for m in ('FO','FULL'):
 rows[m]={}
 for line in gzip.open(O/('gate_'+m+'.jsonl.gz'),'rt',encoding='utf8'):
  row=json.loads(line);rows[m][row['time']]=row
j='cifar10-2';times=[];deltas={};total=None
for t in sorted(set(rows['FO'])&set(rows['FULL'])):
 a,b=rows['FO'][t],rows['FULL'][t];sa=a['actual']['result']['selected'];sb=b['actual']['result']['selected']
 if j not in sa or j not in sb or not(len(sa)>=3 and set(sa)==set(sb)):continue
 if sa[j]['config']['config_id']!='aws:1:2' or sb[j]['config']['config_id']!='aws:1:1':continue
 if t+60 not in rows['FO'] or t+60 not in rows['FULL']:continue
 times.append(t)
 for m in ('FO','FULL'):
  for e in rows[m][t+60]['execution']:
   deltas.setdefault(e['job_id'],{'FO':0.,'FULL':0.})[m]+=e['progress_delta']
 vals=[x for x in b['actual']['continuity_scores'] if x['job']==j]
 v=vals[0]['remaining_work']+b['job_end_state'][j]['progress']
 if total is None:total=v
 assert abs(v-total)<1e-6
assert len(times)>=6 and all(b-a==60 for a,b in zip(times,times[1:]))
gains={k:v['FULL']-v['FO'] for k,v in deltas.items()};share=deltas[j]['FULL']/total
assert share>=.25 and sum(gains.values())>0
result=json.loads((O/'primary_receiver_gate_audit.json').read_text())
assert result['intervals']==len(times) and result['net_gain']==gains and result['full_work_share']==share
print('Primary receiver gate reproduced; no files changed')
