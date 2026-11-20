import gzip,json,math
from pathlib import Path
O=Path(__file__).resolve().parent
def read(n):
 with gzip.open(O/'cells'/n/'rounds.jsonl.gz','rt',encoding='utf8') as f:
  for l in f:yield json.loads(l)
result={}
for method in ('FO','FULL'):
 old=list(read('OLD_'+method));new=list(read('NEW_'+method));first=None
 for i,(a,b) in enumerate(zip(old,new)):
  aa=a['decision_for_next_interval']['result']['selected'];bb=b['decision_for_next_interval']['result']['selected']
  geometry=lambda d:{j:(c['config']['config_id'],c['placement'],c['cpu_per_gpu'],c['ram_gb_per_gpu']) for j,c in d.items()}
  if geometry(aa)==geometry(bb):continue
  assert a['job_end_state']==b['job_end_state']
  pairs=[]
  for j in set(aa)&set(bb):
   x,y=aa[j],bb[j]
   if geometry({j:x})==geometry({j:y}):continue
   ex=next(v for v in old[i+1]['execution'] if v['job_id']==j);ey=next(v for v in new[i+1]['execution'] if v['job_id']==j)
   sx,sy=ex['training_segments'][0],ey['training_segments'][0]
   equal=math.isclose(sx['raw_rate_before_rho'],sy['raw_rate_before_rho'],rel_tol=1e-12)
   pairs.append(dict(job=j,old_cpu=x['cpu_per_gpu'],new_cpu=y['cpu_per_gpu'],old_ram=x['ram_gb_per_gpu'],new_ram=y['ram_gb_per_gpu'],
      old_rho=x['rho'],new_rho=y['rho'],same_first_raw=equal,old_first_raw=sx['raw_rate_before_rho'],new_first_raw=sy['raw_rate_before_rho'],
      effective_rate_ratio=sy['effective_rate']/sx['effective_rate'],rho_ratio=y['rho']/x['rho']))
  first=dict(time=a['interval_end'],same_actual_predecision_state=True,pairs=pairs);break
 result[method]=first
with (O/'transmission.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(result)
