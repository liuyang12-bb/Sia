import collections,gzip,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;RUN=OUT.parent/'sia_r0_full160_comparison/attempt2'
def rounds(cell):
 with gzip.open(RUN/cell/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:yield json.loads(line)
boost=collections.defaultdict(lambda:dict(seconds=0.,rho_max=1.,segments=0));first=None
for a,b in zip(rounds('main_w1_medium_sia_p'),rounds('main_w1_medium_sia_tune')):
 if first is None:
  for x,y in zip(a['execution'],b['execution']):
   if x['progress_after']!=y['progress_after']:
    first=dict(time=a['interval_end'],job=x['job_id'],sia_p=x,sia_tune=y);break
 for e in b['execution']:
  if e['rho']>1.000001 and e['training_seconds']>0:
   v=boost[e['job_id']];v['seconds']+=e['training_seconds'];v['rho_max']=max(v['rho_max'],e['rho']);v['segments']+=len(e['training_segments'])
with (OUT/'execution_detail.json').open('x',encoding='utf-8') as f:json.dump(dict(boost=dict(boost),first_progress_difference=first),f,indent=2,ensure_ascii=False)
print(dict(boost))
