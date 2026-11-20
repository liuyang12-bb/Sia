import gzip,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(method):
 with gzip.open(OUT/'real_cells'/method/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
  return [json.loads(line) for line in f]
def selected(r):
 return {j:(v['config']['config_id'],v['cpu_per_gpu'],v['ram_gb_per_gpu']) for j,v in r['decision_for_next_interval']['result']['selected'].items()}
a=read('medium_G00');b=read('medium_G01');differences=[]
for old,new in zip(a,b):
 if selected(old)!=selected(new):
  differences.append(dict(time=old['time'] if 'time' in old else old['execution'][0]['interval_end'],
   old=selected(old),new=selected(new),
   old_observation=old['execution'],new_observation=new['execution'],
   new_score=[v for v in new['decision_for_next_interval']['net_progress_scores'] if v['job']=='deepspeech2-8']))
result=dict(first_divergence=differences[0] if differences else None,different_decision_rounds=len(differences),
    warning='Only first divergence has a shared prior trajectory; later states are not same-state counterfactuals')
with (OUT/'trajectory_diagnosis.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print('different_rounds',len(differences))
if differences:
 v=differences[0];print(v['time'],v['old'],v['new'])
 for e in v['old_observation']:
  print('prior_execution',e['job_id'],e['raw_goodput'],e['rho'],e['training_seconds'])
