import json,gzip,collections,math,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;out={};means={}
for cell in ['neutral_sia_p','neutral_sia_aux','neutral_joint_o','medium_sia_p','medium_sia_aux','medium_joint_o']:
 groups=collections.defaultdict(lambda:[0.,0.]);first=None;complete={};arrivals={}
 with gzip.open(P/cell/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);d=r['decision_for_next_interval']
   if first is None:first={j:dict(config=c['config']['config_id'],source=d['resources'][j]['goodput_provenance']['goodput_source']) for j,c in d['result']['selected'].items()}
   for e in r['execution']:
    arrivals[e['job_id']]=e['arrival_time']
    if e['completion_time'] is not None:complete[e['job_id']]=e['completion_time']
    if e['job_id']=='bert-28' and e['training_seconds']:
     v=groups[e['resource_allocation']['config_id']];v[0]+=e['training_seconds'];v[1]+=e['training_seconds']*e['raw_goodput']
 summary=json.loads((P/cell/'summary.json').read_text());mean=sum(t-arrivals[j] for j,t in complete.items())/3;assert len(complete)==3 and math.isclose(mean,summary['mean_jct'])
 means[cell]=dict(mean_from_raw_events=mean,completed=3)
 out[cell]=dict(first=first,bert_training_by_config=dict(groups),bert=summary['jobs']['bert-28'])
with (P/'bert_trace.json').open('x') as f:json.dump(out,f,indent=2)
with (P/'completion_event_check.json').open('x') as f:json.dump(means,f,indent=2)
print('MEANS',means)
