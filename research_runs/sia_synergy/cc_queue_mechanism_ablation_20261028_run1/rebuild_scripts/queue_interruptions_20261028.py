"""Raw active-to-wait transitions and FIFO/service counterfactual contrasts."""
import collections,gzip,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1'
results={}
for folder in sorted((OUT/'cells').iterdir()):
 if not (folder/'summary.json').exists():continue
 counts=collections.Counter();events=[];previous_round=None
 with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);d=r['decision_for_next_interval']
   if d['round']==previous_round:continue
   previous_round=d['round'];selected=d['result']['selected']
   for job in d['probe']['specs']:
    j=job['job_id']
    if job['current_config_id'] is None or j in selected:continue
    assert not job['checkpoint_locked'],'CHECKPOINT_ILLEGAL_EVICTION'
    counts[j]+=1
    events.append(dict(time=r['interval_end'],job=j,current_config=job['current_config_id'],
     previous_placement=job['current_placement'],
     active_jobs=[x['job_id'] for x in d['probe']['specs']],
     selected={k:dict(config=v['config']['config_id'],placement=v['placement']) for k,v in selected.items()},
     actual_service_events=d.get('liveness_events',[])))
 results[folder.name]=dict(active_to_wait_transitions=sum(counts.values()),per_job=dict(counts),events=events)
with (OUT/'analysis/active_to_wait_transitions.json').open('x',encoding='utf-8') as f:json.dump(results,f,ensure_ascii=False,indent=2)
print({k:v['active_to_wait_transitions'] for k,v in results.items()})
print('FIRST_SIA',results['SIA_P']['events'][0])
