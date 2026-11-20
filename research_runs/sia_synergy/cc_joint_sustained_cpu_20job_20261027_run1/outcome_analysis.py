"""Observed service mix and initial/post-start waiting; no new simulation."""
import collections,csv,gzip,json
from experiment import OUT,RUNS,load,save

load()
with (OUT/'inputs/workload.csv').open(encoding='utf-8') as f:apps={r['name']:r['application'] for r in csv.DictReader(f)}
result={}
for response,method in RUNS:
    d=OUT/'cells'/response/method
    if not (d/'summary.json').exists():continue
    app_time=collections.defaultdict(collections.Counter);jobs=collections.defaultdict(collections.Counter)
    first_alloc={};first_training={};delay_restarts={};rounds=0
    with gzip.open(d/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
      for line in f:
        r=json.loads(line);rounds+=1
        for e in r['execution']:
          j=e['job_id'];a=apps[j]
          if e['resource_allocation'] and e['allocated_seconds']>0:
            first_alloc.setdefault(j,e['first_allocation_time'])
          if e['training_seconds']>0:first_training.setdefault(j,e['first_productive_training_time'])
          phase='initial_wait' if j not in first_alloc else 'post_allocation_wait'
          jobs[j][phase]+=e['waiting_seconds']
          for field in ['waiting_seconds','checkpoint_delay_consumed','training_seconds']:
            app_time[a][field]+=e[field]
    total=sum(t['training_seconds'] for t in app_time.values())
    result[response+'/'+method]=dict(rounds=rounds,application_times=dict(app_time),
      cifar_productive_wall_fraction=app_time['cifar10']['training_seconds']/total if total else None,
      waiting_phases=dict(jobs),first_allocation=first_alloc,first_training=first_training,
      definitions='Job wall-seconds, not GPU-seconds; initial wait ends at first allocation. No extrapolated JCT.')
save(OUT/'outcome_analysis.json',result)
print('OUTCOME_ANALYSIS_READY')
