"""Link each executed resource allocation to the preceding atomic decision."""
import collections,gzip,json,math
from experiment import OUT,RUNS,load,save
load();results={}
for response,method in RUNS:
    previous={};count=collections.Counter();last=None
    with gzip.open(OUT/'cells'/response/method/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
      for line in f:
        r=json.loads(line)
        for e in r['execution']:
          if e['allocated_seconds']<=0:continue
          a=e['resource_allocation'];p=previous[e['job_id']]
          assert a['config_id']==p['config']['config_id']
          assert a['gpus']==p['config']['gpus']
          assert math.isclose(a['cpu_per_gpu'],p['cpu_per_gpu'],rel_tol=1e-12)
          assert math.isclose(a['ram_gb_per_gpu'],p['ram_gb_per_gpu'],rel_tol=1e-12)
          assert math.isclose(e['rho'],p['rho'],rel_tol=1e-12)
          assert {u['node_id']:u['gpus'] for u in a['node_usage']}==p['placement']
          assert e['cluster']==p['config']['gpu_type']
          count['executed_allocations_match_prior_atomic_commit']+=1
        d=r['decision_for_next_interval']
        if d['round']!=last:
          previous=d['result']['selected'];last=d['round']
    results[response+'/'+method]=dict(valid=True,checks=dict(count))
save(OUT/'cross_round_commit_audit.json',dict(valid=True,results=results))
print('CROSS_ROUND_COMMIT_AUDIT_PASS')
