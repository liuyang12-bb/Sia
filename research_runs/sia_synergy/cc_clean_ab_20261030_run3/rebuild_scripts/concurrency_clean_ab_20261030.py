import collections,gzip,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run3'
def area(spans):
 events=collections.Counter()
 for a,b in spans:
  if b>a:events[a]+=1;events[b]-=1
 out=collections.Counter();count=0;last=None
 for t,change in sorted(events.items()):
  if last is not None:out[count]+=t-last
  count+=change;last=t
 assert count==0
 return out
def main():
 result={}
 for n in ['A00','A01','A10','A11','A_SIA']:
  totals=collections.Counter();training=collections.Counter();allocated=collections.Counter()
  with gzip.open(OUT/'cells'/('A20_'+n)/'rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    r=json.loads(line);train=[];alloc=[]
    for e in r['execution']:
     if not e['resource_allocation']:continue
     a=e['resource_allocation'];start=max(e['interval_start'],e['arrival_time'])
     if e['allocated_seconds']>0:alloc.append((start,start+e['allocated_seconds']))
     for z in e['training_segments']:
      train.append((z['start_time'],z['start_time']+z['seconds']))
      totals['productive_job_seconds']+=z['seconds'];totals['productive_gpu_seconds']+=z['seconds']*a['gpus']
    training.update(area(train));allocated.update(area(alloc))
  result[n]=dict(training_concurrency_seconds=dict(training),allocated_concurrency_seconds=dict(allocated),
    average_gpus_per_productively_training_job=totals['productive_gpu_seconds']/totals['productive_job_seconds'],**totals)
 with (OUT/'service_concurrency.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
 print('CONCURRENCY_RAW_ACCOUNTED',result)
if __name__=='__main__':main()
