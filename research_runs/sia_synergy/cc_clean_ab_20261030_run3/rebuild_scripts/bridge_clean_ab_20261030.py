"""Read-only physical trajectory bridge, excluding solver wall/labels."""
import gzip,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run3'
def rows(p):
 with gzip.open(p,'rt',encoding='utf8') as f:
  for l in f:yield json.loads(l)
def sig(x):
 return dict(execution=[(e['job_id'],e['progress_after'],e['completion_time'],e['cluster'],e['placement']) for e in x['execution']],
  selected={j:(c['config']['config_id'],c['placement'],c['cpu_per_gpu'],c['ram_gb_per_gpu']) for j,c in x['decision_for_next_interval']['result']['selected'].items()},ends=x['job_end_state'])
def main():
 result={}
 for name,new,old in [
  ('SIA20','A20_A_SIA',ROOT/'research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1/cells/SIA_P/rounds.jsonl.gz'),
  ('JO_COMPACT','COMPACT_B_JO',ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1/cells/COMPACT_RNEW_medium_CC_FULL/rounds.jsonl.gz'),
  ('JO_SPREAD','SPREAD_B_JO',ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1/cells/SPREAD_RNEW_medium_CC_FULL/rounds.jsonl.gz')]:
  count=0;equal=True;difference=None
  for a,b in itertools.zip_longest(rows(OUT/'cells'/new/'rounds.jsonl.gz'),rows(old)):
   if a is None or b is None or sig(a)!=sig(b):equal=False;difference=count;break
   count+=1
  result[name]=dict(exact_physical_trajectory=equal,equal_prefix_rounds=count,first_difference=difference,reference=str(old))
 with (OUT/'historical_trajectory_bridge.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
 print('HISTORICAL_TRAJECTORY_BRIDGE',result)
if __name__=='__main__':main()
