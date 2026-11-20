import gzip,json,math
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];P=R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1'
def main():
 states=json.loads((O/'same_state_final.json').read_text());errors=[];counts={}
 with gzip.open(P/'cells/NEW_FULL/rounds.jsonl.gz','rt',encoding='utf8') as f:
  execution={(r['interval_start'],e['job_id']):e for r in map(json.loads,f) for e in r['execution']}
 for state in states['results']:
  now=state['time'];rows=state['calibration_records'];seen=set()
  for row in rows:
   identity=row['job'],row['origin']
   if identity in seen:errors.append('DUPLICATE')
   seen.add(identity)
   if not row['origin']<row['observed_end']<=row['observed_at']<=now:errors.append('FUTURE')
   if not math.isclose(row['ratio'],row['observed_raw']/row['predicted_raw'],rel_tol=1e-12):errors.append('RESIDUAL_RATIO')
   first=execution[(row['origin'],row['job'])]['training_segments'][0]
   if not math.isclose(row['observed_raw'],first['raw_rate_before_rho'],rel_tol=1e-12):errors.append('RAW_SOURCE')
  evidence=state['evaluation'].get('uncertainty',{})
  for key,bound in evidence.get('bounds',{}).items():
   eligible=[r for r in rows if r['key']==key and r['observed_end']<=now and r['observed_at']<=now]
   support=len({r['origin'] for r in eligible})>=2
   if (bound is not None)!=support:errors.append('SUPPORT')
   if bound:
    if bound['low']!=min([1.]+[r['ratio'] for r in eligible]) or bound['high']!=max([1.]+[r['ratio'] for r in eligible]):errors.append('ENVELOPE')
  counts[now]=dict(records=len(rows),distinct_strata=len({r['key'] for r in rows}),latest_observed_at=max(r['observed_at'] for r in rows),decision=evidence)
 result=dict(errors=errors,states=counts,uses_future_completion_labels=False)
 with (O/'calibration_validation.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
 assert not errors
 print('CALIBRATION PASS', {k:v['records'] for k,v in counts.items()})
if __name__=='__main__':main()
