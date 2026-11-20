"""Offline check only: zero-duration epoch boundaries are not checkpoints."""
import gzip,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1'
work=json.loads(next(gzip.open(OUT/'cells/GATE_RNEW_medium_CC_FULL/rounds.jsonl.gz','rt')))['decision_for_next_interval']['declared_work']
values={}
for method in ['SIA_P','CC_CPU','CC_MEM','CC_FULL']:
 eta=0.;boundaries=0
 with gzip.open(OUT/f'cells/GATE_RNEW_medium_{method}/rounds.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   for e in json.loads(line)['execution']:
    progressed=e['progress_before'];skipped=False
    for z in e['training_segments']:
     if z['seconds']>0:
      remain=max(0.,work[e['job_id']]['total_progress']-progressed)
      eta+=remain/z['raw_rate_before_rho']*(1.-1./z['rho']);boundaries+=skipped;break
     progressed+=z['progress_delta'];skipped=True
 values[method]=dict(eta_savings_sum_first_productive_segment=eta,zero_boundary_first_segments_skipped=boundaries)
gate=json.loads((OUT/'opportunity_gate.json').read_text())
assert abs(values['CC_FULL']['eta_savings_sum_first_productive_segment']-gate['methods']['CC_FULL']['totals']['sum_round_same_state_eta_savings'])<1e-7
with (OUT/'gate_eta_boundary_recheck.json').open('x',encoding='utf8') as f:json.dump(dict(values=values,gate_full_unchanged=True,note='First-positive segment handling for zero-second epoch rounding. Original gate preserved. Public declared work common to all methods; RNEW proportional rho=1. No simulation/parameter change.'),f,indent=2)
print(values)
