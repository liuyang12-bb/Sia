import json,sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as S
O=Path(__file__).resolve().parent;T=O.parents[2]/'cc-joint-uncertainty-v4'
sys.path[:0]=[str(T/'sia-simulator'),str(T/'sia-simulator/tests')]
from test_policy_rollout import PolicyRolloutTests
from joint_prototype.policy_rollout import RATE_FACTORS,advance_model
from joint_prototype.uncertainty import calibration_key
case=PolicyRolloutTests();c,j=case.case();pred=dict(raw_rate=2.,atomic_bsz=1,accum_steps=0)
key=calibration_key(j,S(config_id='aws:1:1'),pred);token=RATE_FACTORS.set({key:.5})
try:
 with patch('joint_prototype.policy_rollout.predict_job',return_value=pred):advance_model(c,30.)
finally:RATE_FACTORS.reset(token)
assert j.progress==60. and j.completion_time is None and c.policy.current_resources['x']['rho']==3.
token=RATE_FACTORS.set({});c2,j2=case.case();rejected=False
try:
 with patch('joint_prototype.policy_rollout.predict_job',return_value=pred):advance_model(c2,30.)
except ValueError as e:rejected='UNCOVERED_SCENARIO_KEY' in str(e)
finally:RATE_FACTORS.reset(token)
assert rejected
with (O/'scenario_smoke.json').open('x') as f:json.dump(dict(half_rate_progress=60.,physical_rho_unchanged=3.,uncovered_key_rejected=True),f,indent=2)
print('SCENARIO_KERNEL PASS')
