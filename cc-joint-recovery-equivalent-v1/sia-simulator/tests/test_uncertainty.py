import unittest
from types import SimpleNamespace as S
from unittest.mock import patch
from joint_prototype.uncertainty import ResidualBank,endpoint_screen,observe,calibration_key,ReliableMixin
class UncertaintyTests(unittest.TestCase):
 def bank(self):
  b=ResidualBank()
  for origin,ratio in [(0,.8),(60,1.2)]:b.add('k',origin,origin+30,ratio,origin+60,'x')
  return b
 def test_missing_is_not_one(self):self.assertIsNone(self.bank().bounds('missing',120))
 def test_future_labels_are_excluded(self):
  b=self.bank();self.assertIsNone(b.bounds('k',59))
  self.assertEqual(b.bounds('k',120)['low'],.8)
  with self.assertRaises(ValueError):b.add('k',120,150,1.,140,'x')
 def test_two_distinct_origins_required(self):
  b=ResidualBank();b.add('k',0,30,.8,60,'a');b.add('k',0,40,1.2,60,'b')
  self.assertIsNone(b.bounds('k',60))
 def test_bad_endpoint_rejects_positive_point_gain(self):
  result=endpoint_screen({'k':self.bank().bounds('k',120)},lambda w:1 if w['k']>1 else -1)
  self.assertFalse(result['accepted'])
 def test_all_endpoints_beneficial_can_accept(self):
  result=endpoint_screen({'k':self.bank().bounds('k',120)},lambda w:10*w['k'])
  self.assertTrue(result['accepted']);self.assertEqual(result['worlds'],2)
 def test_new_key_or_censor_rejects(self):
  def unknown(w):raise ValueError('UNCOVERED_SCENARIO_KEY')
  result=endpoint_screen({'k':self.bank().bounds('k',120)},unknown)
  self.assertFalse(result['accepted']);self.assertIn('UNCOVERED',result['reason'])
 def test_checkpoint_does_not_create_error_label(self):
  c=S(current_time=60.,jobs=[S(name='x',joint_step_segments=[])],_residual_bank=ResidualBank(),
      _calibration_pending={'x':dict(key='k',origin=0.,epoch=0,raw=2.)})
  observe(c);self.assertEqual(c._residual_bank.rows,[])
 def test_app_batch_and_epoch_are_not_pooled(self):
  j=S(app_name='bert',epoch=0);cfg=S(config_id='aws:1:1');a={'atomic_bsz':12,'accum_steps':0}
  k=calibration_key(j,cfg,a);b=self.bank();self.assertIsNone(b.bounds(k,120))
  j.epoch=1;self.assertNotEqual(k,calibration_key(j,cfg,a))
  j.epoch=0;self.assertNotEqual(k,calibration_key(j,cfg,dict(a,atomic_bsz=24)))
 def test_delayed_observation_not_available_at_segment_end(self):
  b=ResidualBank();b.add('k',0,30,.8,120,'a');b.add('k',60,90,1.2,120,'a')
  self.assertIsNone(b.bounds('k',100));self.assertIsNotNone(b.bounds('k',120))
 def test_forecast_failure_keeps_audited_original(self):
  controller=ReliableMixin();s=S(audit=lambda bundle:{'valid':True});legacy=['original']
  with patch('joint_prototype.policy_rollout.PolicyRolloutMixin.refine_completion_bundle',side_effect=RuntimeError('injected')):
   selected,ev=controller.refine_completion_bundle(s,[],legacy,60.)
  self.assertEqual(selected,legacy);self.assertEqual(ev['status'],'PREDICTION_FAILURE_KEEP_LEGACY')
if __name__=='__main__':unittest.main()
