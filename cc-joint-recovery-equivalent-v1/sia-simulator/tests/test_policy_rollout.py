import unittest
from types import SimpleNamespace as S
from unittest.mock import patch
from joint_prototype.policy_rollout import advance_model,validate_hypothetical

class PolicyRolloutTests(unittest.TestCase):
 def case(self,delay=10.,rho=3.):
  j=S(name='x',app_name='test',epoch=0,current_cluster='aws',placement=(1,),submission_time=0.,completion_time=None,
      progress=0.,rescale_time=delay,execution_time=0.,attained_service=0.,wasted_gpu_seconds=0.,used_gpu_seconds=0.,current_time=0.,joint_step_segments=[])
  c=S(current_time=0.,jobs=[j],policy=S(current_resources={'x':{'config_id':'aws:1:1','rho':rho}},declared_work={'x':{'total_progress':100.}}))
  return c,j
 def test_delay_does_not_train_and_completion_releases_exactly(self):
  c,j=self.case()
  with patch('joint_prototype.policy_rollout.predict_job',return_value={'raw_rate':2.,'atomic_bsz':1,'accum_steps':0}):ev=advance_model(c,30.)
  self.assertAlmostEqual(j.completion_time,10+100/6);self.assertAlmostEqual(j.execution_time,100/6)
  self.assertEqual(j.rescale_time,0.);self.assertEqual(c.current_time,30.)
  self.assertEqual(ev[0]['checkpoint'],10.);self.assertAlmostEqual(ev[0]['raw_rate']*ev[0]['rho']*ev[0]['training'],100.)
 def test_full_checkpoint_cannot_gain_progress(self):
  c,j=self.case(delay=90.)
  with patch('joint_prototype.policy_rollout.predict_job',side_effect=AssertionError('checkpoint must not query training')):advance_model(c,60.)
  self.assertEqual(j.progress,0.);self.assertEqual(j.rescale_time,30.)
 def test_unknown_rate_is_explicit(self):
  c,j=self.case(delay=0.)
  with patch('joint_prototype.policy_rollout.predict_job',return_value={'raw_rate':None}):
   with self.assertRaisesRegex(ValueError,'UNKNOWN'):advance_model(c,60.)
 def test_hypothetical_initial_does_not_disable_real_guards(self):
  with self.assertRaisesRegex(RuntimeError,'NOT_PREDICTION'):validate_hypothetical(False,{'valid':True},True)
  with self.assertRaisesRegex(RuntimeError,'INVALID'):validate_hypothetical(True,{'valid':False},True)
  with self.assertRaisesRegex(RuntimeError,'INVALID'):validate_hypothetical(True,{'valid':True},False)
  self.assertTrue(validate_hypothetical(True,{'valid':True},True))
if __name__=='__main__':unittest.main()
