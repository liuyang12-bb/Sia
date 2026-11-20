"""Real candidate/oracle/audit/MIP state tests; toy rates explicitly labelled."""
import copy, unittest
from unittest.mock import patch
from types import SimpleNamespace
from test_adaptive_service import spec,cfg,current
from test_clean_ablation import controller,state
from test_tail_recovery import scored
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.bounded_correction import BoundedCorrectionMixin

class BoundedCorrectionTests(unittest.TestCase):
 def fixture(self,mode='joint',new=True):
  jobs=[spec('a',configs=[cfg('aws'),cfg('dgx-ext',rate=10.)]),spec('b',configs=[cfg('rtx')],index=1)]
  a=controller(jobs,gpu=mode,aux='opt')
  if new:a.__class__=type('Correction',(BoundedCorrectionMixin,a.__class__),{})
  s=state(jobs,[NodeSpec(t+':0',t,1,3.,62.5) for t in ('aws','dgx-ext','rtx')])
  cs=scored(s);first=a.decide(s,[c for c in cs if c.config.gpu_type!='dgx-ext'],0.,{})
  a.services['a']['delay_charged']=250.
  s=state([current(j,first.selected) for j in jobs],s.nodes)
  return a,s
 def test_real_solver_recovers_same_state_for_both_methods(self):
  for mode in ('fixed','joint'):
   a,s=self.fixture(mode);r=a.decide(s,scored(s),360.,{'a':100.,'b':100.})
   self.assertEqual(r.selected['a'].config.gpu_type,'dgx-ext')
   self.assertTrue(r.audit['valid']);self.assertTrue(a.services['a']['tail_recovery_used'])
   self.assertEqual(a.services['a']['delay_charged'],300.)
   self.assertEqual(r.selected['b'].config.gpu_type,'rtx')
 def test_legacy_retains_counterexample(self):
  a,s=self.fixture(new=False);r=a.decide(s,scored(s),360.,{'a':100.,'b':100.})
  self.assertEqual(r.selected['a'].config.gpu_type,'aws')
 def test_checkpoint_exact(self):
  a,s=self.fixture();s.job_map['a']=__import__('dataclasses').replace(s.job_map['a'],checkpoint_locked=True)
  s.jobs=[s.job_map[j.job_id] for j in s.jobs]
  r=a.decide(s,scored(s),360.,{'a':100.,'b':100.})
  self.assertEqual(r.selected['a'].config.gpu_type,'aws')
 def test_waiting_disables_extension(self):
  a,s=self.fixture();a.ready=['b']
  r=a.decide(s,scored(s),360.,{'a':100.,'b':100.})
  self.assertEqual(r.selected['a'].config.gpu_type,'aws')
 def test_no_payback_no_permit(self):
  a,s=self.fixture();r=a.decide(s,scored(s,remaining=1.),360.,{'a':100.,'b':100.})
  self.assertEqual(r.selected['a'].config.gpu_type,'aws')
 def test_failure_rolls_back(self):
  a,s=self.fixture();old=copy.deepcopy(a.__dict__)
  with patch.object(a,'_milp',return_value=SimpleNamespace(x=None,status=1,success=False,message='injected')):
   with self.assertRaises(RuntimeError):a.decide(s,scored(s),360.,{'a':100.,'b':100.})
  self.assertEqual(old,a.__dict__)
 def test_no_second_recovery(self):
  a,s=self.fixture();r=a.decide(s,scored(s),360.,{'a':100.,'b':100.})
  self.assertTrue(a.services['a'].get('tail_recovery_used'))
  s=state([current(j,r.selected,2) for j in s.jobs],s.nodes)
  r=a.decide(s,scored(s),480.,{'a':160.,'b':160.})
  self.assertEqual(a.services['a']['delay_charged'],300.)
 def test_unobserved_target(self):
  a,s=self.fixture();a.profile_sources['a']['dgx-ext:1:1']='unprofiled_sia_fallback'
  r=a.decide(s,scored(s),360.,{'a':100.,'b':100.})
  self.assertEqual(r.selected['a'].config.gpu_type,'aws')
 def test_no_spare_capacity_no_displacement(self):
  a,s=self.fixture();peer=s.job_map['b']
  # Move the existing peer to the desired target node; keep a trusted column.
  from dataclasses import replace
  peer=replace(peer,configs=(cfg('dgx-ext'),),current_config_id='dgx-ext:1:1',
   current_nodes=('dgx-ext:0',),current_placement=(('dgx-ext:0',1),))
  s=state([s.job_map['a'],peer],s.nodes)
  cs=scored(s);witness=next(c for c in cs if c.job_id=='b')
  a.services['b']['column']=witness
  a._lifetime_history['b']['column']=witness
  a.profile_sources['b']['dgx-ext:1:1']='direct_profile'
  r=a.decide(s,cs,360.,{'a':100.,'b':100.})
  self.assertEqual(r.selected['a'].config.gpu_type,'aws');self.assertTrue(r.audit['valid'])
 def test_zero_rate_disables_offer(self):
  a,s=self.fixture();cs=scored(s)
  for c in cs:
   if c.config.gpu_type=='dgx-ext':s.current_completion_scores[c.column_id]['raw_rate']=0.
  r=a.decide(s,cs,360.,{'a':100.,'b':100.})
  self.assertEqual(r.selected['a'].config.gpu_type,'aws')
 def test_locked_peer_is_protected(self):
  from dataclasses import replace
  a,s=self.fixture();s=state([s.job_map['a'],replace(s.job_map['b'],checkpoint_locked=True)],s.nodes)
  r=a.decide(s,scored(s),180.,{'a':100.,'b':0.})
  self.assertEqual(r.selected['a'].config.gpu_type,'dgx-ext')
  self.assertEqual(r.selected['b'].placement.node_gpus,s.job_map['b'].current_placement)
  self.assertEqual(r.selected['b'].query.cpu_per_gpu,s.job_map['b'].current_cpu_per_gpu)
 def test_delay_mismatch_disables_offer(self):
  a,s=self.fixture();cs=scored(s)
  for c in cs:
   if c.config.gpu_type=='dgx-ext':s.current_completion_scores[c.column_id]['delay']=0.
  r=a.decide(s,cs,360.,{'a':100.,'b':100.})
  self.assertEqual(r.selected['a'].config.gpu_type,'aws')
 def test_real_unseeded_bert_bounded_chain_with_new_controller(self):
  import contextlib,io
  from test_exchange_integration import ExchangeIntegration
  from joint_prototype.completion_continuity import install as completion
  from joint_prototype.tail_recovery import install as tail
  from joint_prototype.clean_ablation import install as clean
  from joint_prototype.execution_audit import audit_execution
  for mode in ('fixed','joint'):
   with contextlib.redirect_stdout(io.StringIO()):
    c=completion(ExchangeIntegration().build(),proportional=False);tail(c);clean(c,'B',1,1,mode,'opt')
    a=c.policy.liveness;a.__class__=type('BoundedReal',(BoundedCorrectionMixin,a.__class__),{})
    trained=0.;checkpoint=0.;previous=0.
    for _ in range(12):
     c.step(60);self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
     for e in c.joint_round_logs[-1]['execution']:
      self.assertTrue(audit_execution(e)['valid']);trained+=e['training_seconds'];checkpoint+=e['checkpoint_delay_consumed']
     self.assertGreaterEqual(c.jobs[0].progress,previous);previous=c.jobs[0].progress
    self.assertGreater(trained,0.);self.assertGreater(checkpoint,0.)
if __name__=='__main__':unittest.main()
