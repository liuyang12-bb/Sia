"""Pre-registered bounded-tail contracts; production MIP, labelled toy rates."""
import copy,importlib,unittest,contextlib,io
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from test_adaptive_service import spec,cfg,current,admission,columns
from test_completion_continuity_states import state
def impl():return importlib.import_module('joint_prototype.tail_recovery')
def scored(s,remaining=10000.):
 cs=columns(s);scores={};out=[]
 for c in cs:
  j=s.job_map[c.job_id];same=j.current_config_id==c.config.config_id and j.current_placement==c.placement.node_gpus
  delay=0. if same else 50.;rate=c.config.sia_goodput*c.query.rho;eta=delay+remaining/rate
  scores[c.column_id]=dict(estimated_completion=eta,raw_rate=rate,delay=delay)
  out.append(replace(c,scheduling_utility=1/eta))
 s.current_completion_scores=scores;return out
def fixture():
 j=spec(configs=[cfg('aws'),cfg('dgx-ext',rate=10.),cfg('rtx',rate=100.)])
 a=impl().TailRecoveryAdmission(300.,250.,60.);base=admission([j],50.)
 a.delay_costs=base.delay_costs;a.delay_costs['j']['aws']=250.;a.profile_sources=base.profile_sources
 s=state([j]);cs=scored(s);first=a.decide(s,[c for c in cs if c.config.gpu_type=='aws'],0.,{})
 return a,j,first
class TailRecoveryTests(unittest.TestCase):
 def test_exhausted_budget_can_recover_once(self):
  a,j,first=fixture();s=state([current(j,first.selected)])
  r=a.decide(s,scored(s),360.,{'j':100.})
  self.assertEqual(r.selected['j'].config.gpu_type,'rtx');self.assertTrue(r.audit['valid'])
  self.assertTrue(a.services['j']['tail_recovery_used']);self.assertEqual(a.services['j']['delay_charged'],300.)
  self.assertEqual(a.services['j']['tail_extra_budget'],50.)
 def test_checkpoint_is_never_overridden(self):
  a,j,first=fixture();s=state([current(j,first.selected,locked=True)])
  r=a.decide(s,scored(s),60.,{'j':0.})
  self.assertEqual(r.selected['j'].config.gpu_type,'aws');self.assertFalse(a.services['j'].get('tail_recovery_used',False))
 def test_other_active_job_disables_tail(self):
  a,j,first=fixture();peer=spec('peer',arrival=60,index=1);s=state([current(j,first.selected),peer])
  a.delay_costs['peer']=admission([peer],50.).delay_costs['peer']
  r=a.decide(s,scored(s),360.,{'j':100.})
  self.assertEqual(r.selected['j'].config.gpu_type,'aws');self.assertFalse(a.services['j'].get('tail_recovery_used',False))
 def test_unobserved_target_is_rejected(self):
  a,j,first=fixture()
  for t in ['dgx-ext:1:1','rtx:1:1']:a.profile_sources['j'][t]='unprofiled_sia_fallback'
  s=state([current(j,first.selected)]);r=a.decide(s,scored(s),360.,{'j':100.})
  self.assertEqual(r.selected['j'].config.gpu_type,'aws')
 def test_unknown_eta_is_rejected(self):
  a,j,first=fixture();s=state([current(j,first.selected)]);cs=scored(s)
  for c in cs:
   if c.config.gpu_type!='aws':s.current_completion_scores[c.column_id]['estimated_completion']=None
  r=a.decide(s,cs,360.,{'j':100.});self.assertEqual(r.selected['j'].config.gpu_type,'aws')
 def test_no_productive_observation_disables_recovery(self):
  a,j,first=fixture();s=state([current(j,first.selected)])
  r=a.decide(s,scored(s),60.,{'j':0.});self.assertEqual(r.selected['j'].config.gpu_type,'aws')
 def test_event_gate_only_is_also_bounded(self):
  a,j,first=fixture();a.services['j']['delay_charged']=50.;s=state([current(j,first.selected)])
  r=a.decide(s,scored(s),360.,{'j':100.})
  self.assertEqual(r.selected['j'].config.gpu_type,'rtx');self.assertEqual(a.services['j']['tail_extra_budget'],0.)
 def test_migration_must_pay_back(self):
  a,j,first=fixture();s=state([current(j,first.selected)]);r=a.decide(s,scored(s,remaining=1.),360.,{'j':100.})
  self.assertEqual(r.selected['j'].config.gpu_type,'aws')
 def test_cannot_rescue_again_after_model_changes(self):
  a,j,first=fixture();s=state([current(j,first.selected)]);second=a.decide(s,scored(s),360.,{'j':100.})
  s=state([current(j,second.selected,2)]);cs=scored(s)
  for c in cs:
   if c.config.gpu_type=='dgx-ext':s.current_completion_scores[c.column_id]['estimated_completion']=.001
  third=a.decide(s,cs,480.,{'j':170.})
  self.assertEqual(third.selected['j'].config.gpu_type,'rtx');self.assertEqual(a.services['j']['delay_charged'],300.)
 def test_solver_failure_does_not_consume_permit(self):
  a,j,first=fixture();before=copy.deepcopy(a.__dict__);s=state([current(j,first.selected)])
  failed=SimpleNamespace(x=None,status=1,success=False,message='injected')
  with patch.object(impl().TailRecoveryAdmission,'_milp',return_value=failed):
   with self.assertRaises(RuntimeError):a.decide(s,scored(s),360.,{'j':100.})
  self.assertEqual(a.__dict__,before)
 def test_old_mode_retains_the_counterexample(self):
  from joint_prototype.completion_continuity import ContinuityAdmission
  a,j,first=fixture();a.__class__=ContinuityAdmission;s=state([current(j,first.selected)])
  r=a.decide(s,scored(s),360.,{'j':100.});self.assertEqual(r.selected['j'].config.gpu_type,'aws')
 def test_real_unseeded_bert_full_chain(self):
  from test_completion_continuity import CompletionContinuityTests
  with contextlib.redirect_stdout(io.StringIO()):
   c=CompletionContinuityTests().build();impl().install(c)
   previous=0.
   for _ in range(12):
    c.step(60);self.assertGreaterEqual(c.jobs[0].progress,previous);previous=c.jobs[0].progress
    self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
    for e in c.joint_round_logs[-1]['execution']:
     if e['checkpoint_delay_consumed']>=e['observed_active_seconds']>0:self.assertEqual(e['progress_delta'],0)
   self.assertGreater(previous,0.)
if __name__=='__main__':unittest.main()
