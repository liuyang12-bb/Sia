import copy,unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from test_adaptive_service import cfg,spec,current,scheduler,columns,admission
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.neutral_liveness import geometry
try:
 from joint_prototype.protected_admission import ProtectedAdmission
except ModuleNotFoundError:
 from joint_prototype.bundle_adaptation import BundleAdmission as ProtectedAdmission

def controller(jobs):
 a=ProtectedAdmission(300.,250.,60.);b=admission(jobs,50.)
 a.delay_costs=b.delay_costs;a.profile_sources=b.profile_sources;return a
def state(jobs,gpus=3):
 return scheduler(jobs,[NodeSpec('aws:0','aws',gpus,3*gpus,62.5*gpus)])

class ProtectedAdmissionTests(unittest.TestCase):
 def test_negative_head_is_served_but_negative_optional_peer_waits(self):
  jobs=[spec('head'),spec('peer',index=1)];s=state(jobs);a=controller(jobs)
  cs=[replace(c,scheduling_utility=-1.) for c in columns(s)]
  r=a.decide(s,cs,0.,{})
  self.assertEqual(set(r.selected),{'head'});self.assertEqual(a.ready,['peer'])

 def test_optional_set_can_replace_fifo_greedy_packing(self):
  jobs=[spec('head'),spec('middle',index=1),spec('gang',[cfg('aws',2)],index=2)]
  s=state(jobs);a=controller(jobs)
  cs=[replace(c,scheduling_utility={'head':-1.,'middle':.1,'gang':2.}[c.job_id]) for c in columns(s)]
  r=a.decide(s,cs,0.,{})
  self.assertEqual(set(r.selected),{'head','gang'});self.assertEqual(a.ready,['middle'])

 def test_locked_optional_peer_is_still_exactly_protected(self):
  jobs=[spec('head'),spec('peer',index=1)];a=controller(jobs);s=state(jobs)
  first=a.decide(s,[replace(c,scheduling_utility=1.) for c in columns(s)],0.,{})
  self.assertEqual(set(first.selected),{'head','peer'})
  jobs=[current(j,first.selected,locked=True) for j in jobs];s=state(jobs)
  r=a.decide(s,[replace(c,scheduling_utility=-100.) for c in columns(s)],60.,{})
  for j in jobs:self.assertEqual(geometry(first.selected[j.job_id]),geometry(r.selected[j.job_id]))

 def test_drain_issues_no_new_optional_service_and_reaches_gang(self):
  small=spec('small');gang=spec('gang',[cfg('aws',3)],index=1);peer=spec('peer',index=2)
  a=controller([small,gang,peer]);s=state([small]);first=a.decide(s,columns(s),0.,{})
  jobs=[current(small,first.selected),gang,peer];s=state(jobs)
  r=a.decide(s,columns(s),60.,{'small':10.})
  self.assertTrue(r.audit['drain']);self.assertEqual(set(r.selected),{'small'})
  r=a.decide(s,columns(s),360.,{'small':300.})
  self.assertEqual(set(r.selected),{'gang'})

 def test_solver_failure_does_not_commit_queue_or_service(self):
  jobs=[spec('head'),spec('peer',index=1)];a=controller(jobs);s=state(jobs)
  before=copy.deepcopy(a.__dict__)
  with patch('joint_prototype.adaptive_service.milp',return_value=SimpleNamespace(x=None,status=1,success=False,message='injected')):
   with self.assertRaisesRegex(RuntimeError,'SOLVER_UNDECIDED'):a.decide(s,columns(s),0.,{})
  self.assertEqual(before,a.__dict__)

 def test_head_without_solo_candidate_is_explicit_failure(self):
  jobs=[spec('head'),spec('peer',index=1)];a=controller(jobs);s=state(jobs)
  with self.assertRaisesRegex(RuntimeError,'NO_SOLO_FEASIBLE_CANDIDATE'):
   a.decide(s,[c for c in columns(s) if c.job_id=='peer'],0.,{})

 def test_negative_optional_job_eventually_becomes_required_and_repeats(self):
  jobs=[spec('a'),spec('b',index=1)];a=controller(jobs);s=state(jobs)
  def negative(s):return [replace(c,scheduling_utility=-1.) for c in columns(s)]
  r=a.decide(s,negative(s),0.,{});self.assertEqual(set(r.selected),{'a'})
  s=state([current(j,r.selected) for j in jobs])
  r=a.decide(s,negative(s),360.,{'a':300.});self.assertEqual(set(r.selected),{'b'})
  s=state([current(j,r.selected,round_index=2) for j in jobs])
  r=a.decide(s,negative(s),720.,{'a':300.,'b':300.});self.assertEqual(set(r.selected),{'a'})

if __name__=='__main__':unittest.main()
