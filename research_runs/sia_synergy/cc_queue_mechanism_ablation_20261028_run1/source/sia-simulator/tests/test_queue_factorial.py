"""Pre-registered contracts, fixed toy states plus real unseeded chain."""
import contextlib,copy,importlib,io,unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from test_adaptive_service import spec,cfg,current,scheduler,columns,admission
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.neutral_liveness import geometry
from joint_prototype.execution_audit import audit_execution

def impl():return importlib.import_module('joint_prototype.queue_factorial')
def controller(jobs,a=1,b=1,c=1):
 x=impl().QueueFactorAdmission(300.,250.,60.)
 x.factors=(a,b,c);base=admission(jobs,50.)
 x.delay_costs=base.delay_costs;x.profile_sources=base.profile_sources
 return x
def state(jobs):
 s=scheduler(jobs,[NodeSpec('aws:0','aws',1,3.,62.5)])
 s.current_known_configs={(j.job_id,c.config_id) for j in jobs for c in j.configs}
 s.current_completion_scores={}
 return s
def cols(s,weights):
 return [replace(c,scheduling_utility=weights[c.job_id]) for c in columns(s)
         if c.query.cpu_per_gpu==3. and c.query.ram_gb_per_gpu==62.5]

class QueueFactorialTests(unittest.TestCase):
 def test_fifo_factor_requires_oldest_even_when_later_has_higher_score(self):
  js=[spec('old',arrival=0,index=0),spec('new',arrival=30,index=1)];s=state(js)
  for b,expected in [(1,'old'),(0,'new')]:
   x=controller(js,b=b);r=x.decide(s,cols(s,{'old':.1,'new':1.}),60.,{})
   self.assertEqual(set(r.selected),{expected});self.assertTrue(r.audit['valid'])
 def test_negative_scores_free_admission_is_nonempty_not_fabricated_bonus(self):
  js=[spec('old'),spec('new',index=1)];s=state(js);x=controller(js,b=0)
  r=x.decide(s,cols(s,{'old':-2.,'new':-1.}),0.,{})
  self.assertEqual(set(r.selected),{'new'});self.assertEqual(r.scheduling_objective,-1.)
 def test_no_continuity_releases_only_after_300_productive_seconds(self):
  js=[spec('old'),spec('new',arrival=60,index=1)];x=controller(js,c=0)
  s=state(js[:1]);first=x.decide(s,cols(s,{'old':1.}),0.,{})
  s=state([current(js[0],first.selected),js[1]])
  r=x.decide(s,cols(s,{'old':1.,'new':1.}),300.,{'old':250.})
  self.assertEqual(set(r.selected),{'old'})
  r=x.decide(s,cols(s,{'old':1.,'new':1.}),360.,{'old':310.})
  self.assertEqual(set(r.selected),{'new'})
  self.assertTrue(any(e['event']=='REQUEUE_PRODUCTIVE_QUANTUM' for e in x.events))
 def test_continuity_on_keeps_old_despite_service_quantum(self):
  js=[spec('old'),spec('new',arrival=60,index=1)];x=controller(js,c=1)
  s=state(js[:1]);first=x.decide(s,cols(s,{'old':1.}),0.,{})
  s=state([current(js[0],first.selected),js[1]])
  r=x.decide(s,cols(s,{'old':.1,'new':1.}),360.,{'old':310.})
  self.assertEqual(set(r.selected),{'old'})
 def test_checkpoint_full_allocation_protected_in_all_eight_cells(self):
  js=[spec('old'),spec('new',arrival=60,index=1)]
  for a in (0,1):
   for b in (0,1):
    for c in (0,1):
     x=controller(js,a,b,c);s=state(js[:1]);first=x.decide(s,cols(s,{'old':1.}),0.,{})
     s=state([current(js[0],first.selected,locked=True),js[1]])
     r=x.decide(s,cols(s,{'old':-100.,'new':100.}),60.,{'old':0.})
     self.assertEqual(geometry(first.selected['old']),geometry(r.selected['old']))
     self.assertEqual(r.selected['old'].candidate_kind,'locked')
 def test_lifetime_budget_not_reset_when_periodic_service_is_reissued(self):
  j=spec('old');x=controller([j],c=0);s=state([j]);first=x.decide(s,cols(s,{'old':1.}),0.,{})
  s=state([current(j,first.selected)])
  second=x.decide(s,cols(s,{'old':1.}),360.,{'old':310.})
  self.assertEqual(x.services['old']['delay_charged'],50.)
  self.assertEqual(second.audit['service_delay_ledgers']['old']['delay_charged'],50.)
 def test_budget_exhaustion_cannot_evict_a_job_that_cannot_resume(self):
  j=spec('old');x=controller([j],c=0);s=state([j]);first=x.decide(s,cols(s,{'old':1.}),0.,{})
  x.services['old']['delay_charged']=250.;s=state([current(j,first.selected)])
  r=x.decide(s,cols(s,{'old':1.}),360.,{'old':310.})
  self.assertIn('old',r.selected)
  self.assertTrue(any(e['event']=='QUANTUM_HELD_NO_LIFETIME_RESUME_BUDGET' for e in x.events))
 def test_solver_failure_does_not_change_queue_or_lifetime_history(self):
  j=spec('old');x=controller([j],b=0,c=0);s=state([j]);before=copy.deepcopy(x.__dict__)
  with patch('joint_prototype.adaptive_service.milp',return_value=SimpleNamespace(status=1,x=None,success=False,message='injected')):
   with self.assertRaisesRegex(RuntimeError,'SOLVER_UNDECIDED'):x.decide(s,cols(s,{'old':1.}),0.,{})
  self.assertEqual(before,x.__dict__)
 def test_all_on_matches_unchanged_tail_controller_exactly(self):
  from joint_prototype.tail_recovery import TailRecoveryAdmission
  js=[spec('old'),spec('new',arrival=60,index=1)]
  x=controller(js);y=TailRecoveryAdmission(300.,250.,60.)
  y.delay_costs=copy.deepcopy(x.delay_costs);y.profile_sources=copy.deepcopy(x.profile_sources)
  s=state(js[:1]);cs=cols(s,{'old':1.});r1=x.decide(s,cs,0.,{});r2=y.decide(s,cs,0.,{})
  s=state([current(js[0],r1.selected),js[1]]);cs=cols(s,{'old':.1,'new':1.})
  for t,done in [(60.,10.),(360.,310.),(420.,370.)]:
   r1=x.decide(s,cs,t,{'old':done});r2=y.decide(s,cs,t,{'old':done})
   self.assertEqual({k:geometry(v) for k,v in r1.selected.items()},{k:geometry(v) for k,v in r2.selected.items()})
   self.assertEqual(x.ready,y.ready);self.assertEqual(x.services,y.services)
 def test_real_unseeded_bert_chain_each_factor_model_and_once_rho(self):
  from test_exchange_integration import ExchangeIntegration
  from joint_prototype.completion_continuity import install
  from joint_prototype.tail_recovery import install as tail
  for factors in ((1,1,1),(0,1,1),(1,0,1),(1,1,0)):
   with contextlib.redirect_stdout(io.StringIO()):
    c=install(ExchangeIntegration().build(),proportional=True);tail(c);impl().install(c,*factors)
    known=False
    for _ in range(12):
     c.step(60);d=c.policy.last_decision;self.assertTrue(d['result']['audit']['valid'])
     for e in c.joint_round_logs[-1]['execution']:self.assertTrue(audit_execution(e)['valid'])
     known|=any(v['raw_rate'] is not None for v in d['continuity_scores'])
    self.assertTrue(known);self.assertGreater(c.jobs[0].progress,0.)
    self.assertEqual(d['queue_factors'],dict(A=factors[0],B=factors[1],C=factors[2]))
if __name__=='__main__':unittest.main(verbosity=2)
