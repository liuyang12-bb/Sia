"""Real oracle/column/MIP regression; synthetic rates, no workload tuning."""
import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from test_adaptive_service import spec,cfg,current
from test_clean_ablation import controller,state
from test_tail_recovery import scored
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.neutral_liveness import geometry


class RecoveryEquivalenceTests(unittest.TestCase):
 def fixture(self,mode='fixed'):
  j=spec(configs=[cfg('aws'),cfg('dgx-ext',rate=10.)])
  a=controller([j],gpu=mode,aux='opt')
  nodes=[NodeSpec(t+':0',t,1,3.,62.5) for t in ('aws','dgx-ext')]
  s=state([j],nodes);cs=scored(s)
  first=a.decide(s,[c for c in cs if c.config.gpu_type=='aws'],0.,{})
  a.services['j']['delay_charged']=250.
  s=state([current(j,first.selected)],nodes)
  return a,s,first

 def test_equal_gpu_recovery_domain_for_fo_full(self):
  domains=[]
  for mode in ('fixed','joint'):
   a,s,_=self.fixture(mode);r=a.decide(s,scored(s),360.,{'j':100.})
   self.assertEqual(r.selected['j'].config.gpu_type,'dgx-ext')
   self.assertTrue(r.audit['valid']);self.assertTrue(a.services['j']['tail_recovery_used'])
   self.assertEqual(a.services['j']['delay_charged'],300.)
   self.assertEqual(a.services['j']['tail_extra_budget'],50.)
   rec=r.audit['tail_recoveries'][0];self.assertEqual(rec['column_id'],r.selected['j'].column_id)
   self.assertEqual(rec['eta_after'],s.current_completion_scores[rec['column_id']]['estimated_completion'])
   domains.append(r.audit['shared_eligibility']['j']['eligible_geometries'])
  self.assertEqual(domains[0],domains[1])

 def test_proportional_selected_column_consumes_same_permit(self):
  a,s,_=self.fixture('fixed');a.clean_aux='prop'
  r=a.decide(s,scored(s),360.,{'j':100.})
  c=r.selected['j'];self.assertEqual(c.config.gpu_type,'dgx-ext');self.assertTrue(a.is_prop(c))
  self.assertTrue(a.services['j']['tail_recovery_used'])
  self.assertEqual(r.audit['tail_recoveries'][0]['column_id'],c.column_id)

 def test_real_checkpoint_remains_exact(self):
  a,s,first=self.fixture();s.jobs=[replace(s.jobs[0],checkpoint_locked=True)];s.job_map={'j':s.jobs[0]}
  r=a.decide(s,scored(s),60.,{'j':0.})
  self.assertEqual(geometry(first.selected['j']),geometry(r.selected['j']))
  self.assertFalse(a.services['j'].get('tail_recovery_used',False))

 def test_unknown_or_no_payback_does_not_open_prop(self):
  for reason in ('unknown','slow','bad_delay'):
   a,s,_=self.fixture();cs=scored(s)
   for c in cs:
    if c.config.gpu_type=='dgx-ext' and a.is_prop(c):
     score=s.current_completion_scores[c.column_id]
     if reason=='unknown':score['raw_rate']=None
     elif reason=='slow':score['estimated_completion']=1e9
     else:score['delay']=0.
   r=a.decide(s,cs,360.,{'j':100.})
   self.assertEqual(r.selected['j'].config.gpu_type,'aws')
   self.assertFalse(a.services['j'].get('tail_recovery_used',False))

 def test_reordered_columns_same_geometry(self):
  results=[]
  for rev in (False,True):
   a,s,_=self.fixture();cs=scored(s)
   r=a.decide(s,list(reversed(cs)) if rev else cs,360.,{'j':100.})
   self.assertEqual(r.selected['j'].config.gpu_type,'dgx-ext');results.append(geometry(r.selected['j']))
  self.assertEqual(results[0],results[1])

 def test_solver_failure_atomic(self):
  a,s,_=self.fixture();before=copy.deepcopy(a.__dict__)
  with patch.object(a,'_milp',return_value=SimpleNamespace(x=None,status=1,success=False,message='injected')):
   with self.assertRaises(RuntimeError):a.decide(s,scored(s),360.,{'j':100.})
  self.assertEqual(a.__dict__,before)

 def test_used_permit_geometry_frozen(self):
  a,s,_=self.fixture();r=a.decide(s,scored(s),360.,{'j':100.})
  self.assertEqual(r.selected['j'].config.gpu_type,'dgx-ext')
  s=state([current(s.jobs[0],r.selected,2)],s.nodes)
  r2=a.decide(s,scored(s),480.,{'j':170.})
  self.assertEqual(r2.selected['j'].config.gpu_type,'dgx-ext')
  self.assertEqual(a.services['j']['delay_charged'],300.)
  self.assertFalse(r2.audit['tail_recoveries'])

if __name__=='__main__':unittest.main(verbosity=2)
