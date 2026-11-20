"""Same candidates/objectives; literal MILP versus exact redundant-call proof."""
import copy,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core80 import Core80Admission
from test_adaptive_service import spec,current
from test_clean_ablation import controller,state,cs
from joint_prototype.joint_profile import NodeSpec

def solve(js,s,columns,enabled=True):
 a=controller(js);a.__class__=Core80Admission;a.resource_factor='PROP';a.factors=(1,1,1);a.enable_exact_optional_proof=enabled
 return a,a.decide(s,columns,0.,{})
def chosen(r):return {j:(c.column_id,c.scheduling_utility) for j,c in r.selected.items()}

class ExactOptionalTests(unittest.TestCase):
 def test_80_zero_optional_old_and_certified_same_exact_output(self):
  js=[spec(f'j{i:03}',index=i) for i in range(80)];s=state(js,[NodeSpec('aws:0','aws',4,12.,250.)]);cols=cs(s,{j.job_id:0. for j in js})
  _,old=solve(js,s,cols,False);a,new=solve(js,s,cols,True)
  self.assertEqual(chosen(old),chosen(new));self.assertEqual(set(new.selected),{'j000'})
  self.assertEqual(new.audit['exact_zero_optional_saved_solver_calls'],79)
  self.assertTrue(all(x['all_current_rows_revalidated'] for x in new.audit['exact_zero_optional_certificates']))
  self.assertFalse(any(k.startswith('_proof_') for k in a.__dict__))
 def test_positive_utility_optional_still_selected(self):
  js=[spec('a'),spec('b',index=1),spec('c',index=2)];s=state(js,[NodeSpec('aws:0','aws',2,6.,125.)]);cols=cs(s,{'a':.1,'b':.3,'c':0.})
  _,old=solve(js,s,cols,False);_,new=solve(js,s,cols,True)
  self.assertEqual(chosen(old),chosen(new));self.assertEqual(set(new.selected),{'a','b'})
  self.assertFalse(any(x['job']=='b' for x in new.audit['exact_zero_optional_certificates']))
 def test_zero_startup_optional_not_certified_and_can_be_selected(self):
  js=[spec('a'),spec('b',index=1)];s=state(js,[NodeSpec('aws:0','aws',2,6.,125.)]);prior=s.build_proportional_columns()[1]
  js[1]=current(js[1],{'b':prior});s=state(js,s.nodes);cols=cs(s,{'a':1.,'b':0.})
  _,old=solve(js,s,cols,False);_,new=solve(js,s,cols,True)
  self.assertEqual(chosen(old),chosen(new));self.assertIn('b',new.selected)
  self.assertEqual(new.audit['exact_zero_optional_saved_solver_calls'],0)
 def test_nonzero_negative_optional_never_misclassified(self):
  js=[spec('a'),spec('b',index=1)];s=state(js,[NodeSpec('aws:0','aws',2,6.,125.)]);cols=cs(s,{'a':1.,'b':-1.})
  _,old=solve(js,s,cols,False);_,new=solve(js,s,cols,True)
  self.assertEqual(chosen(old),chosen(new));self.assertEqual(new.audit['exact_zero_optional_saved_solver_calls'],0)
 def test_solver_failure_not_replaced_by_certificate_and_atomic(self):
  js=[spec('a'),spec('b',index=1)];s=state(js);cols=cs(s,{'a':0.,'b':0.});a=controller(js);a.__class__=Core80Admission;a.resource_factor='PROP';a.factors=(1,1,1);before=copy.deepcopy(a.__dict__)
  with patch('joint_prototype.adaptive_service.milp',return_value=SimpleNamespace(status=1,x=None,success=False,message='injected')):
   with self.assertRaisesRegex(RuntimeError,'SOLVER_UNDECIDED'):a.decide(s,cols,0.,{})
  self.assertEqual(before,a.__dict__)
 def test_service_group_A_keeps_literal_solver_semantics(self):
  js=[spec('a'),spec('b',index=1)];s=state(js);cols=cs(s,{'a':0.,'b':0.});a=controller(js);a.__class__=Core80Admission;a.resource_factor='PROP'
  r=a.decide(s,cols,0.,{});self.assertEqual(r.audit['exact_zero_optional_saved_solver_calls'],0)

if __name__=='__main__':unittest.main(verbosity=2)
