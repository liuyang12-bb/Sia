"""Actual74-column red fixture, reversible conditioning, strict-row protection."""
import gzip,importlib,json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.optimize import LinearConstraint,OptimizeResult

class BreakpointConditioning(unittest.TestCase):
 def admission(self):
  m=importlib.import_module('joint_prototype.completion_continuity')
  a=m.NumericalAdmission(300.,250.,60.);a._diagnostic_job_order=[];a.numerical_solver_retries=[]
  return m,a
 def fixture(self):
  p=Path(__file__).parent/'fixtures/breakpoint_primary_band_failure.json.gz'
  return json.loads(gzip.open(p,'rt',encoding='utf8').read())
 def test_actual_failed74_column_model_obeys_original_rows(self):
  m,a=self.admission();d=self.fixture();model=d['model']
  cols=[SimpleNamespace(job_id=v['job'],column_id=v['column_id']) for v in model['columns']]
  constraints=[LinearConstraint(v['A'],v['lb'],v['ub']) for v in model['constraints']]
  out=a._milp(np.array(model['c']),cols,constraints)
  self.assertEqual(out.status,0);x=m.canonical_integral_solution(out.x,constraints)
  self.assertTrue(np.all((x==0)|(x==1)))
 def test_power_two_transformation_is_bit_reversible_and_preserves_input(self):
  m,_=self.admission();d=self.fixture();model=d['model']
  rows=[LinearConstraint(v['A'],v['lb'],v['ub']) for v in model['constraints']]
  before=[(np.asarray(v.A).copy(),np.asarray(v.lb).copy(),np.asarray(v.ub).copy()) for v in rows]
  scaled,exp=m.condition_constraint_bands(rows)
  self.assertTrue(any(any(e>0 for e in row) for row in exp))
  for original,new,exps,old in zip(rows,scaled,exp,before):
   self.assertTrue(np.array_equal(np.ldexp(np.asarray(new.A),-np.asarray(exps)[:,None]),np.atleast_2d(original.A)))
   for a,b in zip((original.A,original.lb,original.ub),old):self.assertTrue(np.array_equal(a,b))
 def test_success_status_cannot_hide_small_primary_band_violation(self):
  m,a=self.admission();rows=[LinearConstraint([[1.,1.000000015]],1.000000014,1.000000016)]
  cols=[SimpleNamespace(job_id='a',column_id='a'),SimpleNamespace(job_id='b',column_id='b')]
  bad=OptimizeResult(status=0,success=True,x=np.array([1.,0.]),fun=0.,message='injected within loose resource tolerance but outside primary band')
  with patch('joint_prototype.bundle_adaptation.BundleAdmission._milp',return_value=bad):
   try:out=a._milp(np.zeros(2),cols,rows)
   except RuntimeError as exc:
    # Conditioning is not guaranteed to cure every numerical model. A clear
    # stop is correct; a status0 vector outside the primary band is not.
    self.assertIn(str(exc),['CONDITIONED_SOLVER_NO_OPTIMAL_SOLUTION','CONDITIONED_SOLUTION_FAILS_ORIGINAL_ROWS'])
   else:
    x=m.canonical_integral_solution(out.x,rows)
    self.assertEqual(x.tolist(),[0.,1.])
 def test_illegal_resource_vector_still_cannot_be_canonicalized(self):
  m,_=self.admission()
  with self.assertRaises(ValueError):m.canonical_integral_solution([1.,1.],[LinearConstraint([[1.,1.]],-np.inf,1.)])

if __name__=='__main__':unittest.main()
