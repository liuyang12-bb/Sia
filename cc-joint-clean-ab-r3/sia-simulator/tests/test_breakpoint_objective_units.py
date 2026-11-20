"""Original feasible incumbent disproves loose 'optimal'; numeric units only."""
import gzip,importlib,json,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import LinearConstraint

class ObjectiveUnits(unittest.TestCase):
 def module(self):return importlib.import_module('joint_prototype.completion_continuity')
 def test_common_units_roundtrip_and_relative_coefficients_preserved(self):
  original=np.array([0.,-.913,1e-8,-1.]);scaled,k=self.module().objective_solver_units(original)
  self.assertEqual(k,2.**30);self.assertTrue(np.array_equal(scaled/k,original))
  self.assertTrue(np.array_equal(original,np.array([0.,-.913,1e-8,-1.])))
 def test_integer_tie_costs_need_no_unit_change(self):
  original=np.array([0.,-1.,1.,25.,99.]);scaled,k=self.module().objective_solver_units(original)
  self.assertEqual(k,1.);self.assertTrue(np.array_equal(original,scaled))
 def test_nonfinite_objective_stops_not_zeroed(self):
  for v in [[0.,float('nan')],[float('inf'),1.]]:
   with self.assertRaises(ValueError):self.module().objective_solver_units(v)
 def test_recorded_primary_meets_known_better_feasible_incumbent(self):
  m=self.module();p=Path(__file__).parent/'fixtures'
  d=json.loads(gzip.open(p/'breakpoint_primary_band_failure.json.gz','rt',encoding='utf8').read());proof=json.loads((p/'breakpoint_primary_precision.json').read_text())
  data=d['model'];weights=np.asarray(data['constraints'][1]['A']).reshape(-1)
  rows=[LinearConstraint(data['constraints'][0]['A'],data['constraints'][0]['lb'],data['constraints'][0]['ub'])]
  cols=[SimpleNamespace(job_id=c['job'],column_id=c['column_id']) for c in data['columns']]
  a=m.NumericalAdmission(300.,250.,60.);a._diagnostic_job_order=[];a.numerical_solver_retries=[]
  out=a._milp(-weights,cols,rows);x=m.canonical_integral_solution(out.x,rows)
  self.assertGreaterEqual(float(weights@x)+1e-10,proof['known_feasible_baseline'])
  self.assertAlmostEqual(out.fun,float(-weights@x),places=12)
  self.assertGreater(len(a.objective_unit_certifications),0)

if __name__=='__main__':unittest.main()
