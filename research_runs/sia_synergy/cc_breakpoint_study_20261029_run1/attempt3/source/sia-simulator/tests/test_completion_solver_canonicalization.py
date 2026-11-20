"""Actual failed tie model and adversarial rounding checks; no simulation."""
import importlib,json,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import LinearConstraint

class CompletionNumericsTests(unittest.TestCase):
    def canonical(self,*args):
        m=importlib.import_module('joint_prototype.completion_continuity')
        return m.canonical_integral_solution(*args)

    def test_recorded_660_column_model_rounds_to_all_exact_constraints(self):
        d=json.loads((Path(__file__).parent/'fixtures/reciprocal_tie_failure.json').read_text())['solver_diagnostics']
        constraints=[LinearConstraint(v['A'],v['lb'],v['ub']) for v in d['model']['constraints']]
        raw=np.asarray(d['retry_x']);self.assertGreater(np.max(abs(raw-np.rint(raw))),1e-7)
        rounded=self.canonical(raw,constraints)
        self.assertTrue(np.all((rounded==0)|(rounded==1)))
        for v in constraints:
            values=np.asarray(v.A)@rounded
            self.assertTrue(np.all(values>=np.asarray(v.lb)-1e-10))
            self.assertTrue(np.all(values<=np.asarray(v.ub)+1e-10))

    def test_rounding_cannot_hide_resource_or_primary_band_violation(self):
        for constraint in [LinearConstraint([[1.,1.]],[-np.inf],[1.]),
                           LinearConstraint([[1.,1.000001]],[.999999999],[1.000000001])]:
            with self.subTest(constraint=constraint):
                with self.assertRaises(ValueError):self.canonical([1e-7,1-1e-7] if constraint.A[0,1]>1 else [1-1e-7,1-1e-7],[constraint])

    def test_non_near_binary_nonfinite_and_bounds_rejected(self):
        for vector in [[.5,.5],[float('nan'),1.],[1.001,0.],[-.01,1.]]:
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):self.canonical(vector,[])

if __name__=='__main__':unittest.main()
