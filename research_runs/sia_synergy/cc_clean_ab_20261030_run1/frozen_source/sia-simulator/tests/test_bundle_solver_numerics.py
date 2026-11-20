"""An allegedly optimal integer vector must satisfy every linear constraint."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import joint_prototype.adaptive_service as legacy
from test_adaptive_service import spec,scheduler,columns
from test_bundle_adaptation import controller

class SolverNumericsTests(unittest.TestCase):
    def test_invalid_optimal_vector_retries_identical_model_without_presolve(self):
        j=spec();s=scheduler([j]);a=controller([j]);real=legacy.milp;seen=[]
        def injected(**kwargs):
            seen.append(kwargs)
            if len(seen)==1:return SimpleNamespace(x=np.zeros(len(kwargs['c'])),success=True,status=0,message='injected presolve invalid optimal')
            return real(**kwargs)
        with patch.object(legacy,'milp',side_effect=injected):r=a.decide(s,columns(s),0.,{})
        self.assertEqual(set(r.selected),{'j'});self.assertTrue(r.audit['valid'])
        self.assertFalse(seen[1]['options']['presolve'])
        np.testing.assert_array_equal(seen[0]['c'],seen[1]['c'])
        for x,y in zip(seen[0]['constraints'],seen[1]['constraints']):
            np.testing.assert_array_equal(x.A,y.A);np.testing.assert_array_equal(x.lb,y.lb);np.testing.assert_array_equal(x.ub,y.ub)
        self.assertGreaterEqual(r.audit['numerical_solver_retries'][0]['original_validation']['maximum_violation'],1.)

    def test_second_invalid_vector_stops_and_rolls_back(self):
        j=spec();s=scheduler([j]);a=controller([j]);before=dict(a.__dict__)
        def injected(**kwargs):return SimpleNamespace(x=np.zeros(len(kwargs['c'])),success=True,status=0,message='injected invalid optimal')
        with patch.object(legacy,'milp',side_effect=injected):
            with self.assertRaisesRegex(RuntimeError,'INVALID_SOLVER_RESULT_AFTER_NUMERICAL_RETRY'):a.decide(s,columns(s),0.,{})
        self.assertEqual(a.__dict__,before)
