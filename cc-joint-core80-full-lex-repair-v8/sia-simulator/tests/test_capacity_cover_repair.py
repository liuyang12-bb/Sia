import sys,unittest,itertools,json,gzip
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint,OptimizeResult
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from engineering_repair import repair_context
from joint_prototype.completion_continuity import NumericalAdmission,canonical_integral_solution
from joint_prototype import adaptive_service

class CapacityCoverTests(unittest.TestCase):
 def test_backend_receives_valid_capacity_clique_cut(self):
  cols=[SimpleNamespace(job_id='a',column_id='a'),SimpleNamespace(job_id='b',column_id='b')]
  a=NumericalAdmission(300,250,60);a.numerical_solver_retries=[];a._repair_scope=True;a._repair_job_order=['a','b']
  rows=[LinearConstraint(sparse.csc_matrix([[1.,0.],[0.,1.],[1.,1.],[7.,8.],[10.,10.]]),[0.,0.,-np.inf,-np.inf,-np.inf],[1.,1.,2.,12.,100.])]
  def backend(**kw):
   c=kw['constraints'][0];self.assertGreater(c.A.shape[0],5,'Fractional capacity relaxation must be strengthened')
   self.assertTrue(any(np.array_equal(x,[1.,1.]) and u==1. for x,u in zip(c.A.toarray()[5:],np.asarray(c.ub)[5:])))
   return OptimizeResult(x=np.array([1.,0.]),status=0,success=True,message='global optimum')
  with repair_context(None),patch.object(adaptive_service,'milp',side_effect=backend):r=a._milp(np.array([0.,1.]),cols,rows)
  canonical_integral_solution(r.x,rows)
 def test_all_original_binary_points_survive_cuts_with_capacity_tolerance(self):
  from capacity_cover import strengthen
  cols=[SimpleNamespace(job_id=j,column_id=j) for j in ('a','b','c','d')]
  for cap in (12.,12.-5e-8):
   rows=[LinearConstraint(sparse.csc_matrix(np.vstack([np.eye(4),[1.,1.,1.,1.],[7.,8.,5.,6.],[10.,50.,60.,20.]])),[0.]*4+[-np.inf]*3,[1.]*4+[4.,cap,100.])]
   result=strengthen(SimpleNamespace(_repair_scope=True,_repair_job_order=['a','b','c','d']),cols,rows)
   for x in itertools.product([0.,1.],repeat=4):
    try:canonical_integral_solution(x,rows)
    except ValueError:continue
    canonical_integral_solution(x,result)

if __name__=='__main__':unittest.main()
