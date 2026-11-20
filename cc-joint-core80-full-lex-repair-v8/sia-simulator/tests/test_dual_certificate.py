import itertools,sys,unittest
from decimal import Decimal
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dual_certificate import lower_bound,certificate

class DualProofTests(unittest.TestCase):
 def test_arbitrary_dual_multipliers_never_overstate_exhaustive_optimum(self):
  c=np.array([2.,3.,6.]);a=sparse.csc_matrix([[1.,1.,1.],[-1.,-1.,-1.],[1.,2.,0.]])
  b=np.array([1.,-1.,1.]);opt=min(float(c@x) for bits in itertools.product((0.,1.),repeat=3) if np.all(a@(x:=np.asarray(bits))<=b))
  for y in ([-1.,-2.,-3.],[1e20,-1e20,0.],[100.,-100.,-1e-30],[-.123,-.456,-.789]):
   self.assertLessEqual(lower_bound(c,a,b,np.asarray(y)),Decimal.from_float(opt))
 def test_suboptimal_incumbent_is_not_certified(self):
  r=certificate(np.array([2.,6.]),[LinearConstraint([[1.,1.]],1.,1.)],np.array([0.,1.]))
  self.assertFalse(r['proved']);self.assertGreater(float(r['objective']),float(r['lower_bound']))
 def test_reduced_cost_keeps_every_competitive_exhaustive_solution(self):
  a=np.array([[1.,1.,0.,0.],[0.,0.,1.,1.],[1.,0.,1.,0.]]);lo=np.array([1.,1.,-np.inf]);hi=np.array([1.,1.,1.]);x0=np.array([1.,0.,0.,1.])
  for cost in (np.array([0.,1.,0.,10.]),np.array([-.3,-.1,-.4,.8]),np.array([.25,3.5,.5,1.5])):
   r=certificate(cost,[LinearConstraint(a,lo,hi)],x0);removed=r.get('fix_zero',[])
   for bits in itertools.product((0.,1.),repeat=4):
    x=np.asarray(bits)
    if np.all(a@x>=lo) and np.all(a@x<=hi) and cost@x<=cost@x0:
     self.assertTrue(all(x[i]==0. for i in removed))

if __name__=='__main__':unittest.main(verbosity=2)
