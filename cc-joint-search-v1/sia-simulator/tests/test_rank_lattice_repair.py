"""Original-row and backend-contract regressions for exact lattice rows."""
import sys,unittest,itertools,gzip,json
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

class RankLatticeBackendTests(unittest.TestCase):
 def test_integer_startup_band_is_exact_before_backend(self):
  a=NumericalAdmission(300,250,60);a.numerical_solver_retries=[];a._repair_scope=True
  rows=[LinearConstraint(sparse.csc_matrix([[1.,1.]]),1.,1.),LinearConstraint(sparse.csc_matrix([[25.,50.]]),25.-1e-9,25.+1e-9)]
  before=[(np.array(c.lb,copy=True),np.array(c.ub,copy=True)) for c in rows]
  def solve(**kw):
   r=kw['constraints'][1]
   self.assertEqual(float(np.asarray(r.lb).item()),25.)
   self.assertEqual(float(np.asarray(r.ub).item()),25.)
   self.assertEqual(kw['options'],{'time_limit':30.,'mip_rel_gap':0.})
   return OptimizeResult(x=np.array([1.,0.]),status=0,success=True,message='proved optimum')
  with repair_context(None),patch.object(adaptive_service,'milp',side_effect=solve) as call:
   ans=a._milp(np.array([0.,1.]),[SimpleNamespace(job_id='a',column_id='0'),SimpleNamespace(job_id='a',column_id='1')],rows)
  self.assertEqual(call.call_count,1);canonical_integral_solution(ans.x,rows)
  for c,(lo,hi) in zip(rows,before):np.testing.assert_array_equal(c.lb,lo);np.testing.assert_array_equal(c.ub,hi)

class LatticeRepresentationTests(unittest.TestCase):
 def test_dense_and_sparse_exhaustive_binary_domains_identical(self):
  from integer_row_lattice import tighten_integer_rows
  for A,lo,hi in [([[25.,50.,120.]],74.999999999,75.000000001),([[-10.,15.,25.]],-5.,31.),([[.25,.5,.75]],.49,.76)]:
   for mat in (np.array(A),sparse.csc_matrix(A)):
    original=[LinearConstraint(mat,lo,hi)];rows,_=tighten_integer_rows(original)
    for v in itertools.product([0.,1.],repeat=3):
     def accepts(c):
      try:canonical_integral_solution(v,c);return True
      except ValueError:return False
     self.assertEqual(accepts(original),accepts(rows))
 def test_fractional_rows_are_not_rounded(self):
  from integer_row_lattice import tighten_integer_rows
  c=LinearConstraint([[.25,.5]],.25-1e-9,.25+1e-9);r,p=tighten_integer_rows([c]);self.assertIs(r[0],c);self.assertEqual(p,[])
 def test_tolerance_boundary_conflict_is_left_unchanged(self):
  from integer_row_lattice import tighten_integer_rows
  c=LinearConstraint([[1.,1.]],1.+5e-11,2.);r,p=tighten_integer_rows([c]);self.assertIs(r[0],c);self.assertEqual(p,[])
 def test_oversized_integer_sum_is_not_transformed(self):
  from integer_row_lattice import tighten_integer_rows
  c=LinearConstraint([[2.**52,2.**52]],0.,1e20);r,p=tighten_integer_rows([c]);self.assertIs(r[0],c);self.assertEqual(p,[])
 def test_noncanonical_fractional_duplicates_are_not_reinterpreted(self):
  from integer_row_lattice import tighten_integer_rows
  from primary_band_presolve import primary_certificate
  A=sparse.csr_matrix((np.array([.1,.9]),np.array([0,0]),np.array([0,2])),shape=(1,2))
  self.assertFalse(A.has_canonical_format)
  c=LinearConstraint(A,1.-1e-9,1.+1e-9);r,p=tighten_integer_rows([c]);self.assertIs(r[0],c);self.assertEqual(p,[])
  self.assertIsNone(primary_certificate([LinearConstraint([[1.,1.]],1.,1.),c,LinearConstraint([[25.,25.]],25.,25.)],2))

class PrimaryBandPresolveTests(unittest.TestCase):
 def model(self):
  return [LinearConstraint(sparse.csc_matrix([[1.,1.,0.,0.],[0.,0.,1.,1.]]),[1.,1.],[1.,1.]),LinearConstraint([[.5,.3,.2,.1]],.7-1e-9,.7+1e-9),LinearConstraint([[25.,25.,50.,50.]],75.-1e-9,75.+1e-9)]
 def test_primary_band_alone_prunes_before_startup_is_optimized(self):
  from primary_band_presolve import allowed
  for halfband in (1e-9,0.):
   rows=self.model()[:2];rows[1]=LinearConstraint([[.5,.3,.2,.1]],.7-halfband,.7+halfband)
   keep=allowed(SimpleNamespace(_repair_scope=True),[None]*4,rows,np.array([1.,0.,1.,0.]))
   self.assertIsNotNone(keep);np.testing.assert_array_equal(keep,[0,2])
 def test_current_cost_certificate_uses_certified_feasible_domain_first(self):
  import dual_certificate
  original=dual_certificate.certificate
  def prove(cost,rows,x):
   self.assertEqual(len(cost),2,'Impossible columns must be removed before current-cost LP')
   return original(cost,rows,x)
  a=NumericalAdmission(300,250,60);a.numerical_solver_retries=[];a._repair_scope=True;a._repair_job_order=['a','b']
  cols=[SimpleNamespace(job_id=j,column_id=str(i)) for i,j in enumerate(['a','a','b','b'])];a._repair_last_columns=cols;a._repair_last_x=np.array([1.,0.,1.,0.])
  with repair_context(None),patch('dual_certificate.certificate',side_effect=prove):ans=a._milp(np.array([9.,5.,4.,0.]),cols,self.model())
  self.assertEqual(ans.fun,13.);canonical_integral_solution(ans.x,self.model())
 def test_noncontiguous_domain_lifts_support_residuals_and_lattice(self):
  from primary_band_presolve import certificate_on_domain
  from dual_certificate import refresh_incumbent
  cost=np.array([5.,1.,10.,2.]);x=np.array([1.,0.,1.,0.]);keep=np.array([0,2]);p=certificate_on_domain(cost,self.model(),x,keep)
  self.assertTrue(p['proved']);self.assertEqual(p['gcd'],5);self.assertEqual(len(p['_residuals']),4);self.assertTrue(set(p['_lp_support'])<=set(keep));self.assertTrue(set(p['fix_zero'])<=set(keep));self.assertTrue(refresh_incumbent(p,cost,x)['proved'])
  with self.assertRaisesRegex(ValueError,'OUTSIDE_CERTIFIED_DOMAIN'):certificate_on_domain(cost,self.model(),np.array([0.,1.,1.,0.]),keep)
 def test_exhaustive_keeps_every_original_feasible_binary_point(self):
  from primary_band_presolve import primary_certificate
  rows=self.model();keep,proof=primary_certificate(rows,4);self.assertGreater(proof['removed'],0)
  for v in itertools.product([0.,1.],repeat=4):
   try:canonical_integral_solution(v,rows)
   except ValueError:continue
   self.assertTrue(set(np.flatnonzero(v))<=set(keep))
 def test_future_geometry_masks_reuse_only_same_prefix(self):
  from primary_band_presolve import allowed
  a=SimpleNamespace(_repair_scope=True);cols=[None]*4;rows=self.model();keep=allowed(a,cols,rows)
  allowed(a,cols,rows+[LinearConstraint([[1.,0.,0.,0.]],1.,1.)]);self.assertEqual(a._repair_primary_band_lp_calls,1)
  rows[1].lb=np.array([.4-1e-9]);rows[1].ub=np.array([.4+1e-9]);allowed(a,cols,rows);self.assertEqual(a._repair_primary_band_lp_calls,2)
 def test_scope_and_missing_primary_do_not_prune(self):
  from primary_band_presolve import allowed
  self.assertIsNone(allowed(SimpleNamespace(_repair_scope=False),[None]*4,self.model()))
  self.assertIsNone(allowed(SimpleNamespace(_repair_scope=True),[None]*4,self.model()[:1]))
 def test_corrupt_elimination_cannot_discard_valid_incumbent(self):
  from primary_band_presolve import allowed
  a=SimpleNamespace(_repair_scope=True);record=dict(seconds=0.,removed=3)
  with patch('primary_band_presolve.primary_certificate',return_value=(np.array([1]),record)):
   with self.assertRaisesRegex(RuntimeError,'CONTRADICTS_CURRENT_VALID_INCUMBENT'):
    allowed(a,[None]*4,self.model(),np.array([1.,0.,1.,0.]))
 def test_mixed_sign_primary_preserves_all_original_points(self):
  from primary_band_presolve import primary_certificate
  rows=self.model();rows[1]=LinearConstraint([[-.5,.3,.2,-.1]],-.3-1e-9,-.3+1e-9);keep,_=primary_certificate(rows,4)
  found=0
  for v in itertools.product([0.,1.],repeat=4):
   try:canonical_integral_solution(v,rows)
   except ValueError:continue
   found+=1;self.assertTrue(set(np.flatnonzero(v))<=set(keep))
  self.assertGreater(found,0)
 def test_lp_failure_keeps_all_variables(self):
  from primary_band_presolve import primary_certificate
  bad=OptimizeResult(status=1,message='injected timeout',ineqlin=None)
  with patch('primary_band_presolve.linprog',return_value=bad):keep,p=primary_certificate(self.model(),4)
  np.testing.assert_array_equal(keep,np.arange(4));self.assertEqual(p['removed'],0)
 def test_roundoff_bounds_dominate_small_exact_binary_sums(self):
  from primary_band_presolve import dot_error
  from decimal import Decimal
  a=np.array([.1,.2,.3,.4]);error=dot_error(a,len(a))
  for x in itertools.product([0.,1.],repeat=4):
   exact=sum((Decimal.from_float(float(v)) for v,s in zip(a,x) if s),Decimal(0));actual=Decimal.from_float(float(a@x));self.assertLessEqual(abs(actual-exact),error)

class CapturedRankRegressionTests(unittest.TestCase):
 def test_real_step37_startup_optimum_with_only_primary_fixed(self):
  src=Path(__file__).parent/'fixtures/core80_full_startup_step37.json.gz';d=json.loads(gzip.decompress(src.read_bytes()));rows=[]
  for r in d['model']['constraints']:
   v=r['A'];A=sparse.coo_matrix((v['data'],(v['row'],v['col'])),shape=v['shape']).tocsc() if isinstance(v,dict) else np.atleast_2d(v);rows.append(LinearConstraint(A,r['lb'],r['ub']))
  self.assertEqual(len(rows),2)
  cols=[SimpleNamespace(**q) for q in d['model']['columns']];a=NumericalAdmission(300,250,60);a.numerical_solver_retries=[];a._repair_scope=True;a._repair_allow_cbc=True;a._repair_last_columns=cols;a._repair_last_x=np.rint(d['last_valid_x'])
  B=rows[0].A.tocsr();a._repair_job_order=[cols[B.indices[B.indptr[i]]].job_id for i in range(B.shape[0]-9)];cost=np.asarray(d['model']['c'])
  with repair_context(None):r=a._milp(cost,cols,rows)
  self.assertEqual(r.status,0);self.assertTrue(r.success);self.assertEqual(float(cost@canonical_integral_solution(r.x,rows)),250.)
  self.assertGreater(a._repair_primary_band_variables_removed,150000)
 def test_real_step19_model_global_rank_optimum_with_original_rows(self):
  src=Path(__file__).parent/'fixtures/core80_full_rank_step19.json.gz';d=json.loads(gzip.decompress(src.read_bytes()));rows=[]
  for r in d['model']['constraints']:
   v=r['A'];A=sparse.coo_matrix((v['data'],(v['row'],v['col'])),shape=v['shape']).tocsc() if isinstance(v,dict) else np.atleast_2d(v);rows.append(LinearConstraint(A,r['lb'],r['ub']))
  cols=[SimpleNamespace(**q) for q in d['model']['columns']];a=NumericalAdmission(300,250,60);a.numerical_solver_retries=[];a._repair_scope=True;a._repair_allow_cbc=True;a._repair_last_columns=cols;a._repair_last_x=np.rint(d['last_valid_x'])
  B=rows[0].A.tocsr();a._repair_job_order=[cols[B.indices[B.indptr[i]]].job_id for i in range(B.shape[0]-9)]
  cost=np.array(d['model']['c'])
  with repair_context(None):r=a._milp(cost,cols,rows)
  self.assertEqual(r.status,0);self.assertTrue(r.success);self.assertEqual(float(cost@canonical_integral_solution(r.x,rows)),5.)
  self.assertGreater(a._repair_primary_band_variables_removed,100000)

if __name__=='__main__':unittest.main()
