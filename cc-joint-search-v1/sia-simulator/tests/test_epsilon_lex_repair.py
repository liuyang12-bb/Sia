"""Exact secondary proof must not require a hairline primary equality MIP."""
import sys,unittest,itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint,OptimizeResult
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'sia-simulator')]
from joint_prototype.completion_continuity import NumericalAdmission,canonical_integral_solution
from engineering_repair import repair_context

class EpsilonBackendRegression(unittest.TestCase):
 def test_startup_uses_equivalent_budget_proof_before_hairline_native(self):
  a=NumericalAdmission(300,250,60);a.numerical_solver_retries=[];a._repair_scope=True;a._repair_job_order=['a']
  cols=[SimpleNamespace(job_id='a',column_id='a|g:1:1|3|10|g:0=1'),SimpleNamespace(job_id='a',column_id='a|g:1:2|3|10|g:0=2')]
  rows=[LinearConstraint(sparse.csc_matrix([[1.,1.],[1.,2.],[3.,6.],[10.,20.]]),[1.,-np.inf,-np.inf,-np.inf],[1.,2.,6.,20.]),LinearConstraint(sparse.csc_matrix([[.5,.4]]),.5-1e-9,.5+1e-9)]
  a._repair_last_columns=cols;a._repair_last_x=np.array([1.,0.])
  with repair_context(None),patch('primary_band_presolve.certificate_on_domain',return_value=None),patch('epsilon_lex_repair.solve',return_value=OptimizeResult(x=np.array([1.,0.]),status=0,success=True,fun=5.,message='EXACT_STARTUP_BUDGET_CERTIFICATE')) as proof:
   result=a._milp(np.array([5.,0.]),cols,rows)
  proof.assert_called_once();self.assertEqual(result.fun,5.);canonical_integral_solution(result.x,rows)

class EpsilonProofTests(unittest.TestCase):
 def model(self):
  cols=[SimpleNamespace(job_id='a',column_id='a|g:1:1|'+str(i)+'|10|g:0=1') for i in range(3)]
  rows=[LinearConstraint(sparse.csc_matrix([[1.,1.,1.],[1.,1.,1.],[1.,2.,3.],[10.,10.,10.]]),[1.,-np.inf,-np.inf,-np.inf],[1.,1.,3.,10.]),LinearConstraint(sparse.csc_matrix([[.7,.7,.4]]),.7-1e-9,.7+1e-9)]
  return cols,rows,np.array([10.,5.,0.]),np.array([1.,0.,0.])
 def test_monotone_witness_then_global_bound_matches_exhaustive(self):
  from epsilon_lex_repair import solve
  cols,rows,c,x=self.model();a=SimpleNamespace(_repair_scope=True,_repair_job_order=['a'])
  r=solve(a,c,cols,rows,x,None);self.assertIsNotNone(r);self.assertEqual(r.fun,5.)
  feasible=[]
  for v in itertools.product([0.,1.],repeat=3):
   try:canonical_integral_solution(v,rows);feasible.append(c@v)
   except ValueError:pass
  self.assertEqual(r.fun,min(feasible));self.assertEqual([t['budget'] for t in a._repair_epsilon_trace],[5,0]);self.assertTrue(a._repair_epsilon_trace[-1]['proved'])
 def test_timeout_without_improving_witness_cannot_prove_optimal(self):
  from epsilon_lex_repair import solve
  cols,rows,c,x=self.model();a=SimpleNamespace(_repair_scope=True,_repair_job_order=['a'])
  with patch('joint_prototype.adaptive_service.milp',return_value=OptimizeResult(status=1,success=False,x=None,message='timeout',mip_dual_bound=0.)):
   self.assertIsNone(solve(a,c,cols,rows,x,None))
  self.assertFalse(a._repair_epsilon_trace[-1]['proved'])
 def test_fractional_fake_optimum_cannot_be_used_as_proof(self):
  from epsilon_lex_repair import solve
  cols,rows,c,x=self.model();a=SimpleNamespace(_repair_scope=True,_repair_job_order=['a'])
  def bad(**kw):return OptimizeResult(status=0,success=True,x=np.full(len(kw['c']),.5),message='fault',mip_dual_bound=0.)
  with patch('joint_prototype.adaptive_service.milp',side_effect=bad):self.assertIsNone(solve(a,c,cols,rows,x,None))
  self.assertTrue(a._repair_epsilon_trace[-1]['invalid_query_point'])
 def test_scope_and_nonintegral_cost_are_not_reinterpreted(self):
  from epsilon_lex_repair import solve
  cols,rows,c,x=self.model()
  self.assertIsNone(solve(SimpleNamespace(_repair_scope=False),c,cols,rows,x,None));self.assertIsNone(solve(SimpleNamespace(_repair_scope=True),c+.1,cols,rows,x,None))
 def test_lost_or_invalid_current_incumbent_is_not_accepted(self):
  from epsilon_lex_repair import solve
  cols,rows,c,x=self.model();a=SimpleNamespace(_repair_scope=True,_repair_job_order=['a'])
  with self.assertRaisesRegex(RuntimeError,'LOST_VALID_INCUMBENT'):solve(a,c,cols,rows,x,np.array([1,2]))
  with self.assertRaises(ValueError):solve(a,c,cols,rows,np.zeros(3),None)
 def test_outer_rows_keep_all_original_accepted_binary_points(self):
  from epsilon_lex_repair import outer_rows
  rows=[LinearConstraint([[1.,1.,0.],[.1,.2,.3]],[0.,.1],[1.,.30000000001])];outer=outer_rows(rows,3)
  for v in itertools.product([0.,1.],repeat=3):
   try:canonical_integral_solution(v,rows)
   except ValueError:continue
   canonical_integral_solution(v,outer)

class ExtendedGroupTests(unittest.TestCase):
 def test_binary_extension_is_bijective_and_preserves_objective(self):
  from gpu_group_formulation import prepare
  cols,rows,c,x=EpsilonProofTests().model();c=np.array([5.,5.,5.]);a=SimpleNamespace(_repair_scope=True,_repair_job_order=['a'])
  ce,re,E=prepare(a,c,cols,rows,True)
  for v in itertools.product([0.,1.],repeat=3):
   try:original=canonical_integral_solution(v,rows)
   except ValueError:continue
   extended=np.concatenate([original,E@original]);canonical_integral_solution(extended,re);self.assertEqual(c@original,ce@extended)

class CapturedStartup42(unittest.TestCase):
 def test_real_failure42_global_startup_225_with_unchanged_solver_options(self):
  import gzip,json,hashlib
  from primary_band_presolve import allowed
  from epsilon_lex_repair import solve
  path=Path(__file__).parent/'fixtures/core80_full_startup_step42.json.gz';before=hashlib.sha256(path.read_bytes()).hexdigest();d=json.loads(gzip.decompress(path.read_bytes()));rows=[]
  for r in d['model']['constraints']:
   q=r['A'];A=sparse.coo_matrix((q['data'],(q['row'],q['col'])),shape=q['shape']).tocsc() if isinstance(q,dict) else np.atleast_2d(q);rows.append(LinearConstraint(A,r['lb'],r['ub']))
  cols=[SimpleNamespace(**q) for q in d['model']['columns']];B=rows[0].A.tocsr();nj=B.shape[0]-9;names=[cols[B.indices[B.indptr[i]]].job_id for i in range(nj)];a=SimpleNamespace(_repair_scope=True,_repair_job_order=names);c=np.asarray(d['model']['c']);x=np.asarray(d['last_valid_x']);keep=allowed(a,cols,rows,x)
  ans=solve(a,c,cols,rows,x,keep);self.assertIsNotNone(ans);self.assertEqual(ans.fun,225.);canonical_integral_solution(ans.x,rows);self.assertTrue(a._repair_epsilon_trace[-1]['proved']);self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

if __name__=='__main__':unittest.main()
