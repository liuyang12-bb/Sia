"""Later integer lex phases must preserve ALL earlier fixed constraints."""
import sys,unittest,gzip,json,hashlib
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'sia-simulator')]
from epsilon_lex_repair import solve
from joint_prototype.completion_continuity import canonical_integral_solution

class GeneralIntegerLexTests(unittest.TestCase):
 def test_zero_integer_cost_is_a_real_global_lower_bound_not_a_timeout_promotion(self):
  cols=[SimpleNamespace(job_id='a',column_id='a|g:1:1|'+str(i)+'|10|g:0=1') for i in range(3)]
  rows=[LinearConstraint(sparse.csc_matrix([[1,1,1],[1,1,1],[1,1,1],[10,10,10]]),[1,-np.inf,-np.inf,-np.inf],[1,1,3,10]),LinearConstraint([[.7,.7,.4]],.7-1e-9,.7+1e-9)]
  c=np.array([5.,0,0]);x=np.array([1.,0,0]);a=SimpleNamespace(_repair_scope=True,_repair_job_order=['a']);r=solve(a,c,cols,rows,x,None)
  self.assertEqual(r.fun,0.);canonical_integral_solution(r.x,rows);self.assertTrue(a._repair_epsilon_trace[-1]['proved']);self.assertEqual(a._repair_epsilon_trace[-1]['proof'],'EXACT_NONNEGATIVE_INTEGER_LEX_ATTAINED_ZERO')
 def test_rank_preserves_fixed_startup_and_earlier_job_geometry(self):
  cols=[SimpleNamespace(job_id=j,column_id=j+'|g:1:1|'+str(i)+'|10|g:0=1') for i,j in enumerate(['a','a','a','b','b'])]
  rows=[LinearConstraint(sparse.csc_matrix([[1,1,1,0,0],[0,0,0,1,1],[1,1,1,1,1],[1,2,3,1,2],[10,10,10,10,10]]),[1,1,-np.inf,-np.inf,-np.inf],[1,1,2,10,50]),LinearConstraint([[.5,.5,.3,.2,.2]],.7-1e-9,.7+1e-9),LinearConstraint([[25,0,25,25,50]],50-1e-9,50+1e-9),LinearConstraint([[0,0,0,1,0]],1,1)]
  c=np.array([5.,0,0,0,0]);x=np.array([1.,0,0,1,0]);a=SimpleNamespace(_repair_scope=True,_repair_job_order=['a','b']);r=solve(a,c,cols,rows,x,None)
  self.assertIsNotNone(r);self.assertEqual(r.fun,5.);canonical_integral_solution(r.x,rows);self.assertTrue(a._repair_epsilon_trace[-1]['proved'])
 def test_cutoff_is_directed_downward_at_long_binary_fraction_boundary(self):
  from epsilon_lex_repair import lower_primary_cutoff
  from primary_band_presolve import dot_error
  p=np.array([.12345678912345678,.2345678912345679,.3456789123456789]);L=.7037035927037036
  actual=lower_primary_cutoff(L,p,3);exact=Fraction.from_float(L)-Fraction.from_float(1e-10)-Fraction(dot_error(p,3))
  self.assertLessEqual(Fraction(actual),exact)

class CapturedRank42(unittest.TestCase):
 def test_real_failure42_later_rank_has_proof_without_dropping_earlier_masks(self):
  from primary_band_presolve import allowed
  path=Path(__file__).parent/'fixtures/core80_full_rank_step42.json.gz';before=hashlib.sha256(path.read_bytes()).hexdigest();d=json.loads(gzip.decompress(path.read_bytes()));rows=[]
  for r in d['model']['constraints']:
   q=r['A'];A=sparse.coo_matrix((q['data'],(q['row'],q['col'])),shape=q['shape']).tocsc() if isinstance(q,dict) else np.atleast_2d(q);rows.append(LinearConstraint(A,r['lb'],r['ub']))
  cols=[SimpleNamespace(**q) for q in d['model']['columns']];B=rows[0].A.tocsr();nj=B.shape[0]-9;names=[cols[B.indices[B.indptr[i]]].job_id for i in range(nj)];a=SimpleNamespace(_repair_scope=True,_repair_job_order=names);c=np.asarray(d['model']['c']);x=np.asarray(d['last_valid_x']);keep=allowed(a,cols,rows,x)
  result=solve(a,c,cols,rows,x,keep);self.assertIsNotNone(result);canonical_integral_solution(result.x,rows);self.assertLessEqual(result.fun,8186.);self.assertTrue(a._repair_epsilon_trace[-1]['proved']);self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())
  print('RANK42_CERTIFICATE='+json.dumps(dict(objective=result.fun,trace=a._repair_epsilon_trace),ensure_ascii=False),flush=True)

if __name__=='__main__':unittest.main()
