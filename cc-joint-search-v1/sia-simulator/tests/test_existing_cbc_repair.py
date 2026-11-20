import gzip,json,sys,unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from existing_cbc_retry import solve
from scale_storage import sparse_condition
from joint_prototype.completion_continuity import canonical_integral_solution

class CapturedCBCRegression(unittest.TestCase):
 def test_captured_full_step10_original_rows_and_startup_optimum(self):
  f=Path(__file__).parent/'fixtures/core80_failed_startup_reduced.json.gz'
  with gzip.open(f,'rt',encoding='utf8') as stream:d=json.load(stream)
  rows=[]
  for c in d['constraints']:
   a=c['A'];a=sparse.coo_matrix((a['data'],(a['row'],a['col'])),shape=a['shape']).tocsc();rows.append(LinearConstraint(a,c['lb'],c['ub']))
  conditioned,_=sparse_condition(rows);result=solve(d['cost'],conditioned)
  self.assertEqual(result.status,0);self.assertTrue(result.success)
  x=canonical_integral_solution(result.x,rows)
  self.assertEqual(float(np.dot(d['cost'],x)),d['expected_startup'])
  self.assertEqual(d['original_variables'],121133)

if __name__=='__main__':unittest.main(verbosity=2)
