import contextlib,gzip,io,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.optimize import LinearConstraint,OptimizeResult
from scipy import sparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from engineering_repair import repair_context
from joint_prototype.bundle_adaptation import BundleAdmission
from joint_prototype.adaptive_service import AdaptiveAdmission
from joint_prototype.completion_continuity import NumericalAdmission
from joint_prototype import adaptive_service

class EngineeringNumericTests(unittest.TestCase):
 def fixture(self):
  path=Path(__file__).parent/'fixtures/core80_full_step5.json.gz'
  def hook(x):return float(x['nonfinite']) if set(x)=={'nonfinite'} else x
  d=json.load(gzip.open(path,'rt',encoding='utf8'),object_hook=hook);cs=[]
  for q in d['model']['constraints']:
   a=q['A'];a=sparse.coo_matrix((a['data'],(a['row'],a['col'])),shape=a['shape']).tocsc() if isinstance(a,dict) else np.atleast_2d(a)
   cs.append(LinearConstraint(a,q['lb'],q['ub']))
  return d,cs,[SimpleNamespace(job_id=x.get('job_id',x.get('job')),column_id=x['column_id']) for x in d['model']['columns']]
 def controller(self):
  a=NumericalAdmission(300,250,60);a.numerical_solver_retries=[];a._diagnostic_job_order=[];return a
 def test_captured_optimal_near_binary_validated_before_retry(self):
  d,cs,columns=self.fixture();first=OptimizeResult(x=np.asarray(d['original_x']),status=0,success=True,message='captured optimal')
  retry=OptimizeResult(x=np.asarray(d['retry_x']),status=1,success=False,message='captured timeout')
  with repair_context(None),patch.object(adaptive_service,'milp',side_effect=[first,retry]) as call:
   answer=self.controller()._milp(np.asarray(d['model']['c']),columns,cs)
  self.assertEqual(call.call_count,1);np.testing.assert_array_equal(answer.x,np.rint(first.x))
 def test_narrow_band_false_infeasible_rechecked_without_relaxation(self):
  cs=[LinearConstraint(np.array([[1.,1.]]),1.,1.),LinearConstraint(np.array([[.3,.4]]),.3-1e-9,.3+1e-9)]
  no=OptimizeResult(x=None,status=2,success=False,message='numerical infeasible')
  yes=OptimizeResult(x=np.array([1.,0.]),status=0,success=True,message='conditioned optimal')
  with repair_context(None),patch.object(adaptive_service,'milp',side_effect=[no,yes]) as call:
   ans=self.controller()._milp(np.array([0.,1.]),[SimpleNamespace(job_id='a',column_id='a'),SimpleNamespace(job_id='b',column_id='b')],cs)
  self.assertEqual(ans.status,0);np.testing.assert_array_equal(ans.x,[1,0]);self.assertEqual(call.call_count,2)
  self.assertEqual(call.call_args.kwargs['options'],{'time_limit':30.,'mip_rel_gap':0.,'presolve':False})
 def test_invalid_rounded_capacity_and_primary_band_never_accepted(self):
  c=[LinearConstraint([[1.,1.]],1.,1.),LinearConstraint([[.3,.4]],.3-1e-9,.3+1e-9)]
  bad=OptimizeResult(x=np.array([0.,1.]),status=0,success=True,message='invalid primary')
  with repair_context(None),patch.object(adaptive_service,'milp',return_value=bad):
   with self.assertRaises(RuntimeError):self.controller()._milp(np.array([0.,1.]),[SimpleNamespace(job_id='a',column_id='a'),SimpleNamespace(job_id='b',column_id='b')],c)
 def test_timeout_incumbent_is_not_promoted_to_optimal(self):
  bad=OptimizeResult(x=np.array([1.]),status=1,success=False,message='timeout')
  with repair_context(None),patch.object(adaptive_service,'milp',return_value=bad):
   r=self.controller()._milp(np.array([1.]),[SimpleNamespace(job_id='a',column_id='a')],[LinearConstraint([[1.]],1.,1.)])
  self.assertEqual(r.status,1);self.assertFalse(r.success)
 def test_scoped_timeout_one_fixed_retry_requires_new_optimal_result(self):
  a=self.controller();a._repair_scope=True;col=[SimpleNamespace(job_id='a',column_id='a')]
  no=OptimizeResult(x=np.array([1.]),status=1,success=False,message='timeout incumbent')
  yes=OptimizeResult(x=np.array([1.]),status=0,success=True,message='independently solved retry optimal')
  with repair_context(None),patch.object(adaptive_service,'milp',side_effect=[no,yes]) as call:
   r=a._milp(np.array([1.]),col,[LinearConstraint([[1.]],1.,1.)])
  self.assertEqual(r.status,0);self.assertEqual(call.call_count,2)
  self.assertEqual(call.call_args.kwargs['options'],{'time_limit':30.,'mip_rel_gap':0.,'presolve':False})
 def test_scoped_repeated_timeout_stops_with_model_not_incumbent_commit(self):
  a=self.controller();a._repair_scope=True;no=OptimizeResult(x=np.array([1.]),status=1,success=False,message='timeout')
  with repair_context(None),patch.object(adaptive_service,'milp',return_value=no) as call:
   with self.assertRaises(RuntimeError) as caught:a._milp(np.array([1.]),[SimpleNamespace(job_id='a',column_id='a')],[LinearConstraint([[1.]],1.,1.)])
  self.assertEqual(call.call_count,2);self.assertEqual(caught.exception.solver_diagnostics['retry_status'],1)
 def test_required_job_positive_cost_lower_bound_is_exact_certificate(self):
  a=self.controller();col=[SimpleNamespace(job_id='head',column_id='cheap'),SimpleNamespace(job_id='head',column_id='costly')]
  a._repair_scope=True;a._repair_job_order=['head'];a._repair_last_columns=col;a._repair_last_x=np.array([1.,0.])
  with repair_context(None),patch.object(adaptive_service,'milp',side_effect=RuntimeError('RED: unnecessary backend call')) as call:
   r=a._milp(np.array([120.,125.]),col,[LinearConstraint([[1.,1.]],1.,1.)])
  self.assertEqual(call.call_count,0);self.assertEqual(r.fun,120.);np.testing.assert_array_equal(r.x,[1.,0.])
 def test_capacity_coupled_integer_cost_dual_certificate_skips_mip(self):
  a=self.controller();col=[SimpleNamespace(job_id=j,column_id=str(i)) for i,j in enumerate(['a','a','b','b'])]
  a._repair_scope=True;a._repair_job_order=['a','b'];a._repair_last_columns=col;a._repair_last_x=np.array([1.,0.,0.,1.])
  rows=[LinearConstraint([[1.,1.,0.,0.],[0.,0.,1.,1.],[1.,0.,1.,0.]], [1.,1.,-np.inf],[1.,1.,1.])]
  with repair_context(None),patch.object(adaptive_service,'milp',side_effect=RuntimeError('RED: unnecessary integer solve')) as call:
   r=a._milp(np.array([0.,10.,0.,10.]),col,rows)
  self.assertEqual(call.call_count,0);self.assertEqual(r.fun,10.);self.assertIn('DUAL',r.message)
 def test_lp_reduced_cost_discards_only_objective_worse_variables(self):
  a=self.controller();col=[SimpleNamespace(job_id='a',column_id=str(i)) for i in range(4)]
  a._repair_scope=True;a._repair_job_order=['a'];a._repair_last_columns=col;a._repair_last_x=np.array([0.,1.,0.,0.])
  def backend(**args):
   self.assertEqual(len(args['c']),2)
   return OptimizeResult(x=np.array([1.,0.]),status=0,success=True,message='optimal reduced exact model')
  with repair_context(None),patch.object(adaptive_service,'milp',side_effect=backend):
   r=a._milp(np.array([0.,10.,20.,30.]),col,[LinearConstraint([[1.,1.,1.,1.]],1.,1.)])
  np.testing.assert_array_equal(r.x,[1.,0.,0.,0.])
 def test_lp_support_integer_candidate_is_only_used_with_global_proof(self):
  a=self.controller();col=[SimpleNamespace(job_id='a',column_id=str(i)) for i in range(3)]
  a._repair_scope=True;a._repair_job_order=['a'];a._repair_last_columns=col;a._repair_last_x=np.array([0.,0.,1.])
  def backend(**args):
   self.assertEqual(len(args['c']),2)
   return OptimizeResult(x=np.array([1.,0.]),status=0,success=True,message='optimal restricted model, not yet globally proved')
  with repair_context(None),patch.object(adaptive_service,'milp',side_effect=backend) as call:
   r=a._milp(np.array([0.,10.,20.]),col,[LinearConstraint([[1.,1.,1.]],1.,1.)])
  self.assertEqual(call.call_count,1);self.assertEqual(r.fun,0.);self.assertIn('CERTIFIED',r.message);np.testing.assert_array_equal(r.x,[1.,0.,0.])
 def test_restricted_infeasibility_or_invalid_vector_does_not_fail_full_model(self):
  for hint in (OptimizeResult(x=None,status=2,success=False,message='restricted infeasible'),OptimizeResult(x=np.array([1.,1.]),status=0,success=True,message='invalid restricted vector')):
   a=self.controller();col=[SimpleNamespace(job_id='a',column_id=str(i)) for i in range(3)]
   a._repair_scope=True;a._repair_job_order=['a'];a._repair_last_columns=col;a._repair_last_x=np.array([0.,0.,1.])
   good=OptimizeResult(x=np.array([1.,0.,0.]),status=0,success=True,message='full verified optimum')
   with repair_context(None),patch.object(adaptive_service,'milp',side_effect=[hint,good]) as call:
    r=a._milp(np.array([0.,10.,20.]),col,[LinearConstraint([[1.,1.,1.]],1.,1.)])
   self.assertEqual(call.call_count,2);self.assertEqual(r.fun,0.)
 def test_restricted_optimum_is_not_promoted_without_global_bound(self):
  import dual_certificate
  a=self.controller();col=[SimpleNamespace(job_id='a',column_id=str(i)) for i in range(3)]
  a._repair_scope=True;a._repair_job_order=['a'];a._repair_last_columns=col;a._repair_last_x=np.array([0.,0.,1.]);cost=np.array([0.,10.,20.]);rows=[LinearConstraint([[1.,1.,1.]],1.,1.)]
  evidence=dual_certificate.certificate(cost,rows,a._repair_last_x);evidence['_lp_support']=[1]
  def backend(**args):return OptimizeResult(x=np.array([1.,0.]),status=0,success=True,message='restricted-or-full optimum')
  with repair_context(None),patch.object(dual_certificate,'certificate',return_value=evidence),patch.object(adaptive_service,'milp',side_effect=backend) as call:
   r=a._milp(cost,col,rows)
  self.assertEqual(call.call_count,2);self.assertEqual(r.fun,0.);np.testing.assert_array_equal(r.x,[1.,0.,0.])
 def test_explicit_safety_backend_must_independently_prove_optimality(self):
  a=self.controller();a._repair_scope=True;a._repair_allow_cbc=True
  no=OptimizeResult(x=None,status=1,success=False,message='native timeout without incumbent')
  with repair_context(None),patch.object(adaptive_service,'milp',return_value=no):
   r=a._milp(np.array([1.]),[SimpleNamespace(job_id='a',column_id='a')],[LinearConstraint([[1.]],1.,1.)])
  self.assertEqual(r.status,0);self.assertIn('CBC',r.message);np.testing.assert_array_equal(r.x,[1.])
 def test_safety_backend_timeout_is_not_a_global_optimum(self):
  import existing_cbc_retry
  a=self.controller();a._repair_scope=True;a._repair_allow_cbc=True;no=OptimizeResult(x=np.array([1.]),status=1,success=False,message='timeout incumbent')
  with repair_context(None),patch.object(adaptive_service,'milp',return_value=no),patch.object(existing_cbc_retry,'solve',return_value=no):
   with self.assertRaises(RuntimeError) as c:a._milp(np.array([1.]),[SimpleNamespace(job_id='a',column_id='a')],[LinearConstraint([[1.]],1.,1.)])
  self.assertEqual(c.exception.solver_diagnostics['independent_cbc']['status'],1)
 def test_safety_backend_invalid_optimal_vector_is_rejected(self):
  import existing_cbc_retry
  a=self.controller();a._repair_scope=True;a._repair_allow_cbc=True;no=OptimizeResult(x=None,status=1,success=False,message='timeout');bad=OptimizeResult(x=np.array([0.]),status=0,success=True,message='invalid alleged optimal')
  with repair_context(None),patch.object(adaptive_service,'milp',return_value=no),patch.object(existing_cbc_retry,'solve',return_value=bad):
   with self.assertRaises(RuntimeError) as c:a._milp(np.array([1.]),[SimpleNamespace(job_id='a',column_id='a')],[LinearConstraint([[1.]],1.,1.)])
  self.assertEqual(c.exception.solver_diagnostics['independent_cbc_x'],[0.])

if __name__=='__main__':unittest.main(verbosity=2)
