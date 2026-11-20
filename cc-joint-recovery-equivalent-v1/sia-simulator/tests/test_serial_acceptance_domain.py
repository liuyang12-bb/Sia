"""A physical incumbent need not belong to FO's frozen GPU domain."""
import unittest
from unittest.mock import patch
from dataclasses import replace
from test_adaptive_service import spec,cfg,current,columns
from test_clean_ablation import controller,state
from joint_prototype.joint_profile import NodeSpec

class SerialAcceptanceDomainTests(unittest.TestCase):
 def scenario(self,gpu):
  nodes=[NodeSpec(t+':0',t,1,6.,125.) for t in ('aws','dgx-ext')]
  j=spec('a',configs=[cfg('aws')]);x=controller([j],gpu=gpu,aux='opt');s=state([j],nodes);s.enforce_floor=False
  def score(cols):
   return [replace(c,scheduling_utility=(10. if c.config.gpu_type=='aws' and c.query.cpu_per_gpu==3. and c.query.ram_gb_per_gpu==62.5 else 4. if c.config.gpu_type=='dgx-ext' else 1.)) for c in cols]
  first=x.decide(s,score(columns(s)),0.,{})
  self.assertEqual(first.selected['a'].query.cpu_per_gpu,3.)
  j=replace(j,configs=(cfg('aws'),cfg('dgx-ext',rate=2.)))
  s=state([current(j,first.selected)],nodes);s.enforce_floor=False
  x.profile_sources['a']['dgx-ext:1:1']='direct_profile'
  result=x.decide(s,score(columns(s)),180.,{'a':60.})
  return result
 def test_serial_frozen_gpu_domain_can_exclude_better_old_aux_bundle(self):
  r=self.scenario('fixed')
  self.assertEqual(r.selected['a'].config.gpu_type,'dgx-ext')
  self.assertEqual(r.scheduling_objective,4.)
  self.assertTrue(r.audit['valid'])
  self.assertFalse(r.audit['global_bundle_baseline']['comparable'])
  self.assertEqual(r.audit['global_bundle_baseline']['skip_reason'],'INCUMBENT_OUTSIDE_SERIAL_FIXED_GPU_DOMAIN')
 def test_joint_still_keeps_the_better_feasible_incumbent(self):
  r=self.scenario('joint')
  self.assertEqual(r.selected['a'].config.gpu_type,'aws')
  self.assertEqual(r.scheduling_objective,10.)
  self.assertTrue(r.audit['global_bundle_baseline']['comparable'])
 def test_serial_aux_deterioration_is_still_rejected(self):
  from joint_prototype.clean_ablation import CleanCore
  original=CleanCore._decide
  def corrupt(controller,*args,**kwargs):
   r=original(controller,*args,**kwargs)
   if r.audit.get('serial_proposals'):
    for c in r.audit['serial_proposals'][-1]['selected']:c['utility']-=100.
   return r
  with patch.object(CleanCore,'_decide',corrupt):
   with self.assertRaisesRegex(RuntimeError,'SERIAL_AUX_OBJECTIVE_REGRESSION'):self.scenario('fixed')
 def test_joint_objective_guard_is_not_disabled(self):
  from joint_prototype.clean_ablation import CleanCore
  original=CleanCore._decide;calls=[0]
  def corrupt(controller,*args,**kwargs):
   r=original(controller,*args,**kwargs);calls[0]+=1
   if calls[0]==2:r.scheduling_objective-=1.
   return r
  with patch.object(CleanCore,'_decide',corrupt):
   with self.assertRaisesRegex(RuntimeError,'GLOBAL_BUNDLE_OBJECTIVE_REGRESSION'):self.scenario('joint')
 def test_failure_has_comparable_bundle_diagnostics(self):
  from joint_prototype.clean_ablation import CleanCore
  original=CleanCore._decide;calls=[0]
  def corrupt(controller,*args,**kwargs):
   r=original(controller,*args,**kwargs);calls[0]+=1
   if calls[0]==2:r.scheduling_objective-=1.
   return r
  with patch.object(CleanCore,'_decide',corrupt):
   try:self.scenario('joint')
   except RuntimeError as e:
    self.assertTrue(e.solver_diagnostics['comparison']['comparable'])
    self.assertIn('baseline',e.solver_diagnostics);self.assertIn('selected',e.solver_diagnostics)
   else:self.fail('expected explicit diagnostic failure')
if __name__=='__main__':unittest.main()
