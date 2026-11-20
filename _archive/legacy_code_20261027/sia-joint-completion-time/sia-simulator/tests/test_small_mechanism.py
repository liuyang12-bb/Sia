import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from joint_prototype.joint_profile import NodeSpec,GpuConfig,JobSpec
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.neutral_liveness import FactorScheduler
from joint_prototype.small_mechanism import fixed_set_solve
ROOT=Path(__file__).resolve().parents[3]
class SmallTests(unittest.TestCase):
 def scheduler(self):
  registry=ROOT/'research_runs/sia_synergy/stage2b_neutral/g0_verified/registry.json';models=ROOT/'research_sources/Synergy/simulator/models'
  # Worktree nesting differs from source tree root; use workspace sibling.
  if not registry.exists():registry=ROOT.parent/'research_runs/sia_synergy/stage2b_neutral/g0_verified/registry.json';models=ROOT.parent/'research_sources/Synergy/simulator/models'
  o=R0Oracle(registry,models,{'j':'cifar10'},'medium');row=o.reference('j','rtx:1:1');cfg=GpuConfig('rtx:1:1','rtx',1,1,1,row['phi_c'],row['phi_m'])
  return FactorScheduler([NodeSpec('rtx:0','rtx',8,24,500)],[JobSpec('j','alexnet_1.py',(cfg,))],o,[1,2,3,4,6,9,12,24],[20.83,62.5,125,187.5,250],allocation_bonus=0,verify_exact=False,minimum_cpu_per_gpu=1,minimum_ram_gb_per_gpu=20.83,candidate_policy='b3')
 def test_fixed_set_forces_negative_utility_and_gpu(self):
  from dataclasses import replace
  s=self.scheduler();cols=[replace(c,scheduling_utility=-1/c.query.rho) for c in s.build_joint_columns()];r=fixed_set_solve(s,cols,{'j'});self.assertEqual(set(r.selected),{'j'});self.assertTrue(s.audit(list(r.selected.values()))['valid']);self.assertEqual(r.selected['j'].config.config_id,'rtx:1:1');self.assertGreater(r.selected['j'].query.rho,1)
 def test_no_free_admission(self):
  s=self.scheduler();r=fixed_set_solve(s,s.build_joint_columns(),set());self.assertFalse(r.selected)
 def test_solver_failure_explicit(self):
  s=self.scheduler()
  with patch('joint_prototype.small_mechanism.milp',return_value=type('R',(),dict(success=False,status=1,x=None,message='injected'))()):
   with self.assertRaisesRegex(RuntimeError,'AUX_SOLVER_UNDECIDED'):fixed_set_solve(s,s.build_joint_columns(),{'j'})
 def test_missing_intent_rejected(self):
  s=self.scheduler()
  with self.assertRaisesRegex(RuntimeError,'MISSING_FIXED_INTENT'):fixed_set_solve(s,[],{'j'})
if __name__=='__main__':unittest.main()
