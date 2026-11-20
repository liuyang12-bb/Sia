import sys,unittest
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from test_breakpoint_candidates import BreakpointCandidates
from residual_capacity import resource_points
from joint_prototype.joint_profile import NodeSpec

class ResidualTests(unittest.TestCase):
 def setUp(self):
  BreakpointCandidates.setUp(self)
  self.s.nodes=(NodeSpec('dgx-ext:0','dgx-ext',2,6.,125.),)
  self.s.node_map={n.node_id:n for n in self.s.nodes}
  self.s.minimum_cpu_per_gpu=1.;self.s.minimum_ram_gb_per_gpu=20.83
 def test_shared_floor_complement(self):
  cpu,ram=resource_points(self.s,self.jobs[0],self.jobs[0].configs[0],self.p)
  self.assertIn(5.,cpu);self.assertIn(104.17,ram)
 def test_no_task_identity_or_arrival_dependency(self):
  j=self.jobs[0];a=resource_points(self.s,j,j.configs[0],self.p)
  b=resource_points(self.s,replace(j,arrival_time=999,observed_progress=123),j.configs[0],self.p)
  self.assertEqual(a,b)
 def test_generated_column_is_audited_and_response_unchanged(self):
  self.s.resource_point_provider=resource_points
  cs=self.s.build_joint_columns();c=next(c for c in cs if c.job_id=='cf' and c.query.cpu_per_gpu==5.)
  self.assertTrue(self.s.audit([c])['valid'])
  self.assertEqual(c.query,self.o.query(self.s.job_map[c.job_id],c.config,c.query.cpu_per_gpu,c.query.ram_gb_per_gpu,(3.,62.5)))
 def test_disabled_axes_not_expanded(self):
  self.s.breakpoint_cpu_enabled=False;self.s.aux_cpu_points=(3.,)
  self.assertEqual(resource_points(self.s,self.jobs[0],self.jobs[0].configs[0],self.p)[0],(3.,))

if __name__=='__main__':unittest.main()
