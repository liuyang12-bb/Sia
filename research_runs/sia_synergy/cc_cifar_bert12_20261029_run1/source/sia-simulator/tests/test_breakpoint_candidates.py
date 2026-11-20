import unittest,sys
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from breakpoint_study import resource_points,slope_knots
from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.joint_scheduler import Placement
from joint_prototype.neutral_liveness import FactorScheduler
from resource_study import ORIGINAL_CPU,ORIGINAL_RAM

class BreakpointCandidates(unittest.TestCase):
 def setUp(self):
  self.o=R0Oracle(ROOT/'inputs/registry.json',ROOT/'inputs/models',{'cf':'cifar10','ds':'deepspeech2'},'medium')
  self.jobs=[]
  for name,app in [('cf','cifar10'),('ds','deepspeech2')]:
   r=self.o.reference(name,'dgx-ext:1:1')
   g=GpuConfig('dgx-ext:1:1','dgx-ext',1,1,1.,r['phi_c'],r['phi_m'])
   self.jobs.append(JobSpec(name,APPLICATION_TEMPLATES[app],(g,)))
  self.s=FactorScheduler([NodeSpec('dgx-ext:0','dgx-ext',2,12.,187.5)],self.jobs,self.o,
       ORIGINAL_CPU,ORIGINAL_RAM,enforce_proportional_progress_floor=False,verify_exact=False,candidate_policy='b3')
  self.p=Placement((('dgx-ext:0',1),))
 def test_breakpoint_and_complement_are_generated_and_audited(self):
  self.s.resource_point_provider=resource_points
  cols=self.s.build_joint_columns()
  r=self.o.reference('cf','dgx-ext:1:1');knot=62.5*r['phi_m'];other=187.5-knot
  cf=next(c for c in cols if c.job_id=='cf' and c.query.cpu_per_gpu==6. and c.query.ram_gb_per_gpu==knot)
  ds=next(c for c in cols if c.job_id=='ds' and c.query.cpu_per_gpu==6. and c.query.ram_gb_per_gpu==other)
  self.assertTrue(self.s.audit([cf,ds])['valid'])
  self.assertFalse(cf.query.source_coordinate_clipped or cf.query.baseline_coordinate_clipped)
  self.assertFalse(ds.query.source_coordinate_clipped or ds.query.baseline_coordinate_clipped)
 def test_default_grid_and_forged_quota_stay_rejected(self):
  cols=self.s.build_joint_columns();self.assertTrue(cols)
  r=self.o.reference('cf','dgx-ext:1:1');knot=62.5*r['phi_m']
  self.assertFalse(any(c.query.ram_gb_per_gpu==knot for c in cols))
  col=cols[0];q=self.o.query(self.s.job_map[col.job_id],col.config,6.,101.2345,(6.,93.75))
  fake=replace(col,query=q,node_usage=self.s._node_usage(col.placement,6.,101.2345))
  self.s.resource_point_provider=resource_points
  self.assertFalse(self.s.audit([fake])['valid'])
 def test_axis_ablation_remains_literal(self):
  self.s.aux_cpu_points=(6.,);self.s.breakpoint_cpu_enabled=False
  self.s.resource_point_provider=resource_points
  self.assertTrue(all(c.query.cpu_per_gpu==6. for c in self.s.build_joint_columns()))
 def test_source_slopes_no_new_response_or_lookahead(self):
  matrix=self.o.template('alexnet_1.py').matrix
  from joint_prototype.joint_profile import RAM_COORDS
  self.assertIn(62.5,slope_knots(matrix,RAM_COORDS,1))
  before=self.o.provenance_digest()
  a=resource_points(self.s,self.jobs[0],self.jobs[0].configs[0],self.p)
  b=resource_points(self.s,replace(self.jobs[0],arrival_time=999,observed_progress=100),self.jobs[0].configs[0],self.p)
  self.assertEqual(a,b);self.assertEqual(before,self.o.provenance_digest())

if __name__=='__main__':unittest.main()
