"""Six-method adapter acceptance. Bounded runs: three jobs x five rounds."""
import contextlib,importlib,io,unittest
from pathlib import Path
import sys
def api():
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
 return importlib.import_module('resource_study')
class ResourceStudyTests(unittest.TestCase):
 def build(self,m,scenario='R01',response='medium'):
  fixture=Path(__file__).resolve().parent/'fixtures/resource_study_three.csv'
  with contextlib.redirect_stdout(io.StringIO()):c=api().build(m,scenario,response,fixture)
  return c
 def test_six_contracts_and_sia_not_given_l2(self):
  for m in api().METHODS:
   c=self.build(m)
   self.assertEqual(hasattr(c.policy,'liveness'),m.startswith('CC'))
   self.assertEqual(c.policy.oracle.response_exponent,.5)
 def test_no_grid_expansion_and_proportional_ablation(self):
  for scenario in api().CAPACITIES:
   cpu,ram=api().CAPACITIES[scenario]
   for method in api().METHODS:
    c=self.build(method,scenario)
    self.assertTrue(set(c.policy.aux_cpu_points)<=set(api().ORIGINAL_CPU)|{cpu/2})
    self.assertTrue(set(c.policy.aux_ram_points)<=set(api().ORIGINAL_RAM)|{ram/2})
    if method=='CC_CPU':self.assertEqual(c.policy.aux_ram_points,(ram/2,))
    if method=='CC_MEM':self.assertEqual(c.policy.aux_cpu_points,(cpu/2,))
 def test_neutral_cc_trajectories_equal(self):
  trajectories=[]
  with contextlib.redirect_stdout(io.StringIO()):
   for m in ['CC_P','CC_CPU','CC_MEM','CC_FULL']:
    c=self.build(m,'R11','neutral');states=[]
    for _ in range(5):
     c.step(60);states.append((c.allocations.copy(),[(j.progress,j.num_restarts,j.completion_time) for j in c.jobs]))
     for col in c.policy.last_decision['result']['selected'].values():self.assertEqual(col['rho'],1.)
    trajectories.append(states)
  for t in trajectories[1:]:self.assertEqual(t,trajectories[0])
 def test_all_six_actual_medium_paths_and_capacity(self):
  from joint_prototype.execution_audit import audit_execution
  with contextlib.redirect_stdout(io.StringIO()):
   for m in api().METHODS:
    c=self.build(m)
    for _ in range(5):
     c.step(60);d=c.policy.last_decision;self.assertTrue(d['result']['audit']['valid'])
     for u in d['result']['audit']['node_totals'].values():
      self.assertLessEqual(u['gpus'],2);self.assertLessEqual(u['cpu'],12);self.assertLessEqual(u['ram_gb'],187.5)
     for e in c.joint_round_logs[-1]['execution']:self.assertTrue(audit_execution(e)['valid'])
    self.assertGreater(sum(j.progress for j in c.jobs),0)
 def test_same_state_candidate_factor_inclusion(self):
  from joint_prototype.neutral_liveness import geometry
  with contextlib.redirect_stdout(io.StringIO()):
   c=self.build('CC_FULL');c.step(60);c.step(60);c.step(60)
   jobs=c.get_job_infos();specs,_=c.policy._job_specs(jobs,c.allocations);nodes=c.policy._nodes(c.get_node_infos())
   sets={}
   for m in ['CC_P','CC_CPU','CC_MEM','CC_FULL']:
    p=self.build(m).policy;s=p._scheduler(nodes,specs)
    cols=s.build_proportional_columns() if m=='CC_P' else s.build_joint_columns()
    sets[m]={geometry(x) for x in cols}
   self.assertTrue(sets['CC_P']<=sets['CC_CPU']<=sets['CC_FULL'])
   self.assertTrue(sets['CC_P']<=sets['CC_MEM']<=sets['CC_FULL'])
if __name__=='__main__':unittest.main()
