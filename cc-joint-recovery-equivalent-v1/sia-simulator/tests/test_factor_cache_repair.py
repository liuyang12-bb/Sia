"""Same-state score-vector, domain, execution and normalization equivalence."""
import contextlib,copy,dataclasses,io,sys,tempfile,unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core80 import build

def retained(policy,columns):
 mode=policy.liveness.resource_factor
 if mode=='PROP':return [c for c in columns if abs(c.query.cpu_per_gpu-c.proportional_cpu_per_gpu)<1e-8 and abs(c.query.ram_gb_per_gpu-c.proportional_ram_per_gpu)<1e-8 or c.candidate_kind=='locked']
 if mode=='CPU':return [c for c in columns if abs(c.query.ram_gb_per_gpu-c.proportional_ram_per_gpu)<1e-8 or c.candidate_kind=='locked']
 if mode=='MEM':return [c for c in columns if abs(c.query.cpu_per_gpu-c.proportional_cpu_per_gpu)<1e-8 or c.candidate_kind=='locked']
 return columns

class FactorCacheRepairTests(unittest.TestCase):
 def test_real_same_state_all_factor_scores_and_training(self):
  data=pd.read_csv(ROOT/'inputs/core80_workload.csv').iloc[[0,3,4]].copy();data.time=0
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'three.csv';data.to_csv(path,index=False)
   for method in ('PROP','CPU','MEM','FULL','FO'):
    traces=[]
    for enabled in (False,True):
     with contextlib.redirect_stdout(io.StringIO()):
      c=build(method,workload_path=path,topology={'aws':2,'dgx-ext':2},engineering=enabled);trace=[]
      for _ in range(5):
       c.step(60)
       cols=retained(c.policy,c.policy.exchange_columns)
       trace.append((copy.deepcopy(c.allocations),[(j.progress,j.epoch,j.rescale_time,j.num_restarts,j.completion_time) for j in c.jobs],
            [dataclasses.asdict(x) for x in cols],set(c.policy.liveness.previous_known_configs)))
      if enabled:
       cert=c.policy._repair_factor_certificate
       self.assertGreaterEqual(cert['full_grid_points_considered']+cert['unknown_proportional_columns'],cert['materialized_columns'])
     traces.append(trace)
    self.assertEqual(traces[0],traces[1],method)
 def test_reference_max_cache_key_changes_with_capacity(self):
  from factor_cache import full_max_rho
  from test_clean_ablation import state
  from test_adaptive_service import spec,cfg
  from joint_prototype.joint_scheduler import enumerate_placements
  from joint_prototype.joint_profile import NodeSpec
  s=state([spec('a')],[NodeSpec('aws:0','aws',2,6.,125.)]);j=s.jobs[0];c=j.configs[0];p=enumerate_placements(c,s.nodes)[0]
  full_max_rho(s,j,c,p);n=len(s.oracle._repair_full_grid_rho);full_max_rho(s,j,c,p);self.assertEqual(len(s.oracle._repair_full_grid_rho),n)
  s2=state([j],[NodeSpec('aws:0','aws',2,12.,125.)]);s2.oracle=s.oracle;full_max_rho(s2,j,c,p)
  self.assertEqual(len(s.oracle._repair_full_grid_rho),n+1)
 def test_real_unseeded_bert_full_checkpoint_training_next_optimization(self):
  data=pd.read_csv(ROOT/'inputs/core80_workload.csv').iloc[[4]].copy();data.time=17
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'bert.csv';data.to_csv(path,index=False)
   for method in ('SIA_P','SIA_AUX','FULL'):
    tracks=[]
    for enabled in (False,True):
     with contextlib.redirect_stdout(io.StringIO()):
      c=build(method,workload_path=path,topology={'aws':2,'dgx-ext':2},seed=False,engineering=enabled);track=[]
      for _ in range(12):
       c.step(60);j=c.jobs[0]
       self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
       track.append((copy.deepcopy(c.allocations),j.progress,j.epoch,j.rescale_time,j.num_restarts,j.completion_time,j.first_allocation_time,j.first_productive_training_time,c.joint_round_logs[-1]['execution'][0]['waiting_seconds']))
      self.assertGreater(j.progress,0.);self.assertIsNotNone(j.first_productive_training_time)
     tracks.append(track)
    self.assertEqual(tracks[0],tracks[1],method)

if __name__=='__main__':unittest.main(verbosity=2)
