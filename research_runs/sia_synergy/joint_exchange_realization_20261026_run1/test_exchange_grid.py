import copy,json,sys,unittest
from pathlib import Path
OUT=Path(__file__).resolve().parent;PARENT=OUT.parent/'joint_opportunity_experiments_20261026_run1'
sys.path.insert(0,str(OUT/'source' if (OUT/'source').exists() else PARENT/'source'))
import pandas as pd
from joint_prototype.full160 import make_cluster
from joint_prototype.physical_domain import restrict_single_node_domain
from joint_prototype.service_horizon import install as legacy

def build(extended=True):
 cfg=json.loads((PARENT/'inputs/D3_queued_training_proposal_config.json').read_text())
 cfg['cluster'].update(nnodes={'aws':1},ngpus_per_node={'aws':2},max_physical_nodes={'aws':1})
 w=pd.read_csv(PARENT/'inputs/D3_queued_training_proposal.csv');w=w[w.name.isin(['cifar10-1','deepspeech2-9'])]
 c=make_cluster(cfg,w,PARENT/'inputs/registry.json',PARENT/'inputs/models','joint_o','medium','main',True)
 c.policy.node_capacities={'aws':dict(cpu_cores=12,ram_gb=250)};restrict_single_node_domain(c.policy)
 if extended:
  try:from joint_prototype.cpu_exchange_grid import install
  except ModuleNotFoundError:install=legacy  # Red phase executes actual legacy behavior.
 else:install=legacy
 return install(c)

class GridTests(unittest.TestCase):
 def test_real_optimize_selects_eight_four(self):
  c=build();c.step(60);selected=c.policy.last_decision['result']['selected']
  self.assertEqual(set(selected),{'cifar10-1','deepspeech2-9'})
  self.assertEqual(selected['cifar10-1']['cpu_per_gpu'],8)
  self.assertEqual(selected['deepspeech2-9']['cpu_per_gpu'],4)
  self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
 def test_legacy_grid_unchanged(self):
  c=build(False);self.assertNotIn(8,c.policy.aux_cpu_points)
 def test_grid_is_single_documented_extension(self):
  c=build();self.assertEqual(set(c.policy.aux_cpu_points),{1,2,3,4,6,8,9,12,24})
 def test_actual_training_and_capacity(self):
  c=build()
  for _ in range(8):
   c.step(60);d=c.policy.last_decision;cols=d['result']['selected']
   self.assertLessEqual(sum(v['config']['gpus']*v['cpu_per_gpu'] for v in cols.values()),12)
   self.assertLessEqual(sum(v['config']['gpus']*v['ram_gb_per_gpu'] for v in cols.values()),250)
   for e in c.joint_round_logs[-1]['execution']:
    self.assertTrue(e['independent_execution_audit']['valid'])
  self.assertTrue(all(j.progress>0 for j in c.jobs))
 def test_lock_protects_full_allocation(self):
  c=build();c.step(60)
  j=c.jobs[0];j.rescale_time=120
  before=copy.deepcopy(c.policy.current_resources[j.name]);c.step(60)
  after=c.policy.current_resources[j.name]
  for k in ['config_id','placement','cpu_per_gpu','ram_gb_per_gpu']:self.assertEqual(before[k],after[k])
  e=next(e for e in c.joint_round_logs[-1]['execution'] if e['job_id']==j.name);self.assertEqual(e['progress_delta'],0)
if __name__=='__main__':unittest.main()
