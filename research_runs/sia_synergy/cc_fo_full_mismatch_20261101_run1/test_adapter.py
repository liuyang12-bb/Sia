import json,sys,unittest
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-mismatch-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from mismatch_adapter import build
from core80 import build as old_build

class AdapterTests(unittest.TestCase):
 def test_all_capacities_before_first_decision(self):
  for sid in ('S0_symmetric','S1_cpu_mismatch','S2_ram_mismatch','S3_crossed'):
   for method in ('FO','FULL'):
    c=build(method,O/f'inputs/{sid}.json',O/'inputs/gate2.csv')
    self.assertEqual(c.current_time,0)
    self.assertEqual(c.policy.node_capacities,c.experimental_capacity_receipt)
    c.step(60) # real scheduler verifies every node and atomic capacity audit
    self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
 def test_symmetric_three_round_exact_regression(self):
  for method in ('FO','FULL'):
   a=old_build(method,workload_path=O/'inputs/gate2.csv',topology={'aws':2,'dgx-ext':2})
   from residual_capacity import install
   install(a)
   b=build(method,O/'inputs/S0_symmetric.json',O/'inputs/gate2.csv')
   for _ in range(3):
    a.step(60);b.step(60)
    def state(c):
     d=c.policy.last_decision['result']['selected']
     return ({j:(v['config']['config_id'],v['placement'],v['cpu_per_gpu'],v['ram_gb_per_gpu'],v['rho']) for j,v in d.items()},
      [(j.progress,j.num_restarts,j.rescale_time,j.completion_time) for j in c.jobs])
    self.assertEqual(state(a),state(b))
if __name__=='__main__':unittest.main(verbosity=2)
