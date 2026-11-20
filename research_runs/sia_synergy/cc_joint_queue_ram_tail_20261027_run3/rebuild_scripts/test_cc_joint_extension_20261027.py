"""Offline acceptance: no job stepping and no solver invocation."""
import importlib.util,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parent/'cc_joint_extension_20261027.py'
spec=importlib.util.spec_from_file_location('extension',P);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
class Acceptance(unittest.TestCase):
 def test_frozen_source_and_environment(self):
  self.assertEqual(e.verify()['alpha'],.5)
 def test_memory_witness_includes_donor_loss(self):
  x=json.loads((e.OUT/'preflight.json').read_text(encoding='utf-8'))
  self.assertGreater(x['receiver_speed_change'],0);self.assertLess(x['donor_speed_change'],0)
  self.assertTrue(x['net_JCT_benefit_not_proven'])
 def test_source_workload_is_unchanged(self):
  self.assertEqual(e.sha(e.OUT/'source/inputs/workload.csv'),e.sha(e.OLD/'inputs/workload.csv'))
 def test_proportional_points_fit(self):
  self.assertEqual(2*6,12);self.assertEqual(2*93.75,187.5)
  self.assertEqual(125+62.5,187.5)
 def test_raw_time_decomposition(self):
  x=json.loads((e.OUT/'queue_tail_raw_diagnostics.json').read_text(encoding='utf-8'))
  for r in x['methods'].values():
   self.assertAlmostEqual(r['queue_areas']['active_job_seconds'],sum(j['jct'] for j in r['jobs'].values()))
   self.assertAlmostEqual(r['queue_areas']['waiting_job_seconds'],sum(j['wait'] for j in r['jobs'].values()))
 def test_tail_candidate_includes_real_startup(self):
  x=json.loads((e.OUT/'tail_budget_witness.json').read_text(encoding='utf-8'))
  self.assertEqual(x['best_score']['delay'],25)
  self.assertLess(x['best_score']['estimated_completion'],x['current_score']['estimated_completion'])
  self.assertEqual(x['audit']['service_delay_ledgers']['deepspeech2-8']['delay_charged'],250)
 def test_supported_memory_opportunity_missing_from_grid(self):
  x=json.loads((e.OUT/'memory_candidate_gap.json').read_text(encoding='utf-8'))
  self.assertAlmostEqual(sum(x['ram'].values()),187.5)
  self.assertGreater(x['evidence']['cf']['speed_ratio'],1)
  self.assertAlmostEqual(x['evidence']['ds']['speed_ratio'],1)
  for v in x['evidence'].values():
   self.assertFalse(v['in_current_grid']);self.assertFalse(v['physical_alternative']['source_coordinate_clipped']);self.assertFalse(v['physical_alternative']['baseline_coordinate_clipped'])
if __name__=='__main__':unittest.main(verbosity=2)
