import csv,hashlib,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1';OLD=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
class ArrivalContracts(unittest.TestCase):
 def test_only_actual_arrival_changes(self):
  original=list(csv.DictReader((OLD/'inputs/BASE_workload.csv').open()))
  for mode in ['BURST','COMPACT','SPREAD']:
   rows=list(csv.DictReader((OUT/'inputs'/f'{mode}_workload.csv').open()));self.assertEqual(len(rows),12)
   for a,b in zip(rows,original):self.assertEqual({k:v for k,v in a.items() if k!='time'},{k:v for k,v in b.items() if k!='time'})
 def test_distinct_initial_states(self):
  counts=[]
  for mode in ['BURST','COMPACT','SPREAD']:counts.append(sum(int(r['time'])==0 for r in csv.DictReader((OUT/'inputs'/f'{mode}_workload.csv').open())))
  self.assertEqual(counts,[12,4,1])
 def test_source_is_accepted_repair_and_packages_frozen(self):
  p=json.loads((OUT/'preregistration.json').read_text());old=json.loads((OLD/'preregistration.json').read_text());self.assertEqual(p['source_hashes'],old['source_hashes']);self.assertEqual(p['packages'],old['packages'])
  for n,h in p['source_hashes'].items():self.assertEqual(sha(OUT/'source'/n),h)
 def test_matrix_and_byte_identical_gate(self):
  p=json.loads((OUT/'preregistration.json').read_text());self.assertEqual(len(p['cells']),18);self.assertEqual(len(p['gate_cells']),4);self.assertEqual(p['capacities'],{'RNEW':[6.,125.]});self.assertEqual(sha(OUT/'inputs/GATE_workload.csv'),sha(OUT/'inputs/BURST_workload.csv'));self.assertEqual(sha(OUT/'inputs/COMPACT_workload.csv'),sha(OLD/'inputs/BASE_workload.csv'))
 def test_donor_loss_can_outweigh_receiver_eta_gain(self):
  spec=importlib.util.spec_from_file_location('g',ROOT/'maintenance/gate_arrival18_20261030.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
  receiver=g.response_effect(1,1,10,10,2,1);donor=g.response_effect(1,1,10,10,.5,1)
  self.assertGreater(receiver[0]+donor[0],0);self.assertLess(receiver[1]+donor[1],0)
 def test_checkpoint_has_no_rho_gain(self):
  spec=importlib.util.spec_from_file_location('g',ROOT/'maintenance/gate_arrival18_20261030.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
  self.assertEqual(g.response_effect(1,0,10,10,2,1),(0.,0.))
if __name__=='__main__':unittest.main(verbosity=2)
