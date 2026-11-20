import csv,hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1';OLD=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
class Contract(unittest.TestCase):
 def test_frozen_source_and_inputs(self):
  p=json.loads((OUT/'preregistration.json').read_text())
  for n,h in p['source_hashes'].items():self.assertEqual(sha(OUT/'source'/n),h)
  for n,h in p['input_hashes'].items():self.assertEqual(sha(OUT/'inputs'/n),h)
  self.assertEqual(sha(OUT/'run.py'),p['runner_hash']);self.assertEqual(len(p['cells']),36)
 def test_source_rows_preserve_work_batch_demand(self):
  sources={t:list(csv.DictReader((ROOT/f'official-artifacts/sia-simulator/workloads/saturn/workload-{t}.csv').open())) for t in [1,5,10]}
  p=json.loads((OUT/'preregistration.json').read_text())
  for name,rows in p['cohort_rows'].items():
   self.assertEqual(len(rows),12);self.assertEqual([int(r['time']) for r in rows],[0]*4+[300]*2+[600]*2+[900]*2+[1200]*2)
   for r in rows:
    original=sources[int(r['original_trace'])][int(r['original_source_row_index'])]
    for k,v in original.items():
     if k not in ['name','time']:self.assertEqual(r[k],v)
 def test_domain_supported_mechanism(self):
  w=json.loads((OUT/'strict_witness.json').read_text());r=json.loads((OUT/'fixed_mechanism.json').read_text())
  for mode,v in w['pairs'].items():
   self.assertLessEqual(v['cpu_sum'],6+1e-9);self.assertLessEqual(v['ram_sum'],125+1e-9)
   for app in ['cifar10','bert']:self.assertTrue(v[app]['supported_including_floating_endpoint_tolerance'])
   self.assertTrue(all(j['completed'] for j in r[mode]['jobs'].values()))
  for mode in ['CPU','MEM','FULL']:self.assertEqual(r[mode]['jobs']['bert-28']['jct'],r['P']['jobs']['bert-28']['jct'])
 def test_previous_complete_regression_preserved(self):
  seal=json.loads((OLD/'delivery_hashes.json').read_text())
  for n,h in seal.items():self.assertEqual(sha(OLD/n),h)
  self.assertEqual(json.loads((OLD/'tests_canonical.json').read_text())['exit_code'],0)
 def test_proportional_canonical_red_green(self):
  p=json.loads((OUT/'preregistration.json').read_text());cpu,ram=p['capacities']['AMPLE']
  failure=json.loads((OUT/'cells/BASE_AMPLE_medium_SIA_P/failure.json').read_text())
  col=next(iter(failure['last_decision']['result']['selected'].values()));cp,mm=col['cpu_per_gpu'],col['ram_gb_per_gpu']
  self.assertNotEqual((cp,mm),(cpu/2,ram/2))
  self.assertEqual((cp,mm),(round(cpu/2,9),round(ram/2,9)))
  self.assertNotIn(cp+1e-6,{cpu/2,round(cpu/2,9)})
  self.assertNotIn(mm+1e-6,{ram/2,round(ram/2,9)})
  self.assertLess(abs(cp*2-cpu),1e-7);self.assertLess(abs(mm*2-ram),1e-7)
if __name__=='__main__':unittest.main(verbosity=2)
