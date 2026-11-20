"""Independent offline validation only; no scheduler or simulator import."""
import collections,csv,hashlib,json,sys,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'cc-joint-core80-full-lex-repair-v8'
OUT=ROOT/'research_runs/sia_synergy/cc_paper_sameversion_20261102_run2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(name):return json.loads((OUT/name).read_text())
class OfflineAcceptance(unittest.TestCase):
 def test_full_actual_source_snapshot(self):
  hashes=load('source_hashes.json');self.assertEqual(len(hashes),329)
  with zipfile.ZipFile(OUT/'baseline_source_snapshot.zip') as z:
   self.assertEqual(set(z.namelist()),set(hashes))
   for p,h in hashes.items():
    self.assertEqual(sha(BASE/p),h);self.assertEqual(hashlib.sha256(z.read(p)).hexdigest().upper(),h)
 def test_trace_extraction_preserves_all_original_parameters(self):
  for info in load('input_manifest.json'):
   self.assertEqual(sha(info['path']),info['sha256']);self.assertEqual(sha(info['source']),info['source_sha256'])
   actual=read(info['path']);self.assertEqual(len(actual),80)
   if 'source_csv_data_row_indices' not in info:continue
   source=read(info['source']);ordered=sorted(range(len(source)),key=lambda i:(float(source[i]['time']),i))[:80]
   self.assertEqual(ordered,info['source_csv_data_row_indices'])
   for row,i in zip(actual,ordered):
    for k,v in source[i].items():
     if k!='time':self.assertEqual(row[k],v)
    self.assertAlmostEqual(float(row['time']),float(source[i]['time'])-info['arrival_shift'],places=8)
 def test_controlled_arrivals_and_work_unchanged(self):
  original=read(BASE/'inputs/core80_workload.csv')
  for name in ('waves','uniform'):
   actual=read(OUT/f'inputs/{name}.csv')
   for i,(a,b) in enumerate(zip(actual,original)):
    for k,v in b.items():
     if k!='time':self.assertEqual(a[k],v)
    self.assertAlmostEqual(float(a['time']),float(b['time']) if name=='waves' else 1200*i/79,places=9)
 def test_fixed_matrix_and_static_capacities(self):
  p=load('preregistration.json');cells=p['cells'];self.assertEqual(len(cells),47)
  self.assertEqual(collections.Counter(c['group'] for c in cells),{'service':8,'resources':12,'negative':3,'trace':18,'scale':6})
  self.assertEqual(len({c['id'] for c in cells}),47)
  self.assertTrue(all(c['interval']==60 and c['max_rounds']==6000 for c in cells))
  for cpu,ram in p['capacities_per_gpu'].values():self.assertGreaterEqual(cpu,1);self.assertGreaterEqual(ram,20.83)
  self.assertEqual(p['reference_point'],[3.,62.5]);self.assertEqual(p['alpha'],{'neutral':0.,'medium':.5})
 def test_r0_source_rows_and_template_bytes(self):
  rows=json.loads((BASE/'inputs/registry.json').read_text());self.assertEqual(len(rows),55)
  checked=set()
  for r in rows:
   for src in r['source_rows']:
    name=src['file'].replace('\\','/').rsplit('/',1)[-1]
    target=BASE/'sia-simulator/traces'/r['application']/name
    self.assertEqual(sha(target),src['sha256'].upper());checked.add(str(target))
   self.assertEqual(sha(BASE/'inputs/models'/r['template']),sha(ROOT/'research_sources/Synergy/simulator/models'/r['template']))
  self.assertGreater(len(checked),0)

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(OfflineAcceptance)
 with (OUT/'offline_tests.log').open('x',encoding='utf8') as stream:
  result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
  data={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
   'scope':'Five new offline checks; not the existing335 scheduler tests; no training simulation'}
 (OUT/'offline_test_summary.json').write_text(json.dumps(data,indent=2),encoding='utf8')
 print(json.dumps(data));sys.exit(0 if result.wasSuccessful() else 1)
