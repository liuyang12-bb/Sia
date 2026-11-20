import csv,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_text(encoding='utf8'))
class DesignTests(unittest.TestCase):
    def test_budget_and_prop_feasible(self):
        for caps in read('v2_screen_plan.json')['scenarios'].values():
            self.assertEqual(sum(v[0] for v in caps.values()),12.)
            self.assertEqual(sum(v[1] for v in caps.values()),250.)
            for cpu,ram in caps.values():
                self.assertGreaterEqual(cpu/2,1.)
                self.assertGreaterEqual(ram/2,20.83)
    def test_workload_only_arrival_changes(self):
        def rows(n):
            with (P/'inputs'/n).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
        a,b=rows('compact12.csv'),rows('burst12.csv')
        self.assertEqual(len(a),12);self.assertEqual(len(b),12)
        for x,y in zip(a,b):
            self.assertEqual({k:v for k,v in x.items() if k!='time'},{k:v for k,v in y.items() if k!='time'})
            self.assertEqual(y['time'],'0')
    def test_all_scenarios_kept(self):
        p=read('design_manifest.json');self.assertEqual(len(p['cells']),12)
        for i in range(0,12,2):
            a,b=p['cells'][i:i+2]
            self.assertEqual(a['method'],'FO');self.assertEqual(b['method'],'FULL')
            for k in ('scenario','workload','response'):self.assertEqual(a[k],b[k])
    def test_strict_receiver_support(self):
        r=read('v2_offline_results.json')[-1]
        a=r['fixed_gpu_aux_optimum']['allocation'];b=r['joint_optimum']['allocation']
        self.assertEqual(a[1],b[1])
        self.assertGreater(b[0]['speed'],a[0]['speed'])
        for c in (a[0],b[0]):
            self.assertFalse(c['num_clipped']);self.assertFalse(c['den_clipped'])
        self.assertAlmostEqual(a[0]['gpu']*a[0]['cpu'],b[0]['gpu']*b[0]['cpu'])
        self.assertAlmostEqual(a[0]['gpu']*a[0]['ram'],b[0]['gpu']*b[0]['ram'])
    def test_no_claim_of_execution(self):
        p=read('design_manifest.json')
        self.assertEqual(p['status'],'DESIGN_AND_OFFLINE_PRECHECK_ONLY_NOT_EXECUTION_PREREGISTRATION')
        for r in read('v2_offline_results.json'):self.assertIn('NOT ACTUAL FO/FULL',r['warning'])
    def test_all_gpu_configs_covered_and_gaps_exposed(self):
        c=read('v2_coverage.json');self.assertEqual(len(c),32)
        self.assertTrue(all(x['feasible_points']>0 for x in c))
        self.assertTrue(any(x['fully_supported_points']==0 for x in c))
if __name__=='__main__':unittest.main(verbosity=2)
