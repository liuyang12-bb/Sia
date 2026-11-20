"""Offline analysis tests, not new scheduler/regression/performance tests."""
import math,unittest
from diagnose import OUT,read,interp,matrix,response,CPU,RAM

class DiagnosticsTests(unittest.TestCase):
    def test_matrix_knots(self):
        for name in ('alexnet_1.py','bert_1.py'):
            mat=matrix(name)
            for i,x in enumerate(CPU):
                for j,y in enumerate(RAM):self.assertAlmostEqual(interp(mat,x,y),mat[i][j])
    def test_reference_and_flat_template(self):
        for pc,pm in [(2.236518965970687,1.4954995707022742),(4,2)]:
            cfg=dict(cpu_demand_scale=pc,memory_demand_scale=pm)
            self.assertAlmostEqual(response(matrix('alexnet_1.py'),cfg,3,62.5)['rho'],1)
            for c,m in [(1,20.83),(3,62.5),(6,125)]:
                self.assertEqual(response(matrix('bert_1.py'),cfg,c,m)['rho'],1)
    def test_all_current_selected_responses_match(self):
        # Generated file requires exact match to each selected log before saving.
        self.assertTrue(read(OUT/'diagnostic_validation.json')['independent_rho_recalculation_matches_logged_selected'])
    def test_unknowns_not_mislabelled(self):
        s=read(OUT/'availability_summary.json');u=read(OUT/'unknown_job_rounds.json')
        for method in s:
            r=[x for x in u if x['method']==method]
            self.assertEqual(len(r),s[method]['unknown_job_rounds'])
            self.assertTrue(all(x['used_gpu_seconds']==0 for x in r))
            self.assertTrue(all(x['sources']==['unprofiled_sia_fallback'] for x in r))
    def test_capacity_and_locked_in_enumeration(self):
        states=read(OUT/'fixed_gpu_quota_enumeration.json')
        self.assertLessEqual(len(states),6)
        for s in states:
            for node in s['nodes']:
                best=node['best_relative_speed']
                self.assertLessEqual(best['cpu'],6+1e-7);self.assertLessEqual(best['ram'],125+1e-7);self.assertLessEqual(node['gpus'],2)
                for j,o in best['allocation'].items():
                    if s['selected_jobs'][j]['locked']:
                        self.assertEqual(o['cpu'],s['selected_jobs'][j]['current']['cpu'])
                        self.assertEqual(o['ram'],s['selected_jobs'][j]['current']['ram'])
    def test_physical_slack_not_policy_claim(self):
        rows=read(OUT/'physical_slack_bounds.json')
        positive=[r for r in rows if r['relative_speedup']>1e-9]
        self.assertTrue(positive)
        self.assertTrue(all(not r['in_logged_eligible_grid'] for r in positive))
        self.assertTrue(all(r['within_capacity'] and not r['checkpoint_locked'] for r in positive))
    def test_temporal_residual_summary(self):
        rows=read(OUT/'prequential_calibration_rows.json');summaries=read(OUT/'prequential_calibration_summary.json')
        for method,s in summaries.items():
            subset=[r for r in rows if r['method']==method]
            self.assertEqual(sum(r['supported'] for r in subset),s['prequential_supported'])
            self.assertTrue(all(r['prior_origins']>=2 for r in subset if r['supported']))
            self.assertTrue(all(r['covered'] is None for r in subset if not r['supported']))
    def test_short_horizon_is_archived_arithmetic_only(self):
        for r in read(OUT/'existing_forecast_horizon_rows.json'):
            self.assertIn(r['horizon'],(60,120,300));self.assertGreaterEqual(r['observed_integrated_work'],0)
            self.assertGreaterEqual(r['predicted_integrated_work'],0)
            if r['zero_actual']:self.assertIsNone(r['ape'])

if __name__=='__main__':unittest.main(verbosity=2)
