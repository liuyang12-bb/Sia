"""Verify shared execution contracts, cohort input, and bounded actual execution."""
import collections,contextlib,csv,hashlib,importlib.metadata,io,json,math,platform,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from resource_study import build,METHODS,CAPACITIES,ORIGINAL_CPU,ORIGINAL_RAM
from run_stage2c_cell import independent_rho
from joint_prototype.execution_audit import audit_execution
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
records=[]
def fixture(mix):
    folder=OUT/'acceptance_fixtures';folder.mkdir(exist_ok=True);path=folder/(mix+'_three.csv')
    if not path.exists():
        rows=list(csv.DictReader((OUT/'inputs'/(mix+'_workload.csv')).open(encoding='utf8')))
        picked=[next(r for r in rows if r['application']=='cifar10'),next(r for r in rows if r['application']=='deepspeech2')]
        picked.append(next(r for r in rows if r['name'] not in {v['name'] for v in picked}))
        with path.open('x',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(picked)
    return path
class Contracts(unittest.TestCase):
    def test_all42_input_and_shared_execution_contracts(self):
        reference=None
        cells=[(x,s,'medium',m) for x in ['M1','M2','M3'] for s in ['R01','R11'] for m in METHODS]+[('M2','R11','neutral',m) for m in METHODS]
        for mix,cap,response,method in cells:
            fullrows=list(csv.DictReader((OUT/'inputs'/(mix+'_workload.csv')).open(encoding='utf8')))
            self.assertEqual(len(fullrows),20);self.assertEqual(len({r['name'] for r in fullrows}),20)
            with contextlib.redirect_stdout(io.StringIO()):c=build(method,cap,response,fixture(mix))
            self.assertEqual(len(c.jobs),3);self.assertEqual(hasattr(c.policy,'liveness'),method.startswith('CC'))
            self.assertEqual(c.policy.oracle.response_exponent,.5 if response=='medium' else 0.)
            o=c.policy.oracle
            execution=(type(c.jobs[0]).step.__module__,type(c.jobs[0]).step.__qualname__)
            shared=dict(execution=execution,normalization=o.normalization_mode,fixed_reference=o.fixed_reference,registry=o.registry_sha256,
                models={t:o.template(t).source_sha256 for t in ['alexnet_1.py','deepspeech_1.py']})
            if reference is None:reference=shared
            self.assertEqual(shared,reference)
            cpu,ram=CAPACITIES[cap]
            self.assertEqual(c.policy.node_capacities,{t:dict(cpu_cores=cpu,ram_gb=ram) for t in ['aws','dgx-ext']})
            for j in c.jobs:
                self.assertEqual(j.progress,0);self.assertEqual(j.calibration_factor,c.jobs[[x.app_name for x in c.jobs].index(j.app_name)].calibration_factor)
                self.assertIsNotNone(j.perf_params['aws'])
            records.append(dict(mix=mix,capacity=cap,response=response,method=method,shared=shared,full_input_hash=sha(OUT/'inputs'/(mix+'_workload.csv')),full_job_rows=fullrows,bounded_contract_fixture_jobs=3,jobs=[dict(name=j.name,arrival=j.submission_time,batch=j.target_batch_size,work=j.reference_app.get_progress(j.reference_app.max_epochs)) for j in c.jobs]))
    def test_bounded_actual_six_paths(self):
        for response in ['neutral','medium']:
            for method in METHODS:
                with contextlib.redirect_stdout(io.StringIO()):
                    c=build(method,'R11',response,fixture('M2'))
                    for _ in range(3):
                        c.step(60);d=c.policy.last_decision;self.assertTrue(d['result']['audit']['valid'])
                        for name,col in d['result']['selected'].items():
                            expected=independent_rho(c.policy.oracle,name,col['config']['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu'])
                            self.assertTrue(math.isclose(col['rho'],expected,rel_tol=1e-12))
                        for e in c.joint_round_logs[-1]['execution']:self.assertTrue(audit_execution(e)['valid'])
                self.assertGreater(sum(j.progress for j in c.jobs),0)
if __name__=='__main__':
    with (OUT/'new_acceptance_tests.log').open('x',encoding='utf8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    (OUT/'execution_contracts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'new_acceptance_summary.json').write_text(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),bounded_cases=12,round_limit=3,formal_cells=0)),encoding='utf8')
    if not result.wasSuccessful():sys.exit(1)
    print('NEW_ACCEPTANCE_PASS',len(records),'contracts')
