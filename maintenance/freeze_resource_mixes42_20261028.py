import hashlib,importlib.metadata,json,platform,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1';SRC=OUT/'source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 tests=(OUT/'existing_tests_canonical_paths.log').read_text(encoding='utf8')
 assert 'Ran 229 tests' in tests and '\nOK\n' in tests,'EXISTING_TEST_GATE'
 assert json.loads((OUT/'new_acceptance_summary.json').read_text())['passed'],'NEW_TEST_GATE'
 parent=json.loads((ROOT/'research_runs/sia_synergy/cc_resource_mixes_preflight_20261028_run1/protected_source_hashes.json').read_text())
 for n,h in parent.items():
  rel=Path(n).relative_to('cc-joint-resource-study');assert sha(SRC/rel)==h and sha(ROOT/n)==h,n
 source={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 assert len(source)==len(parent)
 methods=['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']
 cells=[(x,s,'medium',m) for x in ['M1','M2','M3'] for s in ['R01','R11'] for m in methods]+[('M2','R11','neutral',m) for m in methods]
 pre=dict(frozen_unix=time.time(),cells=cells,methods=methods,source_hashes=source,runner_hash=sha(ROOT/'maintenance/run_resource_mixes42_20261028.py'),
  input_hashes={p.name:sha(p) for p in (OUT/'inputs').iterdir()},capacities=dict(R01=[12,187.5],R11=[6,187.5]),gpu_per_node=2,nodes=['aws:0','dgx-ext:0'],
  response_exponents=dict(neutral=0.,medium=.5),interval=60,max_rounds=6000,seed_profiles=[1,8],reference=[3.,62.5],
  boundary='E-legacy: source coordinate clipping to CPU[1,24],RAM[20.83,250], numerator and denominator both; explicitly accepted synthetic constant boundary extension; no data-complete claim',
  response_warning='R0 application/template mapping, phi, interpolation and cross-hardware multiplication synthetic; official source matrix cells only are source profiles',
  phi='fixed original registry, no online update or per-cohort normalization',grid_changed=False,algorithm_changed=False,response_changed=False,
  methods_contract=dict(SIA_P='official Sia optimize+placement, proportional aux, no L2',SIA_AUX='same Sia full GPU/placement intent, existing ECT auxiliary optimizer, no L2; not Synergy-TUNE',
   CC='same unchanged CC-TAIL; four variants differ only auxiliary quota freedom'),
  contracts_hash=sha(OUT/'execution_contracts.json'),tests=dict(existing=229,new_contract_tests=2,bounded_paths=12,round_limit=3),
  test_invocation_caveat='Preserved initial relocation/path failures; all229 tests rerun in canonical layout with byte-identical source; no production source edits. New contracts build 3-job fixtures for42 settings and verify full20-row input hashes;12 short3-round execution cases. Initial redundant full20-job contract construction interrupted and retained before formal freeze.',
  packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','cvxpy','cylp','psutil','autograd']},python=platform.python_version(),platform=platform.platform(),
  gates=dict(memory_gib=2,disk_gib=5),execution='serial independent Python process per cell, streaming gzip; fixed order; no result-based retuning/retry',
  missing='No phi=1, no dropped legal candidates, no discarded jobs',
  censor='6000 rounds; completed-job mean not cohort mean if censored; sample RMST common followup tau=357600s and completion rate; continue next cell if round-limit censoring',
  failure='Stop on implementation/capacity/lock/response audit failure; retain partial dir; no partial overwrite; fix only after scope assessment and new version paired reruns',
  statistics='cohort paired, nested synthetic mixtures not independent workload replications; all20 main; per-job deltas, mean/median/P95/P99/makespan, resource hours, wait/training/checkpoint, restarts, net CPU/RAM interaction',
  effects=dict(CPU='1-J_CC_CPU/J_CC_P',MEM='1-J_CC_MEM/J_CC_P',conditional_MEM='1-J_CC_FULL/J_CC_CPU',conditional_CPU='1-J_CC_FULL/J_CC_MEM',interaction_seconds='J_FULL-J_CPU-J_MEM+J_P; negative synergistic, positive overlap/opposition'),
  original_sia_regression_evidence_hash=sha(ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/official_regression.json'),
  audit_script_hash=sha(ROOT/'maintenance/audit_resource_mixes42_20261028.py'),
  limits='One160-row source trace, controlled arrivals20jobs, source-supported static witnesses do not guarantee JCT; no expansion beyond42',warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
 save('preregistration.json',pre)
 save('preregistration.sha256.json',dict(sha256=sha(OUT/'preregistration.json')))
 prior=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'
 protect={str(p.relative_to(ROOT)):sha(p) for p in (prior/'cells').glob('*/*') if p.is_file() and p.name=='output_hashes.json'}
 save('historical_hash_manifest_files.json',protect)
 for p in ROOT.glob('maintenance/*resource_mixes42_20261028.py'):shutil.copyfile(p,OUT/p.name)
 (OUT/'CURRENT_TASK.md').write_text('FROZEN_READY:42 authorized cells, preregistration.json authoritative. Run maintenance/run_resource_mixes42_20261028.py batch. Before recovery verify no duplicate process, completed prefix and hashes; partial directory never overwrite. Boundaries E-legacy explicitly accepted. Source unchanged282 files. 229 old +2 new contract tests passed. Finish all42 then independent raw-log audit and REPORT.md. No tuning, no next experiments.\n',encoding='utf8')
 print('FROZEN',len(cells),'cells',len(source),'source files')
if __name__=='__main__':main()
