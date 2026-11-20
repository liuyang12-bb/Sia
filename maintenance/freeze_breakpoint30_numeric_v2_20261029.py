"""Freeze the previously authorized30 cells only after all small gates pass."""
import hashlib,importlib.metadata,json,platform,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt2';SRC=OUT/'source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 assert json.loads((OUT/'bounded_16round_summary.json').read_text())['valid']
 assert json.loads((OUT/'small_gate_audit.json').read_text())['formal30_gate_passed']
 assert json.loads((OUT/'neutral_acceptance.json').read_text())['valid']
 assert json.loads((OUT/'adapter_acceptance.json').read_text())['valid']
 assert json.loads((OUT/'accumulator_red_green.json').read_text())['valid']
 assert json.loads((OUT/'independent_offline_audit.json').read_text())['valid']
 assert json.loads((OUT/'tests_result2.json').read_text())['exit_code']==0
 rules=json.loads((OUT/'rules.json').read_text());cells=rules['formal_cells']
 text=(ROOT/'maintenance/run_resource_mixes42_20261028.py').read_text(encoding='utf8')
 text=text.replace("OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1'","OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt2'")
 before="CELLS=[(x,s,'medium',m) for x in ['M1','M2','M3'] for s in ['R01','R11'] for m in METHODS]+[('M2','R11','neutral',m) for m in METHODS]"
 assert before in text;text=text.replace(before,'CELLS='+repr([tuple(c) for c in cells]))
 text=text.replace('from resource_study import build,CAPACITIES','from resource_study import CAPACITIES\n  from breakpoint_study import build')
 text=text.replace('/42','/30').replace('ALL42_ENDED','ALL30_ENDED').replace('total=42','total=30')
 target=ROOT/'maintenance/run_breakpoint30_numeric_v2_20261029.py';target.write_text(text,encoding='utf8')
 old=json.loads((ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1/preregistration.json').read_text())
 source={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 small=json.loads((OUT/'small_gates/small_preregistration.json').read_text());assert source==small['source_hashes']
 pre=dict(old,frozen_unix=time.time(),cells=cells,source_hashes=source,runner_hash=sha(target),
  input_hashes={p.name:sha(p) for p in (OUT/'inputs').iterdir()},grid_changed=True,algorithm_changed='CANDIDATES_ONLY',objective_changed=False,response_changed=False,
  candidate_version='SOURCE_SLOPE_KNOTS_PLUS_ONE_PEER_COMPLEMENTS_V1',
  numerical_repair='Reversible power-of-two scaling only for finite nonzero narrow constraint bands, same cost/constraints/solver options. Every integer result audited against original unscaled rows. No tolerance relaxation; no optimality/admission fallback. Original15 results are not mixed; all30 newly run.',
  bounded_numerical_gate_hash=sha(OUT/'bounded_16round_summary.json'),
  candidate_contract='breakpoint_study.py: own inverse mapped slope-change knots + source endpoints + one-peer capacity complements from whole fixed R0 registry, union original grid and proportional. No response extension change. CPU/MEM axis restrictions literal. Same per-config point contract generation and atomic audit.',
  packages={n:importlib.metadata.version(n) for n in old['packages']},python=platform.python_version(),
  tests=dict(existing=229,new_candidate_tests=4,new_numeric_tests=4,total=237,adapters_medium=6,adapters_neutral=6,small_training_cells=8,independent_small_audit=True,accumulator_repair_fault_test=True),
  preliminary_gate=dict(offline_audit_sha256=sha(OUT/'independent_offline_audit.json'),small_audit_sha256=sha(OUT/'small_gate_audit.json'),small_prereg_sha256=sha(OUT/'small_gates/small_preregistration.json')),
  prior_failure='Parent formal batch stopped after15 completed; M3CPU at t960 secondary HiGHS vector violated original primary band. All original data preserved. Numeric reformulation uses reversible power-of-two rows with identical objective and options; strict original-row checks mandatory. All30 rerun, no mixed versions.',
  analysis='Original old-grid42 remains external historical paired-input comparator. New six-method comparisons primary within same new candidate version. First assess CPU/MEM/FULL versus P and FULL versus CPU, plus neutral/higherCPU controls. No retuning if negative.',
  limits='Only30 declared cells; three nested mixes from one source are not independent traces. Small gates positive but effects modest; no guaranteed full-cohort benefit. Boundary clipping reported per policy. All source support gaps retained, not phi=1.',
  contracts_hash=sha(OUT/'adapter_acceptance.json'),audit_script_hash=sha(ROOT/'maintenance/audit_breakpoint30_numeric_v2_20261029.py'))
 with (OUT/'preregistration.json').open('x',encoding='utf8') as f:json.dump(pre,f,indent=2,ensure_ascii=False)
 shutil.make_archive(str(OUT/'source_snapshot'),'zip',SRC)
 shutil.copyfile(target,OUT/target.name)
 with (OUT/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nFROZEN30_READY: gates passed; main preregistration.json now authoritative. Execute maintenance/run_breakpoint30_numeric_v2_20261029.py batch. Completed prefix hashes required before resume; never overwrite partial cells. Do not change running source/parameters. Only30 new cells, no104/160. After raw-event audit/report STOP.\n')
 print('FROZEN30',len(cells),len(source),sha(OUT/'preregistration.json'))
if __name__=='__main__':main()
