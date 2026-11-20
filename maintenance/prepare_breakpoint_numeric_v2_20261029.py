"""Preserve failed run1; prepare mathematically equivalent numeric repair."""
import gzip,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';OUT=BASE/'attempt2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 OUT.mkdir(exist_ok=False)
 old=json.loads((BASE/'preregistration.json').read_text())
 for n,h in old['source_hashes'].items():assert sha(BASE/'source'/n)==h,n
 shutil.copytree(BASE/'source',OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copytree(BASE/'inputs',OUT/'inputs')
 for n in ['rules.json','offline_coverage.json','offline_validation.json','offline_pair_census.json','mechanism_witnesses.json','independent_offline_audit.json','accumulator_red_green.json']:
  shutil.copyfile(BASE/n,OUT/n)
 (OUT/'old_source_hashes.json').write_text(json.dumps(old['source_hashes'],indent=2))
 shutil.copyfile(BASE/'numeric_diagnosis/solver_diagnostics.json.gz',OUT/'source/sia-simulator/tests/fixtures/breakpoint_primary_band_failure.json.gz')
 (OUT/'CURRENT_TASK.md').write_text('NUMERIC_REPAIR_ONLY after original30 stopped15 complete +M3CPU failed at t960. Parent evidence immutable. Candidate grid/target/physical response/workload/capacities/solver options unchanged. Repair must use reversible power-of-two conditioning and validate all ORIGINAL constraints. No tolerance relaxation, bonus, target change or JCT tuning. Red/green numeric fixture tests +all old tests +bounded16-round same failure gate +repeat8 small +neutral gate BEFORE fresh all30 in this directory; no mixing original15. Read preregistration.json/execution_manifest before any recovery. Never resume104/160, no Low/High. Stop/report if exact numerical recovery cannot be proved.\n',encoding='utf8')
 (OUT/'repair_scope.json').write_text(json.dumps(dict(parent_prereg_hash=sha(BASE/'preregistration.json'),original_complete_cells=15,failed_cell='M3_R11_medium_CC_CPU',failure_time=960,scientific_parameters_changed=False,
  root_cause='HiGHS status0 secondary vector violates primary band by2.6655709839396025e-7; both unconditioned and existing presolve-off retry fail unscaled strict audit.',
  repair='Check every integerized result against all original rows. Only on numeric row failure, power-of-two condition finite nonzero narrow bands to width>=1, unchanged objective and existing retry options; independent original-row audit mandatory. If unsuccessful stop, never accept relaxed solution.',
  rerun='Complete same30 cells after tests; all first attempt data remain diagnostic and historical, never mixed as primary.'),ensure_ascii=False,indent=2),encoding='utf8')
 print(OUT)
if __name__=='__main__':main()
