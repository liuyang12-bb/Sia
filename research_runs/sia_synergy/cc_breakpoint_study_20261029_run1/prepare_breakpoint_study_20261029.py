"""Prepare a versioned, gated candidate-only experiment; never starts training."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1'
OLD=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 OUT.mkdir(exist_ok=False)
 old=json.loads((OLD/'preregistration.json').read_text())
 for n,h in old['source_hashes'].items():assert sha(OLD/'source'/n)==h,n
 shutil.copytree(OLD/'source',OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copytree(OLD/'inputs',OUT/'inputs')
 (OUT/'old_source_hashes.json').write_text(json.dumps(old['source_hashes'],indent=2))
 (OUT/'CURRENT_TASK.md').write_text('AUTHORIZED candidate-breakpoint-only study. Preserve CC-TAIL objective/admission/restart, R0/E-legacy alpha, mapping, workload, capacities. No104/160/9cohort runs. First offline coverage + red/green tests, then independent fixed-set and small full-completion mechanism gates. Only if CPU and MEM mechanism gates pass, freeze and run30 cells: M1/M2/M3 R11 medium six methods18; M2 R01 medium6; M2 R11 neutral6. Failed performance gate is a result: stop, no target/response tuning, no new cohort search. Memory2GiB/disk5GiB per process. All outputs new, never overwrite. Recovery: read rules.json, stage_status.json and logs.\n',encoding='utf8')
 rules=dict(created=time.time(),change='CANDIDATE_REPRESENTATION_ONLY',response_changed=False,objective_changed=False,
  source_parent_sha256=sha(OLD/'preregistration.json'),
  knots='Retain original points/proportional. Add inverse-mapped CPU/RAM source-matrix slope-change knots and endpoints (phi*source_axis). At each node also add capacity complement of one registered peer knot for every compatible single-node GPU configuration, with legal GPU counts; no future arrivals/performance read. Fixed full registry determines peers.',
  eligibility='Same original legal CPU/RAM floors, capacities, quota-axis ablations and checkpoint locks. Expanded grid eligibility independently rechecked. No phi=1, no dropped GPU configurations, no response edits.',
  evidence='Separate strict supported numerator+denominator witnesses from E-legacy boundary-exposed candidates. No use of boundary-exposed donor as strict evidence.',
  mechanism_selection='For each CPU/MEM factor, scan official row order job pairs, aws before dgx-ext, R11 before R01; require two distinct apps, single GPU each, strictly supported baseline+new points, sum of constant-current-rate remaining ETAs lower than proportional and net normalized rate higher. Select first qualifying state, not largest benefit. Fixed complete declared work; no JCT-result selection.',
  fixed_set='Compare proportional versus offline enumerated opportunity holding GPU/batch/service set; actual Job.step to completion with no future rate supplied to scheduler. Check donor and receiver.',
  small_gate='For each CPU and MEM first witness run current CC-P and candidate-only corresponding CC axis to full completion; 6000-round fixed upper bound, no extension. Require all complete, capacity/lock/rho tests, selected non-proportional quota and lower mean JCT versus P; failure stops formal matrix, all negative results retained.',
  formal_cells=[(x,'R11','medium',m) for x in ['M1','M2','M3'] for m in ['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']]+[('M2',s,r,m) for s,r in [('R01','medium'),('R11','neutral')] for m in ['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']],
  interval=60,max_rounds=6000,resources=dict(memory_gib=2,disk_gib=5),
  warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
 (OUT/'rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=2),encoding='utf8')
 print(OUT)
if __name__=='__main__':main()
