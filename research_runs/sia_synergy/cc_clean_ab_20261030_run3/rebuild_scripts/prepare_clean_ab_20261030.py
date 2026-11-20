import hashlib,json,shutil,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run1'
BASE=ROOT/'cc-joint-ample-lex-repair'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def hashes(p):return {f.relative_to(p).as_posix():sha(f) for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
def main():
 OUT.mkdir(exist_ok=False)
 shutil.copytree(BASE,OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
 old=hashes(BASE);new=hashes(OUT/'source');assert old==new
 (OUT/'baseline_source_hashes.json').write_text(json.dumps(old,indent=2),encoding='utf8')
 (OUT/'inputs').mkdir();(OUT/'cells').mkdir();(OUT/'tests').mkdir()
 for tag in ['COMPACT','SPREAD']:
  shutil.copy2(ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1/inputs'/(tag+'_workload.csv'),OUT/'inputs'/(tag+'_workload.csv'))
 shutil.copy2(ROOT/'research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1/source/inputs/workload.csv',OUT/'inputs/A20_workload.csv')
 (OUT/'CURRENT_TASK.md').write_text('''# Current authorized task: clean A/B ablations and C offline precheck

This new source snapshot starts byte-identical to accepted cc-joint-ample-lex-repair (289 files). Historical directories, official sources and old algorithms are immutable. New adapters only; no change to historical score/grid/response/capacity or workload. No 104/160-job runs.

A: one frozen20-job cohort, Neutral, proportional, 2x2 ordered admission x continuous service + external actual Sia. Shared Sia fair score and original cold-start utility (not new completion-score bootstrap), shared eligibility for every previously started job whether held or requeued, lifetime delay budget and actual locks. Quantized300 productive-second expiry has non-destructive resume-budget safety. No individual starvation guarantee for optional admission.
B: existing COMPACT and SPREAD12-job cohorts, Medium, 2x2 serial GPU choice x proportional/optimized auxiliary quotas + external Sia. Shared reciprocal ETA score, FIFO/continuity and eligibility. Serial proposal is a common-controller proportional-column MIP, NOT original Sia. It freezes selected jobs and GPU config; placement remains optimizable in auxiliary stage. Both passes use same utility, resource budgets and locks. A change in placement remains part of optimization, not isolated separately.
C: offline full app/config/quota enumeration for A/B, CPU/RAM numerator and denominator bounds separately; sensitivity classifications and non-flat donor opportunities. E-legacy remains unchanged. No new boundary response enters execution.

Before formal runs: red/green contracts, complete old tests, bounded original Sia regression, offline precheck and preregistration hashes. All formal cells60s/6000rounds, separate processes,2GiB memory/5GiB disk gate. No result-driven tuning. Censoring preserved, JCT means only complete pairs. Failed attempts preserved. Resume entry maintenance/run_clean_ab_20261030.py (to be installed/frozen after tests).
''',encoding='utf8')
 (OUT/'DESIGN.md').write_text('''# Clean two-line attribution, preregistration design

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

## Hypotheses fixed before any new training run
A ordered admission may improve or worsen waiting; continuity may reduce requeue/checkpoint but delay first service. Neither guaranteed beneficial. Sia fair scoring is held fixed to remove the previous unknown-ETA score confound. Completion-priority score is not a factor in A. Keep config-event/profile/delay filtering identical for held and requeued histories, so expiry alone does not relax geometry eligibility.
B joint selection may exploit resource complementarity; serial adaptation may be equally good or better. Shared objective reciprocal current ETA; unknown ETA remains common proportional bootstrap with zero utility; no bonus. Serial vs joint differ only feasible search structure. Serial first solves the proportional subset, freezes GPU type/count/node-count and selected service set; second pass can optimize quotas/placement in fixed configs. The serial baseline is explicitly new, not historical B1/B4 or official Sia.

## Formal matrix fixed
A20: A00,A01,A10,A11,A_SIA. First digit ordered FIFO, second continuous service; CPU6 RAM187.5 per2GPU node, AWS/DGX, Neutral.
COMPACT and SPREAD: B_FP,B_FO,B_JP,B_JO,B_SIA each; CPU6 RAM125 per2GPU node, Medium alpha0.5. Total15 cells, in that exact order (A first, COMPACT then SPREAD).
All fixed real Artifact job records/full declared work, input order and controlled arrival preserved; seeded profiles(1,8). No new workload search.60s,6000rounds. No automatic extension for censoring.

## Shared safety
Full checkpoint equalities; current-round oracle, capacity and candidate audit; lifetime startup250s carried across requeue. One existing tail recovery mechanism, if offered, must obey same history/conditions independent of service factor. On solver undecided or invalid lock stop, no partial fallback. Quantized expiry suppressed when no safe reissue budget; count it separately. A no-FIFO comparator is not universally starvation-free; finite-window completion is empirical.

## Analysis
Unit=cohort, deterministic factor contrasts not independent job replicates. Mean/median/P95/P99/makespan, jobwise differences and censoring; wait split initial/requeue, productive/checkpoint, actual resource hours/restarts, solver/delay and clipping. A marginal/conditional effects and interaction in seconds. B aux effects in each GPU mode and joint effects in each auxiliary mode; interaction=(JO-JP)-(FO-FP), negative indicates extra joint benefit on JCT scale. No adding percentages. Keep all jobs; leave-largest-beneficiary-out diagnostic only. B same-state calls preserve first-pass intent; actual Sia separate external reference.

## Gate
No training before tests and hash freeze. Offline gaps do not get phi1, deletion or altered clipping; formal legacy runs may proceed only with explicit fixed synthetic-boundary disclaimer and full coverage report. No new real-hardware claim. Test or audit failures retain raw evidence, fix correctness only and rerun complete affected pairs under uniform source version.
''',encoding='utf8')
 print(OUT);print('BASE_IDENTICAL',len(old))
if __name__=='__main__':main()
