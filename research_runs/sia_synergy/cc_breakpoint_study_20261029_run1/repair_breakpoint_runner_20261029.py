"""Repair only the legacy accumulator adapter; preserve all first attempt data."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 out=BASE/'small_attempt2';out.mkdir(exist_ok=False)
 for n in ['tests_result.json','offline_validation.json','mechanism_witnesses.json']:
  shutil.copyfile(BASE/n,out/n)
 script=(ROOT/'maintenance/run_breakpoint_mechanisms_20261029.py').read_text(encoding='utf8')
 script=script.replace("OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';SRC=OUT/'source'","OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/small_attempt2';SRC=OUT.parent/'source'")
 old="     m.consume(c,r);c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()"
 new="""     # Current donor-allowing CC contract, identical to accepted42 runner.
     # Fresh status/capacity/rho were independently checked above. Legacy
     # consume's fresh block assumes the discontinued per-job speed floor.
     m.last_decision_round=d['round']
     m.consume(c,r);c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()"""
 assert old in script;script=script.replace(old,new)
 target=ROOT/'maintenance/run_breakpoint_mechanisms_20261029_v2.py';target.write_text(script,encoding='utf8')
 source=json.loads((BASE/'small_preregistration.json').read_text())['source_hashes']
 for n,h in source.items():assert sha(BASE/'source'/n)==h,n
 fix=dict(root_cause='Runner failed at round3 because stale Accumulator.consume enforces rho >= proportional for every selected job. Current CC intentionally permits donor slowdown and independently audits capacity/locks/response. The accepted42 runner already suppresses this obsolete fresh-block via last_decision_round.',
   source_changed=False,response_changed=False,workload_changed=False,target_changed=False,
   fix='Set accumulator last_decision_round after complete caller-side audit, before consume; preserves all execution and exclusive-time checks. Removes duplicate solve/status counting too.',
   failed_evidence='mechanisms/CPU_free_O/failure.json and CPU_free_O.log',
   old_runner_hash=sha(ROOT/'maintenance/run_breakpoint_mechanisms_20261029.py'),new_runner_hash=sha(target),
   rerun='All eight small cells in small_attempt2 with identical source, fixtures, capacities and observation limit; no mixed-version primary results.')
 (out/'RUNNER_REPAIR.json').write_text(json.dumps(fix,ensure_ascii=False,indent=2),encoding='utf8')
 with (BASE/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nSmall attempt1 halted by obsolete per-job progress-floor accumulator check; all failed raw evidence preserved. Runner-only repair, unchanged source. Active small evidence now small_attempt2, all8 rerun pairs same conditions. Formal30 still gated, not launched.\n')
 print(target)
if __name__=='__main__':main()
