"""Create isolated wrappers; no old artifacts/scripts are edited."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE='cc_breakpoint_study_20261029_run1';OUT=ROOT/'research_runs/sia_synergy'/BASE/'attempt2'
def main():
 pairs={
  'test_breakpoint_study_20261029.py':'test_breakpoint_numeric_v2_20261029.py',
  'accept_breakpoint_study_20261029.py':'accept_breakpoint_numeric_v2_20261029.py',
  'neutral_breakpoint_acceptance_20261029.py':'neutral_breakpoint_numeric_v2_20261029.py',
  'run_breakpoint_mechanisms_20261029_v2.py':'run_breakpoint_small_numeric_v2_20261029.py',
  'audit_breakpoint_mechanisms_20261029.py':'audit_breakpoint_small_numeric_v2_20261029.py',
  'freeze_breakpoint30_20261029.py':'freeze_breakpoint30_numeric_v2_20261029.py',
  'audit_breakpoint30_20261029.py':'audit_breakpoint30_numeric_v2_20261029.py',
  'describe_breakpoint30_20261029.py':'describe_breakpoint30_numeric_v2_20261029.py'}
 for old,new in pairs.items():
  text=(ROOT/'maintenance'/old).read_text(encoding='utf8').replace(BASE,BASE+'/attempt2')
  text=text.replace('cc-joint-breakpoint-study','cc-joint-breakpoint-conditioned')
  text=text.replace('small_attempt2','small_gates')
  if new.startswith('freeze_'):
   text=text.replace('run_breakpoint30_20261029.py','run_breakpoint30_numeric_v2_20261029.py')
   text=text.replace('audit_breakpoint30_20261029.py','audit_breakpoint30_numeric_v2_20261029.py')
   text=text.replace('new_candidate_tests=4,total=233','new_candidate_tests=4,new_numeric_tests=4,total=237')
   text=text.replace("prior_failure='Original small attempt preserved: obsolete legacy accumulator progress-floor assumption; runner adapter corrected to accepted42 caller-side audit contract, no source/response/target change. All8 small runs rerun in small_gates.'", "prior_failure='Parent formal batch stopped after15 completed; M3CPU at t960 secondary HiGHS vector violated original primary band. All original data preserved. Numeric reformulation uses reversible power-of-two rows with identical objective and options; strict original-row checks mandatory. All30 rerun, no mixed versions.'")
  if new.startswith('audit_breakpoint30_'):
   text=text.replace('233项测试（229原有+4候选）','237项测试（229原有+4候选+4数值保护）')
  with (ROOT/'maintenance'/new).open('x',encoding='utf8') as f:f.write(text)
 print(json.dumps(pairs,indent=2))
if __name__=='__main__':main()
