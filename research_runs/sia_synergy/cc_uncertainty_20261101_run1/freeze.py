import difflib,hashlib,importlib.metadata,json,re,shutil,sys,zipfile
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-uncertainty-v4';OLD=R/'cc-joint-policy-rollout-v3';P=R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,x):
 with (O/n).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def main():
 log=(O/'full_suite_final.log').read_text(encoding='utf8');assert '\nOK\n' in log and '\nFAILED (' not in log
 assert 'Ran 385 tests' in log
 v=json.loads((O/'state_validation_final.json').read_text());assert v['ranking_gate'] and v['technical_gate']
 assert not json.loads((O/'calibration_validation.json').read_text())['errors']
 assert json.loads((O/'original_sia_regression.json').read_text())['exact']
 assert json.loads((O/'original_full_regression.json').read_text())['exact']
 h={p.relative_to(T).as_posix():sha(p) for p in T.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 base=json.loads((O/'baseline_hashes.json').read_text());assert all(sha(OLD/n)==v for n,v in base.items())
 changed=[n for n in base if h.get(n)!=base[n]];new=sorted(set(h)-set(base));patch=[]
 for n in changed+new:
  before=(OLD/n).read_text(encoding='utf8').splitlines(True) if n in base else []
  patch.extend(difflib.unified_diff(before,(T/n).read_text(encoding='utf8').splitlines(True),fromfile='a/'+n,tofile='b/'+n))
 (O/'source_changes.patch').write_text(''.join(patch),encoding='utf8')
 with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(h):z.write(T/n,n)
 save('source_hashes.json',h);save('source_receipt.json',dict(changed=changed,new=new,files=len(h),old_source_unchanged=True,snapshot_sha256=sha(O/'source_snapshot.zip'),git='Complete actual source copy; no fabricated commit'))
 (O/'inputs').mkdir();shutil.copy2(P/'inputs/workload.csv',O/'inputs/workload.csv')
 inputs={str(p):sha(p) for p in (T/'inputs').rglob('*') if p.is_file()};inputs[str(O/'inputs/workload.csv')]=sha(O/'inputs/workload.csv')
 env=dict(python=sys.version,executable=sys.executable,packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','cvxpy','cylp','psutil']})
 save('environment.json',env);save('input_hashes.json',inputs)
 save('preregistration.json',dict(version='PREDECISION_UNCERTAINTY_V4',source=str(T),source_hashes=h,input_hashes=inputs,environment=env,
  topology={'aws':2,'dgx-ext':2},capacity={'aws':{'cpu':6,'ram':125},'dgx-ext':{'cpu':6,'ram':125}},
  cells=[dict(id=label,method=m,new_evaluation=new) for label,m,new in [('OLD_FO','FO',False),('OLD_FULL','FULL',False),('NEW_FO','FO',True),('NEW_FULL','FULL',True)]],
  workload=str(O/'inputs/workload.csv'),count=12,response='R0_E_LEGACY_MEDIUM',alpha=.5,seed_profiles=[1,8],interval=60,max_rounds=6000,
  memory_gib=2,disk_gib=5,solver={'time_limit':30,'mip_rel_gap':0},
  uncertainty=dict(strata='application/config/atomic/accum/epoch',history='ALL_THIS_RUN_PREDECISION_FIRST_REAL_SEGMENTS',support='TWO_DISTINCT_OBSERVED_INTERVAL_ORIGINS',range='MIN_MAX_OF_OBSERVED_RATIOS_UNION_NOMINAL_ONE_ONLY_IF_SUPPORTED',stress='PAIRED_ALL_ENDPOINT_COMBINATIONS_WITH_EARLY_REJECTION',unknown='KEEP_ORIGINAL',confidence_guarantee=False),
  success='Complete12/12 only; no tuning or expansion.',science_label='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
 save('execution_receipt.json',{n:sha(O/n) for n in ('DESIGN.md','run_small.py','pipeline_small.py','raw_audit_lib.py','preregistration.json')})
 print('FROZEN',len(h),sha(O/'preregistration.json'))
if __name__=='__main__':main()
