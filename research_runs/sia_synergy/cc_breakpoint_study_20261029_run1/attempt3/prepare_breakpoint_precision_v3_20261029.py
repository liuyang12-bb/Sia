import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';OLD=BASE/'attempt2';OUT=BASE/'attempt3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 proof=json.loads((OLD/'bounded_failure_diagnosis/primary_precision_proof.json').read_text());assert proof['valid']
 parent={p.relative_to(OLD/'source').as_posix():sha(p) for p in (OLD/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 with (OLD/'source_hashes_at_stop.json').open('x',encoding='utf8') as f:json.dump(parent,f,indent=2)
 shutil.make_archive(str(OLD/'source_snapshot_at_stop'),'zip',OLD/'source')
 OUT.mkdir(exist_ok=False);shutil.copytree(OLD/'source',OUT/'source');shutil.copytree(OLD/'inputs',OUT/'inputs')
 for n in ['rules.json','offline_coverage.json','offline_validation.json','offline_pair_census.json','mechanism_witnesses.json','independent_offline_audit.json','accumulator_red_green.json']:shutil.copyfile(OLD/n,OUT/n)
 with (OUT/'parent_source_hashes.json').open('x',encoding='utf8') as f:json.dump(parent,f,indent=2)
 shutil.copyfile(OLD/'bounded_failure_diagnosis/primary_precision_proof.json',OUT/'primary_precision_proof.json')
 (OUT/'CURRENT_TASK.md').write_text('ACTIVE precision-only repair attempt3. Parent formal15 completed +one failed, attempt2 only bounded gates failed; all retained. No JCT-driven tuning. Same data-knot grid, targets, response, capacities, workloads,60/6000 and solver options. Add common reversible objective units derived from existing primary1e-9 band, plus prior reversible constraint bands. Re-test all241 tests, true M3 16-round path, all8 small, six adapters/Neutral; only then fresh all30 no mixed source versions. If exact original constraints/candidate/checkpoint audit fail, stop; never relax. No104/160 or parameter search. Read prereg/execution manifest before recovery.\n',encoding='utf8')
 test=OLD/'source/sia-simulator/tests/test_breakpoint_numeric_conditioning.py'
 # Generate only isolated wrappers, never alter old wrappers or artifacts.
 names=['test_breakpoint_numeric_v2_repeat_20261029.py','bounded_breakpoint_numeric_v2_20261029.py','accept_breakpoint_numeric_v2_20261029.py','neutral_breakpoint_numeric_v2_20261029.py','run_breakpoint_small_numeric_v2_20261029.py','audit_breakpoint_small_numeric_v2_20261029.py','freeze_breakpoint30_numeric_v2_20261029.py','audit_breakpoint30_numeric_v2_20261029.py','describe_breakpoint30_numeric_v2_20261029.py','accept_and_run_breakpoint_numeric_v2_20261029.py']
 for name in names:
  text=(ROOT/'maintenance'/name).read_text(encoding='utf8').replace(BASE.name+'/attempt2',BASE.name+'/attempt3')
  text=text.replace('numeric_v2','precision_v3').replace('cc-joint-breakpoint-conditioned-r2','cc-joint-breakpoint-precision')
  text=text.replace('all_tests2.log','all_tests.log').replace('tests_result2.json','tests_result.json')
  text=text.replace('new_numeric_tests=4,total=237','new_numeric_tests=4,new_objective_unit_tests=4,total=241')
  text=text.replace('237项测试（229原有+4候选+4数值保护）','241项测试（229原有+4候选+8数值保护）')
  if name.startswith('bounded_'):
   text=text.replace('conditioning=0;start=', 'conditioning=0;unit_calls=0;start=')
   text=text.replace("     assert record['unscaled_original_rows_validated'];conditioning+=1", "     assert record['unscaled_original_rows_validated'];conditioning+=1")
   text=text.replace("   f.write(json.dumps(r,default=enc,separators=(',',':'))+'\\n')", "   unit_calls+=sum(v['scale']>1 for v in d['result']['audit'].get('objective_unit_certifications',[]))\n   f.write(json.dumps(r,default=enc,separators=(',',':'))+'\\n')")
   text=text.replace(" assert conditioning>0,'RECORDED_FAILURE_REPAIR_PATH_NOT_EXERCISED'", " assert unit_calls>0,'OBJECTIVE_UNIT_REPAIR_NOT_EXERCISED'")
   text=text.replace('conditioning_recoveries=conditioning,all_original_rows_valid=True','conditioning_recoveries=conditioning,objective_unit_calls=unit_calls,all_original_rows_valid=True')
  new=name.replace('numeric_v2','precision_v3')
  with (ROOT/'maintenance'/new).open('x',encoding='utf8') as f:f.write(text)
 print('PREPARED_PRECISION_V3',len(parent))
if __name__=='__main__':main()
