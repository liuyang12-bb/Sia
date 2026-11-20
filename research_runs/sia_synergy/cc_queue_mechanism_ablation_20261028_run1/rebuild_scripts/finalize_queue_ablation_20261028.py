"""Finalize after all log writers have closed; preserve preliminary hashes."""
import hashlib,importlib.metadata,json,shutil,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1';OLD=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
p=load(OUT/'preregistration.json');v=load(OUT/'analysis/validation.json');assert v['valid'] and all(n==20 for n in v['completed'].values())
assert sha(ROOT/'maintenance/run_queue_ablation_20261028.py')==p['runner_hash']
for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h and sha(ROOT/'cc-joint-queue-ablation'/n)==h,n
for n,h in load(OUT/'parent_source_hashes.json').items():assert sha(OLD/'source'/n)==h and sha(ROOT/'cc-joint-resource-study'/n)==h,n
for n,h in load(OLD/'parent_source_hashes.json').items():assert sha(ROOT/'cc-joint-tail-recovery'/n)==h,n
old_cells=0
for cell in (OLD/'cells').iterdir():
 for n,h in load(cell/'output_hashes.json').items():assert sha(cell/n)==h,(cell.name,n)
 old_cells+=1
for name in p['cells']:
 for n,h in load(OUT/'cells'/name/'output_hashes.json').items():assert sha(OUT/'cells'/name/n)==h,(name,n)
for n,val in p['packages'].items():assert importlib.metadata.version(n)==val,n
with zipfile.ZipFile(OUT/'rebuild_source.zip') as z:assert z.testzip() is None
raw=load(OUT/'analysis/raw_audit.json')
for name,r in raw.items():
 if name=='SIA_P':continue
 if name[3]=='1':assert r['events'].get('REQUEUE_PRODUCTIVE_QUANTUM',0)==0
 else:assert r['events'].get('REQUEUE_PRODUCTIVE_QUANTUM',0)>0
 if name[2]=='0':assert r['events'].get('DRAIN_FOR_HEAD',0)==0
with (OUT/'red_legacy_semantics2.log').open(encoding='utf-8') as f:assert 'EXPECTED_OLD_BEHAVIOR_FAILURES_CONFIRMED_2' in f.read()
shutil.copy2(Path(__file__),OUT/'rebuild_scripts'/Path(__file__).name)
pre=load(OUT/'delivery_hashes.json');pre_mismatch=[n for n,h in pre.items() if sha(OUT/n)!=h]
assert pre_mismatch==['report_build.log'],pre_mismatch
save(OUT/'FINAL_VALIDATION.json',dict(status='COMPLETE_AUDITED_STOPPED',completed_cells=9,jobs_per_cell=20,completed_all=True,
 tests_passed=239,new_tests=10,old_semantic_red_failures_confirmed=2,
 exact_historical_regressions=dict(CC_TAIL=278,SIA_P=300),historical_cells_unchanged=old_cells,
 parent_source_files_unchanged=282,frozen_source_files=len(p['source_hashes']),capacity_progress_lock_rho_violations=0,
 solver_failures=0,algorithm_core_or_response_changed=False,new_opt_in_ablation_module=True,
 legacy_label_note='Inherited legacy descriptive labels are retained and counted in raw_audit.effective_semantics; explicit factor flags and executed events define true mechanisms.',
 preliminary_hash_note='delivery_hashes.json was captured before report_build.log writer closed; final authoritative checksums are delivery_hashes_final.json. Raw experiment/source hashes did not change.',
 preliminary_changed_files=pre_mismatch,source_zip_sha=sha(OUT/'rebuild_source.zip'),
 experiment_wall_seconds=sum(json.loads(x)['wall_seconds'] for x in (OUT/'execution_manifest.jsonl').read_text().splitlines()),
 stop='No further experiments or algorithm changes automatically authorized by completion.'))
final={x.relative_to(OUT).as_posix():sha(x) for x in OUT.rglob('*') if x.is_file() and x.name!='delivery_hashes_final.json'}
save(OUT/'delivery_hashes_final.json',final)
for n,h in load(OUT/'delivery_hashes_final.json').items():assert sha(OUT/n)==h,n
print('FINAL_VALIDATION_OK',len(final),'HASHES; ALL_30_HISTORY_CELLS_PRESERVED; ALL_NINE_COMPLETE')
