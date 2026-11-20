"""Freeze an independent engineering rerun; no simulation or data changes."""
import difflib,hashlib,importlib.metadata,json,os,platform,sys,time,zipfile
from pathlib import Path
import psutil
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-repair';OLD=R/'research_runs/sia_synergy/cc_core80_8h_20261030_run3';O=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 prior=json.loads((OLD/'preregistration.json').read_text());baseline=json.loads((O/'baseline_source_hashes.json').read_text())
 if 'files' in baseline:baseline=baseline['files']
 files=sorted(p for p in T.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts)
 hashes={p.relative_to(T).as_posix():sha(p) for p in files}
 inputs={}
 for name,h in prior['input_hashes'].items():
  p=T/Path(name).relative_to(Path(prior['source_root']));assert sha(p)==h,('INPUT_CHANGED',p);inputs[str(p)]=h
 for name,h in prior['source_hashes'].items():
  if name.startswith('sia-simulator/traces/') or name.startswith('inputs/'):assert hashes[name]==h,name
 changes=[n for n,h in hashes.items() if n not in baseline or baseline[n]!=h]
 save(O/'source_delta.json',dict(changed_or_added=changes,removed=sorted(set(baseline)-set(hashes)),unchanged_count=sum(hashes.get(n)==h for n,h in baseline.items()),input_files_verified=len(inputs)))
 assert not(set(baseline)-set(hashes))
 with (O/'source_changes.patch').open('x',encoding='utf8') as f:
  for n in changes:
   a=(R/'cc-joint-core80-8h-v3'/n);b=T/n
   try:f.writelines(difflib.unified_diff(a.read_text(encoding='utf8').splitlines(True) if a.exists() else [],b.read_text(encoding='utf8').splitlines(True),fromfile='baseline/'+n,tofile='repaired/'+n))
   except UnicodeDecodeError:f.write('\nBINARY ADDITION '+n+' SHA256 '+hashes[n]+'\n')
 with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.relative_to(T).as_posix())
 save(O/'source_hashes.json',hashes)
 # Every technical run and complete suite must precede the freeze.
 suite=(O/'tests/full_suite.log').read_text(encoding='utf8');assert 'Ran 289 tests' in suite and '\nOK\n' in suite
 assert all(x['exact_semantic_match'] for x in json.loads((O/'tests/historical_prefix_bridges.json').read_text()))
 assert json.loads((O/'tests/original_sia_exact_regression.json').read_text())['exact_match']
 active=[(p.pid,p.info['cmdline']) for p in psutil.process_iter(['cmdline']) if p.pid!=os.getpid() and p.info['cmdline'] and any('run_core80.py' in v or 'supervise80.py' in v for v in p.info['cmdline'])]
 assert not active,active
 env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in prior['environment']['packages']},solver=prior['environment']['solver'],PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'))
 assert env['packages']==prior['environment']['packages']
 methods=['FULL','A10','PROP','CPU','MEM','FO','SIA_P','SIA_AUX','A00','A01','A11']
 by={c['method']:c for c in prior['cells'] if not c['optional']};cells=[by[n] for n in methods];now=time.time()
 prereg={**prior,'version':'CORE80_ENGINEERING_REPAIR_CANONICAL_FIRST_EXACT_PRESOLVE','output':str(O),'source_root':str(T),'source_hashes':hashes,'input_hashes':inputs,'cells':cells,'environment':env,
  'frozen_unix':now,'formal_stop_unix':now+24*3600,'hard_deadline_unix':now+25*3600,'cell_wall_limit_seconds':6*3600,
  'source_snapshot_sha256':sha(O/'source_snapshot.zip'),'baseline_source_snapshot_sha256':sha(OLD/'frozen_source_snapshot.zip'),
  'execution':'New repair authorization after the previous 8h task closed. Serial independent processes,2GiB/5GiB before launch. Common maximum program compute6h/cell and24h batch; these are engineering watchdogs, not simulated training horizon or MIP parameter changes. No optional4. The order prioritizes previously failed/incomplete methods, fixed before outcomes.',
  'timeout':'Preserve COMPUTE_TIMEOUT distinct from6000 simulated rounds; no full mean for partial jobs. No extension based on JCT.',
  'failure':'Preserve exact failed model/vector/messages and continue registered cells. No unverified incumbent promoted; at most one fixed original no-presolve30sec/gap0 retry. No scientific tuning.',
  'numerical_acceptance':'Existing canonical binary1e-6 and original narrow-row1e-10/general-row1e-7 checks before retry; no relaxed utility band.',
  'computation_equivalence':'Full response grid/normalizer retained. Materialize only already factor-eligible columns; exact same-geometry dominance and zero-optional startup proofs substitute variables only; exact jobwise lower-bound certificates.',
  'optional':'Not authorized for this repair run; removed all4 optional cells.',
  'partial':'No comparisons of complete JCT means until both cells ALL_COMPLETE. Old results preserved as regression references; new full matrix uses this single repaired source.'}
 save(O/'preregistration.json',prereg)
 save(O/'freeze_receipt.json',dict(unix=now,source_files=len(hashes),source_snapshot_sha256=sha(O/'source_snapshot.zip'),preregistration_sha256=sha(O/'preregistration.json'),resources=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(O)).free/2**30)))
 print(json.dumps(json.loads((O/'freeze_receipt.json').read_text())))
if __name__=='__main__':main()
