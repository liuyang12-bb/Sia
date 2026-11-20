import difflib,hashlib,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def run(script,log):
 with (OUT/log).open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'maintenance'/script)],stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'ANALYSIS_FAILURE:{script}:{r.returncode}')
def main():
 records=[json.loads(x) for x in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
 assert len(records)==15 and all(r['exit_code']==0 for r in records),'MATRIX_NOT_COMPLETE'
 base=json.loads((OUT/'baseline_source_hashes.json').read_text());src=OUT/'frozen_source';allh={f.relative_to(src).as_posix():sha(f) for f in src.rglob('*') if f.is_file()}
 assert all(allh[n]==h for n,h in base.items());added={n:h for n,h in allh.items() if n not in base}
 save('source_changes_final.json',dict(base_files_unchanged=len(base),modified=[],added=added))
 patch=[]
 for n in added:patch.extend(difflib.unified_diff([],(src/n).read_text(encoding='utf8').splitlines(True),fromfile='/dev/null',tofile=n))
 (OUT/'source_additions_final.patch').write_text(''.join(patch),encoding='utf8')
 run('audit_clean_ab_r3_20261030.py','independent_audit.log')
 run('response_clean_ab_r3_20261030.py','response_realization.log')
 run('concurrency_clean_ab_20261030.py','service_concurrency.log')
 run('bridge_clean_ab_20261030.py','historical_trajectory_bridge.log')
 run('report_clean_ab_r3_20261030.py','report_build.log')
 # Preserve red sources/evidence without rewriting either interrupted attempt.
 (OUT/'tests/references').mkdir()
 for dirname,names in [('cc_clean_ab_20261030_run1',['red_missing.log','red_old_membership.log']),('cc_clean_ab_20261030_run2',['red_old_universe.log','red_zero_startup_history.log'])]:
  for name in names:
   f=OUT.parent/dirname/'tests'/name
   if f.exists():shutil.copy2(f,OUT/'tests/references'/(dirname+'_'+name))
 with zipfile.ZipFile(OUT/'tests/red_zero_startup_source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in added:z.write(ROOT/'cc-joint-clean-ab-r2'/n,n)
 (OUT/'rebuild_scripts').mkdir()
 for n in ['prepare_clean_ab_20261030.py','implement_clean_ab_20261030.py','run_clean_ab_r3_20261030.py','audit_clean_ab_r3_20261030.py','precheck_clean_c_20261030.py','report_clean_c_20261030.py','response_clean_ab_r3_20261030.py','concurrency_clean_ab_20261030.py','bridge_clean_ab_20261030.py','report_clean_ab_r3_20261030.py','finalize_clean_ab_20261030.py','replay_clean_ab_20261030.py']:
  shutil.copy2(ROOT/'maintenance'/n,OUT/'rebuild_scripts'/n)
 prior_checks={}
 for directory in ['cc_arrival_pressure18_20261030_run1','cc_ample_lex_repair_20261030_run1']:
  seal=OUT.parent/directory/'delivery_hashes_final.json';data=json.loads(seal.read_text())
  # Existing seals use either direct path/hash map or a named files object.
  rows=data.get('files',data)
  count=0
  for n,h in rows.items():
   if not isinstance(h,str) or len(h)!=64:continue
   file=OUT.parent/directory/n
   if file.exists():assert sha(file)==h,(directory,n);count+=1
  prior_checks[directory]=dict(seal_sha256=sha(seal),verified_files=count)
 save('historical_protection.json',prior_checks)
 (OUT/'REPRODUCE.md').write_text('''# Reproduce the frozen experiment

Authoritative scientific definition: preregistration.json + DESIGN.md. All289 base files are unchanged; final adapters and tests are in frozen_source/ and frozen_source_snapshot.zip. Runtime pins: frozen_source/dependencies.lock.txt. Original official sources and old experiments are read-only.

Replay ONE cell, with identical source/input/dependency checks and a NEW destination:

```powershell
D:/科研启航/Sia/.venv-sia-baseline/Scripts/python.exe -X utf8 -B D:/科研启航/Sia/research_runs/sia_synergy/cc_clean_ab_20261030_run3/rebuild_scripts/replay_clean_ab_20261030.py --artifact D:/科研启航/Sia/research_runs/sia_synergy/cc_clean_ab_20261030_run3 --dest D:/科研启航/Sia/research_runs/sia_synergy/NEW_REPLAY --workload COMPACT --name B_JO
```

The destination must not exist. Each replay is an independent process; CPU/RAM/model/work are frozen. The helper changes only output destination, reads the original frozen source, and refuses input/source/dependency drift. Do not replay into the original directory. For all15, use the fixed order from preregistration and separate destinations. No replay was automatically executed during delivery.

Raw data: cells/*/rounds.jsonl.gz, summary.json, output_hashes.json. Recompute: audit_clean_ab_r3_20261030.py (writes exclusive new files, use a copied artifact with original analyses preserved/redirected). Metrics derive from actual completion events, captured raw rates, and allocation-time integrals, not summary alone. Test suites require the retained historical fixture files referenced by existing unit tests; formal-run GPU/profile/model data are included in frozen_source.

Earlier attempts run1/run2 contain interrupted control-variable debugging evidence and must not be merged with run3. C is an offline precheck, not a new response model. All conclusions are synthetic cross-hardware simulation evidence.
''',encoding='utf8')
 with (OUT/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nCOMPLETE15_RAW_AUDITED_STOPPED. REPORT.md, validation.json, factor_effects.json and C_RESPONSE_PREFLIGHT.md authoritative. No further experiment authorized.\n')
 files={f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_hashes_final.json'}
 save('delivery_hashes_final.json',dict(files=files,created=time.time(),formal_simulation_wall_seconds=sum(r['wall_seconds'] for r in records)))
 for n,h in files.items():assert sha(OUT/n)==h,n
 print('COMPLETE15_RAW_AUDITED_STOPPED',len(files),flush=True)
if __name__=='__main__':main()
