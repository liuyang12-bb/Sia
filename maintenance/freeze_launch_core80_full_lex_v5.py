"""Wait for acceptance, freeze seven full pairs, launch once. No tuning."""
import difflib,hashlib,importlib.metadata,json,shutil,subprocess,sys,time,zipfile,psutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-full-lex-repair-v5';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run2';OLD=R/'research_runs/sia_synergy/cc_core80_repair_full_20261031_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 deadline=time.time()+3*3600
 while not (O/'software_acceptance.json').exists():
  if 'Traceback (most recent call last)' in (O/'acceptance_driver.log').read_text(encoding='utf8') or time.time()>deadline:raise RuntimeError('SOFTWARE_GATE_FAILED_NO_FORMAL_LAUNCH')
  time.sleep(5)
 accept=json.loads((O/'software_acceptance.json').read_text());assert accept['status']=='PASS'
 assert not (O/'preregistration.json').exists(),'Never launch twice'
 p=json.loads((OLD/'preregistration.json').read_text());base=Path(p['source_root']);baseline=p['source_hashes']
 h={x.relative_to(T).as_posix():sha(x) for x in T.rglob('*') if x.is_file() and '__pycache__' not in x.parts}
 receipt=O/'tests/accepted_source_hashes.json'
 if not receipt.exists():
  for label in ('FULL_30','A10_20','PROP_12'):
   used=json.loads((O/'tests'/('real80_'+label+'_lattice_primary')/'runtime_source_hashes.json').read_text())
   for n,v in used.items():assert h[n.replace('\\','/')]==v,('SOURCE_CHANGED_FROM_BOUNDED_EXECUTION',label,n)
  save(receipt,h)
  save(O/'tests/acceptance_source_attestation.json',dict(method='All three bounded executions verify current non-test Python bytes at every boundary; final full-suite318 log and new fixture preserved.',source_files=len(h)))
 accepted={n.replace('\\','/'):v for n,v in json.loads(receipt.read_text()).items()}
 assert h==accepted,'SOURCE_CHANGED_AFTER_ACCEPTANCE'
 changed=[n for n,v in baseline.items() if h.get(n)!=v];new=sorted(set(h)-set(baseline))
 assert set(changed)=={'numeric_repair.py','engineering_repair.py','supervise80.py'},changed
 assert set(new)=={'integer_row_lattice.py','primary_band_presolve.py','sia-simulator/tests/test_rank_lattice_repair.py','sia-simulator/tests/fixtures/core80_full_rank_step19.json.gz'},new
 for n,v in baseline.items():assert sha(base/n)==v,('OLD_TREE_CHANGED',n)
 for package,version in p['environment']['packages'].items():assert importlib.metadata.version(package)==version,(package,version)
 inputs={}
 for n,v in p['input_hashes'].items():
  path=Path(n);assert sha(path)==v
  try:path=T/path.relative_to(base)
  except ValueError:pass
  assert sha(path)==v;inputs[str(path)]=v
 active=[(x.pid,x.cmdline()) for x in psutil.process_iter() if x.name().lower()=='python.exe' and any(Path(v).name in ('run_core80.py','supervise80.py') for v in x.cmdline())]
 assert not active,active
 resources=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(R)).free/2**30)
 assert resources['memory_gib']>=2 and resources['disk_gib']>=5,resources
 save(O/'source_hashes.json',h);save(O/'input_hashes.json',inputs);shutil.copyfile(OLD/'source_snapshot.zip',O/'base_source_snapshot.zip')
 save(O/'source_change_inventory.json',dict(base=str(base),changed=changed,new=new,unchanged=sum(h.get(n)==v for n,v in baseline.items()),
  scientific_change=False,orchestration_change='Opt-in stop on engineering failure; defaults and solver/cell limits unchanged.'))
 patch=[]
 for n in changed:
  patch.extend(difflib.unified_diff((base/n).read_text().splitlines(True),(T/n).read_text().splitlines(True),fromfile='a/'+n,tofile='b/'+n))
 for n in new:
  if n.endswith('.py'):patch.extend(difflib.unified_diff([], (T/n).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+n))
 with (O/'source_changes.patch').open('x',encoding='utf8') as f:f.writelines(patch)
 with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(h):z.write(T/n,n)
 for n in ('dependencies.lock.txt','environment.json','external_test_dependencies.zip','external_test_dependency_hashes.json'):
  if (OLD/n).exists():shutil.copyfile(OLD/n,O/n)
 order=['FULL','SIA_P','SIA_AUX','PROP','CPU','MEM','FO'];cells=[next(c for c in p['cells'] if c['method']==name) for name in order]
 assert all(c['scenario']=='dual' and c['response']=='medium' and not c['optional'] for c in cells)
 now=time.time();p.update(version='CORE80_V5_CERTIFIED_DOMAIN_FIRST_7PAIR',status='FROZEN',cells=cells,source_root=str(T),output=str(O),source_hashes=h,input_hashes=inputs,
  source_snapshot_path=str(O/'source_snapshot.zip'),source_snapshot_sha256=sha(O/'source_snapshot.zip'),software_acceptance_path=str(O/'software_acceptance.json'),
  stop_on_engineering_failure=True,frozen_unix=now,formal_stop_unix=now+24*3600,hard_deadline_unix=now+25*3600,cell_wall_limit_seconds=6*3600,
  failure='FULL first. Stop entire batch on engineering FAILED/COMPUTE_TIMEOUT, preserving all raw evidence. ROUND_LIMIT is censoring, no JCT imputation. Never edit frozen source or mix paired versions.',
  source_equivalence='Only integer lattice row representation and directed weak-duality feasibility presolve of variables impossible at the existing primary band. Original objective, all rows/precommit audits, candidate grids, capacities, response, seeds, work and solver options unchanged. Final decisions require original-row validation and global optimum proof.',
  matrix='Seven complete affected resource-method pairs on one version, FULL,SIA_P,SIA_AUX,PROP,CPU,MEM,FO. No Neutral/104/160/optional capacity experiments.',
  historical_base=str(OLD),historical_base_prereg_sha256=sha(OLD/'preregistration.json'))
 (O/'cells').mkdir();save(O/'preregistration.json',p);save(O/'freeze_receipt.json',dict(unix=now,source_files=len(h),inputs=len(inputs),resources=resources,source_zip_sha256=sha(O/'source_snapshot.zip'),prereg_sha256=sha(O/'preregistration.json'),patch_sha256=sha(O/'source_changes.patch')))
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nFROZEN '+p['source_snapshot_sha256']+';319files,38inputs. Acceptance318tests,realFULL30/A1020/PROP12,prefix18/20/12 and originalSia3round exact. This launcher starts supervise80.py ONCE; inspect execution_manifest.jsonl and current process/session before recovery. FULL must complete before six controls; stop on any engineering failure. Do not alter source.\n')
 print('FROZEN_LAUNCH',len(h),p['source_snapshot_sha256'],resources,flush=True)
 return subprocess.run([sys.executable,'-X','utf8','-B',str(T/'supervise80.py'),str(O/'preregistration.json')],cwd=T).returncode
if __name__=='__main__':raise SystemExit(main())
