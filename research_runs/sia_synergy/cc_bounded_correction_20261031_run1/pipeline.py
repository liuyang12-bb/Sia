"""One-shot acceptance, freeze, serial two-cell run, independent raw audit."""
import csv,difflib,gzip,hashlib,itertools,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import psutil
R=Path(r'D:/科研启航/Sia');T=R/'cc-joint-bounded-correction-v1';O=Path(__file__).resolve().parent
OLD=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(name,v):
 with (O/name).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def run(name,args,cwd=R):
 with (O/(name+'.log')).open('xb') as log:
  p=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,args)],cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
 print(name,p.returncode,flush=True)
 if p.returncode:raise RuntimeError('GATE_OR_RUN_FAILED '+name)
def source_hashes():return {p.relative_to(T).as_posix():sha(p) for p in T.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
def main():
 started=time.time();save('pipeline_start.json',dict(pid=os.getpid(),unix=started,command=sys.argv))
 # The exploratory suite already running predates the corrected locked-peer
 # fixture. Wait for that exact PID, never run test processes concurrently.
 wait_pid=int(sys.argv[1]) if len(sys.argv)>1 else None
 if wait_pid:
  while psutil.pid_exists(wait_pid):time.sleep(3)
 run('full_suite_green',['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],T/'sia-simulator')
 text=(O/'full_suite_green.log').read_text(encoding='utf8')
 assert 'Ran 348 tests' in text and '\nOK\n' in text,'UNEXPECTED_SUITE_COUNT_OR_STATUS'
 probes={}
 for tree in ('official-artifacts',T.name):
  label='original_sia_'+tree
  run(label,[T/'sia-simulator/check_stage1_legacy_regression.py','--probe',R/tree/'sia-simulator'])
  probes[tree]=json.loads(next(x.removeprefix('PROBE_JSON=') for x in (O/(label+'.log')).read_text(encoding='utf8').splitlines() if x.startswith('PROBE_JSON=')))
 assert probes['official-artifacts']==probes[T.name]
 save('original_sia_regression.json',dict(exact=True,rounds=3,states=probes))
 # Real 80-job initial-state prefixes, before the recovery trigger, must
 # preserve original allocations/execution. No manual seed or synthetic rate.
 script=(R/'maintenance/bounded_core80_full_lex_v8.py').read_text(encoding='utf8')
 script=script.replace("ROOT=Path(__file__).resolve().parents[1];TREE=ROOT/'cc-joint-core80-full-lex-repair-v8';OUT=ROOT/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'",f"ROOT=Path({str(R)!r});TREE=Path({str(T)!r});OUT=Path({str(O)!r})")
 (O/'bounded_prefix.py').write_text(script,encoding='utf8');(O/'tests').mkdir()
 def normalize(v):
  if isinstance(v,dict):return {k:normalize(x) for k,x in v.items() if k!='template_source'}
  if isinstance(v,list):return [normalize(x) for x in v]
  return v
 bridges=[]
 for method in ('FO','FULL'):
  run('real_prefix_'+method,[O/'bounded_prefix.py',method,3,'new'])
  def frames(path):
   with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(x) for x in itertools.islice(f,3)]
  old=frames(OLD/'cells'/('dual_medium_'+method)/'rounds.jsonl.gz')
  new=frames(O/'tests'/('real80_'+method+'_3_new')/'prefix.jsonl.gz')
  def project(x):
   d=x['decision_for_next_interval']['result'];return normalize(dict(execution=x['execution'],selected=d['selected'],objective=d['scheduling_objective']))
  assert len(new)==3 and [project(x) for x in old]==[project(x) for x in new]
  bridges.append(dict(method=method,rounds=3,exact=True))
 save('software_acceptance.json',dict(status='PASS',existing_tests=335,new_tests=13,total=348,original_sia_exact=True,prefix_bridges=bridges,
  exploratory_fixture_error='First expanded fixture held a zero-progress peer until360s, violating the unchanged310s no-progress watchdog; fixture corrected to180s; failed logs preserved.'))
 base=json.loads((OLD/'preregistration.json').read_text());h=source_hashes()
 before=json.loads((O/'baseline_source_hashes.json').read_text());changes=[n for n,v in before.items() if h.get(n)!=v];new=sorted(set(h)-set(before))
 assert changes==['core80.py'] or set(changes)=={'core80.py'}
 assert set(new)=={'sia-simulator/joint_prototype/bounded_correction.py','sia-simulator/tests/test_bounded_correction.py'}
 for n,v in before.items():assert sha(R/'cc-joint-core80-full-lex-repair-v8'/n)==v
 inputs={}
 for n,v in base['input_hashes'].items():
  path=Path(n);assert sha(path)==v
  try:path=T/path.relative_to(Path(base['source_root']))
  except ValueError:pass
  assert sha(path)==v;inputs[str(path)]=v
 save('source_hashes.json',h);save('input_hashes.json',inputs)
 with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(h):z.write(T/n,n)
 patch=[]
 for n in changes+new:
  prev=(R/'cc-joint-core80-full-lex-repair-v8'/n).read_text(encoding='utf8').splitlines(True) if n in before else []
  patch.extend(difflib.unified_diff(prev,(T/n).read_text(encoding='utf8').splitlines(True),fromfile='a/'+n,tofile='b/'+n))
 (O/'source_changes.patch').write_text(''.join(patch),encoding='utf8')
 resources=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(R)).free/2**30)
 assert resources['memory_gib']>=2 and resources['disk_gib']>=5,resources
 for p in psutil.process_iter(['pid','name','cmdline']):
  if p.info['name'] and p.info['name'].lower()=='python.exe':
   assert not any(Path(a).name in ('supervise80.py','run_core80.py') for a in p.info['cmdline'] or []),p.info
 now=time.time();base.update(version='COMMON_BOUNDED_CORRECTION_V1',source_root=str(T),output=str(O),source_hashes=h,input_hashes=inputs,
  cells=[dict(id='dual_medium_'+m,method=m,response='medium',scenario='dual',optional=False) for m in ('FO','FULL')],
  frozen_unix=now,formal_stop_unix=now+24*3600,hard_deadline_unix=now+25*3600,cell_wall_limit_seconds=6*3600,
  status='FROZEN',stop_on_engineering_failure=True,source_snapshot_sha256=sha(O/'source_snapshot.zip'),
  recovery_rule=dict(trigger='CURRENT_READY_EMPTY_AND_MULTIPLE_ALLOCATED_JOBS',point='EXISTING_PROPORTIONAL_ONLY',
   witness='FIX_ALL_OTHER_CURRENT_CONFIG_PLACEMENT_QUOTAS',eta='CURRENT_MODEL_STRICT_GAIN_INCLUDES_REAL_STARTUP',
   permit='SHARED_WITH_OLD_TAIL_ONCE_PER_JOB',max_extra_startup=250,post_permit='GPU_GEOMETRY_FROZEN_AUX_ADAPTIVE',
   same_rule=['FO','FULL'],future_information=False),
  analysis='Four cells: sealed oldFO/oldFULL plus newFO/newFULL. Positive improvement=1-new/old. Also difference-in-differences of FULL-minus-FO meanJCT; negative means enlarged joint advantage. All80 retained; no re-tuning. P95/P99/makespan/DeepSpeech and wait/checkpoint/train decomposition. Censoring never imputed.',
  historical_reference=str(OLD))
 save('preregistration.json',base);save('freeze_receipt.json',dict(unix=now,source_files=len(h),changed=changes,new=new,resources=resources,prereg_sha256=sha(O/'preregistration.json')))
 (O/'cells').mkdir()
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nFROZEN '+str(now)+';348tests accepted. FO→FULL serial new processes. Resume only after inventory and prefix hashes; no automatic rerun of failed directories.\n')
 run('formal_supervisor',[T/'supervise80.py',O/'preregistration.json'],T)
 # Reuse the existing independent event/fragment auditor, changing only paths
 # and expected count. It never repairs worker logs or infers censored JCT.
 audit=(R/'maintenance/audit_core80_full_lex_v8.py').read_text(encoding='utf8')
 audit=audit.replace("ROOT=Path(__file__).resolve().parents[1];TREE=ROOT/'cc-joint-core80-full-lex-repair-v8';OUT=ROOT/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'",f"ROOT=Path({str(R)!r});TREE=Path({str(T)!r});OUT=Path({str(O)!r})")
 audit=audit.replace('expected_core=7','expected_core=2')
 (O/'audit_raw.py').write_text(audit,encoding='utf8');run('raw_audit',[O/'audit_raw.py'])
 run('final_report',[O/'report.py'])
 save('pipeline_complete.json',dict(unix=time.time(),wall_seconds=time.time()-started))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  if not (O/'pipeline_failure.json').exists():save('pipeline_failure.json',dict(unix=time.time(),error=repr(e),traceback=traceback.format_exc()))
  raise
