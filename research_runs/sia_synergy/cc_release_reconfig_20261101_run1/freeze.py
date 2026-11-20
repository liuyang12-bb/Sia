import difflib,hashlib,importlib.metadata,json,shutil,subprocess,sys,zipfile
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-release-reconfig-v2';OLD=R/'cc-joint-release-evaluation-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
 assert json.loads((O/'same_state_exact.json').read_text())['gate']
 text=(O/'full_suite.log').read_text(encoding='utf8');assert '\nOK\n' in text and '\nFAILED (' not in text
 assert '\nOK\n' in (O/'green_final_v2.log').read_text(encoding='utf8')
 probes={}
 for tree in ('official-artifacts',T.name):
  log=O/('sia_probe_'+tree+'.log')
  with log.open('xb') as f:rc=subprocess.run([sys.executable,'-X','utf8','-B',str(T/'sia-simulator/check_stage1_legacy_regression.py'),'--probe',str(R/tree/'sia-simulator')],cwd=R,stdout=f,stderr=subprocess.STDOUT).returncode
  assert rc==0
  probes[tree]=json.loads(next(s.removeprefix('PROBE_JSON=') for s in log.read_text(encoding='utf8').splitlines() if s.startswith('PROBE_JSON=')))
 assert probes['official-artifacts']==probes[T.name]
 save('original_sia_regression.json',dict(exact=True,rounds=3,states=probes))
 h={p.relative_to(T).as_posix():sha(p) for p in T.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 base=json.loads((O/'baseline_hashes.json').read_text());assert all(sha(OLD/n)==v for n,v in base.items())
 changed=[n for n in base if h.get(n)!=base[n]];new=sorted(set(h)-set(base));patch=[]
 for n in changed+new:
  before=(OLD/n).read_text(encoding='utf8').splitlines(True) if n in base else []
  patch.extend(difflib.unified_diff(before,(T/n).read_text(encoding='utf8').splitlines(True),fromfile='a/'+n,tofile='b/'+n))
 (O/'source_changes.patch').write_text(''.join(patch),encoding='utf8')
 with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(h):z.write(T/n,n)
 save('source_hashes.json',h);save('source_receipt.json',dict(changed=changed,new=new,files=len(h),old_source_unchanged=True,snapshot_sha256=sha(O/'source_snapshot.zip'),git='Complete source copy; no fabricated HEAD'))
 (O/'inputs').mkdir();src=R/'research_runs/sia_synergy/cc_clean_ab_20261030_run3/inputs/COMPACT_workload.csv';shutil.copy2(src,O/'inputs/workload.csv')
 inputs={str(p):sha(p) for p in (T/'inputs').rglob('*') if p.is_file()};inputs[str(O/'inputs/workload.csv')]=sha(src)
 env=dict(python=sys.version,executable=sys.executable,packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','cvxpy','cylp','psutil']})
 save('environment.json',env);save('input_hashes.json',inputs)
 save('preregistration.json',dict(version='RELEASE_RECONFIG_V2',source=str(T),source_hashes=h,input_hashes=inputs,environment=env,
  topology={'aws':2,'dgx-ext':2},capacity={'aws':{'cpu':6,'ram':125},'dgx-ext':{'cpu':6,'ram':125}},
  cells=[dict(id=label,method=m,new_evaluation=new) for label,m,new in [('OLD_FO','FO',False),('OLD_FULL','FULL',False),('NEW_FO','FO',True),('NEW_FULL','FULL',True)]],
  workload=str(O/'inputs/workload.csv'),source_workload=str(src),count=12,response='R0_E_LEGACY_MEDIUM',alpha=.5,seed_profiles=[1,8],interval=60,max_rounds=6000,
  memory_gib=2,disk_gib=5,solver={'time_limit':30,'mip_rel_gap':0},
  success='Compare means only for complete12/12 pairs; no extrapolation, no result-conditioned retuning or expansion.',
  science_label='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
 print('FROZEN',len(h),sha(O/'preregistration.json'),flush=True)
if __name__=='__main__':main()
