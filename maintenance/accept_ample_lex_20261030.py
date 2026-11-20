import difflib,gzip,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1';SRC=OUT/'source';CANON=ROOT/'cc-joint-ample-lex-repair'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(name,data):
 with (OUT/name).open('x',encoding='utf8') as f:json.dump(data,f,indent=2,ensure_ascii=False)
def run(name,args,cwd=None):
 start=time.time()
 with (OUT/name).open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',*args],cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 save(name+'.json',dict(exit_code=r.returncode,wall_seconds=time.time()-start));assert r.returncode==0,name
def main():
 m=json.loads((SRC/'source_manifest.json').read_text())
 for n in m['files']:m['files'][n]=sha(SRC/n)
 m['files']['sia-simulator/tests/test_ample_lex_identity.py']=sha(SRC/'sia-simulator/tests/test_ample_lex_identity.py')
 m['lexicographic_repair']='Count each chosen column once, independent of duplicate candidate-list references; unchanged primary and actual startup secondary objective.'
 (SRC/'source_manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf8')
 shutil.copytree(SRC,CANON)
 for f in SRC.rglob('*'):
  if f.is_file():assert sha(f)==sha(CANON/f.relative_to(SRC))
 run('all_tests.log',['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],CANON/'sia-simulator')
 run('bounded_green.log',[str(ROOT/'maintenance/bounded_ample_green_20261030.py')])
 probe=SRC/'sia-simulator/check_stage1_legacy_regression.py';states={}
 for name,sim in [('official',ROOT/'official-artifacts/sia-simulator'),('repaired',CANON/'sia-simulator')]:
  r=subprocess.run([sys.executable,'-X','utf8','-B',str(probe),'--probe',str(sim)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  (OUT/f'original_sia_{name}.log').write_bytes(r.stdout);assert r.returncode==0
  states[name]=json.loads(next(l.split('PROBE_JSON=',1)[1] for l in r.stdout.decode('utf8').splitlines() if l.startswith('PROBE_JSON=')))
 assert states['official']==states['repaired'];save('original_sia_regression.json',dict(exact_state_match=True,rounds=3,jobs=1,states=states))
 old=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_ample_retry1';p=json.loads((old/'preregistration.json').read_text())
 text=(old/'run.py').read_text().replace('cc_cb_validation_20261029_ample_retry1','cc_ample_lex_repair_20261030_run1');(OUT/'run.py').write_text(text,encoding='utf8')
 p.update(source_hashes={f.relative_to(SRC).as_posix():sha(f) for f in SRC.rglob('*') if f.is_file()},runner_hash=sha(OUT/'run.py'),frozen_unix=time.time(),repair_scope='Only selected startup cost sum fixes alias overcount; primary objective, feasible candidates, response, quotas, capacity, solver options and stable tie ordering unchanged.',parent_hash=sha(old/'preregistration.json'))
 for n in ['preregistration.json','execution_preregistration_v2.json']:save(n,p)
 original=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1/source';patch=[]
 for n in ['sia-simulator/joint_prototype/bundle_adaptation.py','sia-simulator/tests/test_ample_lex_identity.py']:
  before=(original/n).read_text().splitlines(True) if (original/n).exists() else []
  patch.extend(difflib.unified_diff(before,(SRC/n).read_text().splitlines(True),fromfile='baseline/'+n,tofile='repaired/'+n))
 (OUT/'source_diff.patch').write_text(''.join(patch),encoding='utf8');shutil.make_archive(str(OUT/'repaired_source_snapshot'),'zip',SRC)
 save('acceptance.json',dict(tests=245,original_sia_exact=True,bounded_methods=['CC_CPU','CC_MEM'],all_gates_passed=True))
 print('ACCEPTED_FROZEN6',sha(OUT/'preregistration.json'),flush=True)
 run('batch_console.log',[str(OUT/'run.py'),'batch'])
 records=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()];assert len(records)==6 and all(r['exit_code']==0 for r in records)
 run('final_audit.log',[str(ROOT/'maintenance/audit_ample_lex_20261030.py')])
 print('REPAIR_SIX_COMPLETE_AUDITED_STOPPED',flush=True)
if __name__=='__main__':main()
