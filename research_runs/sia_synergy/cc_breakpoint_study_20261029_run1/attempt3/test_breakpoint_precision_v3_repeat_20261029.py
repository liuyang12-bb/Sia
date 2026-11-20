"""Relocate new source to canonical workspace depth for legacy path tests."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt3';SRC=OUT/'source'
CANON=ROOT/'cc-joint-breakpoint-precision'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 m=json.loads((SRC/'source_manifest.json').read_text(encoding='utf8'))
 for n in m['files']:m['files'][n]=sha(SRC/n)
 (SRC/'source_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
 shutil.copytree(SRC,CANON,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for p in SRC.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':assert sha(p)==sha(CANON/p.relative_to(SRC))
 start=time.time()
 with (OUT/'all_tests.log').open('x',encoding='utf8') as f:
  result=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'],cwd=CANON/'sia-simulator',stdout=f,stderr=subprocess.STDOUT)
 (OUT/'tests_result.json').write_text(json.dumps(dict(exit_code=result.returncode,wall_seconds=time.time()-start,canonical_tree=str(CANON),code_matches_snapshot=True),indent=2))
 print('TEST_END',result.returncode,flush=True)
if __name__=='__main__':main()
