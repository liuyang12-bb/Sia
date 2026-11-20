"""Freeze one strategy revision; two fixed 3-job cells, no outcome-based retries."""
import json,hashlib,subprocess,time,zipfile
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;WORK=ROOT/'sia-joint-adaptive-service';SIM=WORK/'sia-simulator';PY=ROOT/'.venv-sia-baseline/Scripts/python.exe';OLD=ROOT/'research_runs/sia_synergy/small_mechanism_20261025'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def main():
 start=time.time();p=json.loads((OLD/'preregistration.json').read_text());prior_source=json.loads((OLD/'source_hashes.json').read_text())
 if not (SIM/'tests/test_adaptive_service.py').exists():raise RuntimeError('ADAPTIVE_UNIT_TESTS_REQUIRED')
 for f,h in prior_source.items():
  if sha(WORK/f)!=h:raise RuntimeError('OLD_IMPLEMENTATION_CHANGED:'+f)
 prior_hashes=json.loads((OLD/'artifact_hashes.json').read_text())
 for f,h in prior_hashes.items():
  if sha(OLD/f)!=h:raise RuntimeError('OLD_ARTIFACT_CHANGED:'+f)
 save(OUT/'baseline_reuse_check.json',dict(source_files=len(prior_source),artifacts=len(prior_hashes),valid=True))
 with (OUT/'all_tests.log').open('x',encoding='utf-8') as f:r=subprocess.run([str(PY),'-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'],cwd=SIM,stdout=f,stderr=subprocess.STDOUT)
 save(OUT/'all_tests.json',dict(exit_code=r.returncode,sha256=sha(OUT/'all_tests.log')))
 if r.returncode:raise RuntimeError('FULL_TEST_GATE')
 sources=list(SIM.rglob('*.py'));p['hashes'].update({str(f):sha(f) for f in sources});p.update(branch='exp/joint-adaptive-service',method='JOINT-R',runs=[['neutral','joint_r'],['medium','joint_r']],prior_preregistration_sha=sha(OLD/'preregistration.json'),selection='EXACT_SAME_PRIOR_3JOB_WORKLOAD_AND_120_ROUND_CAP',service_change='Membership preserved, unlocked geometry adaptation within existing total250s delay budget, online payback, unchanged Q300/objective/response.')
 save(OUT/'preregistration.json',p)
 with zipfile.ZipFile(OUT/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for f in sources:z.write(f,f.relative_to(WORK))
 save(OUT/'source_hashes.json',{str(f.relative_to(WORK)):sha(f) for f in sources})
 for response in ['neutral','medium']:
  mem=psutil.virtual_memory().available/1024**3
  if mem<2:raise RuntimeError(f'MEMORY_GATE {mem}')
  cmd=[str(PY),'-B','-u',str(SIM/'run_adaptive_cell.py'),'--manifest',str(OUT/'preregistration.json'),'--response',response,'--output',str(OUT/(response+'_joint_r'))]
  save(OUT/(response+'_launch.json'),dict(command=cmd,memory_gib=mem,start=time.time()));tick=time.perf_counter()
  with (OUT/(response+'.log')).open('x',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=SIM,stdout=f,stderr=subprocess.STDOUT)
  save(OUT/(response+'_exit.json'),dict(exit_code=r.returncode,wall=time.perf_counter()-tick));print(response,r.returncode,flush=True)
  if r.returncode:raise RuntimeError('CELL_FAILED:'+response)
 save(OUT/'run_wall.json',dict(total_wall=time.time()-start));print('TWO_CELLS_COMPLETE',flush=True)
if __name__=='__main__':main()
