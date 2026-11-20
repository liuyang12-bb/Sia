import csv,json,hashlib,subprocess,sys,time,zipfile
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;WORK=ROOT/'sia-joint-small-mechanism';SIM=WORK/'sia-simulator';PY=ROOT/'.venv-sia-baseline/Scripts/python.exe'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def main():
 old=json.loads((ROOT/'research_runs/sia_synergy/sia_r0_full160_comparison/attempt2/preregistration.json').read_text());OUT.mkdir(exist_ok=True)
 log=OUT/'tests.log'
 with log.open('x',encoding='utf-8') as f:r=subprocess.run([str(PY),'-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'],cwd=SIM,stdout=f,stderr=subprocess.STDOUT)
 save(OUT/'tests.json',dict(exit_code=r.returncode,log_hash=sha(log)))
 if r.returncode:raise RuntimeError('TEST_GATE')
 trace=ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv';rows=list(csv.DictReader(trace.open()));picked=[]
 for app in ('bert','cifar10','deepspeech2'):
  idx,row=next((i,r) for i,r in enumerate(rows) if r['application']==app);r=dict(row);r.update(source_arrival=r['time'],source_row_index=idx,time='0');picked.append(r)
 with (OUT/'workload.csv').open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(picked[0]));w.writeheader();w.writerows(picked)
 sources=list(SIM.rglob('*.py')); hashes=dict(old['input_hashes']);hashes.update({str(f):sha(f) for f in sources});hashes[str(OUT/'workload.csv')]=sha(OUT/'workload.csv')
 p=dict(base_commit=old['commit'],branch='exp/sia-aux-small-mechanism',max_rounds=120,interval=60,workload=str(OUT/'workload.csv'),config=old['config'],registry=old['registry'],models=old['models'],hashes=hashes,seed_profiles=[1,8],Q=300,D_bar=250,capacity='main',runs=[[r,m] for r in ['neutral','medium'] for m in ['sia_p','sia_aux','joint_o']],selection='First original row for bert/cifar10/deepspeech2; controlled simultaneous arrivals; original workload parameters unchanged; 120-round limit fixed before runs; censor retained.',scientific_contract='SIA-AUX uses existing fairness p/lambda/restart and no L2; fixed full official GPU type/count/node intent, auxiliary placement allowed; JO retains existing L2. Same-state comparisons restrict identical task set and locks. No new liveness algorithm.',warning=old['warning'])
 save(OUT/'preregistration.json',p)
 with zipfile.ZipFile(OUT/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for f in sources:z.write(f,f.relative_to(WORK))
 save(OUT/'source_hashes.json',{str(f.relative_to(WORK)):sha(f) for f in sources})
 for response,method in p['runs']:
  mem=psutil.virtual_memory().available/1024**3
  if mem<2:raise RuntimeError(f'MEMORY_GATE {mem}')
  name=response+'_'+method;command=[str(PY),'-B','-u',str(SIM/'run_small_mechanism.py'),'--manifest',str(OUT/'preregistration.json'),'--method',method,'--response',response,'--output',str(OUT/name)]
  save(OUT/(name+'_launch.json'),dict(command=command,memory_gib=mem,start=time.time()))
  tick=time.perf_counter()
  with (OUT/(name+'.log')).open('x',encoding='utf-8') as f:r=subprocess.run(command,cwd=SIM,stdout=f,stderr=subprocess.STDOUT)
  save(OUT/(name+'_exit.json'),dict(exit_code=r.returncode,wall=time.perf_counter()-tick));print(name,r.returncode,flush=True)
  if r.returncode:raise RuntimeError('CELL_FAILED:'+name)
 print('SIX_SMALL_CELLS_COMPLETE',flush=True)
if __name__=='__main__':main()
