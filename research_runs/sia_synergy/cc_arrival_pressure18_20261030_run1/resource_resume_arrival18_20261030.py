"""Bounded resource wait; resumes only before any gate child has started."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1'
assert not (OUT/'cells').exists(),'Refuse duplicate or partial gate restart'
start=time.time();samples=[]
for i in range(21):
 x=dict(time=time.time(),memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30);samples.append(x)
 print('RESOURCE_RECHECK',i,x,flush=True)
 if x['memory_gib']>=2 and x['disk_gib']>=5:
  p=json.loads((OUT/'preregistration.json').read_text())
  for n,h in p['source_hashes'].items():assert hashlib.sha256((OUT/'source'/n).read_bytes()).hexdigest().upper()==h
  for n,h in p['input_hashes'].items():assert hashlib.sha256((OUT/'inputs'/n).read_bytes()).hexdigest().upper()==h
  with (OUT/'resource_rechecks.json').open('x',encoding='utf8') as f:json.dump(dict(resumed=True,samples=samples),f,indent=2)
  with (OUT/'resumed_gate_console.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'maintenance/gate_arrival18_20261030.py')],stdout=f,stderr=subprocess.STDOUT)
  print('GATE_OR_BATCH_EXIT',r.returncode,flush=True);sys.exit(r.returncode)
 if i<20:time.sleep(30)
with (OUT/'resource_rechecks.json').open('x',encoding='utf8') as f:json.dump(dict(resumed=False,samples=samples),f,indent=2)
(OUT/'CURRENT_TASK.md').write_text('BLOCKED_MEMORY_GATE_BEFORE_ANY_SIMULATION. Frozen18 matrix and6 input tests ready. Bounded10min resource recheck ended below2GiB. No gate/formal process running; never lower threshold. Resume only after memory recovers, no existing cell outputs may be overwritten.\n',encoding='utf8')
print('SAFE_STOP_NO_SIMULATION_LAUNCHED',flush=True)
