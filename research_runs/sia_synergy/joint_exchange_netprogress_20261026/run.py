"""Serial controller; never retries or overwrites any strategy output."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
P=json.loads((OUT/'preregistration.json').read_text())
for cell in P['runs']:
 mem=psutil.virtual_memory().available/2**30
 if mem<2.25:raise RuntimeError(f'MEMORY_GATE_WITH_STARTUP_MARGIN:{mem}')
 cmd=[sys.executable,'-B',str(OUT/'real_cell.py'),'--cell',cell['id']]
 tick=time.perf_counter();print('START',cell['id'],mem,flush=True)
 with (OUT/(cell['id']+'.log')).open('x',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 with (OUT/'execution_manifest.jsonl').open('a',encoding='utf-8') as f:
  f.write(json.dumps(dict(cell=cell,command=cmd,exit_code=r.returncode,wall_seconds=time.perf_counter()-tick))+'\n')
 print('END',cell['id'],r.returncode,flush=True)
 if r.returncode:raise RuntimeError('CELL_FAILED_SEE_PRESERVED_LOG')
