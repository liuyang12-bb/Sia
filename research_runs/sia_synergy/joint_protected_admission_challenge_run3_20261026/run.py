import json,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;P=json.loads((OUT/'preregistration.json').read_text())
assert json.loads((OUT/'offline_reference_precheck.json').read_text())['valid']
for method in P['runs']:
 mem=psutil.virtual_memory().available/2**30
 if mem<2.25:raise RuntimeError(f'MEMORY_GATE_WITH_MARGIN:{mem}')
 cmd=[sys.executable,'-X','utf8','-B',str(OUT/'cell.py'),'--method',method]
 print('START',method,mem,flush=True);tick=time.perf_counter()
 with (OUT/(method+'.log')).open('x',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 with (OUT/'execution_manifest.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(method=method,command=cmd,exit_code=r.returncode,wall_seconds=time.perf_counter()-tick))+'\n')
 print('END',method,r.returncode,flush=True)
 if r.returncode:raise RuntimeError('CELL_FAILURE_PRESERVED')
 if method=='G111' and not json.loads((OUT/'cells/G111/opportunity_gate.json').read_text())['passed']:
  print('STOP_GATE_NOT_TRIGGERED',flush=True);break
