import json,subprocess,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent
assert json.loads((O/'persistence_gate.json').read_text())['passed']
p=json.loads((O/'preregistration.json').read_text());start=time.time()
for c in p['cells']:
 with (O/(c['id']+'.log')).open('xb') as f:
  t=time.time();q=subprocess.run([sys.executable,'-X','utf8','-B',str(O/'run_cell.py'),'--cell',c['id']],stdout=f,stderr=subprocess.STDOUT)
 with (O/'execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(cell=c['id'],exit_code=q.returncode,wall=time.time()-t))+'\n')
 if q.returncode:raise RuntimeError('CELL_FAILED '+c['id'])
 print('DONE',c['id'],flush=True)
q=subprocess.run([sys.executable,'-X','utf8','-B',str(O/'summarize.py')]);assert q.returncode==0
with (O/'pipeline_complete.json').open('x') as f:json.dump(dict(completed=4,wall=time.time()-start),f)
