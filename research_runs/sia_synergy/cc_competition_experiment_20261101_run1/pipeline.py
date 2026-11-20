import hashlib,json,subprocess,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent
assert json.loads((O/'gate_result.json').read_text())['passed']
p=json.loads((O/'preregistration.json').read_text());started=time.time()
for cell in p['cells']:
 with (O/(cell['id']+'.log')).open('xb') as log:
  tick=time.time();q=subprocess.run([sys.executable,'-X','utf8','-B',str(O/'run_cell.py'),'--cell',cell['id']],stdout=log,stderr=subprocess.STDOUT)
 with (O/'execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(cell=cell['id'],exit_code=q.returncode,wall=time.time()-tick))+'\n')
 if q.returncode:raise RuntimeError('CELL_FAILED '+cell['id'])
 print('DONE',cell['id'],flush=True)
q=subprocess.run([sys.executable,'-X','utf8','-B',str(O/'summarize.py')]);assert q.returncode==0
with (O/'pipeline_complete.json').open('x') as f:json.dump(dict(completed=8,wall=time.time()-started),f)
