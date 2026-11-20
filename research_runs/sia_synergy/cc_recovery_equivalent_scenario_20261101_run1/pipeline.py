import json,subprocess,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent
p=json.loads((O/'preregistration.json').read_text());g=json.loads((O/'gate_result.json').read_text());start=time.time();done=[];skipped=[]
for cell in p['cells']:
 if cell['id'].startswith('balanced') and not g['passed']:
  skipped.append(cell['id']);continue
 with (O/(cell['id']+'.log')).open('xb') as f:
  tick=time.time();q=subprocess.run([sys.executable,'-X','utf8','-B',str(O/'run_cell.py'),'--cell',cell['id']],stdout=f,stderr=subprocess.STDOUT)
 with (O/'execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(cell=cell['id'],exit_code=q.returncode,wall=time.time()-tick))+'\n')
 if q.returncode:raise RuntimeError('CELL_FAILED '+cell['id'])
 done.append(cell['id']);print('DONE',cell['id'],flush=True)
with (O/'pipeline_complete.json').open('x') as f:json.dump(dict(completed=done,skipped=skipped,wall=time.time()-start),f,indent=2)
