import json,subprocess,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent
def main():
 p=json.loads((O/'preregistration.json').read_text());start=time.time()
 with (O/'execution_manifest.jsonl').open('x',encoding='utf8') as manifest:
  for cell in p['cells']:
   command=[sys.executable,'-X','utf8','-B',str(O/'run_small.py'),'--cell',cell['id']]
   manifest.write(json.dumps(dict(event='START',cell=cell,unix=time.time(),command=command))+'\n');manifest.flush()
   with (O/(cell['id']+'.log')).open('xb') as log:
    rc=subprocess.run(command,cwd=O,stdout=log,stderr=subprocess.STDOUT).returncode
   manifest.write(json.dumps(dict(event='END',cell=cell['id'],code=rc,unix=time.time()))+'\n');manifest.flush()
   print('CELL_END',cell['id'],rc,'wall',time.time()-start,flush=True)
   if rc:raise RuntimeError('CELL_STOP '+cell['id'])
 with (O/'pipeline_complete.json').open('x',encoding='utf8') as f:json.dump(dict(complete=True,wall_seconds=time.time()-start),f,indent=2)
if __name__=='__main__':main()
