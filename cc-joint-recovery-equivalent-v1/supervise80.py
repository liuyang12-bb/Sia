"""Serial deadline watchdog; partial/failed directories are never reused."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def append(p,v):
 with p.open('a',encoding='utf8') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def stop_tree(proc):
 try:
  tree=psutil.Process(proc.pid);children=tree.children(recursive=True)
  for x in reversed(children):
   try:x.terminate()
   except psutil.NoSuchProcess:pass
  try:tree.terminate()
  except psutil.NoSuchProcess:pass
  psutil.wait_procs(children+[tree],timeout=10)
 except psutil.NoSuchProcess:pass
def main(path):
 p=json.loads(Path(path).read_text(encoding='utf8'));out=Path(p['output']);manifest=out/'execution_manifest.jsonl'
 if manifest.exists():raise RuntimeError('REFUSE_SECOND_SUPERVISOR_OR_EXISTING_MANIFEST')
 append(manifest,dict(event='BATCH_START',unix=time.time(),pid=os.getpid(),prereg_sha256=sha(path)))
 completed=[];start=time.time();stop_reason=None
 for i,cell in enumerate(p['cells']):
  now=time.time()
  if cell['optional']:
   # Optional cells are admitted only as entire fixed pairs. No JCT used.
   if i%2==1:
    if len(completed)<11 or any(x['state']!='ALL_COMPLETE' for x in completed[:11]):stop_reason='OPTIONAL_GATE_CORE_NOT_ALL_COMPLETE';break
    if p['formal_stop_unix']-now<2*p['cell_wall_limit_seconds']+60:stop_reason='OPTIONAL_PAIR_BUDGET_NOT_AVAILABLE';break
  if now>=p['formal_stop_unix']:stop_reason='FORMAL_WALL_DEADLINE';break
  resources=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(out)).free/2**30)
  if resources['memory_gib']<2 or resources['disk_gib']<5:
   stop_reason='SAFETY_GATE';append(manifest,dict(event='SAFE_STOP',cell=cell['id'],unix=now,resources=resources));break
  if (out/'cells'/cell['id']).exists():raise RuntimeError('REFUSE_EXISTING_CELL '+cell['id'])
  cmd=[sys.executable,'-X','utf8','-B',str(ROOT/'run_core80.py'),'--manifest',str(Path(path).resolve()),'--cell',cell['id']]
  append(manifest,dict(event='CELL_START',cell=cell['id'],unix=now,command=cmd,resources=resources))
  with (out/f"{cell['id']}.console.log").open('x',encoding='utf8') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   append(manifest,dict(event='CELL_PROCESS',cell=cell['id'],pid=proc.pid,unix=time.time()))
   limit=min(now+p['cell_wall_limit_seconds'],p['formal_stop_unix']);timed=False
   while proc.poll() is None:
    if time.time()>=limit:stop_tree(proc);timed=True;break
    time.sleep(5)
   proc.wait()
  summary=out/'cells'/cell['id']/'summary.json';failure=out/'cells'/cell['id']/'failure.json'
  state='COMPUTE_TIMEOUT' if timed else (json.loads(summary.read_text(encoding='utf8'))['termination'] if summary.exists() else 'FAILED')
  event=dict(event='CELL_END',cell=cell['id'],unix=time.time(),state=state,exit_code=proc.returncode,wall_seconds=time.time()-now,
     output_hashes={str(x.relative_to(out)):sha(x) for x in (out/'cells'/cell['id']).glob('*') if x.is_file()})
  append(manifest,event);completed.append(event)
  print('CELL_END',cell['id'],state,'wall',round(event['wall_seconds'],1),flush=True)
  if timed:
   # Do not edit or close/reconstruct a truncated worker log after a hard kill.
   (out/'cells'/cell['id']/'COMPUTE_TIMEOUT.json').write_text(json.dumps(dict(status=state,wall_limit=p['cell_wall_limit_seconds'],unix=time.time(),reason='Hard watchdog; no JCT imputation, raw partial logs retained.'),indent=2),encoding='utf8')
  if p.get('stop_on_engineering_failure',False) and state in ('FAILED','COMPUTE_TIMEOUT'):
   stop_reason='REGISTERED_ENGINEERING_FAILURE_GATE';break
 append(manifest,dict(event='BATCH_END',unix=time.time(),elapsed=time.time()-start,attempted=len(completed),stop_reason=stop_reason or 'REGISTERED_MATRIX_FINISHED'))

if __name__=='__main__':main(sys.argv[1])
