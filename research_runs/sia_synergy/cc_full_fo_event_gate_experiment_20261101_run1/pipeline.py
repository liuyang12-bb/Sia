import json,subprocess,sys,time,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];start=time.time();completed=[]
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
def run(name,args):
 with (O/f'{name}.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,args)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{name}:exit{r.returncode}')
 print('DONE',name,time.time()-start,flush=True)
try:
 run('gate_tests',[O/'test_event_gate.py']);run('event_gate',[O/'event_gate.py'])
 if not json.loads((O/'event_gate_result.json').read_text())['passed']:
  save('pipeline_complete.json',dict(status='STOPPED_GATE',completed=[],wall=time.time()-start));sys.exit(0)
 for c in json.loads((O/'preregistration.json').read_text())['cells']:
  run(c['id'],[O/'run_cell.py','--cell',c['id']]);completed.append(c['id'])
  with (O/'execution_manifest.jsonl').open('a') as f:f.write(json.dumps(dict(cell=c['id'],summary_sha256=hashlib.sha256((O/'cells'/c['id']/'summary.json').read_bytes()).hexdigest()))+'\n')
 save('pipeline_complete.json',dict(status='FINISHED_RUNS',completed=completed,wall=time.time()-start))
 run('summary',[O/'summarize.py'])
except Exception as e:
 save('pipeline_failure.json',dict(error=repr(e),completed=completed,wall=time.time()-start));raise
