import json,subprocess,sys,time,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1';start=time.time();completed=[]
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
def run(name,args,cwd=R):
 with (O/f'{name}.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,args)],cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{name} exit {r.returncode}; preserved {name}.log')
 print('DONE',name,round(time.time()-start,1),flush=True)
try:
 run('full_tests',['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],T/'sia-simulator')
 run('regressions',[O/'regressions.py'])
 for m in ('FO','FULL'):run('gate_'+m,[O/'gate_owner.py','--owner',m])
 gates={m:json.loads((O/f'gate_{m}_summary.json').read_text()) for m in ('FO','FULL')}
 passed=any(v['passed'] for v in gates.values());save('gate_result.json',dict(passed=passed,owners=gates,criterion='six consecutive supported same-state GPU-different objective-improving rounds; not JCT gate'))
 if not passed:
  save('pipeline_complete.json',dict(status='STOPPED_OPPORTUNITY_GATE',completed=[],wall=time.time()-start));sys.exit(0)
 for cell in json.loads((O/'preregistration.json').read_text())['cells']:
  run(cell['id'],[O/'run_cell.py','--cell',cell['id']]);completed.append(cell['id'])
  with (O/'execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(cell=cell['id'],status='FINISHED',summary_sha256=hashlib.sha256((O/'cells'/cell['id']/'summary.json').read_bytes()).hexdigest()))+'\n')
 save('pipeline_complete.json',dict(status='FINISHED_RUNS',completed=completed,wall=time.time()-start))
 run('summary',[O/'summarize.py'])
except Exception as e:
 save('pipeline_failure.json',dict(error=repr(e),completed=completed,wall=time.time()-start));raise
