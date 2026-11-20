import hashlib,json,subprocess,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[2];start=time.time();done=[]
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
def run(name,args):
 with (O/(name+'.log')).open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name+':exit'+str(r.returncode))
 print('DONE',name,'wall',time.time()-start,flush=True)
try:
 assert json.loads((O/'pipeline_complete.json').read_text())['status']=='STOPPED_GATE'
 assert not (O/'cells').exists()
 run('gate_analysis_v2',[O/'gate_analyze_v2.py'])
 assert json.loads((O/'gate_result_v2.json').read_text())['passed']
 for method in ('FO','FULL'):
  run(method,[O/'run_cell.py','--cell',method]);done.append(method)
  with (O/'execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(cell=method,summary_sha256=hashlib.sha256((O/'cells'/method/'summary.json').read_bytes()).hexdigest()))+'\n')
 run('report_v2',[O/'report_v2.py'])
 save('pair_complete.json',dict(status='FINISHED',completed=done,wall=time.time()-start))
except Exception as e:
 save('pair_failure.json',dict(error=repr(e),completed=done,wall=time.time()-start));raise
