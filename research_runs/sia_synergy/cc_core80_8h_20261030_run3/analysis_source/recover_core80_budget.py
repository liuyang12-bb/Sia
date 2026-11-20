"""One transparent operational-budget amendment; frozen science/source intact."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];TREE=ROOT/'cc-joint-core80-8h-v3';OLD=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3';OUT=OLD/'execution_recovery1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def append(p,v):
 with p.open('a',encoding='utf8') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def stop(proc):
 try:
  parent=psutil.Process(proc.pid);kids=parent.children(recursive=True)
  for q in reversed(kids):
   try:q.terminate()
   except psutil.NoSuchProcess:pass
  try:parent.terminate()
  except psutil.NoSuchProcess:pass
  psutil.wait_procs(kids+[parent],timeout=10)
 except psutil.NoSuchProcess:pass
def main():
 p=json.loads((OLD/'preregistration.json').read_text());assert not (OUT/'preregistration.json').exists()
 for path,h in p['source_hashes'].items():assert sha(TREE/path)==h,path
 reused=('dual_medium_SIA_P','dual_medium_FULL');events=[json.loads(x) for x in (OLD/'execution_manifest.jsonl').read_text().splitlines()]
 proofs={}
 for cell in reused:
  end=next(x for x in events if x['event']=='CELL_END' and x['cell']==cell)
  assert end['wall_seconds']<1800
  for path,h in end['output_hashes'].items():assert sha(OLD/path)==h,path
  proofs[cell]=dict(directory=str(OLD/'cells'/cell),state=end['state'],wall_seconds=end['wall_seconds'],hashes=end['output_hashes'],
    justification='Ended before both process budgets: earlier completion/failure unchanged by increasing a maximum. Same source/input/physical rules; not an independent replicate.')
 order=['dual_neutral_'+x for x in ('A00','A01','A10','A11')]+['dual_medium_'+x for x in ('SIA_AUX','PROP','CPU','MEM','FO')]
 p.update(output=str(OUT),version=p['version']+'_OPERATIONAL_BUDGET_AMENDMENT1',cell_wall_limit_seconds=3000,execution_order=order,
    parent_manifest_sha256=sha(OLD/'preregistration.json'),reused_outputs=proofs,
    amendment=dict(unix=time.time(),reason='Uniform30min process budget was too short for observed candidate/solver overhead. This revision uses only computation-time evidence; missing complete JCTs have not been estimated.',
     unchanged=['All302source file hashes','80-job work/arrivals','capacities','R0/Elegacy/alpha','6000simulated rounds','60sec interval','per-MIP30sec/gap0','seeds/initialization','2GiB/5GiB gates','original global8h deadline'],
     change='Uniform operational process cap50min; service four cells prioritized by smaller model size. No simulation-window extension or performance parameter tuning.',
     old_partials='All1800sec/interrupted attempts retained; every one of nine rerun cells is selected from this new attempt regardless of outcome; no best-result selection.',
     optional='Original optional4 not expanded; only after core fully valid and remaining budget. FULL numerical failure retained, so optional gate currently fails.'))
 save(OUT/'preregistration.json',p)
 mapping={c['id']:str(OLD/'cells'/c['id'] if c['id'] in reused else OUT/'cells'/c['id']) for c in p['cells']}
 save(OLD/'canonical_result_paths.json',dict(manifest=str(OUT/'preregistration.json'),manifest_sha256=sha(OUT/'preregistration.json'),paths=mapping,reused=proofs,
      selection='Source/science identical; only operational budget amended. Rerun paths fixed before any of their results. Original1800sec evidence is supplementary, not mixed as partial means.'))
 ledger=OUT/'execution_manifest.jsonl';append(ledger,dict(event='BATCH_START',unix=time.time(),pid=os.getpid(),prereg_sha256=sha(OUT/'preregistration.json')))
 for cell in order:
  if time.time()>=p['formal_stop_unix']:append(ledger,dict(event='SAFE_STOP',reason='GLOBAL_FORMAL_DEADLINE',unix=time.time()));break
  resources=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
  if resources['memory_gib']<2 or resources['disk_gib']<5:append(ledger,dict(event='SAFE_STOP',cell=cell,resources=resources,unix=time.time()));break
  assert not (OUT/'cells'/cell).exists()
  cmd=[sys.executable,'-X','utf8','-B',str(TREE/'run_core80.py'),'--manifest',str(OUT/'preregistration.json'),'--cell',cell]
  t=time.time();append(ledger,dict(event='CELL_START',cell=cell,unix=t,resources=resources,command=cmd))
  with (OUT/f'{cell}.console.log').open('x',encoding='utf8') as f:
   proc=subprocess.Popen(cmd,cwd=TREE,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));append(ledger,dict(event='CELL_PROCESS',cell=cell,pid=proc.pid,unix=time.time()))
   limit=min(t+3000,p['formal_stop_unix']);timed=False
   while proc.poll() is None:
    if time.time()>=limit:stop(proc);timed=True;break
    time.sleep(5)
   proc.wait()
  d=OUT/'cells'/cell;s=d/'summary.json'
  state='COMPUTE_TIMEOUT' if timed else json.loads(s.read_text())['termination'] if s.exists() else 'FAILED'
  if timed:save(d/'COMPUTE_TIMEOUT.json',dict(status=state,wall_limit_seconds=3000,global_formal_stop=p['formal_stop_unix'],unix=time.time(),raw_partial_untouched=True))
  append(ledger,dict(event='CELL_END',cell=cell,state=state,exit_code=proc.returncode,unix=time.time(),wall_seconds=time.time()-t,output_hashes={str(x.relative_to(OUT)):sha(x) for x in d.glob('*') if x.is_file()}))
  print('CELL_END',cell,state,'wall',round(time.time()-t,2),flush=True)
 append(ledger,dict(event='BATCH_END',unix=time.time(),reason='Fixed new order finished or global/safety gate; no science changes.'))
if __name__=='__main__':main()
