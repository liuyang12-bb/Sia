"""Prepare isolated unchanged-source 42-cell comparison and reusable runner."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1'
TREE=ROOT/'cc-joint-resource-study'
PREF=ROOT/'research_runs/sia_synergy/cc_resource_mixes_preflight_20261028_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
    OUT.mkdir(exist_ok=False)
    baseline=json.loads((PREF/'protected_source_hashes.json').read_text())
    for p,h in baseline.items():assert sha(ROOT/p)==h,p
    shutil.copytree(TREE,OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (OUT/'inputs').mkdir()
    for mix in ['M1','M2','M3']:shutil.copyfile(PREF/(mix+'_workload.csv'),OUT/'inputs'/(mix+'_workload.csv'))
    (OUT/'CURRENT_TASK.md').write_text('PREPARING authorized42 cells: M1/M2/M3 x R01/R11 x six methods Medium=36; M2/R11 six Neutral=6. Unchanged CC-TAIL source, grid, R0/E-legacy and input cohorts. Explicit constant boundary extension accepted by user. Never tune from results. 60seconds/6000rounds, serial independent processes, memory2GiB/disk5GiB gates. Read preregistration.json and execution_manifest.jsonl before recovery; verify completed prefix and no active duplicate runner. Preserve partial failures, stop on correctness failure. Deliver REPORT.md and independent raw-event audit, per-policy clipping and CPU/MEM ablation. Do not resume104 or160 jobs.\n',encoding='utf8')
    old=(ROOT/'maintenance/run_cc_resource_factorial_20261027.py').read_text(encoding='utf8')
    run=old[old.index('def run(scenario,response,method):'):old.index('\ndef batch():')]
    run=run.replace('def run(scenario,response,method):','def run(mix,scenario,response,method):').replace('name=label(scenario,response,method);assert [scenario,response,method]','name=label(mix,scenario,response,method);assert [mix,scenario,response,method]')
    run=run.replace('c=build(method,scenario,response);','c=build(method,scenario,response,OUT/\'inputs\'/(mix+\'_workload.csv\'));')
    run=run.replace('summary.update(scenario=scenario,','summary.update(mix=mix,scenario=scenario,')
    target=ROOT/'maintenance/run_resource_mixes42_20261028.py'
    header='''"""Frozen42-cell runner; scientific run body reused from the accepted30-cell runner."""
import argparse,collections,dataclasses,gzip,hashlib,importlib.metadata,json,math,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1'
METHODS=['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']
CELLS=[(x,s,'medium',m) for x in ['M1','M2','M3'] for s in ['R01','R11'] for m in METHODS]+[('M2','R11','neutral',m) for m in METHODS]
def enc(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def save(p,x):
 with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,default=enc)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def label(mix,s,r,m):return f'{mix}_{s}_{r}_{m}'
def setup():
 p=json.loads((OUT/'preregistration.json').read_text());assert sha(Path(__file__))==p['runner_hash']
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h,n
 for n,h in p['input_hashes'].items():assert sha(OUT/'inputs'/n)==h,n
 for n,v in p['packages'].items():assert importlib.metadata.version(n)==v,n
 sys.path[:0]=[str(OUT/'source'),str(OUT/'source/sia-simulator')]
 from cc_factory import verify
 verify();return p
def resources():
 import psutil
 x=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
 if x['memory_gib']<2 or x['disk_gib']<5:raise RuntimeError('RESOURCE_GATE '+str(x))
 return x
'''
    footer='''
def batch():
 p=setup();records=[]
 if (OUT/'execution_manifest.jsonl').exists():records=[json.loads(x) for x in (OUT/'execution_manifest.jsonl').read_text().splitlines() if x.strip()]
 for i,r in enumerate(records):
  assert r['index']==i and r['cell']==label(*CELLS[i]) and r['exit_code']==0,'NONRESUMABLE_PREFIX'
  folder=OUT/'cells'/r['cell'];h=json.loads((folder/'output_hashes.json').read_text())
  for name,v in h.items():assert sha(folder/name)==v,'PREFIX_HASH_MISMATCH'
 for i,(mix,s,r,m) in enumerate(CELLS):
  if i<len(records):continue
  name=label(mix,s,r,m)
  if (OUT/'cells'/name).exists():raise RuntimeError('REFUSE_PARTIAL_EXISTING '+name)
  try:res=resources()
  except Exception as e:
   save(OUT/f'safe_stop_{i}_{time.time_ns()}.json',dict(index=i,next_cell=name,error=repr(e)));print('SAFE_STOP',name,e,flush=True);return
  with (OUT/(name+'.log')).open('x',encoding='utf8') as f:
   start=time.time();proc=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)),'run','--mix',mix,'--scenario',s,'--response',r,'--method',m],stdout=f,stderr=subprocess.STDOUT)
  with (OUT/'execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(index=i,cell=name,exit_code=proc.returncode,wall_seconds=time.time()-start,resources=res))+'\\n')
  print('CELL_END',i+1,'/42',name,proc.returncode,flush=True)
  (OUT/'CURRENT_PROGRESS.json').write_text(json.dumps(dict(ended=i+1,total=42,last_cell=name,exit_code=proc.returncode,time=time.time()),indent=2),encoding='utf8')
  if proc.returncode:return
 print('ALL42_ENDED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['batch','run']);p.add_argument('--mix');p.add_argument('--scenario');p.add_argument('--response');p.add_argument('--method');a=p.parse_args()
 if a.action=='run':run(a.mix,a.scenario,a.response,a.method)
 else:batch()
'''
    target.write_text(header+run+footer,encoding='utf8')
    shutil.copyfile(target,OUT/target.name)
    (OUT/'runner_reuse_diff.txt').write_text('Run loop identical to accepted30-cell runner except: mix in label, registered tuple and summary; explicit frozen workload_path argument. No scheduling/execution logic changes. Batch matrix and isolation/protection are new.\n',encoding='utf8')
    print('PREPARED',OUT)
if __name__=='__main__':main()
