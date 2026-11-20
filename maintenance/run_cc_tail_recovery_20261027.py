"""Four frozen paired CC trials; no grid/model/score changes."""
import argparse,collections,dataclasses,difflib,gzip,hashlib,json,math,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_joint_tail_recovery_20261027_run1';TREE=ROOT/'cc-joint-tail-recovery'
CELLS=[(250.,'JOINT_CC'),(250.,'JOINT_CC_TAIL'),(187.5,'JOINT_CC'),(187.5,'JOINT_CC_TAIL')]
def enc(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def save(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,default=enc)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def files(root):return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
def prepare():
 log=(OUT/'all_tests.log').read_text(encoding='utf-8');assert 'Ran 224 tests' in log and '\nOK\n' in log
 old=json.loads((OUT/'baseline_hashes.json').read_text());changes=[];patches=[]
 for n,h in files(TREE).items():
  if old.get(n)==h:continue
  changes.append(dict(file=n,before=old.get(n),after=h))
  if n.endswith('.py'):
   before=(OUT/'baseline_source'/n).read_text(encoding='utf-8').splitlines(True) if (OUT/'baseline_source'/n).exists() else []
   patches.extend(difflib.unified_diff(before,(TREE/n).read_text(encoding='utf-8').splitlines(True),fromfile='baseline/'+n,tofile='tail/'+n))
 with (OUT/'code.diff').open('x',encoding='utf-8') as f:f.writelines(patches)
 save(OUT/'source_changes.json',changes)
 # New isolated manifest; original manifest and all original code remain intact.
 manifest=json.loads((TREE/'source_manifest.json').read_text());manifest['parent_source_manifest_sha256']=sha(OUT/'baseline_source/source_manifest.json')
 for n in list(manifest['files']):manifest['files'][n]=sha(TREE/n)
 for n in ['sia-simulator/joint_prototype/tail_recovery.py','sia-simulator/tests/test_tail_recovery.py']:manifest['files'][n]=sha(TREE/n)
 manifest.update(algorithm_changes=True,variant='CC_TAIL_ONE_RECOVERY_V1')
 # The copied old manifest is preserved separately before regenerating this derived manifest.
 shutil.copy2(TREE/'source_manifest.json',OUT/'pre_extension_source_manifest.json')
 with (TREE/'source_manifest.json').open('w',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
 shutil.copytree(TREE,OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 save(OUT/'preregistration.json',dict(cells=CELLS,source_hashes=files(TREE),runner_hash=sha(Path(__file__)),
  workload_sha256=sha(TREE/'inputs/workload.csv'),response='medium',alpha=.5,interval=60,max_rounds=6000,
  cpus_per_node=12,gpus_per_node=2,seed_profiles=[1,8],startup_budget=250,
  mechanism='At most one strict ETA-improving observed GPU recovery for a sole arrived incumbent with empty ready and no checkpoint; extra grant exactly needed, <=one actual startup<=250; GPU frozen afterward',
  future_information=False,grid_change=False,objective_change=False,response_change=False,
  censor='At6000 record censoring; no JCT insertion or horizon extension',gate=dict(memory_gib=2,disk_gib=5)))
 print('FROZEN',len(changes))
def setup():
 sys.path.insert(0,str(OUT/'source'));sys.path.insert(0,str(OUT/'source/sia-simulator'))
 from cc_factory import verify
 verify()
 p=json.loads((OUT/'preregistration.json').read_text());assert sha(Path(__file__))==p['runner_hash']
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h,n
 return p
def gate():
 import psutil
 r=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
 if r['memory_gib']<2 or r['disk_gib']<5:raise RuntimeError('SAFETY_GATE '+str(r))
 return r
def label(ram,method):return f'ram{ram:g}_{method}'
def run(ram,method):
 setup();res=gate();out=OUT/'cells'/label(ram,method);out.mkdir(parents=True,exist_ok=False)
 save(out/'start.json',dict(time=time.time(),resources=res,method=method,ram=ram,manifest_sha256=sha(OUT/'preregistration.json')))
 from cc_factory import construct
 from run_stage2c_cell import Accumulator,independent_rho
 from joint_prototype.execution_audit import audit_execution
 c=construct(method,'medium');c.policy.node_capacities={t:dict(cpu_cores=12.,ram_gb=ram) for t in c.ngpus_per_node}
 m=Accumulator(c);seen=None;events=[];start=time.time()
 try:
  with gzip.open(out/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
   for i in range(6000):
    if c.all_complete():break
    c.step(60);r=c.joint_round_logs[-1]
    r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
    f.write(json.dumps(r,default=enc,separators=(',',':'))+'\n');f.flush()
    d=r['decision_for_next_interval'];a=d['result']['audit'];sel=d['result']['selected']
    if d['round']!=seen:
     seen=d['round'];assert a['valid'];m.status[d['result']['status']]+=1;m.solve.append(d['result']['solve_seconds']);m.events.update(e['event'] for e in d.get('liveness_events',[]))
     specs={j['job_id']:j for j in d['probe']['specs']};usage=collections.defaultdict(lambda:[0.,0.,0.])
     for name,col in sel.items():
      cfg=col['config'];cpu,mem=col['cpu_per_gpu'],col['ram_gb_per_gpu'];assert math.isclose(col['rho'],independent_rho(c.policy.oracle,name,cfg['config_id'],cpu,mem),rel_tol=1e-12)
      j=specs[name]
      if j['checkpoint_locked']:assert cfg['config_id']==j['current_config_id'] and col['placement']==dict(j['current_placement']) and (cpu,mem)==(j['current_cpu_per_gpu'],j['current_ram_gb_per_gpu'])
      for node,g in col['placement'].items():
       for k,x in enumerate([g,g*cpu,g*mem]):usage[node][k]+=x
     for v in usage.values():assert all(x<=y+1e-7 for x,y in zip(v,[2,12,ram]))
     for x in a.get('tail_recoveries',[]):
      assert len(specs)==1 and not a['ready'] and not specs[x['job']]['checkpoint_locked']
      assert 0<=x['extra_budget']<=x['actual_startup']<=250
      assert x['eta_after']<x['eta_before'];events.append(x)
     for name,v in a.get('service_delay_ledgers',{}).items():
      extra=a.get('tail_grants',{}).get(name,{}).get('extra_budget',0.)
      assert v['delay_charged']<=250+extra+1e-9
    for e in r['execution']:assert audit_execution(e)['valid']
    m.last_decision_round=d['round'];m.consume(c,r)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
    if (i+1)%100==0:print(label(ram,method),'ROUND',i+1,flush=True)
  s=m.summary(c,dict(rmst_tau_seconds=360000));s.update(method=method,ram_per_node=ram,tail_recoveries=events,total_wall_seconds=time.time()-start)
  save(out/'summary.json',s);print('RESULT',label(ram,method),s['completed'],s['mean_jct'],flush=True)
 except Exception as e:save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time,last_decision=c.policy.last_decision));raise
def batch():
 setup()
 for ram,method in CELLS:
  name=label(ram,method)
  if (OUT/'cells'/name).exists():raise RuntimeError('EXISTING_CELL '+name)
  try:r=gate()
  except Exception as e:save(OUT/'safe_stop.json',dict(next_cell=name,error=repr(e)));return
  with (OUT/(name+'.log')).open('x',encoding='utf-8') as f:
   start=time.time();p=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)),'run','--ram',str(ram),'--method',method],stdout=f,stderr=subprocess.STDOUT)
  with (OUT/'execution_manifest.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(cell=name,exit_code=p.returncode,wall_seconds=time.time()-start,resources=r))+'\n')
  print('CELL_END',name,p.returncode,flush=True)
  if p.returncode:return
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','batch','run']);p.add_argument('--ram',type=float);p.add_argument('--method');a=p.parse_args()
 if a.action=='run':run(a.ram,a.method)
 else:globals()[a.action]()
