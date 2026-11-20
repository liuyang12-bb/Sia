"""Frozen, independent-process mechanism experiment; no policy code changes."""
import argparse, collections, csv, gzip, hashlib, importlib.metadata, json, math, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
ORIGINAL=ROOT/'sia-joint-service-horizon/sia-simulator'
OLD=ROOT/'research_runs/sia_synergy/joint_service_horizon_trial_run4_20261026'
PYTHON=ROOT/'.venv-sia-baseline/Scripts/python.exe'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,default=default)
def default(v):
 import dataclasses
 if dataclasses.is_dataclass(v):return dataclasses.asdict(v)
 if hasattr(v,'item'):return v.item()
 raise TypeError(type(v).__name__)
def freeze(out):
 import psutil
 if out.exists():raise FileExistsError(out)
 if psutil.virtual_memory().available<2*2**30:raise RuntimeError('MEMORY_GATE_2GiB')
 if psutil.disk_usage(str(BASE)).free<5*2**30:raise RuntimeError('DISK_GATE_5GiB')
 prior=json.loads((OLD/'preregistration.json').read_text())
 for p,h in prior['source_hashes'].items():assert sha(ORIGINAL/p)==h,p
 out.mkdir()
 # Copy the complete actual simulator, including untracked source and profiles.
 shutil.copytree(ORIGINAL,out/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ['metrics_legacy_adapter.py']:
  shutil.copy2(OLD/name,out/name)
 shutil.copy2(__file__,out/'experiment.py')
 shutil.copytree(BASE/'offline_v1',out/'inputs')
 data=json.loads((out/'inputs/offline_evidence.json').read_text(encoding='utf-8'))
 cfg=json.loads((ROOT/'research_runs/sia_synergy/sia_r0_full160_comparison/inputs/config.json').read_text())
 registry=ROOT/'research_runs/sia_synergy/sia_r0_full160_comparison/inputs/registry.json'
 shutil.copy2(registry,out/'inputs/registry.json')
 models=ROOT/'research_sources/Synergy/simulator/models';shutil.copytree(models,out/'inputs/models',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 runs=[]
 for scenario in ['D2_gpu_auxiliary_coupling','D1_auxiliary_exchange','D3_queued_training_proposal']:
  nodes=data['scenarios'][scenario]['nodes'];c=json.loads(json.dumps(cfg))
  c['cluster'].update(nnodes={k:1 for k in nodes},ngpus_per_node={k:v['gpus'] for k,v in nodes.items()},max_physical_nodes={k:1 for k in nodes})
  save(out/'inputs'/f'{scenario}_config.json',c)
  for response in (['medium','neutral','cpu_symmetric'] if scenario.startswith('D3') else ['medium','neutral'] if scenario.startswith('D2') else ['medium']):
   for method in ['sia_p','sia_aux','G112']:
    runs.append(dict(id=f'{scenario}_{response}_{method}',scenario=scenario,response='neutral' if response=='neutral' else 'medium',capacity_control=response,method=method,nodes={k:dict(v,cpu=12.0) if response=='cpu_symmetric' else v for k,v in nodes.items()}))
 p=dict(status='PREREGISTERED',tree=str(out/'source'),runs=runs,interval=60,max_rounds=600,initial_profiles=[1,8],reference_only_no_runtime_injection=True,
  warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE',
  rules='Fixed order. No JCT tuning. Complete each admitted scenario to completion/censoring. D3 starts only if D2 live candidate ranking reversal is observed. A policy failing to select an opportunity does not remove results.',
  packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','cvxpy','psutil','cylp']},
  source_hashes={str(f.relative_to(out/'source')):sha(f) for f in (out/'source').rglob('*.py')},
  input_hashes={str(f.relative_to(out)):sha(f) for f in (out/'inputs').rglob('*') if f.is_file()},
  tool_hashes={k:sha(out/k) for k in ['experiment.py','metrics_legacy_adapter.py']},
  original_head=subprocess.check_output(['git','-C',str(ORIGINAL),'rev-parse','HEAD'],text=True).strip(),
  original_status=subprocess.check_output(['git','-C',str(ORIGINAL),'status','--short'],text=True),
  freeze_memory_gib=psutil.virtual_memory().available/2**30,created_unix=time.time())
 save(out/'preregistration.json',p)
 (out/'CURRENT_TASK.md').write_text('New-data mechanism experiments, frozen 600 rounds, D1/D2 plus gated D3; no policy tuning, full160 remains paused. Resume only cells with no directory; failures never overwritten.\n',encoding='utf-8')
 print('FROZEN',len(runs),out,flush=True)
def load(out):
 p=json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
 for rel,h in p['source_hashes'].items():assert sha(out/'source'/rel)==h,rel
 for rel,h in p['input_hashes'].items():assert sha(out/rel)==h,rel
 for rel,h in p['tool_hashes'].items():assert sha(out/rel)==h,rel
 return p
def cell(out,cid):
 p=load(out);cdef=next(c for c in p['runs'] if c['id']==cid)
 import psutil
 if psutil.virtual_memory().available<2*2**30:raise RuntimeError('MEMORY_GATE_2GiB')
 if psutil.disk_usage(str(out)).free<5*2**30:raise RuntimeError('DISK_GATE_5GiB')
 directory=out/'cells'/cid;directory.mkdir(parents=True,exist_ok=False)
 save(directory/'start.json',dict(cell=cdef,memory_gib=psutil.virtual_memory().available/2**30,preregistration_sha256=sha(out/'preregistration.json')))
 sys.path.insert(0,str(out/'source'));sys.path.insert(0,str(out))
 import pandas as pd
 from joint_prototype.full160 import make_cluster
 from joint_prototype.small_mechanism import ProbeSia,ProbeAux
 from joint_prototype.service_horizon import install
 from joint_prototype.physical_domain import restrict_single_node_domain
 from metrics_legacy_adapter import Metrics
 cfg=json.loads((out/'inputs'/f"{cdef['scenario']}_config.json").read_text())
 c=make_cluster(cfg,pd.read_csv(out/'inputs'/f"{cdef['scenario']}.csv"),out/'inputs/registry.json',out/'inputs/models','sia_p' if cdef['method'].startswith('sia') else 'joint_o',cdef['response'],'main',True)
 c.policy.node_capacities={k:dict(cpu_cores=v['cpu'],ram_gb=v['ram']) for k,v in cdef['nodes'].items()}
 restrict_single_node_domain(c.policy)
 if cdef['method']=='sia_p':c.policy.__class__=ProbeSia
 elif cdef['method']=='sia_aux':c.policy.__class__=ProbeAux;c.policy.mode='b4'
 elif cdef['method']=='G112_G8':
  from joint_prototype.cpu_exchange_grid import install as install_grid
  install_grid(c)
 else:install(c)
 m=Metrics(c);started=time.perf_counter();flips=[]
 try:
  with gzip.open(directory/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
   for i in range(p['max_rounds']):
    if c.all_complete():break
    c.step(60);r=c.joint_round_logs[-1];d=r['decision_for_next_interval']
    r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
    f.write(json.dumps(r,default=default,separators=(',',':'))+'\n');f.flush();m.consume(c,r)
    # Live, same-job/count proportional candidates: reference T0 never used.
    props=d['probe']['proportional_columns'];w=[]
    for a in props:
     for b in props:
      if a['job_id']==b['job_id'] and a['config']['gpus']==b['config']['gpus'] and a['config']['gpu_type']!=b['config']['gpu_type'] and a['config']['sia_goodput']>b['config']['sia_goodput']+1e-8 and a['query']['effective_goodput']<b['query']['effective_goodput']-1e-8:
       w.append(dict(job=a['job_id'],raw_preferred=a,actual_preferred=b))
    if w:flips.append(dict(round=i+1,time=c.current_time,witness=w[0]))
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
    if (i+1)%100==0:print('PROGRESS',cid,i+1,flush=True)
  s=m.summary(c,dict(rmst_tau_seconds=p['max_rounds']*60));s.update(cell=cdef,wall_seconds=time.perf_counter()-started,termination='ALL_COMPLETE' if c.all_complete() else 'ROUND_LIMIT',live_rank_reversal_rounds=len(flips))
  save(directory/'summary.json',s);save(directory/'opportunity.json',dict(live_rank_reversals=flips));print('COMPLETE',cid,s['completed'],s['mean_jct'],flush=True)
 except Exception as e:
  save(directory/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time,last_decision=c.policy.last_decision));raise
def batch(out):
 p=load(out)
 with (out/'execution_manifest.jsonl').open('x',encoding='utf-8') as mf:
  for cd in p['runs']:
   if cd['scenario'].startswith('D3'):
    prior=[out/'cells'/v['id']/'summary.json' for v in p['runs'] if v['scenario'].startswith('D2') and v['response']=='medium']
    if not any(f.exists() and json.loads(f.read_text())['live_rank_reversal_rounds']>0 for f in prior):
     mf.write(json.dumps(dict(cell=cd['id'],status='NOT_STARTED_D2_LIVE_OPPORTUNITY_ABSENT'))+'\n');mf.flush();continue
   tick=time.time()
   import psutil
   if psutil.virtual_memory().available<2*2**30:
    mf.write(json.dumps(dict(cell=cd['id'],status='STOP_MEMORY_GATE',available_gib=psutil.virtual_memory().available/2**30))+'\n');mf.flush();return
   with (out/(cd['id']+'.log')).open('x',encoding='utf-8') as log:
    r=subprocess.run([str(PYTHON),'-X','utf8','-B',str(out/'experiment.py'),'cell','--output',str(out),'--cell',cd['id']],cwd=out/'source',stdout=log,stderr=subprocess.STDOUT)
   record=dict(cell=cd['id'],exit_code=r.returncode,wall_seconds=time.time()-tick)
   mf.write(json.dumps(record)+'\n');mf.flush();print(record,flush=True)
   if r.returncode:raise RuntimeError('CELL_FAILED:'+cd['id'])
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','cell','batch']);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--cell');a=ap.parse_args()
 {'freeze':freeze,'batch':batch,'cell':lambda o:cell(o,a.cell)}[a.mode](a.output.resolve())
