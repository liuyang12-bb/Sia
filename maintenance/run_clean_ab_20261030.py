"""Frozen15-cell clean mechanism ablations, isolated serial subprocesses."""
import argparse,collections,dataclasses,gzip,hashlib,importlib.metadata,json,math,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run1';TREE=ROOT/'cc-joint-clean-ab';SRC=OUT/'frozen_source'
CELLS=[('A20',x) for x in ('A00','A01','A10','A11','A_SIA')]+[(w,x) for w in ('COMPACT','SPREAD') for x in ('B_FP','B_FO','B_JP','B_JO','B_SIA')]
def enc(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def hashes(p):return {f.relative_to(p).as_posix():sha(f) for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,default=enc)
def resources():
 import psutil
 x=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
 if x['memory_gib']<2 or x['disk_gib']<5:raise RuntimeError('RESOURCE_GATE '+str(x))
 return x
def freeze():
 assert '\nOK\n' in (OUT/'tests/full_suite1.log').read_text() and 'Ran 255 tests' in (OUT/'tests/full_suite1.log').read_text()
 assert '\nOK\n' in (OUT/'tests/green_final.log').read_text()
 assert json.loads((OUT/'original_sia_regression.json').read_text())['exact_state_match']
 base=json.loads((OUT/'baseline_source_hashes.json').read_text())
 for n,h in base.items():assert sha(TREE/n)==h,n
 assert not json.loads((OUT/'C_coverage.json').read_text())['missing']
 shutil.copytree(TREE,SRC,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
 p=dict(version='CLEAN_AB_V1',frozen_unix=time.time(),cells=CELLS,source_hashes=hashes(SRC),
    runner_hash=sha(Path(__file__)),design_hash=sha(OUT/'DESIGN.md'),input_hashes=hashes(OUT/'inputs'),
    packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','cvxpy','cylp','psutil']},
    config=dict(interval=60,max_rounds=6000,seed_profiles=[1,8],startup_budget=250,quantum=300),
    nodes=dict(A=dict(gpu_per_node=2,cpu_per_node=6.,ram_per_node=187.5),B=dict(gpu_per_node=2,cpu_per_node=6.,ram_per_node=125.)),
    alpha=dict(A=0.,B=.5),A_objective='SIA_FAIR_UTILITY_AND_ORIGINAL_RESTART_PROXY',B_objective='SUM_RECIPROCAL_CURRENT_COMPLETION_ETA',
    B_serial='Common-controller proportional-domain MIP, frozen selected set and GPU type/count/nodecount; aux/placement optimization second. NOT official Sia.',
    B_expected_identity='FP and JP are algorithmically equivalent when all columns proportional; exact physical trajectory is a negative-control contract.',
    boundary='UNCHANGED_R0_E_LEGACY; all coverage gaps/clipping explicitly retained, no real-hardware evidence',
    failure='Stop on implementation/solver/audit failure, retain all evidence; no result-driven tuning. Censor at6000rounds, do not impute JCT.',
    rmst='Use tau<=minimum unfinished followup if censoring; no completed-only full-mean claims',
    source_version='Complete snapshot+SHA256; base289 unchanged, only explicit new opt-in module and tests.',
    tests=dict(old=245,new=11,full_suite_seen=255,extra_chain_test_in_final_new_suite=True),
    warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
 save(OUT/'preregistration.json',p);shutil.make_archive(str(OUT/'frozen_source_snapshot'),'zip',SRC)
 print('FROZEN15',sha(OUT/'preregistration.json'),flush=True)
def setup():
 p=json.loads((OUT/'preregistration.json').read_text());assert sha(Path(__file__))==p['runner_hash']
 for n,h in p['source_hashes'].items():assert sha(SRC/n)==h,n
 for n,h in p['input_hashes'].items():assert sha(OUT/'inputs'/n)==h,n
 assert sha(OUT/'DESIGN.md')==p['design_hash']
 for n,v in p['packages'].items():assert importlib.metadata.version(n)==v,n
 sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
 from cc_factory import verify
 verify();return p
def cell(workload,name):
 p=setup();assert [workload,name] in p['cells'];res=resources();folder=OUT/'cells'/(workload+'_'+name);folder.mkdir(exist_ok=False)
 save(folder/'start.json',dict(time=time.time(),resources=res,argv=sys.argv,preregistration_sha=sha(OUT/'preregistration.json')))
 from resource_study import CAPACITIES
 from breakpoint_study import build
 from joint_prototype.clean_ablation import install
 from joint_prototype.execution_audit import audit_execution
 from run_stage2c_cell import Accumulator,independent_rho
 group='A' if workload=='A20' else 'B';cap=(6.,187.5) if group=='A' else (6.,125.);CAPACITIES['CLEAN']=cap
 external=name.endswith('SIA');prop=group=='A' or name.endswith('P');gpu='fixed' if name.startswith('B_F') else 'joint'
 response='neutral' if group=='A' else 'medium';base='SIA_P' if external else ('CC_P' if prop else 'CC_FULL')
 c=None;started=time.time()
 try:
  c=build(base,'CLEAN',response,OUT/'inputs'/(workload+'_workload.csv'))
  if not external:install(c,group,int(name[1]) if group=='A' else 1,int(name[2]) if group=='A' else 1,gpu,'prop' if prop else 'opt')
  m=Accumulator(c);last=None;counts=collections.Counter();previous_progress={};first_wait=collections.Counter();later_wait=collections.Counter()
  with gzip.open(folder/'rounds.jsonl.gz','xt',encoding='utf8') as f:
   for i in range(p['config']['max_rounds']):
    if c.all_complete():break
    c.step(60);r=c.joint_round_logs[-1]
    r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
    f.write(json.dumps(r,default=enc,separators=(',',':'))+'\n');f.flush()
    d=r['decision_for_next_interval'];result=d['result'];a=result['audit']
    if d['round']!=last:
     last=d['round'];assert a['valid'];m.status[result['status']]+=1;m.solve.append(result['solve_seconds']);m.events.update(e['event'] for e in d.get('liveness_events',[]))
     specs={s['job_id']:s for s in d['probe']['specs']};usage=collections.defaultdict(lambda:[0.,0.,0.]);selected=result['selected']
     assert {j for j,s in specs.items() if s['checkpoint_locked']}<=set(selected)
     for j,col in selected.items():
      q=specs[j];cp,ram=col['cpu_per_gpu'],col['ram_gb_per_gpu'];cfg=col['config'];rho=independent_rho(c.policy.oracle,j,cfg['config_id'],cp,ram)
      assert math.isclose(col['rho'],rho,rel_tol=1e-12)
      if group=='A':assert rho==1.
      if prop:assert math.isclose(cp,cap[0]/2,abs_tol=1e-8) and math.isclose(ram,cap[1]/2,abs_tol=1e-8)
      if q['checkpoint_locked']:
       assert cfg['config_id']==q['current_config_id'] and col['placement']==dict(q['current_placement'])
       assert math.isclose(cp,q['current_cpu_per_gpu'],abs_tol=1e-9) and math.isclose(ram,q['current_ram_gb_per_gpu'],abs_tol=1e-9);counts['lock_checks']+=1
      for node,g in col['placement'].items():
       for k,v in enumerate((g,g*cp,g*ram)):usage[node][k]+=v
      counts['selected']+=1;counts['nonproportional_cpu']+=abs(cp-cap[0]/2)>1e-8;counts['nonproportional_ram']+=abs(ram-cap[1]/2)>1e-8
     for node,v in usage.items():
      assert all(x<=y+1e-7 for x,y in zip(v,[2,*cap]))
      m.idle_saturated+=int(v[1]>=cap[0]-1e-7 and v[0]<2)
      for k,x in zip(('gpu','cpu','ram'),v):m.node_peaks[node][k]=max(m.node_peaks[node][k],x)
     if external:
      assert d['strict_full_intent'] and c.policy.canonical(c.allocations,c.allocations)==c.policy.canonical(d['original_sia_allocations'],c.allocations)
     else:
      assert d['clean_contract']['objective']==p[group+'_objective']
      counts['serial_proposals']+=len(a.get('serial_proposals',[]))
      for event in a.get('serial_proposals',[]):
       assert {q['job']:q['config'] for q in event['selected']}==event['fixed'];counts['serial_fixed_checks']+=1
      for j,v in a.get('service_delay_ledgers',{}).items():assert v['delay_charged']<=250+a.get('tail_grants',{}).get(j,{}).get('extra_budget',0.)+1e-9
    for e in r['execution']:
     assert audit_execution(e)['valid'];j=e['job_id'];assert e['progress_after']>=previous_progress.get(j,0.)-1e-8;previous_progress[j]=e['progress_after']
     job=next(x for x in c.jobs if x.name==j)
     (first_wait if job.first_allocation_time is None else later_wait)[j]+=e['waiting_seconds']
    # Fresh decision checks handled above, without obsolete per-job no-slowdown assertion.
    m.last_decision_round=d['round'];m.consume(c,r)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
    if (i+1)%100==0:print(workload,name,'ROUND',i+1,'COMPLETE',sum(j.completion_time is not None for j in c.jobs),flush=True)
  summary=m.summary(c,dict(rmst_tau_seconds=360000-max(j.submission_time for j in c.jobs)))
  summary.update(workload=workload,method=name,group=group,checks=dict(counts),first_wait=dict(first_wait),later_wait=dict(later_wait),total_wall_seconds=time.time()-started)
  save(folder/'summary.json',summary);save(folder/'output_hashes.json',{x.name:sha(x) for x in folder.iterdir() if x.is_file()})
  print('RESULT',workload,name,summary['completed'],summary['mean_jct'],flush=True)
 except Exception as e:
  save(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time if c else None,last_decision=c.policy.last_decision if c else None));raise
def batch():
 setup();manifest=OUT/'execution_manifest.jsonl';records=[json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
 for i,r in enumerate(records):
  assert r['index']==i and r['cell']==list(CELLS[i]) and r['exit_code']==0,'UNSAFE_PREFIX'
  folder=OUT/'cells'/'_'.join(CELLS[i])
  for n,h in json.loads((folder/'output_hashes.json').read_text()).items():assert sha(folder/n)==h
 for i,(w,n) in enumerate(CELLS):
  if i<len(records):continue
  if (OUT/'cells'/(w+'_'+n)).exists():raise RuntimeError('REFUSE_EXISTING_PARTIAL_CELL')
  try:res=resources()
  except RuntimeError as e:
   save(OUT/f'safe_stop_{i}_{time.time_ns()}.json',dict(index=i,next_cell=[w,n],error=str(e)));print('SAFE_STOP',e,flush=True);return
  started=time.time()
  with (OUT/(w+'_'+n+'.log')).open('x',encoding='utf8') as f:
   r=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)),'cell','--workload',w,'--name',n],stdout=f,stderr=subprocess.STDOUT)
  record=dict(index=i,cell=[w,n],exit_code=r.returncode,wall_seconds=time.time()-started,resources=res)
  with manifest.open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n')
  print('CELL_END',i+1,len(CELLS),w,n,r.returncode,flush=True)
  if r.returncode:return
 print('ALL15_FINISHED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','batch','cell']);p.add_argument('--workload');p.add_argument('--name');a=p.parse_args()
 if a.action=='freeze':freeze()
 elif a.action=='batch':batch()
 else:cell(a.workload,a.name)
