"""Serial, isolated small-training gates. No automatic formal batch launch."""
import argparse,collections,csv,dataclasses,gzip,hashlib,importlib.metadata,json,math,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/small_attempt2';SRC=OUT.parent/'source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def enc(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(str(type(x)))
def save(p,x):
 with p.open('x',encoding='utf8') as f:json.dump(x,f,default=enc,ensure_ascii=False,indent=2,allow_nan=False)
def resources():
 import psutil
 r=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
 if r['memory_gib']<2 or r['disk_gib']<5:raise RuntimeError('RESOURCE_GATE '+str(r))
 return r
def setup():
 p=json.loads((OUT/'small_preregistration.json').read_text())
 assert sha(Path(__file__))==p['runner_hash']
 for n,h in p['source_hashes'].items():assert sha(SRC/n)==h,n
 for n,h in p['fixtures'].items():assert sha(OUT/'fixtures'/n)==h,n
 for n,v in p['packages'].items():assert importlib.metadata.version(n)==v,n
 sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
 return p
def run(factor,kind,arm):
 p=setup();res=resources();name=f'{factor}_{kind}_{arm}';folder=OUT/'mechanisms'/name;folder.mkdir(parents=True,exist_ok=False)
 save(folder/'start.json',dict(time=time.time(),resources=res,command=sys.argv,preregistration_hash=sha(OUT/'small_preregistration.json')))
 witness=json.loads((OUT/'mechanism_witnesses.json').read_text())[factor]
 from breakpoint_study import build
 from resource_study import CAPACITIES
 c=None;start=time.time()
 try:
  if kind=='fixed':
   c=build('CC_P',witness['capacity'],'medium',OUT/'fixtures'/f'{factor}_pair.csv')
   from joint_prototype.joint_profile import JobSpec,GpuConfig
   from joint_prototype.r0_liveness import APPLICATION_TEMPLATES
   quotas=witness['proportional'] if arm=='P' else witness['alternative']
   typ=witness['gpu'];cpu,ram=CAPACITIES[witness['capacity']]
   assert sum(q['cpu'] for q in quotas)<=cpu+1e-8 and sum(q['ram'] for q in quotas)<=ram+1e-8
   for j,q in zip(c.jobs,quotas):
    ref=c.policy.oracle.reference(j.name,witness['config'])
    cfg=GpuConfig(witness['config'],typ,1,1,1.,ref['phi_c'],ref['phi_m'])
    spec=JobSpec(j.name,APPLICATION_TEMPLATES[j.app_name],(cfg,))
    query=c.policy.oracle.query(spec,cfg,q['cpu'],q['ram'],(cpu/2,ram/2))
    assert not(query.source_coordinate_clipped or query.baseline_coordinate_clipped)
    j.migrate(typ,(1,));j.goodput_multiplier=query.rho;j.first_allocation_time=0.;j.start_time=0.
   counters=collections.Counter();rows=[];last_progress={j.name:j.progress for j in c.jobs}
   with gzip.open(folder/'rounds.jsonl.gz','xt',encoding='utf8') as f:
    for i in range(6000):
     if all(j.completion_time is not None for j in c.jobs):break
     events=[]
     for j,q in zip(c.jobs,quotas):
      if j.completion_time is not None:continue
      before=(j.progress,j.execution_time,j.rescale_time,j.current_time)
      j.step(60)
      assert j.progress>=last_progress[j.name];last_progress[j.name]=j.progress
      for s in j.joint_step_segments:
       assert math.isclose(s['effective_rate'],s['raw_rate_before_rho']*s['rho'],rel_tol=1e-12)
       residual=s['progress_delta']-s['effective_rate']*s['seconds']
       assert math.isclose(residual,s['rounding_progress_residual'],abs_tol=1e-7)
       assert s['start_time']>=before[3]+min(before[2],60)-1e-7
      end=min(j.current_time,j.completion_time) if j.completion_time is not None else j.current_time
      dt=end-before[3];train=j.execution_time-before[1];delay=min(before[2],dt)
      assert abs(dt-train-delay)<1e-7
      counters['gpu_seconds']+=dt;counters['cpu_seconds']+=dt*q['cpu'];counters['ram_seconds']+=dt*q['ram']
      counters['training_seconds']+=train;counters['checkpoint_seconds']+=delay
      events.append(dict(job=j.name,before_progress=before[0],progress=j.progress,execution=train,checkpoint=delay,
         quota=q,completion=j.completion_time,segments=j.joint_step_segments))
     f.write(json.dumps(dict(round=i,time=(i+1)*60,execution=events),default=enc)+'\n')
   for j in c.jobs:rows.append(dict(job=j.name,arrival=j.submission_time,completion=j.completion_time,
      jct=j.completion_time-j.submission_time if j.completion_time is not None else None,
      progress=j.progress,training=j.execution_time,restarts=j.num_restarts or 0))
   complete=all(r['completion'] is not None for r in rows)
   summary=dict(completed=sum(r['completion'] is not None for r in rows),jobs=len(rows),rounds=i,
      mean_jct=sum(r['jct'] for r in rows)/len(rows) if complete else None,per_job=rows,counters=dict(counters),
      resource_hours={k:float(counters[k])/3600 for k in ['gpu_seconds','cpu_seconds','ram_seconds']},
      method='FIXED_DECLARED_GPU_BATCH_AND_QUOTA_OPPORTUNITY_NOT_POLICY',quota=quotas)
  else:
   method='CC_P' if arm=='P' else 'CC_'+factor
   c=build(method,witness['capacity'],'medium',OUT/'fixtures'/f'{factor}_three.csv')
   from run_stage2c_cell import Accumulator,independent_rho
   from joint_prototype.execution_audit import audit_execution
   m=Accumulator(c);last=None;nonprop=0
   with gzip.open(folder/'rounds.jsonl.gz','xt',encoding='utf8') as f:
    for i in range(6000):
     if c.all_complete():break
     c.step(60);r=c.joint_round_logs[-1];d=r['decision_for_next_interval'];a=d['result']['audit']
     r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
     assert a['valid']
     if d['round']!=last:
      last=d['round'];m.status[d['result']['status']]+=1;m.solve.append(d['result']['solve_seconds']);m.events.update(e['event'] for e in d.get('liveness_events',[]))
      cpu,ram=CAPACITIES[witness['capacity']];usage=collections.defaultdict(lambda:[0.,0.,0.])
      for j,col in d['result']['selected'].items():
       cfg=col['config'];cp,mm=col['cpu_per_gpu'],col['ram_gb_per_gpu']
       assert math.isclose(col['rho'],independent_rho(c.policy.oracle,j,cfg['config_id'],cp,mm),rel_tol=1e-12)
       if method=='CC_P':assert(cp,mm)==(cpu/2,ram/2)
       if method=='CC_CPU':assert mm==ram/2
       if method=='CC_MEM':assert cp==cpu/2
       nonprop+=(cp,mm)!=(cpu/2,ram/2)
       for n,g in col['placement'].items():
        for k,v in enumerate([g,g*cp,g*mm]):usage[n][k]+=v
      for u in usage.values():assert all(x<=y+1e-7 for x,y in zip(u,[2,cpu,ram]))
     for e in r['execution']:assert audit_execution(e)['valid']
     f.write(json.dumps(r,default=enc,separators=(',',':'))+'\n');f.flush()
     # Current donor-allowing CC contract, identical to accepted42 runner.
     # Fresh status/capacity/rho were independently checked above. Legacy
     # consume's fresh block assumes the discontinued per-job speed floor.
     m.last_decision_round=d['round']
     m.consume(c,r);c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
     if (i+1)%100==0:print(name,'ROUND',i+1,flush=True)
   summary=m.summary(c,dict(rmst_tau_seconds=360000));summary.update(nonproportional_selected_columns=nonprop,method=method)
  summary.update(factor=factor,kind=kind,arm=arm,wall_seconds=time.time()-start)
  save(folder/'summary.json',summary);save(folder/'output_hashes.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
  print('RESULT',name,summary['completed'],summary['mean_jct'],flush=True)
 except Exception as exc:
  save(folder/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=c.current_time if c else None));raise
def freeze():
 assert json.loads((OUT/'tests_result.json').read_text())['exit_code']==0
 assert json.loads((OUT/'offline_validation.json').read_text())['strict_CPU_and_MEM_opportunities']
 fixtures=OUT/'fixtures';fixtures.mkdir(exist_ok=False)
 original=list(csv.DictReader((ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv').open(encoding='utf8')))
 by={r['name']:r for r in original};w=json.loads((OUT/'mechanism_witnesses.json').read_text())
 for factor,case in w.items():
  pair=[by[n] for n in case['jobs']];third=next(r for r in original if r['name'] not in case['jobs'] and r['application'] in ['cifar10','deepspeech2'])
  for name,rows in [('pair',pair),('three',pair+[third])]:
   with (fixtures/f'{factor}_{name}.csv').open('x',encoding='utf8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])+['original_arrival','source_row_index']);writer.writeheader()
    for r in rows:writer.writerow(dict(r,time=0,original_arrival=r['time'],source_row_index=original.index(r)))
 cells=[(factor,kind,arm) for kind in ['fixed','free'] for factor in ['CPU','MEM'] for arm in ['P','O']]
 p=dict(time=time.time(),cells=cells,runner_hash=sha(Path(__file__)),source_hashes={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'},fixtures={p.name:sha(p) for p in fixtures.iterdir()},
  packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','cvxpy','cylp','psutil','autograd']},
  max_rounds=6000,interval=60,seed_profiles=[1,8],response='same R0 E-legacy Medium alpha=.5',
  fixed='Both jobs one GPU each on witness node, constant quotas, no resource reuse after a job completes; real Job.step and adaptive batch for fixedGPU, same full declared work. Batch selection identical between arms at identical state; epoch evolution changes gradients normally. Oracle rho fixed by R0.',
  free='Witness pair plus first other source-order CF/DS job; all arrive0, unchanged declared workload/batch. Original two-node capacity, unrestricted original GPU choices and unchanged CC-TAIL; only grid extends. No artificial forced co-residency.',
  gate='Both fixed and free CPU/MEM all complete, audited; corresponding O lower mean JCT than P; free O selects nonproportional points. Otherwise stop, no formal30 and no automatic new-scene search.',
  disclosure='Small controlled simultaneous arrival differs from full cohort timing. No claim of true hardware profiles or global JCT optimality.')
 save(OUT/'small_preregistration.json',p)
 print('SMALL_FROZEN',sha(OUT/'small_preregistration.json'))
def batch():
 p=setup()
 for i,(factor,kind,arm) in enumerate(p['cells']):
  res=resources();name=f'{factor}_{kind}_{arm}'
  if (OUT/'mechanisms'/name).exists():raise RuntimeError('NO_OVERWRITE '+name)
  with (OUT/f'{name}.log').open('x',encoding='utf8') as f:
   start=time.time();r=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)),'run','--factor',factor,'--kind',kind,'--arm',arm],stdout=f,stderr=subprocess.STDOUT)
  with (OUT/'small_execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(index=i,cell=name,exit_code=r.returncode,wall_seconds=time.time()-start,resources=res))+'\n')
  print('SMALL_END',i,name,r.returncode,flush=True)
  if r.returncode:return
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','batch','run']);a.add_argument('--factor');a.add_argument('--kind');a.add_argument('--arm');v=a.parse_args()
 if v.action=='freeze':freeze()
 elif v.action=='batch':batch()
 else:run(v.factor,v.kind,v.arm)
