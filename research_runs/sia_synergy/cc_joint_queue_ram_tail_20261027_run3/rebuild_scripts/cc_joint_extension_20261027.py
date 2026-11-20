"""Frozen RAM-pressure extension and read-only queue/tail attribution."""
import argparse,collections,dataclasses,gzip,hashlib,json,math,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_joint_queue_ram_tail_20261027_run3'
OLD=ROOT/'research_runs/sia_synergy/cc_joint_sustained_cpu_20job_20261027_run1'
METHODS=['SIA_P','SIA_ECT_AUX','JOINT_CC_P','JOINT_CC_CPU','JOINT_CC']
def encode(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,default=encode)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def setup():
 sys.path.insert(0,str(OUT/'source'));sys.path.insert(0,str(OUT/'source/sia-simulator'));from cc_factory import verify
 verify()
def cluster(method,response='medium'):
 from cc_factory import construct
 c=construct('JOINT_CC' if method=='JOINT_CC_CPU' else method,response)
 c.policy.node_capacities={t:dict(cpu_cores=12.,ram_gb=187.5) for t in c.ngpus_per_node}
 if method=='JOINT_CC_CPU':c.policy.aux_ram_points=(93.75,)
 return c
def prepare():
 OUT.mkdir(exist_ok=False);shutil.copytree(ROOT/'cc-joint',OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 setup()
 from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
 from joint_prototype.joint_profile import JobSpec,GpuConfig
 oracle=R0Oracle(OUT/'source/inputs/registry.json',OUT/'source/inputs/models',{'cf':'cifar10','ds':'deepspeech2'},'medium')
 evidence=[]
 for typ in ['aws','dgx-ext']:
  for job in ['cf','ds']:
   r=oracle.reference(job,f'{typ}:1:1');g=GpuConfig(f'{typ}:1:1',typ,1,1,1.,r['phi_c'],r['phi_m'])
   j=JobSpec(job,APPLICATION_TEMPLATES[oracle.job_applications[job]],(g,))
   qs=[oracle.query(j,g,6.,ram,(6.,93.75)) for ram in [62.5,93.75,125.]]
   evidence.append(dict(job=job,gpu=typ,queries=[q.to_dict() for q in qs],ratio_high=qs[2].rho/qs[1].rho,ratio_low=qs[0].rho/qs[1].rho))
 cf,ds=[next(e for e in evidence if e['gpu']=='dgx-ext' and e['job']==j) for j in ['cf','ds']]
 support=all(not(q['source_coordinate_clipped'] or q['baseline_coordinate_clipped']) for e in [cf,ds] for q in e['queries'])
 assert support and cf['ratio_high']>1 and ds['ratio_low']>0
 save(OUT/'preflight.json',dict(valid=True,evidence=evidence,witness='DGX: CF125+DS62.5=187.5 versus93.75 each; CPU6 each; same GPU/batch; all witness coordinates supported',
  donor_loss_included=True,receiver_speed_change=cf['ratio_high']-1,donor_speed_change=ds['ratio_low']-1,
  net_JCT_benefit_not_proven=True,no_jct_based_selection=True))
 files={p.relative_to(OUT/'source').as_posix():sha(p) for p in (OUT/'source').rglob('*') if p.is_file()}
 save(OUT/'preregistration.json',dict(methods=METHODS,response='medium',alpha=.5,interval=60,max_rounds=6000,
  jobs=20,workload='source/inputs/workload.csv',cpu_per_node=12,ram_per_node=187.5,gpu_per_node=2,
  cpu_ablation_ram_per_gpu=93.75,source_files=files,runner_sha256=sha(Path(__file__)),old_preregistration_sha256=sha(OLD/'preregistration.json'),
  changes=['RAM capacity250->187.5 per node','CPU-only auxiliary ablation fixes RAM93.75'],
  unchanged=['all scheduler source','response/phi/mapping','GPU/CPU','workload/work','initialization','solver','observation window'],
  gates=dict(memory_gib=2,disk_gib=5),stop='Any implementation/audit failure stops batch; no tuning or retries; censor at6000',
  warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
 print('FROZEN',cf['ratio_high'],ds['ratio_low'],flush=True)
def verify():
 p=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
 assert sha(Path(__file__))==p['runner_sha256']
 for n,h in p['source_files'].items():assert sha(OUT/'source'/n)==h,n
 setup();return p
def gate():
 import psutil
 d=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
 if d['memory_gib']<2 or d['disk_gib']<5:raise RuntimeError('RESOURCE_GATE '+str(d))
 return d
def run(method):
 verify();resources=gate();dest=OUT/'cells'/method;dest.mkdir(parents=True,exist_ok=False)
 save(dest/'start.json',dict(method=method,resources=resources,preregistration_sha256=sha(OUT/'preregistration.json'),time=time.time()))
 c=None;start=time.time()
 try:
  c=cluster(method)
  from run_stage2c_cell import Accumulator,independent_rho
  from joint_prototype.execution_audit import audit_execution
  m=Accumulator(c);counts=collections.Counter();last=None
  with gzip.open(dest/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
   for i in range(6000):
    if c.all_complete():break
    c.step(60);r=c.joint_round_logs[-1]
    f.write(json.dumps(r,default=encode,separators=(',',':'))+'\n');f.flush()
    d=r['decision_for_next_interval'];res=d['result']
    if d['round']!=last:
     last=d['round'];assert res['audit']['valid'];m.status[res['status']]+=1;m.solve.append(res['solve_seconds']);m.events.update(e['event'] for e in d.get('liveness_events',[]))
     usage={t+':0':[0.,0.,0.] for t in c.ngpus_per_node};specs={s['job_id']:s for s in d['probe']['specs']}
     for name,col in res['selected'].items():
      cfg=col['config'];cpu,ram=col['cpu_per_gpu'],col['ram_gb_per_gpu'];expected=independent_rho(c.policy.oracle,name,cfg['config_id'],cpu,ram)
      assert math.isclose(expected,col['rho'],rel_tol=1e-12)
      if method in ['SIA_P','JOINT_CC_P']:assert (cpu,ram)==(6.,93.75)
      if method=='JOINT_CC_CPU':assert ram==93.75
      s=specs[name]
      if s['checkpoint_locked']:
       assert cfg['config_id']==s['current_config_id'] and col['placement']==dict(s['current_placement'])
       assert (cpu,ram)==(s['current_cpu_per_gpu'],s['current_ram_gb_per_gpu'])
       counts['full_lock_checks']+=1
      counts['selected']+=1;counts['nonproportional_ram']+=ram!=93.75
      row=c.policy.oracle.reference(name,cfg['config_id'])
      for label,cp,ra in [('numerator',cpu,ram),('denominator',3.,62.5)]:
       counts[label+'_cpu_clip']+=not 1<=cp/row['phi_c']<=24
       counts[label+'_ram_clip']+=not 20.83<=ra/row['phi_m']<=250
      for n,g in col['placement'].items():
       for k,v in enumerate([g,g*cpu,g*ram]):usage[n][k]+=v
     for n,v in usage.items():
      assert all(x<=b+1e-7 for x,b in zip(v,[2,12,187.5]))
      for k,x in zip(['gpu','cpu','ram'],v):m.node_peaks[n][k]=max(m.node_peaks[n][k],x)
     if d.get('strict_full_intent'):assert c.policy.canonical(c.allocations,c.allocations)==c.policy.canonical(d['original_sia_allocations'],c.allocations)
    for e in r['execution']:assert audit_execution(e)['valid']
    # Suppress only legacy decision progress-floor assertions; donors may slow.
    # All current decisions are independently checked above; execution integration unchanged.
    m.last_decision_round=d['round'];m.consume(c,r)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
    if (i+1)%100==0:print(method,'ROUND',i+1,flush=True)
  s=m.summary(c,dict(rmst_tau_seconds=360000));s.update(method=method,checks=dict(counts),total_wall_seconds=time.time()-start)
  save(dest/'summary.json',s);save(dest/'hashes.json',{x.name:sha(x) for x in dest.iterdir() if x.is_file()})
  print('RESULT',method,s['completed'],s['mean_jct'],flush=True)
 except Exception as e:
  save(dest/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time if c else None));raise
def batch():
 verify()
 for method in METHODS:
  if (OUT/'cells'/method).exists():raise RuntimeError('REFUSE_EXISTING_CELL '+method)
  try:resources=gate()
  except Exception as e:save(OUT/'safe_stop.json',dict(method=method,error=repr(e),time=time.time()));return
  with (OUT/(method+'.log')).open('x',encoding='utf-8') as log:
   start=time.time();r=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)),'run',method],stdout=log,stderr=subprocess.STDOUT)
  with (OUT/'execution_manifest.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(method=method,exit_code=r.returncode,wall_seconds=time.time()-start,resources=resources))+'\n')
  print('CELL_END',method,r.returncode,flush=True)
  if r.returncode:return
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','batch','run']);a.add_argument('method',nargs='?');v=a.parse_args()
 if v.action=='run':run(v.method)
 else:globals()[v.action]()
