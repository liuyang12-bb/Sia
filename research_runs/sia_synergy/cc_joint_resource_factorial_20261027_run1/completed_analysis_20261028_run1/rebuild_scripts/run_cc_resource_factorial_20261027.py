"""Freeze and run exactly30 cells after opportunity and acceptance gates."""
import argparse,collections,dataclasses,gzip,hashlib,importlib.metadata,json,math,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1';TREE=ROOT/'cc-joint-resource-study'
METHODS=['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']
CELLS=[(s,'medium',m) for s in ['R00','R10','R01','R11'] for m in METHODS]+[('R11','neutral',m) for m in METHODS]
def enc(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def save(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,default=enc)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def allfiles(p):return {x.relative_to(p).as_posix():sha(x) for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc'}
def prepare():
 gate=json.loads((OUT/'opportunity_preflight.json').read_text());assert gate['run_gate'],'G0_STOP'
 assert '\nOK\n' in (OUT/'adapter_green2.log').read_text() and 'Ran 5 tests' in (OUT/'adapter_green2.log').read_text()
 assert '\nOK\n' in (OUT/'existing_tests.log').read_text() and 'Ran 224 tests' in (OUT/'existing_tests.log').read_text()
 assert json.loads((OUT/'official_regression.json').read_text())['exact']
 parent=json.loads((OUT/'parent_source_hashes.json').read_text())
 for n,h in parent.items():assert sha(TREE/n)==h and sha(ROOT/'cc-joint-tail-recovery'/n)==h,n
 shutil.copytree(TREE,OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 save(OUT/'preregistration.json',dict(cells=CELLS,methods=METHODS,source_hashes=allfiles(TREE),runner_hash=sha(Path(__file__)),
  capacities={'R00':[12,250],'R10':[6,250],'R01':[12,187.5],'R11':[6,187.5]},response_exponents=dict(neutral=0.,medium=.5),
  interval=60,max_rounds=6000,seed_profiles=[1,8],gpu_per_node=2,nodes=['aws:0','dgx-ext:0'],
  g0_sha256=sha(OUT/'opportunity_preflight.json'),coverage_sha256=sha(OUT/'full_domain_coverage.json'),
  workload_sha256=sha(TREE/'inputs/workload.csv'),registry_sha256=sha(TREE/'inputs/registry.json'),
  packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','cvxpy','cylp','psutil']},
  algorithm_changed=False,grid_changed=False,response_changed=False,
  gate=dict(memory_gib=2,disk_gib=5),censor='At6000 retain completion rate andRMST; no imputed JCT; continue other cells',
  failure='Stop batch on implementation/audit failure; retain partial output; no tuning or selective retry',
  warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
 print('FROZEN30')
def setup():
 p=json.loads((OUT/'preregistration.json').read_text());assert sha(Path(__file__))==p['runner_hash']
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h,n
 sys.path.insert(0,str(OUT/'source'));sys.path.insert(0,str(OUT/'source/sia-simulator'))
 from cc_factory import verify
 verify();return p
def resources():
 import psutil
 x=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
 if x['memory_gib']<2 or x['disk_gib']<5:raise RuntimeError('RESOURCE_GATE '+str(x))
 return x
def label(s,r,m):return f'{s}_{r}_{m}'
def run(scenario,response,method):
 p=setup();name=label(scenario,response,method);assert [scenario,response,method] in p['cells']
 out=OUT/'cells'/name;res=resources();out.mkdir(parents=True,exist_ok=False)
 save(out/'start.json',dict(time=time.time(),resources=res,command=sys.argv,preregistration_sha256=sha(OUT/'preregistration.json')))
 c=None;start=time.time()
 try:
  from resource_study import build,CAPACITIES
  from run_stage2c_cell import Accumulator,independent_rho
  from joint_prototype.execution_audit import audit_execution
  c=build(method,scenario,response);cpu,ram=CAPACITIES[scenario];m=Accumulator(c);last=None;cnt=collections.Counter()
  with gzip.open(out/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
   for i in range(6000):
    if c.all_complete():break
    c.step(60);r=c.joint_round_logs[-1];r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
    f.write(json.dumps(r,default=enc,separators=(',',':'))+'\n');f.flush();d=r['decision_for_next_interval'];a=d['result']['audit']
    if d['round']!=last:
     last=d['round'];assert a['valid'];m.status[d['result']['status']]+=1;m.solve.append(d['result']['solve_seconds']);m.events.update(e['event'] for e in d.get('liveness_events',[]))
     specs={x['job_id']:x for x in d['probe']['specs']};usage=collections.defaultdict(lambda:[0.,0.,0.])
     for j,col in d['result']['selected'].items():
      cfg=col['config'];cp,mm=col['cpu_per_gpu'],col['ram_gb_per_gpu'];expected=independent_rho(c.policy.oracle,j,cfg['config_id'],cp,mm)
      assert math.isclose(col['rho'],expected,rel_tol=1e-12)
      if response=='neutral':assert col['rho']==1.
      if method in ['SIA_P','CC_P']:assert(cp,mm)==(cpu/2,ram/2)
      if method=='CC_CPU':assert mm==ram/2
      if method=='CC_MEM':assert cp==cpu/2
      cnt['selected_columns']+=1;cnt['nonproportional_cpu']+=cp!=cpu/2;cnt['nonproportional_ram']+=mm!=ram/2
      row=c.policy.oracle.reference(j,cfg['config_id'])
      for tag,cp0,mm0 in [('numerator',cp,mm),('denominator',3,62.5)]:
       cnt[tag+'_cpu_clip']+=not 1<=cp0/row['phi_c']<=24;cnt[tag+'_ram_clip']+=not 20.83<=mm0/row['phi_m']<=250
      v=specs[j]
      if v['checkpoint_locked']:
       assert cfg['config_id']==v['current_config_id'] and col['placement']==dict(v['current_placement']) and(cp,mm)==(v['current_cpu_per_gpu'],v['current_ram_gb_per_gpu']);cnt['full_lock_checks']+=1
      for node,g in col['placement'].items():
       for k,x in enumerate([g,g*cp,g*mm]):usage[node][k]+=x
     for node,v in usage.items():
      assert all(x<=y+1e-7 for x,y in zip(v,[2,cpu,ram]))
      m.idle_saturated+=int(v[1]>=cpu-1e-7 and v[0]<2)
      for key,x in zip(['gpu','cpu','ram'],v):m.node_peaks[node][key]=max(m.node_peaks[node][key],x)
     if method.startswith('SIA'):
      assert d['strict_full_intent'] and c.policy.canonical(c.allocations,c.allocations)==c.policy.canonical(d['original_sia_allocations'],c.allocations)
     else:
      assert d['shared_contract']['objective']=='SUM_RECIPROCAL_CURRENT_COMPLETION_ETA' and d['shared_contract']['tail_recovery']
      for j,v in a.get('service_delay_ledgers',{}).items():assert v['delay_charged']<=250+a.get('tail_grants',{}).get(j,{}).get('extra_budget',0.)+1e-9
      for v in a.get('tail_recoveries',[]):assert len(specs)==1 and not a['ready'] and not specs[v['job']]['checkpoint_locked'] and v['eta_after']<v['eta_before']
    for e in r['execution']:assert audit_execution(e)['valid']
    m.last_decision_round=d['round'];m.consume(c,r)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
    if (i+1)%100==0:print(name,'ROUND',i+1,flush=True)
  summary=m.summary(c,dict(rmst_tau_seconds=360000));summary.update(scenario=scenario,response=response,method=method,checks=dict(cnt),total_wall_seconds=time.time()-start)
  save(out/'summary.json',summary);save(out/'output_hashes.json',{x.name:sha(x) for x in out.iterdir() if x.is_file()})
  print('RESULT',name,summary['completed'],summary['mean_jct'],flush=True)
 except Exception as e:save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time if c else None,last_decision=c.policy.last_decision if c else None));raise
def batch():
 setup()
 for i,(s,r,m) in enumerate(CELLS):
  name=label(s,r,m)
  if (OUT/'cells'/name).exists():raise RuntimeError('REFUSE_EXISTING '+name)
  try:res=resources()
  except Exception as e:save(OUT/'safe_stop.json',dict(index=i,next_cell=name,error=repr(e)));return
  with (OUT/(name+'.log')).open('x',encoding='utf-8') as f:
   start=time.time();p=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)),'run','--scenario',s,'--response',r,'--method',m],stdout=f,stderr=subprocess.STDOUT)
  with (OUT/'execution_manifest.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(index=i,cell=name,exit_code=p.returncode,wall_seconds=time.time()-start,resources=res))+'\n')
  print('CELL_END',i+1,'/30',name,p.returncode,flush=True)
  if p.returncode:return
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','batch','run']);p.add_argument('--scenario');p.add_argument('--response');p.add_argument('--method');a=p.parse_args()
 if a.action=='run':run(a.scenario,a.response,a.method)
 else:globals()[a.action]()
