"""Frozen42-cell runner; scientific run body reused from the accepted30-cell runner."""
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
def run(mix,scenario,response,method):
 p=setup();name=label(mix,scenario,response,method);assert [mix,scenario,response,method] in p['cells']
 out=OUT/'cells'/name;res=resources();out.mkdir(parents=True,exist_ok=False)
 save(out/'start.json',dict(time=time.time(),resources=res,command=sys.argv,preregistration_sha256=sha(OUT/'preregistration.json')))
 c=None;start=time.time()
 try:
  from resource_study import build,CAPACITIES
  from run_stage2c_cell import Accumulator,independent_rho
  from joint_prototype.execution_audit import audit_execution
  c=build(method,scenario,response,OUT/'inputs'/(mix+'_workload.csv'));cpu,ram=CAPACITIES[scenario];m=Accumulator(c);last=None;cnt=collections.Counter()
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
  summary=m.summary(c,dict(rmst_tau_seconds=360000));summary.update(mix=mix,scenario=scenario,response=response,method=method,checks=dict(cnt),total_wall_seconds=time.time()-start)
  save(out/'summary.json',summary);save(out/'output_hashes.json',{x.name:sha(x) for x in out.iterdir() if x.is_file()})
  print('RESULT',name,summary['completed'],summary['mean_jct'],flush=True)
 except Exception as e:save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time if c else None,last_decision=c.policy.last_decision if c else None));raise
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
  with (OUT/'execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(index=i,cell=name,exit_code=proc.returncode,wall_seconds=time.time()-start,resources=res))+'\n')
  print('CELL_END',i+1,'/42',name,proc.returncode,flush=True)
  (OUT/'CURRENT_PROGRESS.json').write_text(json.dumps(dict(ended=i+1,total=42,last_cell=name,exit_code=proc.returncode,time=time.time()),indent=2),encoding='utf8')
  if proc.returncode:return
 print('ALL42_ENDED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['batch','run']);p.add_argument('--mix');p.add_argument('--scenario');p.add_argument('--response');p.add_argument('--method');a=p.parse_args()
 if a.action=='run':run(a.mix,a.scenario,a.response,a.method)
 else:batch()
