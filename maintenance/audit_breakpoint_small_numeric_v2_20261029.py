"""Independent raw-event small-gate audit, no training or scheduling."""
import collections,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt2';OUT=BASE/'small_gates';SRC=BASE/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from joint_prototype.joint_profile import load_synergy_template,CPU_COORDS,RAM_COORDS
REG={r['key']:r for r in json.loads((SRC/'inputs/registry.json').read_text())}
M={r['template']:load_synergy_template(SRC/'inputs/models'/r['template']).matrix for r in REG.values()}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def rho(name,cid,cpu,ram):
 r=REG[f"{name.rsplit('-',1)[0]}/{cid}"]
 def f(c,m):return float(np.interp(m,RAM_COORDS,[np.interp(c,CPU_COORDS,M[r['template']][:,i]) for i in range(5)]))
 return (f(cpu/r['phi_c'],ram/r['phi_m'])/f(3/r['phi_c'],62.5/r['phi_m']))**.5
def main():
 prereg=json.loads((OUT/'small_preregistration.json').read_text());records=[json.loads(l) for l in (OUT/'small_execution_manifest.jsonl').read_text().splitlines()]
 assert len(records)==8 and all(x['exit_code']==0 for x in records)
 audit={};w=json.loads((BASE/'mechanism_witnesses.json').read_text())
 for factor,kind,arm in prereg['cells']:
  name=f'{factor}_{kind}_{arm}';folder=OUT/'mechanisms'/name;s=json.loads((folder/'summary.json').read_text())
  for p,h in json.loads((folder/'output_hashes.json').read_text()).items():assert sha(folder/p)==h
  checks=collections.Counter();complete={};totals=collections.Counter();jobs=collections.defaultdict(collections.Counter)
  cap=12. if w[factor]['capacity']=='R01' else 6.;last=None
  with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    r=json.loads(line)
    if kind=='fixed':
     for e in r['execution']:
      if e['completion'] is not None:complete[e['job']]=e['completion']
      q=e['quota'];expected=rho(e['job'],w[factor]['config'],q['cpu'],q['ram']);assert math.isclose(q['rho'],expected,rel_tol=1e-12)
      jobs[e['job']]['training']+=e['execution'];jobs[e['job']]['checkpoint']+=e['checkpoint']
      for seg in e['segments']:
       assert math.isclose(seg['effective_rate'],seg['raw_rate_before_rho']*expected,rel_tol=1e-12)
       assert abs(seg['progress_delta']-seg['effective_rate']*seg['seconds']-seg['rounding_progress_residual'])<1e-7
       checks['raw_multiplier_segments']+=1
    else:
     d=r['decision_for_next_interval'];specs={j['job_id']:j for j in d['probe']['specs']}
     if d['round']!=last:
      last=d['round'];usage=collections.defaultdict(lambda:[0.,0.,0.])
      for j,col in d['result']['selected'].items():
       cfg=col['config'];cp,mm=col['cpu_per_gpu'],col['ram_gb_per_gpu'];assert math.isclose(col['rho'],rho(j,cfg['config_id'],cp,mm),rel_tol=1e-12)
       if specs[j]['checkpoint_locked']:
        sp=specs[j];assert(cfg['config_id'],cp,mm,sorted(col['placement'].items()))==(sp['current_config_id'],sp['current_cpu_per_gpu'],sp['current_ram_gb_per_gpu'],sorted(map(tuple,sp['current_placement'])))
        checks['full_locks']+=1
       for node,g in col['placement'].items():
        for k,x in enumerate([g,g*cp,g*mm]):usage[node][k]+=x
      for v in usage.values():assert all(x<=y+1e-7 for x,y in zip(v,[2,cap,187.5]));checks['node_capacity']+=1
     for e in r['execution']:
      n=e['job_id'];a=e['resource_allocation']
      if e['completion_time'] is not None:complete[n]=e['completion_time']
      assert math.isclose(e['observed_active_seconds'],e['waiting_seconds']+e['checkpoint_delay_consumed']+e['training_seconds'],abs_tol=1e-7)
      if e['checkpoint_delay_consumed']>=e['observed_active_seconds']>0:assert e['progress_delta']==0
      for k,col in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:jobs[n][k]+=e[col]
      for seg in e['training_segments']:
       expected=rho(n,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'])
       assert math.isclose(seg['effective_rate'],seg['raw_rate_before_rho']*expected,rel_tol=1e-12)
       assert abs(seg['progress_delta']-seg['effective_rate']*seg['seconds']-seg['rounding_progress_residual'])<1e-7
       checks['raw_multiplier_segments']+=1
      if a:
       dt=e['allocated_seconds']/3600;totals['gpu_hours']+=a['gpus']*dt
       for v in a['node_usage']:totals['cpu_core_hours']+=v['cpu']*dt;totals['ram_gb_hours']+=v['ram_gb']*dt
  expected_jobs=2 if kind=='fixed' else 3;assert len(complete)==expected_jobs
  mean=sum(complete.values())/len(complete);assert math.isclose(mean,s['mean_jct'],abs_tol=1e-7)
  for n,end in complete.items():assert math.isclose(sum(jobs[n][k] for k in ['wait','training','checkpoint']),end,abs_tol=1e-7)
  if kind=='free':
   for k,v in totals.items():assert math.isclose(v,s['totals'][k],abs_tol=1e-6)
  audit[name]=dict(valid=True,completed=len(complete),mean_jct=mean,completion=complete,per_job_exclusive_times={n:dict(v) for n,v in jobs.items()},resource_hours=dict(totals),checks=dict(checks),summary=s)
 comparisons={}
 for factor in ['CPU','MEM']:
  for kind in ['fixed','free']:
   a=audit[f'{factor}_{kind}_O'];b=audit[f'{factor}_{kind}_P']
   comparisons[f'{factor}_{kind}']=dict(improvement=1-a['mean_jct']/b['mean_jct'],mean_delta=a['mean_jct']-b['mean_jct'],per_job_delta={n:a['completion'][n]-t for n,t in b['completion'].items()},passed=a['mean_jct']<b['mean_jct'] and(kind=='fixed' or a['summary']['nonproportional_selected_columns']>0))
 result=dict(valid=True,formal30_gate_passed=all(x['passed'] for x in comparisons.values()),comparisons=comparisons,results=audit,capacity_violations=0,lock_violations=0,response_once_violations=0)
 with (BASE/'small_gate_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print('SMALL_GATE',result['formal30_gate_passed'],comparisons)
if __name__=='__main__':main()
