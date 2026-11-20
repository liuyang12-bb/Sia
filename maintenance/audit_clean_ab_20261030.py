"""Independent raw-event audit and factorial analysis. No simulation."""
import collections,csv,functools,gzip,hashlib,itertools,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run1';SRC=OUT/'frozen_source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from joint_prototype.joint_profile import load_synergy_template
from joint_prototype.execution_audit import audit_execution
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
REG={r['key']:r for r in json.loads((SRC/'inputs/registry.json').read_text())} if SRC.exists() else {}
@functools.lru_cache(maxsize=None)
def matrix(name):return load_synergy_template(SRC/'inputs/models'/name).matrix
@functools.lru_cache(maxsize=None)
def response(app,cid,cpu,ram,alpha):
 ref=REG[app+'/'+cid];a=matrix(ref['template'])
 def f(c,m):return float(np.interp(c,[1,2,3,4,5,6,9,12,24],[np.interp(m,[20.83,62.5,125,187.5,250],row) for row in a]))
 return (f(cpu/ref['phi_c'],ram/ref['phi_m'])/f(3/ref['phi_c'],62.5/ref['phi_m']))**alpha
def stream(p):
 with gzip.open(p,'rt',encoding='utf8') as f:
  for line in f:yield json.loads(line)
def geom(c):
 if not c:return None
 return (c.get('config_id') or c['config']['config_id'],tuple(sorted(c['placement'].items())),c['cpu_per_gpu'],c['ram_gb_per_gpu'])
def signature(r):
 return dict(selected={j:geom(c) for j,c in r['decision_for_next_interval']['result']['selected'].items()},
    execution=[(e['job_id'],e['progress_after'],e['completion_time'],e['cluster'],e['placement']) for e in r['execution']],ends=r['job_end_state'])
def audit(w,n,p):
 folder=OUT/'cells'/(w+'_'+n);s=json.loads((folder/'summary.json').read_text());alpha=0. if w=='A20' else .5;cpu,ram=6.,187.5 if w=='A20' else 125.
 jobs=collections.defaultdict(lambda:collections.Counter());checks=collections.Counter();clips=collections.Counter();states=collections.Counter();last=None;previous={};previous_progress={};frozen={};cand_counts=[];actualsource=[]
 for r in stream(folder/'rounds.jsonl.gz'):
  for e in r['execution']:
   name=e['job_id'];j=jobs[name];assert audit_execution(e)['valid'];checks['execution_events']+=1
   assert e['progress_after']>=previous_progress.get(name,0.)-1e-8;previous_progress[name]=e['progress_after']
   for k,z in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:j[k]+=e[z]
   # Initial vs subsequent waiting is computed from timestamps, not current
   # post-decision Job.first_allocation value (which can change this boundary).
   first=s['jobs'][name]['first_allocation'];initial=e['waiting_seconds'] if first is None else min(e['waiting_seconds'],max(0.,first-max(e['interval_start'],e['arrival_time'])))
   j['first_wait']+=initial;j['later_wait']+=e['waiting_seconds']-initial
   if e['allocated_seconds']>0:
    a=e['resource_allocation'];assert geom(a)==geom(previous[name]);checks['atomic_cross_round']+=1
    app=name.rsplit('-',1)[0];rho=response(app,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'],alpha);assert math.isclose(rho,e['rho'],rel_tol=1e-12)
    dt=e['allocated_seconds']/3600;j['gpu_hours']+=a['gpus']*dt
    j['cpu_core_hours']+=sum(v['cpu'] for v in a['node_usage'])*dt;j['ram_gb_hours']+=sum(v['ram_gb'] for v in a['node_usage'])*dt
    for z in e['training_segments']:
     assert math.isclose(z['effective_rate'],z['raw_rate_before_rho']*rho,rel_tol=1e-12);checks['rho_once']+=1
    ref=REG[app+'/'+a['config_id']]
    for tag,c,m in [('numerator',a['cpu_per_gpu'],a['ram_gb_per_gpu']),('denominator',3.,62.5)]:
     clips[tag+'_cpu']+=e['training_seconds']*int(not 1.-1e-12<=c/ref['phi_c']<=24.+1e-12)
     clips[tag+'_ram']+=e['training_seconds']*int(not 20.83-1e-12<=m/ref['phi_m']<=250.+1e-12)
   j['arrival']=e['arrival_time'];j['completion']=e['completion_time'];j['restarts']=r['job_end_state'][name]['restarts'];j['wasted_gpu_seconds']=r['job_end_state'][name]['wasted_gpu_seconds']
  d=r['decision_for_next_interval'];sel=d['result']['selected'];a=d['result']['audit']
  if d['round']!=last:
   last=d['round'];assert a['valid'];specs={j['job_id']:j for j in d['probe']['specs']};usage=collections.defaultdict(lambda:[0.,0.,0.]);states[d['result']['status']]+=1
   assert {j for j,q in specs.items() if q['checkpoint_locked']}<=set(sel)
   if not n.endswith('SIA'):
    expected=p['A_objective' if w=='A20' else 'B_objective'];assert d['clean_contract']['objective']==expected
    assert d['shared_contract']['objective']==expected
    if w=='A20':assert d['clean_contract']['ordered']==bool(int(n[1])) and d['clean_contract']['continuous']==bool(int(n[2]))
    cand_counts.append(len(d['probe']['joint_columns']))
    for evt in a.get('serial_proposals',[]):
     assert {x['job']:x['config'] for x in evt['selected']}==evt['fixed'];checks['serial_frozen_intent']+=1
     assert sum(x['utility'] for x in evt['selected'])>=sum(x['utility'] for x in evt['proposal'])-1e-7
    for name,v in a.get('service_delay_ledgers',{}).items():assert v['delay_charged']<=250+a.get('tail_grants',{}).get(name,{}).get('extra_budget',0.)+1e-9
   else:
    assert d['strict_full_intent'];intent=d['original_sia_allocations']
    assert set(sel)=={j for j,v in intent.items() if v and v[0] is not None and v[1]}
    for j,c in sel.items():
     assert c['config']['gpu_type']==intent[j][0]
     assert c['placement']=={f'{intent[j][0]}:{x}':g for x,g in collections.Counter(intent[j][1]).items()};checks['official_intent']+=1
   for j,c in sel.items():
    q=specs[j];assert c['config'] in q['configs'];assert c['query_round']==q['query_round'];checks['candidate_current_round']+=1
    rho=response(j.rsplit('-',1)[0],c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu'],alpha);assert math.isclose(rho,c['rho'],rel_tol=1e-12)
    if alpha==0:assert rho==1.
    if w=='A20' or n.endswith('P') or n.endswith('SIA'):assert abs(c['cpu_per_gpu']-cpu/2)<1e-8 and abs(c['ram_gb_per_gpu']-ram/2)<1e-8
    if q['checkpoint_locked']:
     assert geom(c)==(q['current_config_id'],tuple(tuple(x) for x in q['current_placement']),q['current_cpu_per_gpu'],q['current_ram_gb_per_gpu']);checks['full_lock']+=1
    if j in frozen:assert geom(c)[:2]==frozen[j]
    for node,g in c['placement'].items():
     for k,v in enumerate([g,g*c['cpu_per_gpu'],g*c['ram_gb_per_gpu']]):usage[node][k]+=v
   for v in usage.values():assert all(x<=y+1e-7 for x,y in zip(v,[2,cpu,ram]));checks['capacity']+=1
   for t in a.get('tail_recoveries',[]):
    assert len(specs)==1 and not specs[t['job']]['checkpoint_locked'];assert t['eta_after']<t['eta_before'];frozen[t['job']]=geom(sel[t['job']])[:2];checks['tail_recovery']+=1
  previous=sel
 values=[]
 for name,j in jobs.items():
  for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds']:assert math.isclose(j[k],s['jobs'][name][k],abs_tol=1e-6),(w,n,name,k)
  if j['completion'] is not None:
   j['jct']=j['completion']-j['arrival'];assert math.isclose(j['jct'],j['wait']+j['checkpoint']+j['training'],abs_tol=1e-6);values.append(j['jct'])
 if len(values)==len(jobs):
  for k,v in [('mean_jct',np.mean(values)),('median_jct',np.median(values)),('p95_jct',np.percentile(values,95)),('p99_jct',np.percentile(values,99))]:assert math.isclose(s[k],float(v),abs_tol=1e-6)
 assert dict(states)==s['solver_status']
 return dict(valid=True,summary=s,jobs={j:dict(v) for j,v in jobs.items()},checks=dict(checks),clipped_productive_job_seconds=dict(clips),
   candidate_counts=dict(minimum=min(cand_counts,default=0),maximum=max(cand_counts,default=0)),
   source='INDEPENDENT_RAW_COMPLETION_EVENTS_AND_RATE_INTEGRALS')
def contrast(x,y):
 if x['summary']['mean_jct'] is None or y['summary']['mean_jct'] is None:return dict(incomplete=True)
 a,b=x['summary'],y['summary'];d={j:v['jct']-b['jobs'][j]['jct'] for j,v in a['jobs'].items()}
 best=min(d,key=d.get)
 return dict(mean_delta=a['mean_jct']-b['mean_jct'],improvement=1-a['mean_jct']/b['mean_jct'],
   improved=sum(v<0 for v in d.values()),worsened=sum(v>0 for v in d.values()),equal=sum(v==0 for v in d.values()),
   per_job_delta=d,totals_delta={k:a['totals'][k]-b['totals'][k] for k in a['totals']},
   first_wait_delta=sum(v['first_wait'] for v in x['jobs'].values())-sum(v['first_wait'] for v in y['jobs'].values()),
   later_wait_delta=sum(v['later_wait'] for v in x['jobs'].values())-sum(v['later_wait'] for v in y['jobs'].values()),
   p95_change=a['p95_jct']/b['p95_jct']-1,p99_change=a['p99_jct']/b['p99_jct']-1,makespan_change=a['makespan']/b['makespan']-1,
   largest_beneficiary=best,largest_contribution_fraction=d[best]/sum(d.values()) if sum(d.values()) else None,
   leave_largest_beneficiary_out_mean_delta=float(np.mean([v for j,v in d.items() if j!=best])))
def main():
 p=json.loads((OUT/'preregistration.json').read_text());results={};coverage=[];comparisons={};effects={};identities={};perjob=[]
 for w,n in p['cells']:
  name=w+'_'+n;folder=OUT/'cells'/name
  coverage.append(dict(cell=name,state='AVAILABLE' if (folder/'summary.json').exists() else 'MISSING'))
  if not (folder/'summary.json').exists():continue
  results[name]=audit(w,n,p);print('AUDITED',name,flush=True)
  for j,v in results[name]['jobs'].items():perjob.append(dict(cell=name,job=j,**v))
 for w in ('A20','COMPACT','SPREAD'):
  names=('A00','A01','A10','A11') if w=='A20' else ('B_FP','B_FO','B_JP','B_JO')
  if not all(w+'_'+n in results and results[w+'_'+n]['summary']['mean_jct'] is not None for n in names):continue
  r={n:results[w+'_'+n] for n in names};m={n:v['summary']['mean_jct'] for n,v in r.items()}
  if w=='A20':
   pairs=[('A10','A00'),('A11','A01'),('A01','A00'),('A11','A10')]
   effects[w]=dict(order_main=.5*((m['A10']-m['A00'])+(m['A11']-m['A01'])),continuity_main=.5*((m['A01']-m['A00'])+(m['A11']-m['A10'])),
    interaction=m['A11']-m['A10']-m['A01']+m['A00'],definition='JCT seconds: negative beneficial; interaction negative=superadditive reduction')
  else:
   pairs=[('B_FO','B_FP'),('B_JO','B_JP'),('B_JP','B_FP'),('B_JO','B_FO')]
   effects[w]=dict(aux_main=.5*((m['B_FO']-m['B_FP'])+(m['B_JO']-m['B_JP'])),joint_main=.5*((m['B_JP']-m['B_FP'])+(m['B_JO']-m['B_FO'])),
    interaction=m['B_JO']-m['B_JP']-m['B_FO']+m['B_FP'],definition='JCT seconds; negative beneficial; FP=JP expected negative control')
   x=stream(OUT/'cells'/(w+'_B_FP')/'rounds.jsonl.gz');y=stream(OUT/'cells'/(w+'_B_JP')/'rounds.jsonl.gz');count=0
   for a,b in itertools.zip_longest(x,y):assert a is not None and b is not None and signature(a)==signature(b),(w,count,'FP_JP_NOT_IDENTICAL');count+=1
   identities[w]=dict(FP_JP_exact_physical_trajectory=True,rounds=count)
  external=results.get(w+('_A_SIA' if w=='A20' else '_B_SIA'))
  for n,v in r.items():
   if external:comparisons[w+'/'+n+'_vs_SIA']=contrast(v,external)
  for a,b in pairs:comparisons[w+'/'+a+'_vs_'+b]=contrast(r[a],r[b])
 save('raw_audit.json',dict(results=results,coverage=coverage,valid=len(results)==15 and all(r['valid'] for r in results.values()),identities=identities))
 save('comparisons.json',comparisons);save('factor_effects.json',effects)
 with (OUT/'per_job.csv').open('x',encoding='utf8',newline='') as f:
  fields=sorted({k for r in perjob for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(perjob)
 save('metrics.json',{n:v['summary'] for n,v in results.items()})
 save('validation.json',dict(cells=len(results),all_cells_present=len(results)==15,all_completed=all(r['summary']['completion_rate']==1 for r in results.values()),
    capacity_violations=0,lock_violations=0,rho_violations=0,raw_independent_audit=True,
    completed_only_mean_never_used=True,negative_control=identities,coverage=coverage))
 print('AUDIT_DONE',len(results),effects,flush=True)
if __name__=='__main__':main()
