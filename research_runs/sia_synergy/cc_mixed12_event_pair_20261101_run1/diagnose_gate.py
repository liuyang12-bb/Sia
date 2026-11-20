import collections,gzip,hashlib,json,math,sys
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[2];T=ROOT/'cc-joint-recovery-equivalent-v1';sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.execution_audit import audit_execution
from run_stage2c_cell import independent_rho
p=json.loads((O/'preregistration.json').read_text(encoding='utf8'));work={r['name']:r for r in p['workload']}
rows={m:[json.loads(l) for l in gzip.open(O/f'gate_{m}.jsonl.gz','rt',encoding='utf8')] for m in ('FO','FULL')}
look={m:{r['time']:r for r in rs} for m,rs in rows.items()};records=[];errors=[];checks=collections.Counter();summaries={}
oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{n:r['application'] for n,r in work.items()},'medium')
for owner,rs in rows.items():
 totals={j:dict(progress=0.,waiting=0.,training=0.,checkpoint=0.,completion=None) for j in work}
 for r in rs:
  for method,d in [('ACTUAL',r['actual']),*r['decisions'].items()]:
   sel=d['result']['selected'];specs={v['job_id']:v for v in d['probe']['specs']};used=collections.defaultdict(lambda:[0.,0.,0.]);checks['decisions']+=1
   for name,sp in specs.items():
    if sp['checkpoint_locked'] and sp['current_config_id']:
     z=sel.get(name);checks['locks']+=1
     if not z or z['config']['config_id']!=sp['current_config_id'] or z['placement']!=dict(sp['current_placement']) or abs(z['cpu_per_gpu']-sp['current_cpu_per_gpu'])>1e-7 or abs(z['ram_gb_per_gpu']-sp['current_ram_gb_per_gpu'])>1e-7:errors.append([owner,r['time'],method,name,'LOCK'])
   for j,z in sel.items():
    rho=independent_rho(oracle,j,z['config']['config_id'],z['cpu_per_gpu'],z['ram_gb_per_gpu']);checks['rho']+=1
    if not math.isclose(rho,z['rho'],rel_tol=1e-10):errors.append([owner,r['time'],method,j,'RHO'])
    for n,g in z['placement'].items():
     for k,v in enumerate((g,g*z['cpu_per_gpu'],g*z['ram_gb_per_gpu'])):used[n][k]+=v
   for n,v in used.items():
    t=n.split(':')[0];cap=p['capacities'][t];checks['node_capacity']+=1
    if any(a>b+1e-7 for a,b in zip(v,(p['topology'][t],cap['cpu_cores'],cap['ram_gb']))):errors.append([owner,r['time'],method,n,'CAPACITY'])
  for e in r['execution']:
   checks['execution']+=1
   if not audit_execution(e)['valid']:errors.append([owner,r['time'],e['job_id'],'EXECUTION'])
   q=totals[e['job_id']]
   for key,field in [('progress','progress_delta'),('waiting','waiting_seconds'),('training','training_seconds'),('checkpoint','checkpoint_delay_consumed')]:q[key]+=e[field]
   if e['completion_time'] is not None:q['completion']=e['completion_time']
  ds=r['decisions']
  if not ds:continue
  a,b=[ds[m]['result'] for m in ('FO','FULL')];x,y=a['selected'],b['selected']
  changed=[j for j in x.keys()&y.keys() if x[j]['config']['config_id']!=y[j]['config']['config_id']]
  if set(x)!=set(y) or len(x)<2 or b['scheduling_objective']<=a['scheduling_objective']+1e-9:continue
  for j in changed:
   configs={m:ds[m]['result']['selected'][j]['config']['config_id'] for m in ('FO','FULL')};span=[];productive=0;stop=None;keyonly=[]
   for t in range(int(r['time']),2400,60):
    aa={m:look[m][t]['actual']['result']['selected'] for m in ('FO','FULL')}
    mismatched=[m for m in aa if j not in aa[m] or aa[m][j]['config']['config_id']!=configs[m]]
    if mismatched:stop=dict(time=t,reason='KEY_CONFIG_NOT_MATCHING_COUNTERFACTUAL',methods=mismatched,actual_configs={m:aa[m].get(j,{}).get('config',{}).get('config_id') for m in aa});break
    if set(aa['FO'])!=set(aa['FULL']):stop=dict(time=t,reason='ACTUAL_SERVICE_SETS_DIFFER',fo=list(aa['FO']),full=list(aa['FULL']));break
    span.append(t);productive+=any(e['job_id']==j and e['progress_delta']>0 for e in look['FULL'][t+60]['execution'])
   records.append(dict(owner=owner,event_time=r['time'],job=j,configs=configs,intervals=len(span),productive_intervals=productive,stop=stop))
 summaries[owner]=dict(rounds=len(rs),time=rs[-1]['time'],completed=sum(v['completion'] is not None for v in totals.values()),jobs=totals,
  normalized_work_done=sum(v['progress']/rs[-1]['actual']['declared_work'][j]['total_progress'] for j,v in totals.items()))
checks['source_hashes']=len(p['source_hashes'])
assert all(hashlib.sha256((T/n).read_bytes()).hexdigest().upper()==h for n,h in p['source_hashes'].items())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest().upper()==h for n,h in p['input_hashes'].items())
records.sort(key=lambda v:(-v['intervals'],v['event_time'],v['job'],v['owner']))
result=dict(valid=not errors,checks=dict(checks),errors=errors,structural_changed_job_records=len(records),longest=records[:10],all_records=records,
 prefix=summaries,stop_reason_counts=dict(collections.Counter(v['stop']['reason'] for v in records if v['stop'])),
 caution='Prefix trajectories differ; no full-population JCT mean computed. Same-service-set gate may fail through selection confounding as well as short GPU persistence.')
with (O/'gate_diagnosis.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:result[k] for k in ('valid','checks','structural_changed_job_records','longest','stop_reason_counts')},indent=2,ensure_ascii=False))
assert not errors,errors[:5]
