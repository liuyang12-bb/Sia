"""Independent raw-event reconstruction; never derives mean from summaries."""
import collections,csv,gzip,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-uncertainty-v4'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6)
def audit(cell,p,work):
 out=O/'cells'/cell['id'];summary=json.loads((out/'summary.json').read_text())
 oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{j:v['application'] for j,v in work.items()},'medium')
 rows={j:dict(wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,completion=None,progress=0.) for j in work}
 counts=collections.Counter();plans=collections.Counter();evaluations=[];last_decision=None;rounds=0;errors=[];clipped=0;selected=0
 with gzip.open(out/'rounds.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);rounds+=1;d=r['decision_for_next_interval'];res=d['result']
   if d['round']!=last_decision:
    last_decision=d['round'];counts['decisions']+=1
    if not res['audit']['valid']:errors.append('INVALID_COMMIT')
    specs={x['job_id']:x for x in d['probe']['specs']};used={t+':0':[0.,0.,0.] for t in p['topology']}
    for j,c in res['selected'].items():
     selected+=1;clipped+=bool(c['source_coordinates']['clipped'] or c['baseline_source_coordinates']['clipped'])
     rho=independent_rho(oracle,j,c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu'])
     if not close(rho,c['rho']):errors.append('RHO_QUERY')
     for n,g in c['placement'].items():
      for i,x in enumerate((g,g*c['cpu_per_gpu'],g*c['ram_gb_per_gpu'])):used[n][i]+=x
     sp=specs[j]
     if sp['checkpoint_locked'] and sp['current_config_id']:
      counts['lock_checks']+=1
      if not (sp['current_config_id']==c['config']['config_id'] and dict(sp['current_placement'])==c['placement'] and close(sp['current_cpu_per_gpu'],c['cpu_per_gpu']) and close(sp['current_ram_gb_per_gpu'],c['ram_gb_per_gpu'])):errors.append('LOCK_CHANGED')
    for n,u in used.items():
     t=n.split(':')[0];g=p['topology'][t]
     if any(x>y+1e-7 for x,y in zip(u,(g,3*g,62.5*g))):errors.append('CAPACITY')
    ev=res['audit'].get('release_completion')
    if ev:
     counts['evaluation_'+ev['status']]+=1;plans[ev['selected_plan']]+=1
     if ev['status']=='OK':
      waiting=set(specs)-set(ev['fixed_service_set'])
      counts['evaluation_ok_with_waiters']+=bool(waiting)
      counts['predicted_waiter_gain_seconds']+=sum(ev['per_job_predicted_saved'][j] for j in waiting)
      delta=ev['baseline']['sum_completion']-ev['selected']['sum_completion']
      if delta < -1e-7 or not close(delta,sum(ev['per_job_predicted_saved'].values())):errors.append('PREDICTED_NET_ACCEPTANCE')
      if set(ev['per_job_predicted_saved'])!=set(specs):errors.append('EXCLUDED_CURRENT_JOB')
      if ev['fo_fixed_gpu'] is not None and ev['fo_fixed_gpu']!={j:c['config']['config_id'] for j,c in res['selected'].items()}:errors.append('FO_GPU_CHANGED')
      if ev['selected_plan']!='LEGACY':evaluations.append(dict(time=r['interval_end'],**ev))
   for e in r['execution']:
    j=e['job_id'];v=rows[j]
    if not audit_execution(e)['valid']:errors.append('EXECUTION')
    counts['execution_checks']+=1
    for key,field in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:v[key]+=e[field]
    if e['progress_after']<e['progress_before']-1e-8:errors.append('PROGRESS_MONOTONIC')
    v['progress']=e['progress_after']
    if e['completion_time'] is not None:v['completion']=e['completion_time']
    a=e['resource_allocation']
    if a:
     rho=independent_rho(oracle,j,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'])
     if not close(e['rho'],rho):errors.append('EXEC_RHO')
     for seg in e['training_segments']:
      if not close(seg['raw_rate_before_rho']*rho,seg['effective_rate']):errors.append('DOUBLE_RHO')
     dt=e['allocated_seconds']/3600;v['gpu_hours']+=a['gpus']*dt
     for u in a['node_usage']:v['cpu_core_hours']+=u['cpu']*dt;v['ram_gb_hours']+=u['ram_gb']*dt
   for j,v in r['job_end_state'].items():
    rows[j]['restarts']=v['restarts'];rows[j]['wasted_gpu_seconds']=v['wasted_gpu_seconds']
 for j,v in rows.items():
  v['arrival']=float(work[j]['time']);v['jct']=v['completion']-v['arrival'] if v['completion'] is not None else None
  if v['jct'] is not None and not close(v['jct'],v['wait']+v['training']+v['checkpoint']):errors.append('EXCLUSIVE_TIME')
  for k in ('jct','wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds'):
   if v[k] is not None and not close(v[k],summary['jobs'][j][k]):errors.append('SUMMARY_'+k)
 complete=all(v['jct'] is not None for v in rows.values());vals=[v['jct'] for v in rows.values() if v['jct'] is not None]
 stats=dict(completed=len(vals),rounds=rounds,complete=complete,mean=float(np.mean(vals)) if complete else None,
  median=float(np.median(vals)) if complete else None,p95=float(np.percentile(vals,95)) if complete else None,p99=float(np.percentile(vals,99)) if complete else None,
  makespan=max(v['completion'] for v in rows.values())-min(v['arrival'] for v in rows.values()) if complete else None,
  totals={k:sum(v[k] for v in rows.values()) for k in ('wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds')},
  errors=errors,checks=dict(counts),plans=dict(plans),clipped=clipped,selected=selected,clipped_fraction=clipped/selected if selected else None,
  wall=summary['total_wall_seconds'])
 return stats,rows,evaluations
