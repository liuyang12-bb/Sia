import csv,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import raw_audit_lib as lib
O=Path(__file__).resolve().parent;R=O.parents[2];P=R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1'
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def load(p):return json.loads(Path(p).read_text())
def geo(c):return (c['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu'],tuple(sorted(c['placement'].items())))
def cgeo(c):return (c['config'],c['cpu'],c['ram'],tuple(tuple(x) for x in c['placement']))
def main():
 p=load(O/'preregistration.json');work={x['name']:x for x in csv.DictReader((O/'inputs/workload.csv').open())};stats={};jobs={};frames={};forks={}
 for cell in p['cells']:
  name=cell['id'];stats[name],jobs[name],_=lib.audit(cell,p,work)
  assert stats[name]['complete'] and not stats[name]['errors']
  forks[name]=load(O/'cells'/name/'fork.json')
  with gzip.open(O/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8') as f:frames[name]=[json.loads(x) for x in f]
 save('raw_audit.json',stats);save('per_job.json',jobs)
 comparisons=[];details={};trajectories={}
 for t in (4800,5820):
  A=f'T{t}_LEGACY';B=f'T{t}_MIN_STATIC_SUM_ETA';fa=forks[A];fb=forks[B]
  assert fa['jobs']==fb['jobs'] and fa['scores']==fb['scores'] and fa['columns']==fb['columns']
  ea=fa['evaluation']['selected'];eb=fb['evaluation']['selected'];active=set(ea['completion_times'])
  ds={j:jobs[A][j]['jct']-jobs[B][j]['jct'] for j in jobs[A]}
  comparisons.append(dict(time=t,legacy_mean=stats[A]['mean'],alternative_mean=stats[B]['mean'],actual_sum_saved=sum(ds.values()),actual_mean_saved=sum(ds.values())/12,
   predicted_sum_saved=ea['sum_completion']-eb['sum_completion'],per_job_saved=ds,better=sum(v>0 for v in ds.values()),worse=sum(v<0 for v in ds.values()),equal=sum(v==0 for v in ds.values()),
   prediction_error_sum_difference=(ea['sum_completion']-eb['sum_completion'])-sum(ds.values())))
  for name in (A,B):
   fork=forks[name];ev=fork['evaluation']['selected'];cols=fork['columns'];domain={cgeo(c) for c in cols};byid={c['id']:c for c in cols}
   plan=next(z for z in fork['evaluation']['plans'] if z['name']==('LEGACY' if name==A else 'MIN_STATIC_SUM_ETA'))
   errs={};changes=[];future_segments=[]
   for j in active:
    col=byid[plan['columns'][j]];pred=fork['scores'][col['id']];seq=[];first=None;last=None;checkpoints=0;training=0
    for r in frames[name]:
     if r['interval_start']<t:continue
     for e in r['execution']:
      if e['job_id']!=j or e['completion_time'] is not None and e['observed_active_seconds']==0:continue
      checkpoint=e['checkpoint_delay_consumed'];checkpoints+=checkpoint;training+=e['training_seconds'];a=e['resource_allocation']
      if a:
       key=geo(a)
       if key!=last:
        seq.append(dict(time=e['interval_start'],config=a['config_id'],cpu=a['cpu_per_gpu'],ram=a['ram_gb_per_gpu'],rho=a['rho'],in_fork_eligible_domain=key in domain,checkpoint_consumed=checkpoint))
        if last is not None:changes.append(dict(job=j,**seq[-1]))
        last=key
       for seg in e['training_segments']:
        if first is None:first=dict(config=a['config_id'],cpu=a['cpu_per_gpu'],ram=a['ram_gb_per_gpu'],rho=a['rho'],segment=seg,same_as_initial=key==cgeo(col),predicted_raw=pred['raw_rate'],raw_ratio=seg['raw_rate_before_rho']/pred['raw_rate'])
        future_segments.append(dict(job=j,config=a['config_id'],cpu=a['cpu_per_gpu'],ram=a['ram_gb_per_gpu'],**seg))
    actual=jobs[name][j]['completion']-t
    initial_segments=[z for z in future_segments if z['job']==j and z['config']==col['config'] and abs(z['cpu']-col['cpu'])<1e-8 and abs(z['ram']-col['ram'])<1e-8]
    # This is ex-post calibration evidence, never an online oracle input.
    duration=sum(z['seconds'] for z in initial_segments)
    observed=sum(z['seconds']*z['raw_rate_before_rho'] for z in initial_segments)
    errs[j]=dict(predicted_remaining=ev['completion_times'][j],actual_remaining=actual,error=ev['completion_times'][j]-actual,
      initial_column=col,initial_prediction=pred,first_training=first,checkpoint_after_fork=checkpoints,training_after_fork=training,
      same_point_raw_calibration=dict(seconds=duration,predicted=pred['raw_rate'],actual_time_weighted=observed/duration if duration else None))
    trajectories[name+'/'+j]=seq
   recoveries=[];budget_checks=0
   for r in frames[name]:
    audit=r['decision_for_next_interval']['result']['audit']
    for j,ledger in audit.get('service_delay_ledgers',{}).items():
     extra=audit.get('tail_grants',{}).get(j,{}).get('extra_budget',0.)
     assert ledger['delay_charged']<=250.+extra+1e-7,(name,r['interval_end'],j,ledger,extra)
     budget_checks+=1
    if r['interval_end']>=t:
     recoveries.extend(dict(z,decision_time=r['interval_end']) for z in audit.get('tail_recoveries',[]))
   details[name]=dict(per_job=errs,predicted_sum=ev['sum_completion'],actual_sum=sum(jobs[name][j]['completion']-t for j in active),
      future_changes=changes,future_outside_current_domain=[z for z in changes if not z['in_fork_eligible_domain']],predicted_events=ev['events'])
   details[name].update(actual_recovery_permits=recoveries,lifetime_budget_checks=budget_checks)
   save(name+'_segments.json',future_segments)
 save('fork_comparisons.json',comparisons);save('prediction_errors.json',details);save('trajectories.json',trajectories)
 # Same initial state must extend to identical productive history; no "near" prefix matching.
 validation=dict(all_four_12_complete=True,raw_audit_errors=sum(len(x['errors']) for x in stats.values()),
  sources_unchanged=all(lib.sha(R/'cc-joint-release-reconfig-v2'/n)==h for n,h in p['source_hashes'].items()),
  inputs_unchanged=all(lib.sha(n)==h for n,h in p['input_hashes'].items()),
  matched_state_and_candidate_pairs=2,one_intervention_each=True,
  prefix_frames={c['id']:load(O/'cells'/c['id']/'summary.json')['prefix_frames_exact'] for c in p['cells']},
  old_full_4800_legacy_exact=None)
 with gzip.open(P/'cells/OLD_FULL/rounds.jsonl.gz','rt',encoding='utf8') as f:old=[json.loads(x) for x in f]
 cur=frames['T4800_LEGACY']
 validation['old_full_4800_legacy_exact']=len(cur)==len(old) and all(a['execution']==b['execution'] and a['job_end_state']==b['job_end_state'] and a['decision_for_next_interval']['result']['selected']==b['decision_for_next_interval']['result']['selected'] for a,b in zip(cur,old))
 assert all(validation[k] for k in ('sources_unchanged','inputs_unchanged','old_full_4800_legacy_exact'))
 save('validation.json',validation)
 print(json.dumps(comparisons,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
