"""Independent raw completion, queue, capacity and factorial audit; no simulator."""
import collections,gzip,hashlib,itertools,json,math,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1'
DEST=OUT/'analysis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6),(a,b)
def geom(x):
 return (x.get('config_id',x.get('config',{}).get('config_id')),tuple(sorted(x['placement'].items())),x['cpu_per_gpu'],x['ram_gb_per_gpu'])
def quantile(v,q):
 s=sorted(v);k=(len(s)-1)*q;i=int(k);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(k-i)
def integral(events):
 events=sorted(events);count=0.;last=events[0][0] if events else 0.;area=0.;hist=collections.defaultdict(float)
 for t,d in events:
  dt=t-last;area+=count*dt;hist[str(int(round(count)))]+=dt;count+=d;last=t
 close(count,0.)
 return dict(area=area,histogram_seconds=dict(hist))
def audit(name):
 folder=OUT/'cells'/name;s=json.loads((folder/'summary.json').read_text());jobs=collections.defaultdict(lambda:collections.defaultdict(float))
 prior={};last=None;locks=capacities=segments=0;rounds=0;events=collections.Counter();statuses=collections.Counter();allocated_events=[];productive_events=[];waiting_events=[];budgeth=0
 previousprogress={};first={};firstprod={};optional=required=selectedcnt=unknown_selected=0;legacy_labels=collections.Counter()
 nodehours=collections.defaultdict(lambda:collections.defaultdict(float))
 for file,h in json.loads((folder/'output_hashes.json').read_text()).items():assert sha(folder/file)==h
 with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);rounds+=1
   for e in r['execution']:
    j=e['job_id'];v=jobs[j];a=e['resource_allocation'];start=max(e['interval_start'],e['arrival_time'])
    assert e['progress_after']>=e['progress_before']-1e-8
    if j in previousprogress:close(e['progress_before'],previousprogress[j])
    previousprogress[j]=e['progress_after'];assert e['rho']==1.
    close(e['observed_active_seconds'],e['waiting_seconds']+e['checkpoint_delay_consumed']+e['training_seconds'])
    for k,key in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:v[k]+=e[key]
    if e['waiting_seconds']>0:waiting_events.extend([(start,1),(start+e['waiting_seconds'],-1)])
    if e['allocated_seconds']>0:
     assert a and j in prior and geom(a)==geom(prior[j])
     dt=e['allocated_seconds']/3600;v['gpu_hours']+=a['gpus']*dt
     allocated_events.extend([(start,1),(start+e['allocated_seconds'],-1)])
     v['one_gpu_training_seconds']+=e['training_seconds']*int(a['gpus']==1)
     v['two_gpu_training_seconds']+=e['training_seconds']*int(a['gpus']==2)
     for u in a['node_usage']:
      v['cpu_core_hours']+=u['cpu']*dt;v['ram_gb_hours']+=u['ram_gb']*dt
      for k,key in [('gpu','gpus'),('cpu','cpu'),('ram','ram_gb')]:nodehours[u['node_id']][k]+=u[key]*dt
    if e['training_seconds']>0:
     productive_events.extend([(start+e['checkpoint_delay_consumed'],1),(start+e['checkpoint_delay_consumed']+e['training_seconds'],-1)])
     firstprod.setdefault(j,start+e['checkpoint_delay_consumed'])
    for z in e['training_segments']:
     assert z['rho']==1.;close(z['effective_rate'],z['raw_rate_before_rho'])
     close(z['progress_delta'],z['raw_rate_before_rho']*z['seconds']+z['rounding_progress_residual']);segments+=1
    if e['checkpoint_delay_consumed']>=e['observed_active_seconds']>0:close(e['progress_delta'],0.)
    end=r['job_end_state'][j];v.update(arrival=e['arrival_time'],completion=e['completion_time'],restarts=end['restarts'],wasted_gpu_seconds=end['wasted_gpu_seconds'])
   d=r['decision_for_next_interval'];sel=d['result']['selected'];au=d['result']['audit']
   if d['round']!=last:
    last=d['round'];assert au['valid'];specs={j['job_id']:j for j in d['probe']['specs']};totals=collections.defaultdict(lambda:[0.,0.,0.])
    statuses[d['result']['status']]+=1;events.update(e['event'] for e in d.get('liveness_events',[]))
    if name!='SIA_P':
     assert d['queue_factors']==dict(zip('ABC',map(int,name[1:])));budgeth+=len(au.get('budget_safety_holds',[]))
     flags=dict(zip('ABC',map(int,name[1:])))
     if not flags['C'] and au.get('periodic_service_expiry') is False:legacy_labels['inherited_periodic_service_expiry_false']+=1
     if not flags['C'] and d['shared_contract'].get('nonpreemptive_completion') is True:legacy_labels['inherited_nonpreemptive_completion_true']+=1
     if not flags['A'] and d['probe']['contract'].get('objective')=='SUM_RECIPROCAL_CURRENT_COMPLETION_ETA':legacy_labels['inherited_probe_objective_completion']+=1
     required+=len(au.get('required_jobs',[]));optional+=len(au.get('optional_selected',[]));selectedcnt+=len(sel)
     for j,v in au['service_delay_ledgers'].items():assert v['delay_charged']<=250+au.get('tail_grants',{}).get(j,{}).get('extra_budget',0.)+1e-9
    for j,c in sel.items():
     first.setdefault(j,r['interval_end']);q=specs[j];assert c['rho']==1. and(c['cpu_per_gpu'],c['ram_gb_per_gpu'])==(3.,93.75)
     if name!='SIA_P':
      score=next(z for z in d['continuity_scores'] if z['column_id']==c['column_id'])
      unknown_selected+=int(score['estimated_completion'] is None)
     assert c['config'] in q['configs'] and c['query_round']==q['query_round']
     if q['checkpoint_locked']:
      assert geom(c)==(q['current_config_id'],tuple(tuple(x) for x in q['current_placement']),q['current_cpu_per_gpu'],q['current_ram_gb_per_gpu']);locks+=1
     for n,g in c['placement'].items():
      for k,x in enumerate([g,g*3,g*93.75]):totals[n][k]+=x
    for n,v in totals.items():
     assert n in ['aws:0','dgx-ext:0'];assert all(x<=y+1e-7 for x,y in zip(v,[2,6,187.5]));capacities+=1
   prior=sel
 assert rounds==s['rounds']
 vals=[];unfinished=[];unfinished_events=[]
 for j,v in jobs.items():
  q=s['jobs'][j]
  for k in ['wait','training','checkpoint','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds']:close(v[k],q[k])
  assert v['completion']==q['completion'];assert first.get(j)==q['first_allocation'];assert firstprod.get(j)==q['first_training']
  v['first_allocation']=first.get(j);v['first_training']=firstprod.get(j)
  end=v['completion'] if v['completion'] is not None else s['simulated_seconds']
  close(end-v['arrival'],v['wait']+v['training']+v['checkpoint'])
  unfinished_events.extend([(v['arrival'],1),(end,-1)])
  v['initial_wait']=(first[j] if j in first else end)-v['arrival'];v['later_wait']=v['wait']-v['initial_wait']
  v['jct']=v['completion']-v['arrival'] if v['completion'] is not None else None
  v['censor_time']=None if v['completion'] is not None else end-v['arrival']
  if v['jct'] is not None:vals.append(v['jct'])
  else:unfinished.append(j)
 qarea=integral(waiting_events);narea=integral(unfinished_events)
 close(qarea['area'],sum(j['wait'] for j in jobs.values()));close(narea['area'],sum(j['wait']+j['training']+j['checkpoint'] for j in jobs.values()))
 assert len(vals)==s['completed']
 if len(vals)==20:
  close(statistics.mean(vals),s['mean_jct']);close(statistics.median(vals),s['median_jct']);close(quantile(vals,.95),s['p95_jct']);close(quantile(vals,.99),s['p99_jct'])
 return dict(valid=True,summary=s,jobs={j:dict(v) for j,v in jobs.items()},independent_checks=dict(lock=locks,capacity=capacities,rho_segments=segments),
  queue_area=qarea,unfinished_area=narea,allocated_concurrency=integral(allocated_events),productive_concurrency=integral(productive_events),
  states=dict(required_job_rounds=required,optional_selected=optional,selected_job_rounds=selectedcnt,budget_safety_holds=budgeth,unknown_eta_selected_job_rounds=unknown_selected),
  totals={k:sum(v[k] for v in jobs.values()) for k in ['initial_wait','later_wait','wait','training','checkpoint','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds']},
  events=dict(events),solver_status=dict(statuses),unfinished=unfinished,node_hours={n:dict(v) for n,v in nodehours.items()},
  effective_semantics=dict(queue_factors=None if name=='SIA_P' else dict(zip('ABC',map(int,name[1:]))),
   authority='Explicit queue_factors,shared_contract.objective/service,audit.service_rule plus executed events; inherited legacy labels are not factor authority',
   inherited_label_discrepancies=dict(legacy_labels)))
def differences(x,y):
 if x['summary']['completed']!=20 or y['summary']['completed']!=20:return dict(comparable_full_mean=False)
 per={j:x['jobs'][j]['jct']-y['jobs'][j]['jct'] for j in y['jobs']}
 return dict(comparable_full_mean=True,mean_delta=x['summary']['mean_jct']-y['summary']['mean_jct'],improvement=1-x['summary']['mean_jct']/y['summary']['mean_jct'],
  improved=sum(v<0 for v in per.values()),worsened=sum(v>0 for v in per.values()),unchanged=sum(v==0 for v in per.values()),per_job_delta=per,
  decomposition={k:x['totals'][k]-y['totals'][k] for k in ['wait','initial_wait','later_wait','training','checkpoint']})
def main():
 DEST.mkdir(exist_ok=False);p=json.loads((OUT/'preregistration.json').read_text());results={}
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h
 for name in p['cells']:
  assert (OUT/'cells'/name/'summary.json').exists(),('INCOMPLETE_MATRIX',name)
  results[name]=audit(name);print('AUDITED',name,results[name]['summary']['completed'],flush=True)
 comparisons={a+'_vs_'+b:differences(results[a],results[b]) for a,b in [('Q111','SIA_P'),('Q000','SIA_P'),('Q111','Q011'),('Q111','Q101'),('Q111','Q110'),('Q111','Q000')]}
 if all(results['Q'+''.join(map(str,z))]['summary']['completed']==20 for z in itertools.product([0,1],repeat=3)):
  y={z:results['Q'+''.join(map(str,z))]['summary']['mean_jct'] for z in itertools.product([0,1],repeat=3)}
  effects={};shapley={};pairs={}
  for i,n in enumerate('ABC'):
   contrasts=[];sh=0.
   for z in itertools.product([0,1],repeat=3):
    if z[i]:continue
    zz=tuple(1 if k==i else v for k,v in enumerate(z));delta=y[zz]-y[z];k=sum(z)
    contrasts.append(dict(off=z,on=zz,delta=delta));sh+=math.factorial(k)*math.factorial(2-k)/6*(-delta)
   effects[n]=dict(mean_delta=sum(x['delta'] for x in contrasts)/4,contrasts=contrasts);shapley[n]=sh
  for i,j in itertools.combinations(range(3),2):
   k=3-i-j;values=[]
   for v in [0,1]:
    def val(a,b):
     z=[0,0,0];z[i]=a;z[j]=b;z[k]=v;return y[tuple(z)]
    values.append(val(1,1)-val(1,0)-val(0,1)+val(0,0))
   pairs['ABC'[i]+'ABC'[j]]=dict(by_third_factor=values,average=sum(values)/2)
  close(sum(shapley.values()),y[(0,0,0)]-y[(1,1,1)])
  factorial=dict(main_effects=effects,pair_interactions=pairs,shapley_mean_jct_seconds_saved=shapley,
   baseline000=y[(0,0,0)],full111=y[(1,1,1)],note='Negative main delta helps. Positive pair interaction means less-than-additive reduction. Single fixed workload, no population inference.')
 else:factorial=dict(status='CENSORED_NO_FULL_MEAN_FACTORIAL_DECOMPOSITION')
 save(DEST/'raw_audit.json',results);save(DEST/'paired_comparisons.json',comparisons);save(DEST/'factorial_attribution.json',factorial)
 # Predefined scoring pair at their first genuinely identical incoming state.
 initial={}
 for n in ['Q111','Q011','SIA_P']:
  with gzip.open(OUT/'cells'/n/'rounds.jsonl.gz','rt',encoding='utf-8') as f:first_round=json.loads(next(f))
  d=first_round['decision_for_next_interval'];sel=d['result']['selected']
  initial[n]=dict(interval_end=first_round['interval_end'],incoming_execution=first_round['execution'],
   specs=d['probe']['specs'],scores=d.get('continuity_scores',[]),selected=sel,
   selected_gpus=sum(c['config']['gpus'] for c in sel.values()),
   objective=d['result'].get('scheduling_objective'),required=d['result']['audit'].get('required_jobs',[]),
   optional_selected=d['result']['audit'].get('optional_selected',[]))
 assert initial['Q111']['specs']==initial['Q011']['specs']
 def physical_keys(v):return sorted((x['job'],x['column_id']) for x in v['scores'])
 assert physical_keys(initial['Q111'])==physical_keys(initial['Q011'])
 save(DEST/'identical_initial_state_scoring.json',dict(same_specs_and_candidate_ids=True,states=initial,
  interpretation='Same state, B/C fixed, A includes unknown-model utility. First divergence explains underfilling, not a unique decomposition of all later JCT differences.'))
 with gzip.open(OUT/'cells/Q100/rounds.jsonl.gz','rt',encoding='utf-8') as x,gzip.open(OUT/'cells/Q101/rounds.jsonl.gz','rt',encoding='utf-8') as y:
  for n,(a,b) in enumerate(zip(x,y),1):
   aa,bb=json.loads(a),json.loads(b)
   def chosen(z):return {j:geom(c) for j,c in z['decision_for_next_interval']['result']['selected'].items()}
   if chosen(aa)==chosen(bb):continue
   def incoming(z):return [(e['job_id'],e['progress_after'],e['epoch_after'],e['completion_time'],e['cluster'],e['placement']) for e in z['execution']]
   save(DEST/'continuity_first_geometry_divergence.json',dict(raw_line=n,time=aa['interval_end'],incoming_state_equal=incoming(aa)==incoming(bb),
    periodic=dict(selected=chosen(aa),decision=aa['decision_for_next_interval']),
    continuous=dict(selected=chosen(bb),decision=bb['decision_for_next_interval']),
    caution='C changes held membership, hence held-specific event/provenance filters as well as eviction. Not a pure checkpoint-cost-only effect.'))
   break
 parent=json.loads((OUT/'parent_source_hashes.json').read_text())
 for n,h in parent.items():assert sha(ROOT/'cc-joint-resource-study'/n)==h and sha(ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source'/n)==h
 save(DEST/'validation.json',dict(valid=True,cells=len(results),completed={n:r['summary']['completed'] for n,r in results.items()},parent_files_unchanged=len(parent),
  historical_regression={n:json.loads((OUT/(n+'_historical_regression.json')).read_text()) for n in ['Q111','SIA_P']},simulation_calls=0))
 print('INDEPENDENT_NINE_CELL_AUDIT_COMPLETE')
if __name__=='__main__':main()
