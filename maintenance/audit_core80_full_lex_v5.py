"""Read-only reconstruction from raw events; writes only new audit products."""
import collections,csv,gzip,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];TREE=ROOT/'cc-joint-core80-full-lex-repair-v5';OUT=ROOT/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run2'
CANON=json.loads((OUT/'canonical_result_paths.json').read_text()) if (OUT/'canonical_result_paths.json').exists() else None
sys.path[:0]=[str(TREE),str(TREE/'sia-simulator')]
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))
def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6)
def audit(cell,p,work):
 d=Path(CANON['paths'][cell['id']]) if CANON else OUT/'cells'/cell['id'];summary=json.loads((d/'summary.json').read_text()) if (d/'summary.json').exists() else None
 failure=json.loads((d/'failure.json').read_text()) if (d/'failure.json').exists() else None
 oracle=R0Oracle(TREE/'inputs/registry.json',TREE/'inputs/models',{j:r['application'] for j,r in work.items()},cell['response'])
 from core80 import saturation_per_gpu,GPUS
 caps={t:dict(gpu=g,cpu=3*g,ram=62.5*g) for t,g in GPUS.items()}
 if cell['scenario']!='dual':
  sat,_=saturation_per_gpu()
  for t,g in GPUS.items():caps[t]['ram' if cell['scenario']=='cpu_only' else 'cpu']=g*sat[t]['ram' if cell['scenario']=='cpu_only' else 'cpu']
 rows={j:dict(wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,completion=None,first_allocation=None,first_training=None,progress=0.,restarts=0,migrations=0,gpu_changes=0,wasted_gpu_seconds=0.) for j in work}
 errors=[];statuses=collections.Counter();events=collections.Counter();clipping=collections.Counter();seen=set();nround=0;lasttime=0.;lockchecks=0;shared_examples=[];pairseconds=0.;local_gain=0.;local_loss=0.;queue_rounds=0;cpuidle=0;maxwaiting=0;execution_audits=0;last_decision=None;checkpoint_rounds=0;reconfigurations=0
 node_hours=collections.defaultdict(lambda:dict(gpu=0.,cpu=0.,ram=0.));solver_calls=collections.Counter();certified=0;round_wall=0.;job_wall=0.;initial=[];supported_pairseconds=0.;supported_training=0.;clipped_training=0.;snapshots=[]
 local_by_support={k:dict(receiver_gain=0.,donor_loss=0.) for k in ('fully_supported','clipped')}
 numeric_events=collections.Counter();selector_seconds=[];peak_scored=0;engineering=collections.Counter();presolve_peak=0;zero_peak=0;lp_peak=0;primary_peak=0
 previous_execution_config={}
 intent_checks=0;factor_quota_checks=0
 def supported_allocation(job,a):
  # Preserve the source's exact requested coordinates/ULP clipping rule;
  # independent rho is re-evaluated separately below.
  return not (a['source_coordinates']['clipped'] or a['baseline_source_coordinates']['clipped'])
 raw=d/'rounds.jsonl.gz'
 if not raw.exists():
  for j,x in rows.items():x.update(arrival=float(work[j]['time']),completed=False,jct=None,first_wait=None,later_wait=None,observed_until=None,arrived_by_observed_end=False,censor_time=None)
  return dict(cell=cell['id'],status='FAILED_BEFORE_RAW_LOG',completed=0,total=80,rounds=0,mean_jct=None,median_jct=None,p95_jct=None,p99_jct=None,makespan=None,
    totals={k:None for k in ('wait','first_wait','later_wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds')},
    capacity_and_execution_violations=0,errors=[],unobserved=True,clipping_counts={},execution_audits=0,checkpoint_lock_checks=0,waiting_rounds=0,max_waiting_jobs=0,cpu_saturated_gpu_idle_node_rounds=0,co_resident_different_quota_pair_seconds=0),rows
 truncated=False
 try:
  with gzip.open(raw,'rt',encoding='utf8') as f:
   for line in f:
    try:r=json.loads(line)
    except json.JSONDecodeError:truncated=True;break
    nround+=1;round_wall+=r.get('round_wall_seconds',0.);job_wall+=r.get('job_step_wall_seconds',0.);decision=r['decision_for_next_interval'];res=decision['result'];now=max((e['interval_end'] for e in r['execution']),default=lasttime+60)
    peak_scored=max(peak_scored,max((decision.get(k+'_count',0) for k in ('continuity_scores','completion_scores','net_progress_scores','factor_candidate_keys')),default=0))
    if now<=lasttime:errors.append('NONMONOTONIC_TIME')
    lasttime=now
    if nround<=3:
     initial.append(dict(time=now,selected_jobs=len(res['selected']),waiting_jobs=len(res['unallocated_jobs']),
        gpu=sum(c['config']['gpus'] for c in res['selected'].values()),cpu=sum(c['config']['gpus']*c['cpu_per_gpu'] for c in res['selected'].values()),ram=sum(c['config']['gpus']*c['ram_gb_per_gpu'] for c in res['selected'].values()),
        rho_range=[min((c['rho'] for c in res['selected'].values()),default=1),max((c['rho'] for c in res['selected'].values()),default=1)],cold_contract=decision.get('clean_contract',{}).get('cold_start'),
        definition='Allocation for next interval, not already productive training. Initial differences must not be attributed solely to CPU/RAM.'))
    if decision['round']!=last_decision:
     last_decision=decision['round'];statuses[res['status']]+=1;events.update(x['event'] for x in decision.get('liveness_events',[]))
     selector_seconds.append(float(res['solve_seconds']))
     stats=res['audit'].get('engineering_solver',{});engineering.update({k:stats.get(k,0) for k in ('native_calls','native_seconds','bound_certificates','lp_calls','lp_seconds','primary_band_lp_calls','primary_band_lp_seconds','integer_rows_tightened')});presolve_peak=max(presolve_peak,stats.get('variables_removed',0));zero_peak=max(zero_peak,stats.get('zero_variables_removed',0));lp_peak=max(lp_peak,stats.get('lp_variables_removed',0));primary_peak=max(primary_peak,stats.get('primary_band_variables_removed',0))
     if not res['audit'].get('valid',False):errors.append('COMMITTED_INVALID_AUDIT')
     certified+=res['audit'].get('exact_zero_optional_saved_solver_calls',0)
     for retry in res['audit'].get('numerical_solver_retries',[]):
      numeric_events[retry.get('independent_backend',retry.get('reformulation','OTHER_NUMERICAL_EVENT'))]+=1
     solver_calls.update(str(x['status']) for x in res['audit'].get('solver_trace',[]) if not x['message'].startswith('CERTIFIED_'))
     waiting=res['unallocated_jobs'];queue_rounds+=bool(waiting);maxwaiting=max(maxwaiting,len(waiting))
     totals={t+':0':[0.,0.,0.] for t in caps};specs={j['job_id']:j for j in decision['probe']['specs']}
     if decision.get('strict_full_intent'):
      intent_checks+=1
      expected_intent={j:(v[0],dict(collections.Counter(f'{v[0]}:{n}' for n in v[1]))) for j,v in decision['original_sia_allocations'].items() if v[0] is not None and v[1]}
      actual_intent={j:(c['config']['gpu_type'],c['placement']) for j,c in res['selected'].items()}
      if expected_intent!=actual_intent:errors.append('OFFICIAL_SIA_INTENT_CHANGED')
     if cell['method']=='FO' and res['audit'].get('serial_proposals'):
      fixed=res['audit']['serial_proposals'][-1]['fixed']
      if fixed!={j:c['config']['config_id'] for j,c in res['selected'].items()}:errors.append('SERIAL_PROPOSAL_GPU_INTENT_CHANGED')
     # Reconstruct mandatory selection from the current state, independently
     # of the solver's own valid flag and of columns which survived selection.
     # Checking only selected columns would miss a silently dropped lock.
     for j,s in specs.items():
      if s['checkpoint_locked'] and s['current_config_id'] and j not in res['selected']:
       errors.append('CHECKPOINT_LOCK_DROPPED')
     for j,c in res['selected'].items():
      cfg=c['config'];expected=independent_rho(oracle,j,cfg['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu'])
      pc=caps[cfg['gpu_type']]['cpu']/GPUS[cfg['gpu_type']];pm=caps[cfg['gpu_type']]['ram']/GPUS[cfg['gpu_type']]
      cpu_fixed=cell['method'] in ('SIA_P','PROP','MEM','A00','A01','A10','A11')
      ram_fixed=cell['method'] in ('SIA_P','PROP','CPU','A00','A01','A10','A11')
      factor_quota_checks+=1
      if cpu_fixed and not close(c['cpu_per_gpu'],pc) or ram_fixed and not close(c['ram_gb_per_gpu'],pm):errors.append('AUXILIARY_FACTOR_CONTRACT')
      if not close(expected,c['rho']):errors.append('DECISION_RHO')
      if cell['response']=='neutral' and c['rho']!=1.:errors.append('NON_NEUTRAL')
      for n,g in c['placement'].items():
       totals[n][0]+=g;totals[n][1]+=g*c['cpu_per_gpu'];totals[n][2]+=g*c['ram_gb_per_gpu']
      s=specs[j]
      if s['checkpoint_locked'] and s['current_config_id']:
       lockchecks+=1
       if not(cfg['config_id']==s['current_config_id'] and c['placement']==dict(s['current_placement']) and c['cpu_per_gpu']==s['current_cpu_per_gpu'] and c['ram_gb_per_gpu']==s['current_ram_gb_per_gpu']):errors.append('CHECKPOINT_LOCK')
      for label,coords in [('num',c['source_coordinates']),('den',c['baseline_source_coordinates'])]:
       clipping[label+'_cpu']+=not 1<=coords['requested']['cpu']<=24;clipping[label+'_ram']+=not 20.83<=coords['requested']['ram_gb']<=250
      clipping['num_any']+=bool(c['source_coordinates']['clipped']);clipping['den_any']+=bool(c['baseline_source_coordinates']['clipped']);clipping['any_point']+=bool(c['source_coordinates']['clipped'] or c['baseline_source_coordinates']['clipped'])
      clipping['selected']+=1
     for n,v in totals.items():
      cap=caps[n.rsplit(':',1)[0]]
      if any(x>y+1e-7 for x,y in zip(v,(cap['gpu'],cap['cpu'],cap['ram']))):errors.append('CAPACITY')
      cpuidle+=v[1]>=cap['cpu']-1e-7 and v[0]<cap['gpu']
    executed=[]
    for e in r['execution']:
     j=e['job_id'];x=rows[j];valid=audit_execution(e);execution_audits+=1
     if not valid['valid']:errors.extend(valid['violations'])
     if e['progress_before']<x['progress']-1e-6:errors.append('PROGRESS_REWIND')
     x['progress']=e['progress_before']+e['progress_delta']
     x['wait']+=e['waiting_seconds'];x['checkpoint']+=e['checkpoint_delay_consumed'];x['training']+=e['training_seconds']
     if e['checkpoint_delay_consumed']>0:checkpoint_rounds+=1
     if e['completion_time'] is not None:x['completion']=e['completion_time']
     a=e['resource_allocation']
     if not a:continue
     if j in previous_execution_config and previous_execution_config[j]!=a['config_id']:x['gpu_changes']+=1
     previous_execution_config[j]=a['config_id']
     if x['first_allocation'] is None:x['first_allocation']=e['interval_start']
     expected=independent_rho(oracle,j,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'])
     if not close(expected,e['rho']):errors.append('EXECUTION_RHO')
     dt=e['allocated_seconds']/3600;x['gpu_hours']+=a['gpus']*dt
     for u in a['node_usage']:
      x['cpu_core_hours']+=u['cpu']*dt;x['ram_gb_hours']+=u['ram_gb']*dt
      for dim,field in [('gpu','gpus'),('cpu','cpu'),('ram','ram_gb')]:node_hours[u['node_id']][dim]+=u[field]*dt
     for s in e['training_segments']:
      if x['first_training'] is None:x['first_training']=s['start_time']
      if not close(s['effective_rate'],s['raw_rate_before_rho']*expected):errors.append('RHO_DOUBLE_OR_RAW_MISMATCH')
      # Frozen same-state proportional reference: no replay/future runtime.
      prop=independent_rho(oracle,j,a['config_id'],caps[a['gpu_type']]['cpu']/GPUS[a['gpu_type']],caps[a['gpu_type']]['ram']/GPUS[a['gpu_type']])
      equivalent=(s['rho']/prop-1)*s['seconds']
      support=supported_allocation(j,a)
      if support:supported_training+=s['seconds']
      else:clipped_training+=s['seconds']
      local_by_support['fully_supported' if support else 'clipped']['receiver_gain' if equivalent>=0 else 'donor_loss']+=abs(equivalent)
      if equivalent>=0:local_gain+=equivalent
      else:local_loss-=equivalent
     if e['training_seconds']>0:executed.append(e)
    for state_j,state in r.get('job_end_state',{}).items():
     rows[state_j]['restarts']=state['restarts'];rows[state_j]['migrations']=state['migrations'];rows[state_j]['wasted_gpu_seconds']=state['wasted_gpu_seconds']
    if now in (3600.,7200.,10800.):
     snapshots.append(dict(calendar_time=now,completed=sum(x['completion'] is not None for x in rows.values()),unfinished=sum(x['completion'] is None for x in rows.values()),
       cumulative_jobtime=sum(min(x['completion'] if x['completion'] is not None else now,now)-float(work[j]['time']) for j,x in rows.items()),
       meaning='Descriptive at fixed1h/2h/3h simulated calendar ticks, all jobs arrived. Not a full JCT mean or a fixed-age RMST; future outcomes can reverse apparent ordering.'))
    # Record genuine simultaneous training with unequal quotas, receiver gain,
    # and a below-proportional peer. Compare elapsed interval intersections.
    for i,a in enumerate(executed):
     aa=a['resource_allocation'];typ=aa['gpu_type'];pc=caps[typ]['cpu']/GPUS[typ];pm=caps[typ]['ram']/GPUS[typ]
     for b in executed[i+1:]:
      bb=b['resource_allocation'];shared=set(aa['placement'])&set(bb['placement'])
      if not shared or aa['config_id'].split(':')[0]!=bb['config_id'].split(':')[0]:continue
      donor=(aa['cpu_per_gpu']<pc-1e-8 or aa['ram_gb_per_gpu']<pm-1e-8 or bb['cpu_per_gpu']<pc-1e-8 or bb['ram_gb_per_gpu']<pm-1e-8)
      prop_a=independent_rho(oracle,a['job_id'],aa['config_id'],pc,pm);prop_b=independent_rho(oracle,b['job_id'],bb['config_id'],pc,pm)
      if not donor or max(a['rho']/prop_a,b['rho']/prop_b)<=1+1e-9:continue
      overlap=sum(max(0.,min(x['start_time']+x['seconds'],y['start_time']+y['seconds'])-max(x['start_time'],y['start_time'])) for x in a['training_segments'] for y in b['training_segments'])
      pairseconds+=overlap
      both_supported=supported_allocation(a['job_id'],aa) and supported_allocation(b['job_id'],bb)
      if both_supported:supported_pairseconds+=overlap
      if overlap and len(shared_examples)<30:shared_examples.append(dict(round=nround,jobs=[a['job_id'],b['job_id']],nodes=sorted(shared),overlap_seconds=overlap,rhos=[a['rho'],b['rho']],relative_to_proportional=[a['rho']/prop_a,b['rho']/prop_b],all_coords_supported=both_supported,quotas=[[aa['cpu_per_gpu'],aa['ram_gb_per_gpu']],[bb['cpu_per_gpu'],bb['ram_gb_per_gpu']]]))
 except (EOFError,OSError):truncated=True
 if summary and truncated:errors.append('COMPLETED_RAW_STREAM_TRUNCATED')
 if summary and summary['rounds']!=nround:errors.append('SUMMARY_ROUND_COUNT_MISMATCH')
 completed=0
 for j,x in rows.items():
  x['arrival']=float(work[j]['time']);x['completed']=x['completion'] is not None;x['jct']=x['completion']-x['arrival'] if x['completed'] else None;completed+=x['completed']
  x['observed_until']=lasttime;x['arrived_by_observed_end']=x['arrival']<=lasttime
  x['censor_time']=None if x['completed'] or not x['arrived_by_observed_end'] else max(0.,lasttime-x['arrival'])
  if x['completion'] is not None and not close(x['wait']+x['checkpoint']+x['training'],x['jct']):errors.append('EXCLUSIVE_TIME_TOTAL')
  if x['first_allocation'] is not None:x['first_wait']=x['first_allocation']-x['arrival'];x['later_wait']=x['wait']-x['first_wait']
  else:x['first_wait']=x['wait'];x['later_wait']=0.
 if summary:
  for j,x in rows.items():
   for k in ('wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','migrations','gpu_changes','wasted_gpu_seconds','jct','completion'):
    a=x[k];b=summary['jobs'][j][k]
    if (a is None)!=(b is None) or a is not None and not close(a,b):errors.append('SUMMARY_RECOMPUTATION '+j+'/'+k)
 vals=[x['jct'] for x in rows.values() if x['completed']];complete=completed==80
 if complete and not cell['method'].startswith('SIA') and events['ARRIVE_READY']!=80:errors.append('INCOMPLETE_ARRIVAL_LEDGER')
 window=max(x['completion'] for x in rows.values()) if complete else lasttime
 utilization={node:{dim:(hours*3600/(window*caps[node.rsplit(':',1)[0]][dim]) if window else 0.) for dim,hours in usage.items()} for node,usage in node_hours.items()}
 result=dict(cell=cell['id'],status=('INVALID_COMPLETED_EVENTS' if errors else 'ALL_COMPLETE' if summary and summary.get('termination')=='ALL_COMPLETE' else 'COMPLETED_EVENTS_WITHOUT_SUCCESSFUL_FINALIZATION') if complete else ('FAILED' if failure else 'COMPUTE_TIMEOUT_OR_INTERRUPTED' if summary is None else summary['termination']),completed=completed,total=80,rounds=nround,
   failure_error=failure.get('error') if failure else None,failure_simtime=failure.get('time') if failure else None,
   unlogged_failed_interval_seconds=max(0.,failure['time']-lasttime) if failure and failure.get('time') is not None else 0.,
   mean_jct=float(np.mean(vals)) if complete else None,median_jct=float(np.median(vals)) if complete else None,p95_jct=float(np.percentile(vals,95)) if complete else None,p99_jct=float(np.percentile(vals,99)) if complete else None,
   makespan=max(x['completion'] for x in rows.values()) if complete else None,totals={k:sum(x[k] for x in rows.values()) for k in ('wait','first_wait','later_wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','migrations','gpu_changes','wasted_gpu_seconds')},
   capacity_and_execution_violations=len(errors),capacity_violations=sum(x=='CAPACITY' for x in errors),checkpoint_lock_violations=sum(x.startswith('CHECKPOINT_LOCK') for x in errors),errors=errors[:100],truncated_worker_log=truncated,observed_simulated_end=lasttime,solver_status=dict(statuses),liveness_events=dict(events),
   fallback_count=sum(v for k,v in statuses.items() if k.startswith('FALLBACK')),clipping_counts=dict(clipping),checkpoint_lock_checks=lockchecks,execution_audits=execution_audits,
   waiting_rounds=queue_rounds,max_waiting_jobs=maxwaiting,cpu_saturated_gpu_idle_node_rounds=cpuidle,co_resident_different_quota_pair_seconds=pairseconds,co_resident_examples=shared_examples,
   node_resource_hours=dict(node_hours),node_utilization=utilization,outer_uncertified_call_statuses=dict(solver_calls),certified_redundant_calls=certified,
   initial_boundary_states=initial,fixed_calendar_snapshots=snapshots,fully_supported_co_resident_pair_seconds=supported_pairseconds,training_response_exposure_seconds=dict(fully_supported=supported_training,clipped=clipped_training),
   computation_wall_seconds=dict(round_total=round_wall,job_step=job_wall,optimize_and_commit=max(0.,round_wall-job_wall),definition='Raw timers; optimize_and_commit includes model/candidate building as well as selection, not merely MILP solve_seconds.'),
   local_proportional_reference_equivalent_training_seconds=dict(receiver_gain=local_gain,donor_loss=local_loss,net=local_gain-local_loss,interpretation='Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'))
 result['local_reference_gain_by_data_support']={k:dict(**v,net=v['receiver_gain']-v['donor_loss']) for k,v in local_by_support.items()}
 result['arrived_by_observed_end']=sum(x['arrived_by_observed_end'] for x in rows.values())
 result['selection_reported_seconds']=dict(total=sum(selector_seconds),p99=float(np.percentile(selector_seconds,99)) if selector_seconds else None,definition='Logged PolicyResult.solve_seconds; includes multi-MIP selection/control work, not an isolated single-backend-call timer.')
 result['peak_logged_score_catalog_columns']=peak_scored
 result['clipping_rates_percent']={k:100*v/clipping['selected'] for k,v in clipping.items() if k!='selected'} if clipping['selected'] else {}
 result['numerical_event_counts']=dict(numeric_events)
 result['engineering_solver']=dict(**engineering,peak_variables_removed=presolve_peak,peak_zero_optional_removed=zero_peak,peak_lp_variables_removed=lp_peak,peak_primary_band_variables_removed=primary_peak)
 result['official_sia_intent_checks']=intent_checks;result['factor_quota_checks']=factor_quota_checks
 return result,rows

def main():
 p=json.loads((OUT/'preregistration.json').read_text());work={r['name']:r for r in csv.DictReader((TREE/'inputs/core80_workload.csv').open(encoding='utf8'))}
 out=OUT/'audit';out.mkdir(exist_ok=False);results={};perjob={}
 ledger=Path(CANON['manifest']).parent/'execution_manifest.jsonl' if CANON else OUT/'execution_manifest.jsonl'
 launched={json.loads(l)['cell'] for l in ledger.read_text().splitlines() if json.loads(l)['event']=='CELL_START'}
 if CANON:launched.update(CANON['reused'])
 for cell in p['cells']:
  directory=Path(CANON['paths'][cell['id']]) if CANON else OUT/'cells'/cell['id']
  if directory.exists() or cell['id'] in launched:
   print('AUDIT',cell['id'],flush=True);r,js=audit(cell,p,work);results[cell['id']]=r;perjob[cell['id']]=js;save(out/(cell['id']+'.json'),r)
 pairs=[]
 comparisons=[('dual_medium_FULL','dual_medium_'+x) for x in ('SIA_P','SIA_AUX','PROP','CPU','MEM','FO')]
 comparisons += [('dual_medium_SIA_AUX','dual_medium_SIA_P')]
 comparisons += [('dual_medium_'+x,'dual_medium_SIA_P') for x in ('PROP','CPU','MEM','FO')]
 comparisons += [('dual_medium_'+x,'dual_medium_PROP') for x in ('CPU','MEM','FO')]
 comparisons += [('dual_medium_FO','dual_medium_SIA_AUX')]
 # No additional capacity or Neutral scenarios authorized.
 for candidate,baseline in comparisons:
  a=results.get(candidate);b=results.get(baseline)
  if not a or not b or a['status']!='ALL_COMPLETE' or b['status']!='ALL_COMPLETE' or a['mean_jct'] is None or b['mean_jct'] is None:
   pairs.append(dict(candidate=candidate,baseline=baseline,complete_pair=False));continue
  deltas={j:perjob[baseline][j]['jct']-perjob[candidate][j]['jct'] for j in work};top=max(deltas,key=deltas.get);net=sum(deltas.values())
  decomposition={k:(b['totals'][k]-a['totals'][k])/80 for k in ('wait','first_wait','later_wait','checkpoint','training')}
  assert close(sum(decomposition[k] for k in ('wait','checkpoint','training')),b['mean_jct']-a['mean_jct'])
  byapp={app:dict(count=sum(w['application']==app for w in work.values()),sum_saved_seconds=sum(v for j,v in deltas.items() if work[j]['application']==app),improved=sum(v>1e-7 for j,v in deltas.items() if work[j]['application']==app),worse=sum(v< -1e-7 for j,v in deltas.items() if work[j]['application']==app)) for app in ('cifar10','deepspeech2','bert')}
  pairs.append(dict(candidate=candidate,baseline=baseline,complete_pair=True,mean_jct_saved=b['mean_jct']-a['mean_jct'],improvement_percent=100*(1-a['mean_jct']/b['mean_jct']),mean_jct_decomposition_seconds=decomposition,by_application=byapp,
      p95_improvement_percent=100*(1-a['p95_jct']/b['p95_jct']),p99_improvement_percent=100*(1-a['p99_jct']/b['p99_jct']),makespan_improvement_percent=100*(1-a['makespan']/b['makespan']),
      improved_jobs=sum(x>1e-7 for x in deltas.values()),worse_jobs=sum(x< -1e-7 for x in deltas.values()),ties=sum(abs(x)<=1e-7 for x in deltas.values()),
      top_beneficiary=top,top_original_name=work[top]['original_name'],top_source_trace=work[top]['source_trace'],top_saved_seconds=deltas[top],top_fraction_of_net=deltas[top]/net if net>0 else None,
      diagnostic_without_top_saved_mean=sum(v for j,v in deltas.items() if j!=top)/79,
      diagnostic_without_top_improvement_percent=100*sum(v for j,v in deltas.items() if j!=top)/sum(x['jct'] for j,x in perjob[baseline].items() if j!=top),per_job_saved_seconds=deltas))
 factor={}
 ids=['dual_neutral_A'+x for x in ('00','01','10','11')]
 if all(x in results and results[x]['status']=='ALL_COMPLETE' and results[x]['mean_jct'] is not None for x in ids):
  y00,y01,y10,y11=[results[x]['mean_jct'] for x in ids]
  factor=dict(fifo_saved_mean_seconds=((y00-y10)+(y01-y11))/2,continuity_saved_mean_seconds=((y00-y01)+(y10-y11))/2,
       jct_interaction=y11-y10-y01+y00,sign='Negative interaction means an additional JCT decrease beyond additive component changes. Do not add these effects to resource effects measured under a different objective.')
 resource_factor={}
 ids=['dual_medium_'+x for x in ('PROP','CPU','MEM','FULL')]
 if all(x in results and results[x]['status']=='ALL_COMPLETE' and results[x]['mean_jct'] is not None for x in ids):
  yp,yc,ym,yf=[results[x]['mean_jct'] for x in ids]
  resource_factor=dict(cpu_saved_mean_seconds=((yp-yc)+(ym-yf))/2,memory_saved_mean_seconds=((yp-ym)+(yc-yf))/2,jct_interaction=yf-yc-ym+yp,
      sign='Positive main effects reduce JCT; negative interaction is additional reduction. Same completion objective/FIFO/continuity, variable auxiliary factors only.')
 save(out/'raw_recomputed_metrics.json',results);save(out/'paired_comparisons.json',pairs);save(out/'service_factor_effects.json',factor);save(out/'resource_factor_effects.json',resource_factor)
 with (out/'per_job.csv').open('x',encoding='utf8',newline='') as f:
  keys=['cell','job','original_name','application','arrival','completion','completed','jct','wait','first_wait','later_wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','migrations','gpu_changes','wasted_gpu_seconds','observed_until','arrived_by_observed_end','censor_time']
  w=csv.DictWriter(f,keys);w.writeheader()
  for cell,js in perjob.items():
   for j,x in js.items():w.writerow(dict(cell=cell,job=j,original_name=work[j]['original_name'],application=work[j]['application'],**{k:x[k] for k in keys[4:]}))
 save(out/'coverage.json',dict(expected_core=7,expected_optional=0,attempted=len(results),complete=sum(r['status']=='ALL_COMPLETE' for r in results.values()),violations=sum(r['capacity_and_execution_violations'] for r in results.values()),
    cells={c['id']:results.get(c['id'],dict(status='NOT_RUN'))['status'] for c in p['cells']}))
 print('RAW_AUDIT_COMPLETE',len(results),flush=True)

if __name__=='__main__':main()
