"""Describe auxiliary exchanges and neutral controls from raw completed runs."""
import collections,csv,gzip,json,math,sys
from experiment import OUT,SOURCE,METHODS,RUNS,load,save

def main():
    load();sys.path.insert(0,str(SOURCE))
    from joint_prototype.r0_liveness import R0Oracle
    from run_stage2c_cell import independent_rho
    with (OUT/'inputs/workload.csv').open(encoding='utf-8') as f:apps={r['name']:r['application'] for r in csv.DictReader(f)}
    result={};neutral_signatures={}
    for response,method in RUNS:
      path=OUT/'cells'/response/method/'rounds.jsonl.gz'
      if not (path.parent/'summary.json').exists():continue
      oracle=R0Oracle(OUT/'inputs/registry.json',OUT/'inputs/models',apps,response)
      checks=collections.Counter();jobs=collections.defaultdict(collections.Counter);examples=[];signatures=[];last=None
      with gzip.open(path,'rt',encoding='utf-8') as f:
       for line in f:
        r=json.loads(line);d=r['decision_for_next_interval']
        signatures.append(dict(selected={j:(c['config']['config_id'],c['placement']) for j,c in d['result']['selected'].items()},
          execution=[{k:e[k] for k in ['job_id','progress_before','progress_after','completion_time','waiting_seconds','training_seconds','checkpoint_delay_consumed']} for e in r['execution']],
          end=r['job_end_state']))
        for e in r['execution']:
          a=e['resource_allocation']
          if not a or not e['training_seconds']:continue
          prop=independent_rho(oracle,e['job_id'],a['config_id'],6.,125.);ratio=e['rho']/prop
          key='above_proportional_training_seconds' if ratio>1+1e-9 else 'below_proportional_training_seconds' if ratio<1-1e-9 else 'equal_proportional_training_seconds'
          jobs[e['job_id']][key]+=e['training_seconds']
          jobs[e['job_id']]['same_state_proportional_time_equivalent_seconds']+=e['training_seconds']*(ratio-1)
        if last==d['round']:continue
        last=d['round'];positive=[];negative=[];ram=set()
        for j,c in d['result']['selected'].items():
          prop=independent_rho(oracle,j,c['config']['config_id'],6.,125.);ratio=c['rho']/prop
          ref=oracle.reference(j,c['config']['config_id'])
          supported=all(1<=cpu/ref['phi_c']<=24 and 20.83<=ram/ref['phi_m']<=250
            for cpu,ram in [(c['cpu_per_gpu'],c['ram_gb_per_gpu']),(6.,125.),(3.,62.5)])
          row=dict(job=j,config=c['config']['config_id'],cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu'],rho_ratio=ratio,
            placement=c['placement'],numerator_and_reference_supported=supported)
          if ratio>1+1e-9:positive.append(row)
          elif ratio<1-1e-9:negative.append(row)
          checks['selected_columns']+=1;checks['nonproportional_ram']+=c['ram_gb_per_gpu']!=125.
        checks['decisions']+=1;checks['accelerated_job_decisions']+=bool(positive);checks['slowdown_job_decisions']+=bool(negative)
        if positive and negative:
          checks['mixed_exchange_decisions']+=1
          shared=set(n for v in positive for n in v['placement'])&set(n for v in negative for n in v['placement'])
          checks['mixed_shared_node_decisions']+=bool(shared)
          if shared and len(examples)<3:examples.append(dict(time=r['interval_end'],round=d['round'],shared_nodes=sorted(shared),receivers=positive,donors=negative))
      if response=='neutral':neutral_signatures[method]=signatures
      result[response+'/'+method]=dict(counts=dict(checks),jobs=dict(jobs),first_three_mixed_exchange_decisions=examples)
    control={}
    for a,b in [('SIA_P','SIA_ECT_AUX'),('JOINT_CC_P','JOINT_CC')]:
      if a in neutral_signatures and b in neutral_signatures:
        aa=neutral_signatures[a];bb=neutral_signatures[b]
        first=next((i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),None)
        control[a+'_vs_'+b]=dict(exact_gpu_progress_completion_restart_trajectory_equal=aa==bb,
          first_different_interval_index=first,round_counts=[len(aa),len(bb)])
    save(OUT/'mechanism_analysis.json',dict(results=result,neutral_controls=control,
      interpretation='Same-state proportional-time equivalents condition on observed GPU/batch/rate path; they are not realized JCT savings or isolated interventions. No new training run.'))
    print('MECHANISM_ANALYSIS_READY',control,flush=True)

if __name__=='__main__':main()
