"""Independent raw-event audit for the fixed eight-cell workload expansion."""
import collections, csv, gzip, json, math, sys, time
import numpy as np
from experiment import OUT, SOURCE, PARENT, RUNS, METHODS, load, save, sha, WARNING

def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-5),(a,b)
def main():
    p=load();sys.path.insert(0,str(SOURCE))
    from joint_prototype.r0_liveness import R0Oracle
    from joint_prototype.execution_audit import audit_execution
    from run_stage2c_cell import independent_rho
    with (OUT/'inputs/workload.csv').open(encoding='utf-8') as f:work=list(csv.DictReader(f))
    apps={r['name']:r['application'] for r in work};arrival={r['name']:float(r['time']) for r in work}
    results={};per_job=[]
    for response,method in RUNS:
      key=response+'/'+method;directory=OUT/'cells'/response/method
      if not (directory/'summary.json').exists():results[key]=dict(status='FAILED_OR_NOT_RUN');continue
      summary=json.loads((directory/'summary.json').read_text());oracle=R0Oracle(OUT/'inputs/registry.json',OUT/'inputs/models',apps,response)
      totals=collections.defaultdict(collections.Counter);progress=collections.defaultdict(float);completed={}
      count=collections.Counter();statuses=collections.Counter();events=collections.Counter();clipping=collections.Counter()
      by_config=collections.defaultdict(collections.Counter);node_hours=collections.defaultdict(collections.Counter)
      queue=collections.Counter();prior_selected={};prior_ledgers={};last_round=None;last_state={};solve=[];rhos=[]
      with gzip.open(directory/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
       for line in f:
        r=json.loads(line)
        for e in r['execution']:
          assert audit_execution(e)['valid'];j=e['job_id'];close(progress[j],e['progress_before']);progress[j]=e['progress_after']
          assert e['progress_after']>=e['progress_before']-1e-8
          for k,field in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:totals[j][k]+=e[field]
          if e['completion_time'] is not None:completed[j]=e['completion_time']
          if e['checkpoint_delay_consumed']>=e['observed_active_seconds']>0:close(e['progress_delta'],0.);count['full_delay_no_training']+=1
          a=e['resource_allocation']
          if a:
            rho=independent_rho(oracle,j,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu']);close(e['rho'],rho);rhos.append(rho)
            if response=='neutral':assert rho==1.
            for seg in e['training_segments']:
              close(seg['effective_rate'],seg['raw_rate_before_rho']*rho)
              close(seg['progress_delta'],seg['effective_rate']*seg['seconds']+seg['rounding_progress_residual']);count['training_segments']+=1
            dt=e['allocated_seconds']/3600;totals[j]['gpu_hours']+=a['gpus']*dt
            for u in a['node_usage']:
              totals[j]['cpu_core_hours']+=u['cpu']*dt;totals[j]['ram_gb_hours']+=u['ram_gb']*dt
              for k,v in [('gpu',u['gpus']),('cpu',u['cpu']),('ram',u['ram_gb'])]:node_hours[u['node_id']][k]+=v*dt
            by_config[j][a['config_id']]+=e['training_seconds']
          count['execution_records']+=1
        last_state=r['job_end_state']
        d=r['decision_for_next_interval']
        if d['round']==last_round:continue
        last_round=d['round'];res=d['result'];assert res['audit']['valid'];selected=res['selected'];audit=res['audit']
        statuses[res['status']]+=1;solve.append(res['solve_seconds']);events.update(e['event'] for e in d.get('liveness_events',[]))
        count['decision_rounds']+=1
        specs={s['job_id']:s for s in d['probe']['specs']}
        waiting=set(specs)-set(selected);queue['rounds_with_waiters']+=bool(waiting)
        queue['max_unserved_active_jobs']=max(queue['max_unserved_active_jobs'],len(waiting))
        queue['sum_unserved_job_boundaries']+=len(waiting)
        used=collections.defaultdict(lambda:[0.,0.,0.]);scores={s['column_id']:s for s in d.get('completion_scores',[])};objective=0.
        for j,col in selected.items():
          cfg=col['config'];rho=independent_rho(oracle,j,cfg['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu']);close(col['rho'],rho)
          if response=='neutral':assert rho==1.
          if method in ['SIA_P','JOINT_CC_P']:assert (col['cpu_per_gpu'],col['ram_gb_per_gpu'])==(6.,125.)
          row=oracle.reference(j,cfg['config_id']);close(cfg['cpu_demand_scale'],row['phi_c']);close(cfg['memory_demand_scale'],row['phi_m'])
          assert cfg['gpu_type'] in ['aws','dgx-ext'] and cfg['num_nodes']==1 and cfg['gpus'] in [1,2]
          assert sum(col['placement'].values())==cfg['gpus']
          for label,cpu,ram in [('numerator',col['cpu_per_gpu'],col['ram_gb_per_gpu']),('denominator',3.,62.5)]:
            clipping[label+'_cpu']+=not 1<=cpu/row['phi_c']<=24
            clipping[label+'_ram']+=not 20.83<=ram/row['phi_m']<=250
          count['selected_columns']+=1
          for node,g in col['placement'].items():used[node][0]+=g;used[node][1]+=g*col['cpu_per_gpu'];used[node][2]+=g*col['ram_gb_per_gpu']
          if scores:
            score=scores[col['column_id']];objective+=score['utility']
            if score['estimated_completion'] is not None:
              eta=score['delay']+score['remaining_work']/(score['raw_rate']*score['rho']);close(eta,score['estimated_completion'])
              if 'completion_priority' in score:
                close(score['completion_priority'],1/eta);close(score['utility'],1/eta/score['common_numerical_scale'])
            count['score_checks']+=1
        if scores:close(objective,res['scheduling_objective'])
        for node,v in used.items():
          assert all(x<=y+1e-7 for x,y in zip(v,[2,12,250]));count['node_capacity_checks']+=1
          queue['cpu_saturated_gpu_idle_node_rounds']+=v[1]>=12-1e-8 and v[0]<2
        for j,spec in specs.items():
          if spec['checkpoint_locked'] and spec['current_config_id']:
            c=selected[j];assert c['config']['config_id']==spec['current_config_id'] and c['placement']==dict(spec['current_placement'])
            close(c['cpu_per_gpu'],spec['current_cpu_per_gpu']);close(c['ram_gb_per_gpu'],spec['current_ram_gb_per_gpu']);count['checkpoint_locks']+=1
        for diag in audit.get('numerical_solver_retries',[]):
          if diag.get('canonical_binary_accepted'):
            binary=np.asarray(diag['canonical_x']);assert np.all((binary==0)|(binary==1))
            for con in diag['model']['constraints']:
              value=np.asarray(con['A'])@binary;lo=np.asarray(con['lb']);hi=np.asarray(con['ub'])
              narrow=np.isfinite(lo)&np.isfinite(hi)&(hi-lo<1e-6);tol=np.where(narrow,1e-10,1e-7)
              assert np.all(value>=lo-tol) and np.all(value<=hi+tol)
            count['canonical_binary_constraint_checks']+=1
        if audit.get('completion_continuity'):
          assert not any(e['event']=='REQUEUE_PRODUCTIVE_QUANTUM' for e in d.get('liveness_events',[]))
          for j,old in prior_selected.items():
            if j in completed:continue
            assert j in selected
            if not audit['geometry_event_reasons']:
              assert old['config']['config_id']==selected[j]['config']['config_id'] and old['placement']==selected[j]['placement'];count['event_geometry_checks']+=1
          ledgers=audit['service_delay_ledgers']
          for j,v in ledgers.items():
            assert v['delay_charged']<=250+1e-8
            if j in prior_ledgers:assert v['issued_time']==prior_ledgers[j]['issued_time'] and v['delay_charged']>=prior_ledgers[j]['delay_charged']
            count['service_ledger_checks']+=1
          prior_ledgers=ledgers
        if d.get('strict_full_intent'):
          # Explicit per-job intent retained by the adapter; full independent
          # canonical check is also performed by the unchanged run-time guard.
          count['strict_sia_intent_guard_rounds']+=1
        prior_selected=selected
      for j,row in summary['jobs'].items():
        assert row['completed']==(j in completed)
        observed=(completed[j] if j in completed else summary['simulated_seconds'])-arrival[j]
        close(sum(totals[j][k] for k in ['wait','checkpoint','training']),observed)
        if row['completed']:close(row['completion'],completed[j]);close(row['jct'],completed[j]-arrival[j])
        else:assert row['jct'] is None;close(row['censor_time'],observed)
        for k,v in totals[j].items():close(row[k],v)
        close(row['restarts'],last_state[j]['restarts']);close(row['wasted_gpu_seconds'],last_state[j]['wasted_gpu_seconds'])
        per_job.append(dict(response=response,method=method,job=j,**row))
      for node,values in node_hours.items():
        for k,v in values.items():close(v,summary['node_resource_hours'][node][k])
      close(summary['restricted_mean_jct'],np.mean([min(completed.get(j,summary['simulated_seconds'])-arrival[j],360000) for j in apps]))
      if len(completed)==12:
        vals=[completed[j]-arrival[j] for j in apps]
        for k,v in [('mean_jct',np.mean(vals)),('median_jct',np.median(vals)),('p95_jct',np.percentile(vals,95)),('p99_jct',np.percentile(vals,99)),('makespan',max(completed.values()))]:close(summary[k],v)
      else:assert summary['mean_jct'] is None
      actual_fallbacks=sum(v for k,v in statuses.items() if k.startswith('FALLBACK_'));close(actual_fallbacks,summary['fallback_count'])
      results[key]=dict(valid=True,summary=summary,checks=dict(count),queue=dict(queue),clipping=dict(clipping),
        capacity_violations=0,events=dict(events),solver_status=dict(statuses),fallback_count=actual_fallbacks,
        source_log_sha256=sha(directory/'rounds.jsonl.gz'),productive_seconds_by_config=dict(by_config),
        utilization={n:{'gpu':v['gpu']/(2*summary['simulated_seconds']/3600),
          'cpu':v['cpu']/(12*summary['simulated_seconds']/3600),'ram':v['ram']/(250*summary['simulated_seconds']/3600)} for n,v in node_hours.items()})
    comparisons={};parent6=json.loads((PARENT/'comparison_summary.json').read_text())
    parent_original=set(json.loads((PARENT/'cells/JOINT_CC/summary.json').read_text())['jobs'])
    for response in ['neutral','medium']:
      paired={m:results.get(response+'/'+m,{}).get('summary') for m in METHODS}
      if not all(v and v['completion_rate']==1 for v in paired.values()):continue
      comparisons[response]={}
      for baseline in METHODS[:-1]:
        a=paired['JOINT_CC'];b=paired[baseline];delta={j:a['jobs'][j]['jct']-b['jobs'][j]['jct'] for j in apps}
        best=min(delta,key=delta.get);others=set(apps)-{best}
        groups={label:dict(jobs=sorted(names),mean_delta=sum(delta[j] for j in names)/len(names),
          improvement=1-sum(a['jobs'][j]['jct'] for j in names)/sum(b['jobs'][j]['jct'] for j in names))
          for label,names in [('original_six',parent_original),('added_six',set(apps)-parent_original)]}
        comparisons[response][baseline]=dict(improvement=1-a['mean_jct']/b['mean_jct'],mean_delta=a['mean_jct']-b['mean_jct'],
          per_job_delta=delta,improved=sum(v<0 for v in delta.values()),worsened=sum(v>0 for v in delta.values()),
          unchanged=sum(v==0 for v in delta.values()),largest_beneficiary=best,largest_saving=-delta[best],
          without_largest_diagnostic_improvement=1-sum(a['jobs'][j]['jct'] for j in others)/sum(b['jobs'][j]['jct'] for j in others),
          time_delta={k:a['totals'][k]-b['totals'][k] for k in ['wait','checkpoint','training']},groups=groups)
    result=dict(status='EIGHT_COMPLETE_VALID' if len(results)==8 and all(r.get('summary',{}).get('completion_rate')==1 for r in results.values()) else 'INCOMPLETE_NO_IMPUTATION',
      results=results,comparisons=comparisons,warning=WARNING,
      note='One controlled12-job cohort. Jobs are not independent repeated experiments. No factorial interaction claim.')
    save(OUT/'independent_audit.json',result)
    save(OUT/'comparison_summary.json',dict(status=result['status'],comparisons=comparisons,methods={k:v.get('summary') for k,v in results.items()}))
    with (OUT/'per_job.csv').open('x',encoding='utf-8',newline='') as f:
      fields=['response','method','job','completed','jct','completion','censor_time','wait','checkpoint','training','restarts','gpu_hours','cpu_core_hours','ram_gb_hours','wasted_gpu_seconds']
      w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(per_job)
    print('RAW_AUDIT',result['status'],json.dumps(comparisons,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
