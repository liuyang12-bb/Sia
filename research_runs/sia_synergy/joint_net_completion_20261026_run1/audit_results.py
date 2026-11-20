"""Recompute endpoints and constraints from raw events, not summary ratios."""
from pathlib import Path
import collections, csv, gzip, json, math, sys, time
import numpy as np
from experiment import OUT, METHODS, PREVIOUS, load, sha, save

def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-5),(a,b)

def signature(round_):
    d=round_['decision_for_next_interval']
    return dict(round=d['round'],selected={j:{k:v[k] for k in
        ['config','placement','cpu_per_gpu','ram_gb_per_gpu','rho']} for j,v in d['result']['selected'].items()},
        execution=[{k:e[k] for k in ['job_id','progress_before','progress_after','completion_time',
            'waiting_seconds','training_seconds','checkpoint_delay_consumed','rho']} for e in round_['execution']],
        job_end=round_['job_end_state'])

def main():
    p=load();sys.path.insert(0,str(OUT/'source'))
    from joint_prototype.r0_liveness import R0Oracle
    from joint_prototype.execution_audit import audit_execution
    from run_stage2c_cell import independent_rho
    rows=list(csv.DictReader((OUT/'inputs/workload.csv').open(encoding='utf-8')))
    apps={r['name']:r['application'] for r in rows};arrivals={r['name']:float(r['time']) for r in rows}
    oracle=R0Oracle(OUT/'inputs/registry.json',OUT/'inputs/models',apps,'medium')
    results={};perjob=[];oldnames={m:m for m in ['SIA_P','SIA_ECT_AUX','JOINT_CC_P','JOINT_CC']}
    for method in METHODS:
      directory=OUT/'cells'/method
      if not (directory/'summary.json').exists():
        results[method]=dict(status='FAILED_OR_NOT_RUN');continue
      summary=json.loads((directory/'summary.json').read_text())
      totals=collections.defaultdict(collections.Counter);progress=collections.defaultdict(float);completed={}
      count=collections.Counter();byconfig=collections.defaultdict(collections.Counter)
      previous_selected={};previous_services={};last_round=None;signatures=[];events=collections.Counter()
      clipping=collections.Counter();decision_rows=[]
      with gzip.open(directory/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
          r=json.loads(line);signatures.append(signature(r))
          for e in r['execution']:
            assert audit_execution(e)['valid'];j=e['job_id'];close(progress[j],e['progress_before']);progress[j]=e['progress_after']
            assert e['progress_after']>=e['progress_before']-1e-8
            for key,field in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:totals[j][key]+=e[field]
            if e['completion_time'] is not None:completed[j]=e['completion_time']
            allocation=e['resource_allocation']
            if allocation:
              rho=independent_rho(oracle,j,allocation['config_id'],allocation['cpu_per_gpu'],allocation['ram_gb_per_gpu']);close(e['rho'],rho)
              for seg in e['training_segments']:
                close(seg['effective_rate'],seg['raw_rate_before_rho']*rho)
                close(seg['progress_delta'],seg['effective_rate']*seg['seconds']+seg['rounding_progress_residual'])
                count['training_segments']+=1
              dt=e['allocated_seconds']/3600
              totals[j]['gpu_hours']+=allocation['gpus']*dt
              for u in allocation['node_usage']:
                totals[j]['cpu_core_hours']+=u['cpu']*dt;totals[j]['ram_gb_hours']+=u['ram_gb']*dt
              byconfig[j][allocation['config_id']]+=e['training_seconds']
            count['execution_records']+=1
          d=r['decision_for_next_interval']
          if d['round']==last_round:continue
          last_round=d['round'];res=d['result'];assert res['audit']['valid'];selected=res['selected']
          used=collections.defaultdict(lambda:[0.,0.,0.]);scores={v['column_id']:v for v in d.get('completion_scores',[])}
          objective=0.
          for j,col in selected.items():
            config=col['config'];rho=independent_rho(oracle,j,config['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu']);close(col['rho'],rho)
            if method in ['SIA_P','JOINT_CC_P','JOINT_NC_P']:assert (col['cpu_per_gpu'],col['ram_gb_per_gpu'])==(6.,125.)
            row=oracle.reference(j,config['config_id'])
            for label,cpu,ram in [('numerator',col['cpu_per_gpu'],col['ram_gb_per_gpu']),('denominator',3.,62.5)]:
              clipping[label+'_cpu']+=not 1<=cpu/row['phi_c']<=24
              clipping[label+'_ram']+=not 20.83<=ram/row['phi_m']<=250
            count['selected_columns']+=1
            for node,g in col['placement'].items():used[node][0]+=g;used[node][1]+=g*col['cpu_per_gpu'];used[node][2]+=g*col['ram_gb_per_gpu']
            if scores:
              v=scores[col['column_id']];objective+=v['utility']
              if v['estimated_completion'] is not None:
                cost=v['delay']+v['remaining_work']/(v['raw_rate']*v['rho']);close(v['estimated_completion'],cost)
                if 'net_completion_cost' in v:
                  close(v['utility'],-cost/v['net_completion_scale'])
                elif 'completion_priority' in v:
                  close(v['completion_priority'],1/cost);close(v['utility'],1/cost/v['common_numerical_scale'])
              count['score_checks']+=1
          if scores:close(objective,res['scheduling_objective'])
          for node,v in used.items():
            assert all(x<=y+1e-7 for x,y in zip(v,[2,12,250]));count['node_capacity_checks']+=1
          for spec in d['probe']['specs']:
            if spec['checkpoint_locked'] and spec['current_config_id']:
              col=selected[spec['job_id']];assert col['config']['config_id']==spec['current_config_id'] and col['placement']==dict(spec['current_placement'])
              close(col['cpu_per_gpu'],spec['current_cpu_per_gpu']);close(col['ram_gb_per_gpu'],spec['current_ram_gb_per_gpu']);count['checkpoint_locks']+=1
          audit=res['audit'];continuity=audit.get('completion_continuity',False)
          if method.startswith('JOINT_NC'):
            net=audit['net_completion']
            assert set(selected)==set(net['reference_jobs'])
            close(net['chosen_eta_sum'],sum(scores[v['column_id']]['estimated_completion'] or 0. for v in selected.values()))
            assert net['chosen_eta_sum']<=net['reference_eta_sum']+1e-5
            for j,cid in net['unknown_frozen'].items():assert selected[j]['column_id']==cid
            count['net_cost_audits']+=1
          for diagnostic in audit.get('numerical_solver_retries',[]):
            if diagnostic.get('canonical_binary_accepted'):
              binary=np.asarray(diagnostic['canonical_x'])
              assert np.all((binary==0)|(binary==1))
              for constraint in diagnostic['model']['constraints']:
                values=np.asarray(constraint['A'])@binary
                lower=np.asarray(constraint['lb']);upper=np.asarray(constraint['ub'])
                narrow=np.isfinite(lower)&np.isfinite(upper)&(upper-lower<1e-6)
                tol=np.where(narrow,1e-10,1e-7)
                assert np.all(values>=lower-tol) and np.all(values<=upper+tol)
              count['canonical_binary_full_model_checks']+=1
          # R0 emits per-round events and then clears its event buffer.
          new_events=d.get('liveness_events',[]);events.update(e['event'] for e in new_events)
          if continuity:
            assert not any(e['event']=='REQUEUE_PRODUCTIVE_QUANTUM' for e in new_events)
            for j,old in previous_selected.items():
              if j in completed:continue
              assert j in selected,(method,j,'evicted before completion')
              if not audit['geometry_event_reasons']:
                assert old['config']['config_id']==selected[j]['config']['config_id'] and old['placement']==selected[j]['placement']
                count['non_event_geometry_checks']+=1
            ledgers=audit['service_delay_ledgers']
            for j,v in ledgers.items():
              assert v['delay_charged']<=250+1e-8
              if j in previous_services:
                assert v['issued_time']==previous_services[j]['issued_time']
                assert v['delay_charged']>=previous_services[j]['delay_charged']
              count['lifetime_startup_checks']+=1
            previous_services=ledgers
          previous_selected=selected
          decision_rows.append(dict(round=d['round'],status=res['status'],selected=selected,
            geometry_event_reasons=audit.get('geometry_event_reasons'),completion_scores=list(scores.values()),
            ready=audit.get('ready'),drain=audit.get('drain'),service_ledgers=audit.get('service_delay_ledgers')))
      for j,v in summary['jobs'].items():
        if v['completed']:
          close(v['completion'],completed[j]);close(v['jct'],completed[j]-arrivals[j]);close(v['jct'],sum(totals[j][k] for k in ['wait','checkpoint','training']))
        for k,val in totals[j].items():close(v[k],val)
        perjob.append(dict(method=method,job=j,**v))
      if len(completed)==6:close(summary['mean_jct'],np.mean([completed[j]-arrivals[j] for j in apps]))
      regression=None
      if method in oldnames:
        with gzip.open(PREVIOUS/'cells'/oldnames[method]/'rounds.jsonl.gz','rt',encoding='utf-8') as f:old=[signature(json.loads(l)) for l in f]
        regression=dict(exact_trajectory_equal=old==signatures,old_rounds=len(old),new_rounds=len(signatures))
        assert regression['exact_trajectory_equal'],(method,'historical regression changed')
      with gzip.open(directory/'audited_decisions.jsonl.gz','xt',encoding='utf-8') as f:
        for row in decision_rows:f.write(json.dumps(row)+'\n')
      results[method]=dict(valid=True,summary=summary,checks=dict(count),clipping=dict(clipping),events=dict(events),
        capacity_violations=0,regression=regression,productive_seconds_by_config=dict(byconfig))
    comparisons={}
    if all(results.get(m,{}).get('summary',{}).get('completion_rate')==1 for m in METHODS):
      for target,base in [('JOINT_NC',m) for m in METHODS if m!='JOINT_NC']+[
          ('JOINT_NC_P','JOINT_CC_P')]:
        a=results[target]['summary'];b=results[base]['summary'];delta={j:a['jobs'][j]['jct']-b['jobs'][j]['jct'] for j in apps}
        best=min(delta,key=delta.get);rest=[j for j in apps if j!=best]
        comparisons[target+'_vs_'+base]=dict(mean_delta=a['mean_jct']-b['mean_jct'],
          improvement=1-a['mean_jct']/b['mean_jct'],job_delta=delta,improved=sum(v<0 for v in delta.values()),
          worsened=sum(v>0 for v in delta.values()),largest_beneficiary=best,largest_saving=-delta[best],
          without_largest_diagnostic_improvement=1-sum(a['jobs'][j]['jct'] for j in rest)/sum(b['jobs'][j]['jct'] for j in rest),
          time_deltas={k:a['totals'][k]-b['totals'][k] for k in ['wait','checkpoint','training']})
    interaction=None
    result=dict(status='ALL_COMPLETE_VALID' if comparisons else 'INCOMPLETE_NO_IMPUTATION',results=results,
      comparisons=comparisons,interaction_jct_seconds=interaction,
      interaction_definition='Not a factorial experiment; no interaction estimate',
      prereg_to_audit_seconds=time.time()-p['created_unix'],warning=p['warning'])
    save(OUT/'independent_audit.json',result)
    save(OUT/'comparison_summary.json',dict(status=result['status'],comparisons=comparisons,
      interaction_jct_seconds=interaction,methods={m:r.get('summary') for m,r in results.items()}))
    with (OUT/'per_job.csv').open('x',encoding='utf-8',newline='') as f:
      w=csv.DictWriter(f,fieldnames=['method','job','jct','completion','completed','wait','checkpoint','training','restarts','gpu_hours','cpu_core_hours','ram_gb_hours','wasted_gpu_seconds'],extrasaction='ignore');w.writeheader();w.writerows(perjob)
    print(json.dumps(dict(status=result['status'],comparisons=comparisons,interaction=interaction),indent=2))

if __name__=='__main__':main()
