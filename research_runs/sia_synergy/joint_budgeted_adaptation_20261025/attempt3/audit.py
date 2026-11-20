import json,gzip,collections,hashlib,sys,math
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;p=json.loads((OUT/'preregistration.json').read_text())
sys.path.insert(0,str(Path(p['tree'])/'sia-simulator'))
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def close(a,b):return math.isclose(a,b,abs_tol=1e-5,rel_tol=1e-9)
def geom(c):return c['job_id'],c['config']['config_id'],c['placement']['node_gpus'],c['query']['cpu_per_gpu'],c['query']['ram_gb_per_gpu']
def selected_key(c,j):return j,c['config']['config_id'],list(map(list,c['placement'].items())),c['cpu_per_gpu'],c['ram_gb_per_gpu']
def main(n):
    results=[];checks=[];details={}
    for response in ['neutral','medium']:
        cid=f'n{n}_{response}_joint_g';folder=OUT/cid;s=json.loads((folder/'summary.json').read_text())
        costs=json.loads((folder/'start.json').read_text())['actual_delay_costs'];oracle=R0Oracle(p['registry'],p['models'],{j:j.split('-')[0] for j in s['jobs']},response)
        states=collections.defaultdict(lambda:dict(wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,progress=0.,completion=None))
        configs=collections.defaultdict(lambda:collections.defaultdict(float));cnt=collections.Counter();actions=collections.Counter();clipping=collections.Counter();statuses=collections.Counter()
        lastd=None;boost=0.;training=0.;wrho=0.;maxbudget=0.;nsel=0;changes=[];selectionevents=[]
        with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);d=r['decision_for_next_interval'];sel=d['result']['selected'];audit=d['result']['audit']
                for e in r['execution']:
                    assert audit_execution(e)['valid'];assert e['progress_after']+1e-8>=e['progress_before'];cnt['execution']+=1
                    cnt['zero_duration_epoch_settlements']+=sum(v['seconds']==0 and v['progress_delta']>0 for v in e['training_segments'])
                    v=states[e['job_id']];assert close(v['progress'],e['progress_before']);v['progress']=e['progress_after'];v['completion']=e['completion_time']
                    for k,src in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:v[k]+=e[src]
                    training+=e['training_seconds'];boost+=e['training_seconds'] if e['rho']>1.000001 else 0;wrho+=e['training_seconds']*e['rho'];a=e['resource_allocation']
                    if a:
                        assert close(e['rho'],independent_rho(oracle,e['job_id'],a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu']))
                        if response=='neutral':assert e['rho']==1.
                        for k,mul in [('gpu_hours',1.),('cpu_core_hours',a['cpu_per_gpu']),('ram_gb_hours',a['ram_gb_per_gpu'])]:v[k]+=a['gpus']*mul*e['allocated_seconds']/3600
                        configs[e['job_id']][a['config_id']]+=e['training_seconds']
                        if e['checkpoint_remaining_before_decision']>0 and e['completion_time'] is None:
                            c=sel[e['job_id']];assert c['config']['config_id']==a['config_id']
                            for k in ['placement','cpu_per_gpu','ram_gb_per_gpu']:assert c[k]==a[k]
                            cnt['checkpoint_lock']+=1
                if lastd==d['round']:continue
                lastd=d['round'];cnt['decisions']+=1;assert audit['valid'];statuses[d['result']['status']]+=1
                for retry in audit.get('numerical_solver_retries',[]):
                    model=retry['model'];h=hashlib.sha256(np.asarray(model['c'],dtype=float).tobytes())
                    for con in model['constraints']:
                        for key in ['A','lb','ub']:h.update(np.asarray(con[key],dtype=float).tobytes())
                    assert h.hexdigest()==retry['model_fingerprint']
                    def violation(x):
                        x=np.asarray(x);worst=max(float(np.max(-x)),float(np.max(x-1)),float(np.max(np.abs(x-np.round(x)))))
                        for con in model['constraints']:
                            values=np.asarray(con['A'])@x;worst=max(worst,float(np.max(np.asarray(con['lb'])-values)),float(np.max(values-np.asarray(con['ub']))))
                        return worst
                    assert retry['original_status']==0 and violation(retry['original_x'])>1e-7
                    assert retry['retry_status']==0 and violation(retry['retry_x'])<=1e-7
                    assert retry['retry_options']==dict(time_limit=30.,mip_rel_gap=0.,presolve=False)
                    cnt['independently_verified_numerical_retries']+=1
                cols=d['probe']['joint_columns'];specs={v['job_id']:v for v in d['probe']['specs']}
                usage={t:[0.,0.,0.] for t in ['aws','dgx-ext','rtx']};objective=0.;startup=0.;selectedcols={}
                for j,c in sel.items():
                    raw=next(v for v in cols if geom(v)==selected_key(c,j));selectedcols[j]=raw;objective+=raw['scheduling_utility'];nsel+=1
                    q=raw['query'];assert close(c['rho'],independent_rho(oracle,j,c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu']))
                    t=c['config']['gpu_type'];g=c['config']['gpus'];usage[t][0]+=g;usage[t][1]+=g*c['cpu_per_gpu'];usage[t][2]+=g*c['ram_gb_per_gpu']
                    same=specs[j]['current_config_id']==c['config']['config_id'] and specs[j]['current_placement']==list(map(list,c['placement'].items()))
                    startup+=0 if same else costs[j][t]
                    for prefix,pt in [('numerator',c['source_coordinates']['requested']),('denominator',c['baseline_source_coordinates']['requested'])]:
                        clipping[prefix+'_cpu']+=int(not 1<=pt['cpu']<=24);clipping[prefix+'_ram']+=int(not 20.83<=pt['ram_gb']<=250)
                assert close(objective,d['result']['scheduling_objective']);assert close(startup,audit['secondary_tie']['minimum_actual_delay'])
                cnt['waiting_job_rounds']+=len(specs)-len(sel)
                for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]:
                    u=usage[t];assert u[0]<=g and u[1]<=g*3+1e-7 and u[2]<=g*62.5+1e-7;cnt['node_capacity']+=1
                    cnt['cpu_saturated_gpu_idle']+=int(u[1]>=g*3-1e-7 and u[0]<g)
                before=d['probe']['l2_before'];baseline=[]
                for j,c in selectedcols.items():
                    service=before['services'].get(j)
                    if service and before['productive'][j]-service['productive_base']<300.-1e-9:
                        baseline.append(next(v for v in cols if geom(v)==geom(service['column'])))
                    else:baseline.append(c)
                bu={t:[0.,0.,0.] for t in usage}
                for c in baseline:
                    t=c['config']['gpu_type'];g=c['config']['gpus'];q=c['query'];bu[t][0]+=g;bu[t][1]+=g*q['cpu_per_gpu'];bu[t][2]+=g*q['ram_gb_per_gpu']
                feasible=all(bu[t][0]<=g and bu[t][1]<=g*3+1e-7 and bu[t][2]<=g*62.5+1e-7 for t,g in [('aws',4),('dgx-ext',8),('rtx',8)])
                assert feasible==audit['global_bundle_baseline']['feasible']
                baseobj=sum(c['scheduling_utility'] for c in baseline);assert close(baseobj,audit['global_bundle_baseline']['baseline_objective'])
                if feasible:assert objective+1e-7>=baseobj;cnt['nondecreasing_global_objective']+=1
                actual={e[0]:e[-1] for e in r['rescale_events']+r['migrate_events'] if e[1]==r['interval_end']}
                for j,ch in audit['service_budget'].items():
                    old=ch['before'];new=ch['after'];charged=ch['charged_this_round'];assert close(charged,actual.get(j,0.))
                    assert close(new['delay_charged'],(old['delay_charged'] if old else 0)+charged);assert new['delay_charged']<=250.;maxbudget=max(maxbudget,new['delay_charged'])
                    assert r['interval_end']-new['issued_time']<=610
                    if old:assert old['issued_time']==new['issued_time'] and old['productive_base']==new['productive_base'];cnt['clock_not_reset']+=1
                    if ch['change']=='GPU_GEOMETRY':
                        c=sel[j];assert d['probe']['provenance'][j][c['config']['config_id']]['goodput_source'] in ['direct_profile','projected_profile']
                        assert states[j]['training']>old['last_geometry_productive'];changes.append(dict(time=r['interval_end'],job=j,config=c['config']['config_id'],delay=charged))
                    ev=audit['adaptive_candidate_filters'].get(j,{})
                    if sel[j]['column_id'] in {v['column_id'] for v in ev.get('reintroduced',[])}:cnt['selected_previously_vetoed_column']+=1
                actions.update(v['event'] for v in d['liveness_events'])
                selectionevents.append(dict(time=r['interval_end'],required=sorted(sel),objective=objective,baseline=baseobj,baseline_feasible=feasible,startup=startup))
        jcts=[]
        for j,v in states.items():
            row=s['jobs'][j]
            for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours']:assert close(v[k],row[k]),(cid,j,k)
            assert v['completion']==row['completion']
            if row['completion'] is not None:
                jt=row['completion']-row['arrival'];assert close(jt,v['wait']+v['checkpoint']+v['training']);jcts.append(jt)
        assert len(states)==n
        if len(jcts)==n:assert close(np.mean(jcts),s['mean_jct'])
        else:assert s['mean_jct'] is None
        s['independent_jct_metrics']={k:float(v) if len(jcts)==n else None for k,v in [('mean',np.mean(jcts)),('median',np.median(jcts)),('p95',np.percentile(jcts,95)),('p99',np.percentile(jcts,99))]}
        results.append(s);checks.append(dict(cell=cid,valid=True,all_complete=len(jcts)==n,counts=dict(cnt),statuses=dict(statuses),actions=dict(actions),max_delay=maxbudget,
            boosted_training_seconds=boost,training_seconds=training,weighted_rho=wrho/training,selected_count=nsel,clipping=dict(clipping)))
        details[cid]=dict(geometry_adaptations=changes,training_by_config={j:dict(v) for j,v in configs.items()},decisions=selectionevents)
    for fn,data in [(f'comparison_{n}.json',results),(f'validation_{n}.json',dict(valid=True,all_complete=all(v['all_complete'] for v in checks),cells=checks)),(f'mechanism_{n}.json',details)]:
        with (OUT/fn).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
    print('RAW_GATE',n,all(v['all_complete'] for v in checks),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
