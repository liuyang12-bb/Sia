"""Independent raw execution, allocation, admission and completion checks."""
import collections,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;p=json.loads((OUT/'preregistration.json').read_text())
sys.path.insert(0,str(Path(p['tree'])/'sia-simulator'))
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def close(a,b):return math.isclose(a,b,abs_tol=1e-5,rel_tol=1e-9)
def geom(c):return c['job_id'],c['config']['config_id'],c['placement']['node_gpus'],c['query']['cpu_per_gpu'],c['query']['ram_gb_per_gpu']
def key(c,j):return j,c['config']['config_id'],list(map(list,c['placement'].items())),c['cpu_per_gpu'],c['ram_gb_per_gpu']
def save(f,v):
    with f.open('x') as out:json.dump(v,out,indent=2)
def audit(cell):
    cid=cell['id'];folder=OUT/cid;s=json.loads((folder/'summary.json').read_text());joint=cell['method']=='joint_s'
    costs=json.loads((folder/'start.json').read_text())['actual_delay_costs']
    oracle=R0Oracle(p['registry'],p['models'],{j:j.split('-')[0] for j in s['jobs']},cell['response'])
    states=collections.defaultdict(lambda:dict(wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,progress=0.,completion=None))
    cnt=collections.Counter();actions=collections.Counter();statuses=collections.Counter();clipping=collections.Counter()
    trajectories=[];configs=collections.defaultdict(lambda:collections.defaultdict(float));lastd=None;maxbudget=0.;weighted=0.;training=0.
    with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);d=r['decision_for_next_interval'];sel=d['result']['selected'];a=d['result']['audit']
            for e in r['execution']:
                assert audit_execution(e)['valid'];v=states[e['job_id']];assert close(v['progress'],e['progress_before']);assert e['progress_after']+1e-8>=e['progress_before']
                v['progress']=e['progress_after'];v['completion']=e['completion_time'];cnt['execution']+=1
                cnt['epoch_zero_duration_settlements']+=sum(x['seconds']==0 and x['progress_delta']>0 for x in e['training_segments'])
                for k,src in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:v[k]+=e[src]
                training+=e['training_seconds'];weighted+=e['rho']*e['training_seconds'];res=e['resource_allocation']
                if res:
                    assert close(e['rho'],independent_rho(oracle,e['job_id'],res['config_id'],res['cpu_per_gpu'],res['ram_gb_per_gpu']))
                    if cell['response']=='neutral':assert e['rho']==1
                    for k,mul in [('gpu_hours',1.),('cpu_core_hours',res['cpu_per_gpu']),('ram_gb_hours',res['ram_gb_per_gpu'])]:v[k]+=res['gpus']*mul*e['allocated_seconds']/3600
                    configs[e['job_id']][res['config_id']]+=e['training_seconds']
                    if e['checkpoint_remaining_before_decision']>0 and e['completion_time'] is None:
                        c=sel[e['job_id']];assert c['config']['config_id']==res['config_id']
                        for k in ['placement','cpu_per_gpu','ram_gb_per_gpu']:assert c[k]==res[k]
                        cnt['checkpoint_locks']+=1
            if lastd==d['round']:continue
            lastd=d['round'];assert a['valid'];cnt['decisions']+=1;statuses[d['result']['status']]+=1
            cols=d['probe']['joint_columns'];specs={v['job_id']:v for v in d['probe']['specs']}
            usage={t:[0.,0.,0.] for t in ['aws','dgx-ext','rtx']};objective=0.;startup=0.;selectedcols={}
            for j,c in sel.items():
                raw=next(v for v in cols if geom(v)==key(c,j));selectedcols[j]=raw;objective+=raw['scheduling_utility'];cnt['selected_columns']+=1
                assert close(c['rho'],independent_rho(oracle,j,c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu']))
                t=c['config']['gpu_type'];g=c['config']['gpus'];usage[t][0]+=g;usage[t][1]+=g*c['cpu_per_gpu'];usage[t][2]+=g*c['ram_gb_per_gpu']
                same=specs[j]['current_config_id']==c['config']['config_id'] and specs[j]['current_placement']==list(map(list,c['placement'].items()))
                startup+=0 if same else costs[j][t]
                if cell['method']=='sia_p':assert (c['cpu_per_gpu'],c['ram_gb_per_gpu'])==(3.,62.5)
                for prefix,pt in [('numerator',c['source_coordinates']['requested']),('denominator',c['baseline_source_coordinates']['requested'])]:
                    clipping[prefix+'_cpu']+=int(not 1<=pt['cpu']<=24);clipping[prefix+'_ram']+=int(not 20.83<=pt['ram_gb']<=250)
            if cell['method']!='sia_p':assert close(objective,d['result']['scheduling_objective'])
            for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]:
                u=usage[t];assert u[0]<=g and u[1]<=3*g+1e-7 and u[2]<=62.5*g+1e-7;cnt['node_capacity']+=1
                cnt['cpu_saturated_gpu_idle']+=int(u[1]>=g*3-1e-7 and u[0]<g)
            cnt['waiting_job_rounds']+=len(specs)-len(sel);actions.update(e['event'] for e in d.get('liveness_events',[]))
            if joint:
                assert close(startup,a['secondary_tie']['minimum_actual_delay'])
                before=d['probe']['l2_before'];services=before['services'];prod=before['productive']
                held={j for j,v in services.items() if j in specs and prod[j]-v['productive_base']<300.-1e-9}
                ready=[j for j in before['ready'] if j in specs]
                ready += [v['job_id'] for v in sorted(specs.values(),key=lambda v:(v['arrival_time'],v['source_row_index'],v['job_id'])) if v['job_id'] not in set(ready)|set(services)]
                expired=[j for j,v in services.items() if j in specs and j not in held]
                ready += sorted(expired,key=lambda j:services[j]['sequence'])
                required=held|({ready[0]} if ready and not a['drain'] else set())
                optional=set(ready[1:]) if ready and not a['drain'] else set()
                assert set(a['admission_required'])==required and set(a['admission_optional'])==optional
                assert required<=set(sel)<=required|optional
                assert a['ready']==[j for j in ready if j not in sel]
                assert set(a['service_commitments'])==set(sel)
                cnt['omitted_optional_columns']+=len(optional-set(sel));cnt['head_required_rounds']+=int(bool(ready) and not a['drain'])
                for j,v in specs.items():
                    if v['checkpoint_locked']:assert j in sel
                if a['global_bundle_baseline']['feasible']:
                    assert objective+1e-7>=a['global_bundle_baseline']['baseline_objective'];cnt['global_baseline']+=1
                actual={e[0]:e[-1] for e in r['rescale_events']+r['migrate_events'] if e[1]==r['interval_end']}
                for j,ch in a['service_budget'].items():
                    old=ch['before'];new=ch['after'];cost=ch['charged_this_round']
                    assert close(cost,actual.get(j,0.));assert close(new['delay_charged'],(old['delay_charged'] if old else 0)+cost)
                    assert new['delay_charged']<=250;assert r['interval_end']-new['issued_time']<=610;maxbudget=max(maxbudget,new['delay_charged'])
                    if old:assert old['issued_time']==new['issued_time'] and old['productive_base']==new['productive_base']
                    if ch['change']=='GPU_GEOMETRY':
                        assert prod[j]>old['last_geometry_productive']
                        assert d['probe']['provenance'][j][sel[j]['config']['config_id']]['goodput_source'] in ['direct_profile','projected_profile']
                for retry in a['numerical_solver_retries']:
                    model=retry['model'];h=hashlib.sha256(np.asarray(model['c'],dtype=float).tobytes())
                    for con in model['constraints']:
                        for k in ['A','lb','ub']:h.update(np.asarray(con[k],dtype=float).tobytes())
                    assert h.hexdigest()==retry['model_fingerprint']
                    def violation(x):
                        x=np.array(x);v=max(float(np.max(-x)),float(np.max(x-1)),float(np.max(abs(x-np.round(x)))))
                        for con in model['constraints']:
                            val=np.array(con['A'])@x;v=max(v,float(np.max(np.array(con['lb'])-val)),float(np.max(val-np.array(con['ub']))))
                        return v
                    assert violation(retry['original_x'])>1e-7 and violation(retry['retry_x'])<=1e-7;cnt['numerical_retries']+=1
            trajectories.append(dict(time=r['interval_end'],active=len(specs),selected=len(sel),objective=objective,usage=usage,
                required=a.get('admission_required'),optional=a.get('admission_optional'),drain=a.get('drain')))
    jcts=[]
    for j,v in states.items():
        row=s['jobs'][j]
        for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours']:assert close(v[k],row[k]),(cid,j,k)
        assert v['completion']==row['completion']
        if row['completion'] is not None:
            jt=row['completion']-row['arrival'];assert close(jt,v['wait']+v['checkpoint']+v['training']);jcts.append(jt)
    assert len(states)==s['size']
    if len(jcts)==s['size']:assert close(np.mean(jcts),s['mean_jct'])
    else:assert s['mean_jct'] is None
    result=dict(valid=True,all_complete=len(jcts)==s['size'],cell=cid,counts=dict(cnt),statuses=dict(statuses),actions=dict(actions),maxbudget=maxbudget,
        weighted_rho=weighted/training,clipping=dict(clipping),independent_mean_jct=float(np.mean(jcts)) if len(jcts)==s['size'] else None)
    save(folder/'validation.json',result);save(folder/'mechanism.json',dict(rounds=trajectories,training_by_config={j:dict(v) for j,v in configs.items()}))
    print('RAW_AUDIT',cid,result['all_complete'],flush=True);return result
if __name__=='__main__':
    cell=next(c for c in p['runs'] if c['id']==sys.argv[1]);audit(cell)
