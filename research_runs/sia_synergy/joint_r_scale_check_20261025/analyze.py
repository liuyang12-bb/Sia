"""Independent complete-event, execution-segment, capacity and service audit."""
import collections,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
p=json.loads((OUT/'preregistration_v2.json').read_text())
sys.path.insert(0,str(Path(p['tree'])/'sia-simulator'))
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
def close(a,b):return math.isclose(a,b,abs_tol=1e-5,rel_tol=1e-9)
def main(n):
    results=[];checks=[];traces={}
    for response in p['responses']:
        for method in p['methods']:
            cid=f'n{n}_{response}_{method}';folder=OUT/cid;s=json.loads((folder/'summary.json').read_text())
            oracle=R0Oracle(p['registry'],p['models'],{j:j.split('-')[0] for j in s['jobs']},response)
            states=collections.defaultdict(lambda:dict(wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,progress=0.,completion=None))
            cnt=collections.Counter();actions=collections.Counter();statuses=collections.Counter();clipping=collections.Counter()
            boosts=0.;training=0.;rho_weight=0.;lastd=None;maxactive=0;maxwaiters=0;maxbudget=0;changes=[];trace=[];sumselected=0
            opportunities=collections.Counter()
            with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);d=r['decision_for_next_interval'];fresh=d['round']!=lastd;sel=d['result']['selected'];audit=d['result']['audit']
                    for e in r['execution']:
                        assert audit_execution(e)['valid'];cnt['execution_segments']+=1
                        v=states[e['job_id']];assert close(v['progress'],e['progress_before']);v['progress']=e['progress_after'];v['completion']=e['completion_time']
                        for k,src in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:v[k]+=e[src]
                        training+=e['training_seconds'];boosts+=e['training_seconds'] if e['rho']>1.000001 else 0;rho_weight+=e['rho']*e['training_seconds']
                        a=e['resource_allocation']
                        if a:
                            assert close(e['rho'],independent_rho(oracle,e['job_id'],a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu']))
                            if response=='neutral':assert e['rho']==1
                            dt=e['allocated_seconds']/3600
                            for k,mul in [('gpu_hours',1.),('cpu_core_hours',a['cpu_per_gpu']),('ram_gb_hours',a['ram_gb_per_gpu'])]:v[k]+=dt*a['gpus']*mul
                            if e['checkpoint_remaining_before_decision']>0 and e['completion_time'] is None:
                                c=sel[e['job_id']];assert c['config']['config_id']==a['config_id']
                                for k in ['placement','cpu_per_gpu','ram_gb_per_gpu']:assert c[k]==a[k]
                                cnt['checkpoint_locks']+=1
                    if not fresh:continue
                    lastd=d['round'];cnt['decisions']+=1;assert audit['valid'];statuses[d['result']['status']]+=1
                    specs=d['probe']['specs'];maxactive=max(maxactive,len(specs));maxwaiters=max(maxwaiters,len(specs)-len(sel))
                    cnt['waiting_job_rounds']+=len(specs)-len(sel)
                    usage={t:[0.,0.,0.] for t in ['aws','dgx-ext','rtx']}
                    for j,c in sel.items():
                        t=c['config']['gpu_type'];g=c['config']['gpus'];usage[t][0]+=g;usage[t][1]+=g*c['cpu_per_gpu'];usage[t][2]+=g*c['ram_gb_per_gpu'];sumselected+=1
                        assert close(c['rho'],independent_rho(oracle,j,c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu']))
                        for prefix,point in [('numerator',c['source_coordinates']['requested']),('denominator',c['baseline_source_coordinates']['requested'])]:
                            clipping[prefix+'_cpu']+=int(not 1<=point['cpu']<=24);clipping[prefix+'_ram']+=int(not 20.83<=point['ram_gb']<=250)
                    for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]:
                        u=usage[t];assert u[0]<=g and u[1]<=g*3+1e-7 and u[2]<=g*62.5+1e-7
                        cnt['node_capacity_checks']+=1
                        cnt['cpu_saturated_gpu_idle_node_rounds']+=int(u[1]>=g*3-1e-7 and u[0]<g)
                    # Columns already individually feasible; potential is not a claim of joint attainability.
                    props={ (c['job_id'],c['config']['config_id']):c for c in d['probe']['proportional_columns'] }
                    for j in [z['job_id'] for z in specs]:
                        base=[v for (name,_),v in props.items() if name==j]
                        joint=[v for v in d['probe']['joint_columns'] if v['job_id']==j]
                        gains=[v for v in joint if (j,v['config']['config_id']) in props and
                               v['query']['effective_goodput']>props[(j,v['config']['config_id'])]['query']['effective_goodput']+1e-8]
                        if gains:opportunities['job_round_with_individually_feasible_acceleration']+=1
                    trace.append(dict(time=r['interval_end'],active=len(specs),selected=len(sel),usage=usage,drain=audit.get('drain',False)))
                    actions.update(e['event'] for e in d.get('liveness_events',[]))
                    if method=='joint_r':
                        actual={e[0]:e[-1] for e in r['rescale_events']+r['migrate_events'] if e[1]==r['interval_end']}
                        for j,ch in audit['service_budget'].items():
                            before=ch['before'];after=ch['after'];cost=ch['charged_this_round'];assert close(cost,actual.get(j,0))
                            assert close(after['delay_charged'],(before['delay_charged'] if before else 0)+cost)
                            assert after['delay_charged']<=250;maxbudget=max(maxbudget,after['delay_charged'])
                            assert r['interval_end']-after['issued_time']<=610
                            if before:assert before['issued_time']==after['issued_time'] and before['productive_base']==after['productive_base']
                            if ch['change']=='GPU_GEOMETRY':
                                ev=audit['adaptive_candidate_filters'][j];c=sel[j];H=ev['productive_remaining']
                                assert (H-cost)*c['effective_goodput']>H*ev['current_effective_goodput']
                                assert d['probe']['provenance'][j][c['config']['config_id']]['goodput_source'] in ['direct_profile','projected_profile']
                                changes.append(dict(time=r['interval_end'],job=j,new_config=c['config']['config_id'],cost=cost))
            jcts=[]
            for j,v in states.items():
                row=s['jobs'][j]
                for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours']:assert close(v[k],row[k]),(cid,j,k,v[k],row[k])
                assert v['completion']==row['completion']
                if v['completion'] is not None:
                    jct=v['completion']-row['arrival'];assert close(jct,v['wait']+v['checkpoint']+v['training']);jcts.append(jct)
            assert len(states)==n
            if len(jcts)==n:assert close(np.mean(jcts),s['mean_jct'])
            else:assert s['mean_jct'] is None
            result=dict(s);result['independent_jct_metrics']=dict(mean=float(np.mean(jcts)) if len(jcts)==n else None,
                median=float(np.median(jcts)) if len(jcts)==n else None,p95=float(np.percentile(jcts,95)) if len(jcts)==n else None,
                p99=float(np.percentile(jcts,99)) if len(jcts)==n else None)
            results.append(result);checks.append(dict(cell=cid,valid=True,all_complete=len(jcts)==n,counts=dict(cnt),statuses=dict(statuses),
                actions=dict(actions),max_active=maxactive,max_waiters=maxwaiters,maximum_delay_charged=maxbudget,
                boosted_training_seconds=boosts,training_seconds=training,training_weighted_rho=rho_weight/training,
                selected_count=sumselected,clipping_counts=dict(clipping),opportunity_counts=dict(opportunities)))
            traces[cid]=dict(round_resources=trace,geometry_adaptations=changes)
    save(f'comparison_{n}.json',results);save(f'trace_{n}.json',traces)
    gate=dict(valid=all(v['valid'] for v in checks),all_complete=all(v['all_complete'] for v in checks),cells=checks)
    save(f'validation_{n}.json',gate)
    print('GATE',n,gate['valid'],gate['all_complete'])
if __name__=='__main__':main(int(sys.argv[1]))
