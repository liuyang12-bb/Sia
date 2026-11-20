"""Independently integrate raw segments/events; never infer raw rate from rho."""
import collections,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;p=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(Path(p['tree'])/'sia-simulator'))
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def close(a,b):return math.isclose(a,b,abs_tol=1e-5,rel_tol=1e-9)
def geom(c):return c['job_id'],c['config']['config_id'],c['placement']['node_gpus'],c['query']['cpu_per_gpu'],c['query']['ram_gb_per_gpu']
def key(j,c):return j,c['config']['config_id'],list(map(list,c['placement'].items())),c['cpu_per_gpu'],c['ram_gb_per_gpu']
def save(f,v):
 with f.open('x',encoding='utf-8') as o:json.dump(v,o,indent=2)
def audit(cell,folder=None,destination=None):
    folder=Path(folder or OUT/cell['id']);s=json.loads((folder/'summary.json').read_text())
    oracle=R0Oracle(p['registry'],p['models'],{j:j.split('-')[0] for j in s['jobs']},cell['response'])
    states=collections.defaultdict(lambda:dict(wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,progress=0.,completion=None))
    cnt=collections.Counter();statuses=collections.Counter();clipping=collections.Counter();actions=collections.Counter()
    configs=collections.defaultdict(lambda:collections.defaultdict(float));lastd=None;trajectory=[];wrho=0.;train=0.
    with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
      for line in f:
        r=json.loads(line);d=r['decision_for_next_interval'];sel=d['result']['selected'];a=d['result']['audit']
        for e in r['execution']:
            assert audit_execution(e)['valid'];v=states[e['job_id']];assert close(v['progress'],e['progress_before'])
            assert e['progress_after']+1e-8>=e['progress_before'];v['progress']=e['progress_after'];v['completion']=e['completion_time'];cnt['execution']+=1
            for k,src in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:v[k]+=e[src]
            train+=e['training_seconds'];wrho+=e['training_seconds']*e['rho'];res=e['resource_allocation']
            if res:
                assert close(e['rho'],independent_rho(oracle,e['job_id'],res['config_id'],res['cpu_per_gpu'],res['ram_gb_per_gpu']))
                if cell['response']=='neutral':assert e['rho']==1
                for k,mul in [('gpu_hours',1.),('cpu_core_hours',res['cpu_per_gpu']),('ram_gb_hours',res['ram_gb_per_gpu'])]:v[k]+=res['gpus']*mul*e['allocated_seconds']/3600
                configs[e['job_id']][res['config_id']]+=e['training_seconds']
                if e.get('checkpoint_remaining_before_decision',0)>0 and e['completion_time'] is None:
                    c=sel[e['job_id']];assert c['config']['config_id']==res['config_id']
                    for k in ['placement','cpu_per_gpu','ram_gb_per_gpu']:assert c[k]==res[k]
                    cnt['checkpoint_locks']+=1
        if lastd==d['round']:continue
        lastd=d['round'];assert a['valid'];cnt['decisions']+=1;statuses[d['result']['status']]+=1
        cols=d['probe']['joint_columns'];specs={v['job_id']:v for v in d['probe']['specs']}
        usage={t:[0.,0.,0.] for t in ['aws','dgx-ext','rtx']};objective=0.
        for j,c in sel.items():
            raw=next(v for v in cols if geom(v)==key(j,c));objective+=raw['scheduling_utility'];cnt['selected_columns']+=1
            assert close(c['rho'],independent_rho(oracle,j,c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu']))
            t=c['config']['gpu_type'];g=c['config']['gpus'];usage[t][0]+=g;usage[t][1]+=g*c['cpu_per_gpu'];usage[t][2]+=g*c['ram_gb_per_gpu']
            if cell['method']=='sia_p':assert (c['cpu_per_gpu'],c['ram_gb_per_gpu'])==(3.,62.5)
            for prefix,pt in [('numerator',c['source_coordinates']['requested']),('denominator',c['baseline_source_coordinates']['requested'])]:
                clipping[prefix+'_cpu']+=int(not 1<=pt['cpu']<=24);clipping[prefix+'_ram']+=int(not 20.83<=pt['ram_gb']<=250)
        if cell['method']!='sia_p':assert close(objective,d['result']['scheduling_objective'])
        for spec in specs.values():
            if spec['checkpoint_locked'] and spec['current_config_id']:
                c=sel[spec['job_id']];assert c['config']['config_id']==spec['current_config_id'];assert list(map(list,c['placement'].items()))==spec['current_placement']
                assert c['cpu_per_gpu']==spec['current_cpu_per_gpu'] and c['ram_gb_per_gpu']==spec['current_ram_gb_per_gpu']
                cnt['lock_selection_checks']+=1
        for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]:
            u=usage[t];assert u[0]<=g and u[1]<=3*g+1e-7 and u[2]<=62.5*g+1e-7;cnt['node_capacity']+=1
            cnt['cpu_saturated_gpu_idle']+=int(u[1]>=g*3-1e-7 and u[0]<g)
        cnt['waiting_job_rounds']+=len(specs)-len(sel);cnt['empty_with_active']+=int(bool(specs) and not sel)
        cnt['true_fallback']+=int(d['result']['status'].startswith('FALLBACK'))
        actions.update(e['event'] for e in d.get('liveness_events',[]))
        if cell['method']=='b3_free':
            assert not d['probe']['l2_before'] and not d['probe']['l2_after']
            assert a.get('bridge_admission')=='FREE_OPTIONAL_JOBS_EXACT_CHECKPOINT_LOCKS_ONLY'
        if cell['method']=='joint_g':
            assert set(a['service_commitments'])==set(sel)
            for v in a['service_delay_ledgers'].values():assert v['delay_charged']<=250
        for retry in a.get('numerical_solver_retries',[]):
            model=retry['model'];h=hashlib.sha256(np.asarray(model['c'],dtype=float).tobytes())
            for con in model['constraints']:
                for k in ['A','lb','ub']:h.update(np.asarray(con[k],dtype=float).tobytes())
            assert h.hexdigest()==retry['model_fingerprint']
            x=np.asarray(retry['retry_x']);assert np.max(abs(x-np.round(x)))<=1e-7
            for con in model['constraints']:
                val=np.asarray(con['A'])@x;assert np.all(val>=np.asarray(con['lb'])-1e-7) and np.all(val<=np.asarray(con['ub'])+1e-7)
            cnt['numerical_retries']+=1
        trajectory.append(dict(time=r['interval_end'],selected=sorted(sel),objective=objective,usage=usage,
            configs={j:c['config']['config_id'] for j,c in sel.items()},drain=a.get('drain'),waiting=sorted(set(specs)-set(sel))))
    jcts=[]
    for j,v in states.items():
        row=s['jobs'][j]
        for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours']:assert close(v[k],row[k]),(j,k,v[k],row[k])
        assert v['completion']==row['completion']
        if row['completion'] is not None:
            jt=row['completion']-row['arrival'];assert close(jt,v['wait']+v['checkpoint']+v['training']);jcts.append(jt)
    assert len(states)==s['total'];assert len(jcts)==s['completed']
    if len(jcts)==s['total']:assert close(np.mean(jcts),s['mean_jct'])
    else:assert s['mean_jct'] is None
    result=dict(valid=True,all_complete=len(jcts)==s['total'],counts=dict(cnt),statuses=dict(statuses),actions=dict(actions),
        clipping=dict(clipping),weighted_rho=wrho/train if train else None,independent_mean_jct=float(np.mean(jcts)) if len(jcts)==s['total'] else None)
    dest=Path(destination or folder);dest.mkdir(parents=True,exist_ok=True)
    save(dest/'validation.json',result);save(dest/'mechanism.json',dict(rounds=trajectory,training_by_config={j:dict(v) for j,v in configs.items()}))
    print('RAW_AUDIT',cell.get('id',cell['method']),result['all_complete'],flush=True);return result
if __name__=='__main__':audit(next(c for c in p['runs'] if c['id']==sys.argv[1]))
