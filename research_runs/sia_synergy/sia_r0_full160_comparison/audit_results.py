"""Recompute completion/time/resource metrics from raw rounds, not summaries."""
import collections,csv,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];WORK=ROOT/'sia-joint-full160/sia-simulator'
sys.path.insert(0,str(WORK))
from joint_prototype.full160 import capacities
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
from joint_prototype.execution_audit import audit_execution

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6)

def cell_audit(folder,cell,p):
    with Path(p['traces'][str(cell['trace'])]['file']).open() as f:workload=list(csv.DictReader(f))
    jobs={r['name']:dict(application=r['application'],arrival=float(r['time']),completion=None,
        wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,progress=0.) for r in workload}
    o=R0Oracle(p['registry'],p['models'],{r['name']:r['application'] for r in workload},cell['response'])
    cap=capacities(cell['capacity']);ng={'aws':4,'dgx-ext':8,'rtx':8};checks=collections.Counter();clip=collections.Counter();selected_count=0
    previous_d=None;previous_resources={};last_end=0.;endstate={};status=collections.Counter();fallback=collections.Counter();solverwall=[];schedulewall=0.
    with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);assert r['interval_start']==last_end;last_end=r['interval_end'];checks['rounds']+=1
            schedulewall+=r['round_wall_seconds']-r['job_step_wall_seconds'];endstate=r['job_end_state']
            d=r['decision_for_next_interval'];fresh=d.get('round')!=previous_d
            if fresh:
                previous_d=d.get('round');res=d['result'];status[res['status']]+=1;solverwall.append(res['solve_seconds'])
                if res['status'].startswith('FALLBACK_'):fallback[res['status']]+=1
                node={f'{t}:0':[0.,0.,0.] for t in cap}
                for name,col in res['selected'].items():
                    cfg=col['config'];row=o.reference(name,cfg['config_id']);selected_count+=1
                    assert (cfg['cpu_demand_scale'],cfg['memory_demand_scale'])==(row['phi_c'],row['phi_m'])
                    expected=independent_rho(o,name,cfg['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu'])
                    assert close(expected,col['rho']);checks['selected_rho']+=1
                    if cell['response']=='neutral':assert col['rho']==1.
                    if cell['method'] in ('sia_p','joint_p'):
                        assert (col['cpu_per_gpu'],col['ram_gb_per_gpu'])==(cap[cfg['gpu_type']]['cpu_cores']/ng[cfg['gpu_type']],62.5)
                    for prefix,cpu,ram in [('numerator',col['cpu_per_gpu'],col['ram_gb_per_gpu']),('denominator',3.,62.5)]:
                        clip[prefix+'_cpu']+=int(not 1<=cpu/row['phi_c']<=24)
                        clip[prefix+'_ram']+=int(not 20.83<=ram/row['phi_m']<=250)
                    assert sum(col['placement'].values())==cfg['gpus']
                    for n,g in col['placement'].items():
                        node[n][0]+=g;node[n][1]+=g*col['cpu_per_gpu'];node[n][2]+=g*col['ram_gb_per_gpu']
                for n,(g,cpu,ram) in node.items():
                    typ=n.rsplit(':',1)[0];assert g<=ng[typ] and cpu<=cap[typ]['cpu_cores']+1e-7 and ram<=cap[typ]['ram_gb']+1e-7;checks['node_capacity']+=1
                if cell['method'].startswith('sia_'):
                    intent=d['original_sia_allocations'];sel=res['selected']
                    for name,a in intent.items():
                        if a[0] is None or not a[1]:assert name not in sel;continue
                        assert name in sel and sel[name]['config']['gpu_type']==a[0] and sel[name]['config']['gpus']==len(a[1])
                    checks['sia_intent_rounds']+=1
            for e in r['execution']:
                j=jobs[e['job_id']];assert close(j['progress'],e['progress_before']);j['progress']=e['progress_after']
                assert e['progress_delta']>=-1e-9 and audit_execution(e)['valid'];checks['execution']+=1
                if e['completion_time'] is not None:
                    if j['completion'] is not None:assert j['completion']==e['completion_time']
                    j['completion']=e['completion_time']
                active=max(0.,min(last_end,j['completion'] if j['completion'] is not None else last_end)-max(r['interval_start'],j['arrival']))
                assert close(active,e['waiting_seconds']+e['training_seconds']+e['checkpoint_delay_consumed'])
                for key,src in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:j[key]+=e[src]
                a=e['resource_allocation']
                if not a:continue
                expected=independent_rho(o,e['job_id'],a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'])
                assert close(expected,e['rho'])
                for segment in e['training_segments']:
                    assert close(segment['effective_rate'],segment['raw_rate_before_rho']*expected);checks['direct_rate_segments']+=1
                if e['training_seconds']==0:assert close(e['progress_delta'],0)
                dt=e['allocated_seconds']/3600
                # Hours independently reconstructed from GPU geometry and quotas.
                gpu=sum(a['placement'].values());assert gpu==a['gpus']
                j['gpu_hours']+=gpu*dt;j['cpu_core_hours']+=gpu*a['cpu_per_gpu']*dt;j['ram_gb_hours']+=gpu*a['ram_gb_per_gpu']*dt
                if fresh and e['checkpoint_remaining_before_decision']>0 and e['completion_time'] is None:
                    new=d['result']['selected'].get(e['job_id'])
                    assert new and new['candidate_kind']=='locked'
                    for key in ('cpu_per_gpu','ram_gb_per_gpu','placement'):assert new[key]==a[key]
                    assert new['config']['config_id']==a['config_id'];checks['checkpoint_lock']+=1
    summary=json.loads((folder/'summary.json').read_text());tau=p['traces'][str(cell['trace'])]['rmst_tau'];finished=[]
    for name,j in jobs.items():
        j.update(endstate[name]);done=j['completion'] is not None;j['completed']=done
        j['jct']=j['completion']-j['arrival'] if done else None;j['censor_time']=None if done else last_end-j['arrival']
        observed=j['jct'] if done else j['censor_time'];assert close(j['wait']+j['checkpoint']+j['training'],observed)
        j['restricted_jct']=min(observed,tau)
        if done:finished.append(j['jct'])
        for key in ('jct','completion','wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds'):
            old=summary['jobs'][name][key];assert old==j[key] if old is None or j[key] is None else close(old,j[key]),(name,key,old,j[key])
    complete=len(finished)==160
    metrics=dict(completed=len(finished),total=160,complete=complete,mean_jct=float(np.mean(finished)) if complete else None,
        median_jct=float(np.median(finished)) if complete else None,p95_jct=float(np.percentile(finished,95)) if complete else None,
        p99_jct=float(np.percentile(finished,99)) if complete else None,makespan=max(j['completion'] for j in jobs.values()) if complete else None,
        restricted_mean_jct=float(np.mean([j['restricted_jct'] for j in jobs.values()])),rmst_tau=tau,
        totals={key:sum(j[key] for j in jobs.values()) for key in ('wait','training','checkpoint','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds')},
        fallback=dict(fallback),solver_status=dict(status),solver_seconds=sum(solverwall),scheduling_wall_seconds=schedulewall,
        clipping_counts=dict(clip),selected_count=selected_count,clipping_rates={k:v/selected_count for k,v in clip.items()})
    for key in ('completed','mean_jct','median_jct','p95_jct','p99_jct','makespan','restricted_mean_jct'):
        assert summary[key]==metrics[key] if summary[key] is None else close(summary[key],metrics[key]),key
    return dict(cell=cell,metrics=metrics,jobs=jobs,validation=dict(valid=True,violations=0,checks=dict(checks)),raw_hash=sha(folder/'rounds.jsonl.gz'))

def main(label):
    dest=OUT/label;p=json.loads((dest/'preregistration.json').read_text());review=dest/'raw_review';review.mkdir(exist_ok=False)
    allresults={};coverage=[]
    for cell in p['runs']:
        folder=dest/cell['id']
        if (folder/'summary.json').exists():
            result=cell_audit(folder,cell,p);save(review/(cell['id']+'.json'),result);allresults[cell['id']]=result
            coverage.append(dict(**cell,status='COMPLETE' if result['metrics']['complete'] else 'CENSORED',completed=result['metrics']['completed']))
        else:coverage.append(dict(**cell,status='FAILED' if (folder/'failure.json').exists() else 'NOT_RUN',completed=None))
    contrasts=[]
    for capacity,response,traces in [('main',r,range(1,11)) for r in ('neutral','medium')]+[(c,'medium',(1,5,10)) for c in ('half','double')]:
        for other in ('sia_p','sia_tune','joint_p'):
            pairs=[]
            for trace in traces:
                a=allresults.get(f'{capacity}_w{trace}_{response}_{other}');b=allresults.get(f'{capacity}_w{trace}_{response}_joint_o')
                if a is None or b is None or not a['metrics']['complete'] or not b['metrics']['complete']:continue
                diffs={j:a['jobs'][j]['jct']-b['jobs'][j]['jct'] for j in a['jobs']}
                top=sorted(diffs,key=diffs.get,reverse=True);net=sum(diffs.values());mean_a=a['metrics']['mean_jct'];mean_b=b['metrics']['mean_jct']
                pairs.append(dict(trace=trace,baseline=mean_a,joint_o=mean_b,absolute_reduction=mean_a-mean_b,improvement_percent=100*(mean_a-mean_b)/mean_a,
                    jobs_improved=sum(v>0 for v in diffs.values()),jobs_worse=sum(v<0 for v in diffs.values()),jobs_tied=sum(v==0 for v in diffs.values()),
                    net_jct_seconds=net,top5=[dict(job=j,application=a['jobs'][j]['application'],difference=diffs[j]) for j in top[:5]],
                    top5_fraction_of_net=sum(diffs[j] for j in top[:5])/net if net>0 else None,
                    diagnostic_remove_top1_mean_reduction=(net-diffs[top[0]])/159,
                    per_job_difference=diffs,tail_changes={key:b['metrics'][key]-a['metrics'][key] for key in ('median_jct','p95_jct','p99_jct','makespan')}))
            rates=[x['improvement_percent'] for x in pairs];base=[x['baseline'] for x in pairs];joint=[x['joint_o'] for x in pairs]
            contrasts.append(dict(capacity=capacity,response=response,baseline=other,complete_pairs=len(pairs),planned_pairs=len(traces),pairs=pairs,
                mean_baseline=float(np.mean(base)) if pairs else None,mean_joint_o=float(np.mean(joint)) if pairs else None,
                equal_trace_mean_improvement_percent=100*(sum(base)-sum(joint))/sum(base) if pairs else None,
                median_improvement_percent=float(np.median(rates)) if pairs else None,range=[min(rates),max(rates)] if pairs else None,
                wins=sum(v>1e-9 for v in rates),ties=sum(abs(v)<=1e-9 for v in rates),losses=sum(v< -1e-9 for v in rates)))
    save(review/'coverage.json',coverage);save(review/'comparisons.json',contrasts)
    save(review/'validation.json',dict(audited=len(allresults),complete=sum(v['metrics']['complete'] for v in allresults.values()),
        planned=104,violations=0,source='Independent raw completion events, segments and placement/quota integrals; not summary-only.',
        wall_seconds=sum(e.get('wall_seconds',0) for e in [json.loads(l) for l in (dest/'execution_manifest.jsonl').read_text().splitlines()])))
    print('AUDITED',len(allresults))
if __name__=='__main__':main(sys.argv[1])
