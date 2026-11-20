"""Read-only 30-cell diagnostic scan. Never imports or executes a simulator."""
import collections
import gzip
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'cc_joint_resource_factorial_20261027_run1'
ROOT = HERE.parents[2]


def save(name, obj):
    with (HERE/name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest().upper()


def rows(name):
    with gzip.open(BASE/'cells'/name/'rounds.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            yield line_no, json.loads(line)


def geometry(c):
    if not c:
        return None
    return (c.get('config_id') or c['config']['config_id'], c['cpu_per_gpu'],
            c['ram_gb_per_gpu'], tuple(sorted(c['placement'].items())))


def numeric_summary(samples):
    if not samples:
        return {'count': 0}
    w = sum(x[1] for x in samples)
    return dict(count=len(samples), productive_seconds=w,
                median=statistics.median(x[0] for x in samples),
                weighted_mean=sum(x[0]*x[1] for x in samples)/w,
                minimum=min(x[0] for x in samples), maximum=max(x[0] for x in samples))


def summary_pairs(summaries):
    out = {}
    for scenario in ['R00','R10','R01','R11']:
        for left, right in [('CC_P','SIA_P'),('CC_FULL','SIA_P'),('CC_FULL','CC_P'),
                            ('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_CPU')]:
            a, b = [summaries[f'{scenario}_medium_{x}'] for x in (left,right)]
            assert a['completed'] == b['completed'] == 20
            changes, apps = {}, collections.defaultdict(collections.Counter)
            for name in b['jobs']:
                x, y = a['jobs'][name], b['jobs'][name]
                delta = {k:x[k]-y[k] for k in ['jct','wait','training','checkpoint','restarts','gpu_hours','cpu_core_hours','ram_gb_hours']}
                first_a, first_b = x['first_allocation']-x['arrival'], y['first_allocation']-y['arrival']
                delta.update(initial_wait=first_a-first_b,
                             later_wait=(x['wait']-first_a)-(y['wait']-first_b),
                             first_allocation=x['first_allocation']-y['first_allocation'],
                             first_training=x['first_training']-y['first_training'])
                assert math.isclose(delta['jct'],sum(delta[k] for k in ['wait','training','checkpoint']),abs_tol=1e-7)
                changes[name] = delta
                apps[name.rsplit('-',1)[0]].update(delta)
            totals = {k:sum(v[k] for v in changes.values()) for k in next(iter(changes.values()))}
            ranks = {}
            for label, s in [('left',a),('right',b)]:
                ordered = sorted(s['jobs'].items(), key=lambda v:(v[1]['jct'],v[0]))
                low, high = ordered[-2:]
                p99 = .19*low[1]['jct']+.81*high[1]['jct']
                p95 = .95*low[1]['jct']+.05*high[1]['jct']
                assert math.isclose(s['p99_jct'],p99,abs_tol=1e-7)
                assert math.isclose(s['p95_jct'],p95,abs_tol=1e-7)
                ranks[label] = dict(rank19=dict(job=low[0],jct=low[1]['jct']),
                                    rank20=dict(job=high[0],jct=high[1]['jct']),p95=p95,p99=p99)
            largest_gain = min(changes, key=lambda j:changes[j]['jct'])
            largest_loss = max(changes, key=lambda j:changes[j]['jct'])
            def omit(name):
                x = sum(v['jct'] for j,v in a['jobs'].items() if j != name)/19
                y = sum(v['jct'] for j,v in b['jobs'].items() if j != name)/19
                return dict(excluded=name,left_mean=x,right_mean=y,improvement=1-x/y)
            save_key = scenario+'/'+left+'_vs_'+right
            out[save_key] = dict(left_mean=a['mean_jct'],right_mean=b['mean_jct'],
                mean_improvement=1-a['mean_jct']/b['mean_jct'],totals=totals,
                improved=sum(x['jct']<0 for x in changes.values()),
                worsened=sum(x['jct']>0 for x in changes.values()),
                per_job=changes,applications={k:dict(v) for k,v in apps.items()},
                quantile_ranks=ranks,p99_delta=a['p99_jct']-b['p99_jct'],
                p99_rank_contributions=dict(rank19=.19*(ranks['left']['rank19']['jct']-ranks['right']['rank19']['jct']),
                                           rank20=.81*(ranks['left']['rank20']['jct']-ranks['right']['rank20']['jct'])),
                diagnostics=dict(omit_largest_gain=omit(largest_gain),omit_largest_loss=omit(largest_loss)))
    return out


def slim_column(c, score=None):
    return dict(column_id=c['column_id'],config=c['config']['config_id'],
        gpu_type=c['config']['gpu_type'],gpus=c['config']['gpus'],placement=c['placement'],
        cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu'],rho=c['rho'],
        score=score)


def scan(name, summary, capacities):
    previous = None
    stats = collections.Counter()
    hist = collections.Counter()
    events = collections.Counter()
    jobs = collections.defaultdict(collections.Counter)
    residence = collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    predictions = collections.defaultdict(list)
    predictions_same_epoch = collections.defaultdict(list)
    residuals = collections.Counter()
    blocked = collections.defaultdict(collections.Counter)
    snapshots = {}
    cpu, ram = capacities
    trace = HERE/'slim_timelines'/f'{name}.jsonl.gz'
    with gzip.open(trace,'xt',encoding='utf-8') as output:
        for line_no,r in rows(name):
            d = r['decision_for_next_interval'];a=d['result']['audit'];sel=d['result']['selected']
            scores={x['column_id']:x for x in (d.get('continuity_scores') or d.get('completion_scores') or [])}
            prev_scores = {} if previous is None else {x['column_id']:x for x in (previous.get('continuity_scores') or previous.get('completion_scores') or [])}
            execution_slim=[]
            for e in r['execution']:
                j=jobs[e['job_id']]
                for k,field in [('wait','waiting_seconds'),('training','training_seconds'),('checkpoint','checkpoint_delay_consumed')]:j[k]+=e[field]
                j['progress']+=e['progress_delta']
                residuals[e['job_id']]+=e['epoch_rounding_progress_residual']
                resource=e['resource_allocation']
                segments=[x for x in e['training_segments'] if x['seconds']>0]
                if resource and e['allocated_seconds']>0:
                    assert previous is not None
                    old=previous['result']['selected'][e['job_id']]
                    assert geometry(old)==geometry(resource)
                    cfg=residence[e['job_id']][resource['config_id']]
                    cfg['allocated_seconds']+=e['allocated_seconds'];cfg['training_seconds']+=e['training_seconds']
                    cfg['checkpoint_seconds']+=e['checkpoint_delay_consumed']
                    cfg['gpu_training_seconds']+=resource['gpus']*e['training_seconds']
                    cfg['cpu_training_seconds']+=resource['cpu_per_gpu']*resource['gpus']*e['training_seconds']
                    cfg['ram_training_seconds']+=resource['ram_gb_per_gpu']*resource['gpus']*e['training_seconds']
                    cfg['integrated_raw_progress']+=sum(x['raw_rate_before_rho']*x['seconds'] for x in segments)
                    cfg['integrated_effective_progress']+=sum(x['effective_rate']*x['seconds'] for x in segments)
                    cfg['rho_seconds']+=e['rho']*e['training_seconds']
                    score=prev_scores.get(old['column_id'])
                    if score and score['raw_rate'] is not None and segments:
                        first=segments[0]; ratio=score['raw_rate']/first['raw_rate_before_rho']
                        assert math.isclose(first['rho'],score['rho'],rel_tol=1e-12)
                        predictions[e['job_id']].append((ratio,e['training_seconds']))
                        if first['epoch_before']==e['epoch_before']:
                            predictions_same_epoch[e['job_id']].append((ratio,e['training_seconds']))
                        stats['prediction_comparisons']+=1
                    b=previous['result']['audit'].get('adaptive_candidate_filters',{}).get(e['job_id'],{})
                    reasons=set(x['reason'] for x in b.get('rejected',[]))
                    if b.get('non_event_geometry_rejected'):reasons.add('NON_EVENT_GEOMETRY')
                    if b.get('tail_rule')=='USED_ONCE_GPU_FROZEN':reasons.add('TAIL_GPU_FROZEN')
                    for reason in reasons:blocked[e['job_id']][reason+'_training_seconds']+=e['training_seconds']
                execution_slim.append(dict(job=e['job_id'],arrival=e['arrival_time'],completion=e['completion_time'],
                    wait=e['waiting_seconds'],checkpoint=e['checkpoint_delay_consumed'],training=e['training_seconds'],
                    progress_before=e['progress_before'],progress_after=e['progress_after'],epoch_before=e['epoch_before'],
                    epoch_after=e['epoch_after'],resource=None if not resource else dict(config=resource['config_id'],gpus=resource['gpus'],cpu=resource['cpu_per_gpu'],ram=resource['ram_gb_per_gpu'],rho=e['rho'],placement=resource['placement']),
                    first_raw_rate=segments[0]['raw_rate_before_rho'] if segments else None,
                    mean_raw_rate=sum(x['raw_rate_before_rho']*x['seconds'] for x in segments)/e['training_seconds'] if e['training_seconds'] else None,
                    mean_effective_rate=sum(x['effective_rate']*x['seconds'] for x in segments)/e['training_seconds'] if e['training_seconds'] else None))
            # Exact subinterval integration, including completion inside a 60s period.
            bounds={r['interval_start'],r['interval_end']}
            for e in r['execution']:
                for t in [e['arrival_time'],e['completion_time']]:
                    if t is not None and r['interval_start']<t<r['interval_end']:bounds.add(t)
                for seg in e['training_segments']:
                    if seg['seconds']>0:bounds.update([seg['start_time'],seg['start_time']+seg['seconds']])
            bounds=sorted(x for x in bounds if r['interval_start']<=x<=r['interval_end'])
            for t0,t1 in zip(bounds,bounds[1:]):
                t=(t0+t1)/2;dt=t1-t0;running=[];active=[];productive=[]
                usage=collections.defaultdict(lambda:[0.,0.,0.])
                for e in r['execution']:
                    if e['arrival_time']<=t and (e['completion_time'] is None or t<e['completion_time']):
                        active.append(e)
                        if e['resource_allocation']:
                            running.append(e)
                            for u in e['resource_allocation']['node_usage']:
                                for k,key in enumerate(['gpus','cpu','ram_gb']):usage[u['node_id']][k]+=u[key]
                    if any(z['start_time']<=t<z['start_time']+z['seconds'] for z in e['training_segments']):productive.append(e)
                hist['allocated_jobs_'+str(len(running))]+=dt
                hist['productive_jobs_'+str(len(productive))]+=dt
                hist['unfinished_job_seconds']+=len(active)*dt
                hist['queued_job_seconds']+=(len(active)-len(running))*dt
                for vals in usage.values():
                    assert vals[0]<=2+1e-7 and vals[1]<=cpu+1e-7 and vals[2]<=ram+1e-7
                    if vals[0]<2 and vals[1]>=cpu-1e-7:hist['cpu_saturated_gpu_idle_node_seconds']+=dt
                    if vals[0]<2 and vals[2]>=ram-1e-7:hist['ram_saturated_gpu_idle_node_seconds']+=dt
            stats['decision_rounds']+=1
            stats['drain_rounds']+=bool(a.get('drain'))
            stats['optional_admission_rounds']+=bool(a.get('optional_admission_enabled'))
            for item in d.get('liveness_events',[]):events[item['event']]+=1
            for j,fil in a.get('adaptive_candidate_filters',{}).items():
                for reason,n in fil.get('rejection_counts',{}).items():blocked[j][reason+'_candidate_rounds']+=n
                if fil.get('non_event_geometry_rejected'):blocked[j]['NON_EVENT_GEOMETRY_rounds']+=1
                if fil.get('tail_rule'):blocked[j]['TAIL_REASON_'+fil['tail_rule']]+=1
            selected_slim={j:slim_column(c,scores.get(c['column_id'])) for j,c in sel.items()}
            record=dict(line=line_no,start=r['interval_start'],time=r['interval_end'],selected=selected_slim,
                status=d['result']['status'],objective=d['result']['scheduling_objective'],
                required=a.get('required_jobs'),ready=a.get('ready'),drain=a.get('drain'),
                geometry_events=a.get('geometry_event_reasons'),service_budget=a.get('service_delay_ledgers'),
                tail_recoveries=a.get('tail_recoveries'),tail_grants=a.get('tail_grants'),
                filters={j:{k:v for k,v in fil.items() if k in ['current_column_id','delay_charged','tail_rule','tail_offer','rejection_counts','non_event_geometry_rejected','eligible_count']} for j,fil in a.get('adaptive_candidate_filters',{}).items()},
                execution=execution_slim,events=d.get('liveness_events'))
            output.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
            # Structural witnesses selected independently of which policy won.
            def capture(tag):
                snapshots[tag]=dict(cell=name,line=line_no,time=r['interval_end'],raw_round=r)
            if name=='R00_medium_SIA_P' and 'S1' not in snapshots:
                ds=[c for j,c in sel.items() if j.startswith('deepspeech2')]
                if len(ds)==2 and all(c['config']['gpus']==2 for c in ds) and any(j.startswith('cifar10') for j in d['result']['unallocated_jobs']):capture('S1')
            if name=='R00_medium_CC_FULL' and 'S2' not in snapshots:
                for j,c in sel.items():
                    if not j.startswith('cifar10') or c['config']['gpus']!=1:continue
                    for k,v in sel.items():
                        if k.startswith('deepspeech2') and v['config']['gpus']==1 and c['placement']==v['placement'] and c['cpu_per_gpu']!=v['cpu_per_gpu'] and all(scores.get(x['column_id'],{}).get('raw_rate') is not None for x in [c,v]):capture('S2');break
                    if 'S2' in snapshots:break
            if name=='R11_medium_CC_FULL':
                if 'S4' not in snapshots:
                    for j,c in sel.items():
                        if not j.startswith('cifar10') or c['config']['gpus']!=1:continue
                        for k,v in sel.items():
                            if k.startswith('deepspeech2') and v['config']['gpus']==1 and c['placement']==v['placement'] and any(x['ram_gb_per_gpu']!=ram/2 for x in [c,v]) and all(scores.get(x['column_id'],{}).get('raw_rate') is not None for x in [c,v]):capture('S4');break
                        if 'S4' in snapshots:break
                if 'S5' not in snapshots and len(d['probe']['specs'])>1:
                    if any(j.startswith('deepspeech2') and fil.get('rejection_counts',{}).get('SERVICE_DELAY_BUDGET',0)>0 for j,fil in a.get('adaptive_candidate_filters',{}).items()):capture('S5')
                if 'S6' not in snapshots and len(d['probe']['specs'])==1:
                    reasons={v.get('tail_rule') for v in a.get('adaptive_candidate_filters',{}).values()}
                    if reasons & {'ONE_BOUNDED_RECOVERY_OFFERED','NO_COST_PAYING_IMPROVEMENT','USED_ONCE_GPU_FROZEN'}:capture('S6')
            previous=d
    for j,v in jobs.items():
        for k in ['wait','training','checkpoint']:assert math.isclose(v[k],summary['jobs'][j][k],abs_tol=1e-6)
    assert math.isclose(hist['unfinished_job_seconds'],sum(j['jct'] for j in summary['jobs'].values()),abs_tol=1e-6)
    assert math.isclose(hist['queued_job_seconds'],summary['totals']['wait'],abs_tol=1e-6)
    residence_out={}
    for j,configs in residence.items():
        residence_out[j]={}
        for cfg,v in configs.items():
            x=dict(v);t=v['training_seconds']
            x.update(mean_raw_rate=v['integrated_raw_progress']/t if t else None,
                     mean_effective_rate=v['integrated_effective_progress']/t if t else None,
                     mean_rho=v['rho_seconds']/t if t else None)
            residence_out[j][cfg]=x
    return dict(statistics=dict(stats),concurrency_and_area=dict(hist),events=dict(events),
        per_job_residence=residence_out,blocked_candidates={j:dict(v) for j,v in blocked.items()},
        prediction_ratio={j:numeric_summary(v) for j,v in predictions.items()},
        prediction_ratio_epoch_aligned={j:numeric_summary(v) for j,v in predictions_same_epoch.items()},
        epoch_rounding_progress_residual=dict(residuals)),snapshots


def first_divergences():
    cpu=list(rows('R11_medium_CC_CPU'));full=list(rows('R11_medium_CC_FULL'))
    found={};prior_same=True;shared_snapshot=None
    for (na,a),(nb,b) in zip(cpu,full):
        assert a['interval_end']==b['interval_end']
        da,db=[x['decision_for_next_interval'] for x in [a,b]]
        sa,sb=da['result']['selected'],db['result']['selected']
        def project(sel,kind):
            if kind=='jobs':return sorted(sel)
            if kind=='gpu':return {j:(c['config']['config_id'],c['placement']) for j,c in sel.items()}
            if kind=='cpu':return {j:c['cpu_per_gpu'] for j,c in sel.items()}
            return {j:c['ram_gb_per_gpu'] for j,c in sel.items()}
        incoming_equal=[{k:e[k] for k in ['job_id','progress_after','epoch_after','completion_time','cluster','placement','checkpoint_remaining_before_decision']} for e in a['execution']]==[{k:e[k] for k in ['job_id','progress_after','epoch_after','completion_time','cluster','placement','checkpoint_remaining_before_decision']} for e in b['execution']]
        for kind in ['jobs','gpu','cpu','ram']:
            if kind not in found and project(sa,kind)!=project(sb,kind):
                found[kind]=dict(time=a['interval_end'],cpu_line=na,full_line=nb,
                    incoming_execution_state_equal=incoming_equal,
                    cpu_selected=sa,full_selected=sb)
                if prior_same and shared_snapshot is None:
                    shared_snapshot=dict(cell='R11_medium_CC_FULL',line=nb,time=b['interval_end'],raw_round=b,
                                         paired_cpu_cell='R11_medium_CC_CPU',paired_cpu_raw_round=a)
        if 'progress' not in found and not incoming_equal:
            found['progress']=dict(time=a['interval_end'],cpu_line=na,full_line=nb)
        prior_same=prior_same and incoming_equal and all(project(sa,k)==project(sb,k) for k in ['jobs','gpu','cpu','ram'])
    return found,shared_snapshot


def main():
    pre=json.loads((BASE/'preregistration.json').read_text(encoding='utf-8'))
    inventory={}
    for n,h in pre['source_hashes'].items():
        assert sha(BASE/'source'/n)==h,n
        assert sha(ROOT/'cc-joint-resource-study'/n)==h,n
    protected={str(p.relative_to(BASE)):sha(p) for p in [BASE/'preregistration.json',BASE/'execution_manifest.jsonl',BASE/'completed_analysis_20261028_run1/COMPLETED_REPORT.md',BASE/'completed_analysis_20261028_run1/final_validation.json']}
    summaries={}
    for cell in pre['cells']:
        name='_'.join(cell);folder=BASE/'cells'/name
        hashes=json.loads((folder/'output_hashes.json').read_text())
        for n,h in hashes.items():assert sha(folder/n)==h,(name,n)
        inventory[name]=dict(output_hashes_sha256=sha(folder/'output_hashes.json'),raw_sha256=sha(folder/'rounds.jsonl.gz'),summary_sha256=sha(folder/'summary.json'))
        summaries[name]=json.loads((folder/'summary.json').read_text())
        assert summaries[name]['completed']==20
    save('integrity_before.json',dict(source_files_verified=len(pre['source_hashes']),current_working_tree_matches_frozen=True,
         source_version_type='FULL_SOURCE_SNAPSHOT_NO_GIT_IN_COPY',protected=protected,cells=inventory,script_sha256=sha(Path(__file__))))
    save('summary_decomposition.json',summary_pairs(summaries))
    (HERE/'slim_timelines').mkdir()
    allstats={};snapshots={}
    for name,s in summaries.items():
        stats,states=scan(name,s,pre['capacities'][s['scenario']]);allstats[name]=stats;snapshots.update(states)
        print('SCANNED',name,flush=True)
    div,shared=first_divergences()
    save('first_divergence_R11_CPU_FULL.json',div)
    if shared:snapshots['S3']=shared
    (HERE/'states').mkdir()
    manifest={}
    for key,snapshot in sorted(snapshots.items()):
        with (HERE/'states'/f'{key}.json').open('x',encoding='utf-8') as f:json.dump(snapshot,f,ensure_ascii=False)
        manifest[key]={k:v for k,v in snapshot.items() if not k.endswith('raw_round')}
        manifest[key]['state_snapshot_sha256']=sha(HERE/'states'/f'{key}.json')
    save('selected_states.json',manifest)
    save('timeline_statistics.json',allstats)
    assert all(sha(BASE/n)==h for n,h in protected.items())
    save('scan_validation.json',dict(cells=30,all_20_complete=True,exact_JCT_time_identity=True,
         exact_queue_area_identity=True,source_and_history_unchanged=True,simulator_invocations=0,
         offline_scheduler_solves=0,selected_states=len(manifest)))
    print('SCAN_COMPLETE',len(allstats),{k:v['time'] for k,v in manifest.items()},flush=True)


if __name__=='__main__':main()
