"""Read-only diagnostics of CLOSED existing logs; never builds a Cluster."""
import collections, csv, gzip, hashlib, json, math, sys, time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
TREE = ROOT/'cc-joint-core80-full-lex-repair-v8'
sys.path[:0] = [str(TREE), str(TREE/'sia-simulator')]
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024), b''): h.update(x)
    return h.hexdigest().upper()
def save(name,x):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def mean(xs):return sum(xs)/len(xs) if xs else None
def cnt():return collections.Counter()

def main():
    start=time.time()
    with (TREE/'inputs/core80_workload.csv').open(encoding='utf8') as f:work={r['name']:r for r in csv.DictReader(f)}
    oracle=R0Oracle(TREE/'inputs/registry.json',TREE/'inputs/models',{j:r['application'] for j,r in work.items()},'medium')
    metrics=json.loads((RUN/'audit/raw_recomputed_metrics.json').read_text(encoding='utf8'))
    with (RUN/'audit/per_job.csv').open(encoding='utf8') as f:
        table={}
        for r in csv.DictReader(f):table.setdefault(r['cell'].removeprefix('dual_medium_'),{})[r['job']]=r
    pairs=[]
    for j in work:
        a,b=table['FULL'][j],table['FO'][j]
        row=dict(job=j,application=work[j]['application'],original_name=work[j]['original_name'],source_trace=work[j]['source_trace'])
        for key in ('jct','wait','checkpoint','training','completion','first_wait','restarts','gpu_hours','cpu_core_hours','ram_gb_hours'):
            row[key+'_FULL']=float(a[key]);row[key+'_FO']=float(b[key]);row[key+'_saved']=float(b[key])-float(a[key])
        assert math.isclose(row['jct_saved'],sum(row[k+'_saved'] for k in ('wait','checkpoint','training')),abs_tol=1e-6)
        pairs.append(row)
    save('per_job_comparison.json',pairs)
    byapp={a:dict(count=sum(r['application']==a for r in pairs),
                 **{k+'_saved':sum(r[k+'_saved'] for r in pairs if r['application']==a) for k in ('jct','wait','checkpoint','training')})
           for a in ('cifar10','bert','deepspeech2')}
    save('application_decomposition.json',byapp)
    states=[];trajectories={};methods={};timelines={}
    for method in ('FO','FULL'):
        print('READ',method,flush=True)
        raw=RUN/'cells'/('dual_medium_'+method)/'rounds.jsonl.gz'
        stat=dict(decisions=0,drain_rounds=0,optional_selected=0,required_selected=0,
                  no_optional_rounds=0,serial_rounds=0,serial_gpu_violations=0,
                  serial_changed_quota_jobs=0,serial_positive_utility_rounds=0,
                  serial_no_positive_utility_rounds=0,serial_delta_utility_sum=0.,
                  rejection_column_round_counts=cnt(),rejection_job_round_counts=cnt(),
                  application_execution={},initial_configs={},last_configs={})
        appstat={a:dict(training=0.,rho_seconds=0.,accelerated_seconds=0.,slowed_seconds=0.,
                       cpu_above_prop_seconds=0.,ram_above_prop_seconds=0.,
                       quota_below_prop_seconds=0.,equivalent_progress_gain_seconds=0.,
                       config_training=cnt()) for a in byapp}
        jobs={j:dict(config_training=cnt(),rho_training=0.,training=0.,allocations=[],geometry_events=[]) for j in work}
        seen=set();last=None;traj={};serial_examples=[]
        with gzip.open(raw,'rt',encoding='utf8') as f:
            for line in f:
                r=json.loads(line);d=r['decision_for_next_interval'];res=d['result'];a=res['audit'];t=r['interval_end'];sel=res['selected']
                for e in r['execution']:
                    j=e['job_id'];al=e['resource_allocation'];seconds=e['training_seconds']
                    if not al:continue
                    app=work[j]['application'];x=appstat[app];y=jobs[j];cid=al['config_id']
                    x['training']+=seconds;x['rho_seconds']+=seconds*e['rho'];y['training']+=seconds;y['rho_training']+=seconds*e['rho']
                    x['config_training'][cid]+=seconds;y['config_training'][cid]+=seconds
                    x['accelerated_seconds']+=seconds*(e['rho']>1+1e-8);x['slowed_seconds']+=seconds*(e['rho']<1-1e-8)
                    x['cpu_above_prop_seconds']+=seconds*(al['cpu_per_gpu']>3+1e-8)
                    x['ram_above_prop_seconds']+=seconds*(al['ram_gb_per_gpu']>62.5+1e-8)
                    x['quota_below_prop_seconds']+=seconds*(al['cpu_per_gpu']<3-1e-8 or al['ram_gb_per_gpu']<62.5-1e-8)
                    x['equivalent_progress_gain_seconds']+=seconds*(e['rho']-1)
                    key=(cid,al['cpu_per_gpu'],al['ram_gb_per_gpu'])
                    if not y['allocations'] or y['allocations'][-1]['key']!=list(key):
                        y['allocations'].append(dict(time=e['interval_start'],key=list(key),rho=e['rho'],progress=e['progress_before']))
                if d['round']==last:continue
                last=d['round'];stat['decisions']+=1;stat['drain_rounds']+=bool(a['drain'])
                stat['optional_selected']+=len(a['optional_selected']);stat['required_selected']+=len(a['required_jobs']);stat['no_optional_rounds']+=not bool(a['optional_selected'])
                for j,v in sel.items():
                    stat['initial_configs'].setdefault(j,v['config']['config_id']);stat['last_configs'][j]=v['config']['config_id']
                for ev in d['liveness_events']:
                    if ev.get('job') in jobs and ev['event'] in ('ADAPT_GEOMETRY','TAIL_RECOVERY'):
                        jobs[ev['job']]['geometry_events'].append(ev)
                for j,v in a['shared_eligibility'].items():
                    rc=v['rejection_counts'];stat['rejection_column_round_counts'].update(rc)
                    stat['rejection_job_round_counts'].update(k for k,n in rc.items() if n)
                serial=a.get('serial_proposals',[])
                if serial:
                    p=serial[-1];old={v['job']:v for v in p['proposal']};new={v['job']:v for v in p['selected']}
                    stat['serial_rounds']+=1
                    stat['serial_gpu_violations']+=sum(old[j]['config']!=new[j]['config'] for j in old)
                    changed=[j for j in old if abs(old[j]['cpu']-new[j]['cpu'])>1e-8 or abs(old[j]['ram']-new[j]['ram'])>1e-8]
                    delta=sum(v['utility'] for v in new.values())-sum(v['utility'] for v in old.values())
                    stat['serial_changed_quota_jobs']+=len(changed);stat['serial_delta_utility_sum']+=delta
                    stat['serial_positive_utility_rounds']+=delta>1e-9;stat['serial_no_positive_utility_rounds']+=delta<=1e-9
                    assert delta>=-1e-7
                    if changed and len(serial_examples)<4:
                        serial_examples.append(dict(time=t,round=d['round'],proposal=list(old.values()),selected=list(new.values()),utility_increase=delta,
                            rho_after={j:sel[j]['rho'] for j in new},note='Same state, fixed GPU and task set; only aux/placement stage changes.'))
                totals={n:dict(gpu=0.,cpu=0.,ram=0.) for n in ('aws:0','dgx-ext:0','rtx:0')}
                for j,c in sel.items():
                    for n,g in c['placement'].items():
                        totals[n]['gpu']+=g;totals[n]['cpu']+=g*c['cpu_per_gpu'];totals[n]['ram']+=g*c['ram_gb_per_gpu']
                chosen_apps={work[j]['application'] for j in sel}
                tests={
                  'arrivals_done_queue_aux_idle':t>=1200 and len(res['unallocated_jobs'])>=10 and any(v['cpu']<.75*cap or v['ram']<.75*ram for n,v in totals.items() for cap,ram in [(12 if n=='aws:0' else 24,250 if n=='aws:0' else 500)]),
                  'deepspeech_mixed_co_resident':any('deepspeech2' in {work[j]['application'] for j,c in sel.items() if n in c['placement']} and len({work[j]['application'] for j,c in sel.items() if n in c['placement']})>1 for n in totals),
                  'gpu_full_aux_tight':any(v['gpu']==(4 if n=='aws:0' else 8) and (v['cpu']>=.95*(12 if n=='aws:0' else 24) or v['ram']>=.95*(250 if n=='aws:0' else 500)) for n,v in totals.items())}
                for label,ok in tests.items():
                    if not ok or label in seen:continue
                    seen.add(label)
                    # Frozen other allocations; only increase each selected job's
                    # own quota using its remaining node budget. No new placement,
                    # GPU, job admission or training simulation is invoked.
                    local=[]
                    for j,c in sel.items():
                        if c['candidate_kind']=='locked':continue
                        cfg=c['config'];n=next(iter(c['placement']));g=cfg['gpus'];v=totals[n]
                        cap=12 if n=='aws:0' else 24;ramcap=250 if n=='aws:0' else 500
                        cpu=c['cpu_per_gpu'];ram=c['ram_gb_per_gpu'];cr=cpu+(cap-v['cpu'])/g;mr=ram+(ramcap-v['ram'])/g
                        cpu_rho=independent_rho(oracle,j,cfg['config_id'],cr,ram)
                        ram_rho=independent_rho(oracle,j,cfg['config_id'],cpu,mr)
                        both=independent_rho(oracle,j,cfg['config_id'],cr,mr)
                        ref=oracle.reference(j,cfg['config_id']);inside=lambda a,b:1<=a/ref['phi_c']<=24 and 20.83<=b/ref['phi_m']<=250
                        local.append(dict(job=j,config=cfg['config_id'],current_cpu=cpu,current_ram=ram,current_rho=c['rho'],
                            cpu_only_upper=cr,cpu_only_rho=cpu_rho,ram_only_upper=mr,ram_only_rho=ram_rho,both_rho=both,
                            requested_numerator_inside=inside(cr,mr),denominator_inside=inside(3,62.5),
                            source='Physical residual upper bound, NOT a verified candidate-grid point or shared global allocation; one-job-at-a-time only.'))
                    states.append(dict(method=method,criterion=label,time=t,round=d['round'],selected=sel,
                        required=a['required_jobs'],optional_selected=a['optional_selected'],waiting=res['unallocated_jobs'],drain=a['drain'],
                        node_totals=totals,serial_proposals=serial,local_residual_bounds=local,
                        missing='Full per-candidate physical ETA/gradient/batch predictions were hash-only logged; no exact same-state FULL-vs-FO reranking reconstructed.'))
                traj[t]={j:(c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu']) for j,c in sel.items()}
        for x in appstat.values():x['time_weighted_rho']=x['rho_seconds']/x['training'] if x['training'] else None
        stat['application_execution']=appstat;stat['serial_examples']=serial_examples
        methods[method]=stat;trajectories[method]=jobs;timelines[method]=traj
    save('method_mechanisms.json',methods);save('job_trajectories.json',trajectories);save('structural_states.json',states)
    same_time=[]
    for t in sorted(set(timelines['FULL'])&set(timelines['FO'])):
        a,b=timelines['FULL'][t],timelines['FO'][t];common=set(a)&set(b)
        same_time.append(dict(time=t,selected_only_full=len(set(a)-set(b)),selected_only_fo=len(set(b)-set(a)),common=len(common),
                              different_gpu=sum(a[j][0]!=b[j][0] for j in common),same_gpu_different_aux=sum(a[j][0]==b[j][0] and a[j][1:]!=b[j][1:] for j in common)))
    save('descriptive_same_clock.json',same_time)
    save('summary.json',dict(application_decomposition=byapp,
        best=sorted(pairs,key=lambda r:r['jct_saved'],reverse=True)[:8],worst=sorted(pairs,key=lambda r:r['jct_saved'])[:8],
        initial_gpu_differences=sum(methods['FULL']['initial_configs'][j]!=methods['FO']['initial_configs'][j] for j in work),
        final_gpu_differences=sum(methods['FULL']['last_configs'][j]!=methods['FO']['last_configs'][j] for j in work),
        explanation='Per-job time differences are exact accounting, not isolated causal module effects. Same-clock trajectories have diverged state. Local rho uplift is not JCT.'))
    frozen=json.loads((RUN/'final_integrity_hashes.json').read_text(encoding='utf8'))
    used=['preregistration.json','audit/raw_recomputed_metrics.json','audit/per_job.csv','cells/dual_medium_FULL/rounds.jsonl.gz','cells/dual_medium_FO/rounds.jsonl.gz']
    hashes={n:sha(RUN/n) for n in used};assert all(v==frozen[n] for n,v in hashes.items())
    p=json.loads((RUN/'preregistration.json').read_text(encoding='utf8'));assert all(sha(TREE/n)==v for n,v in p['source_hashes'].items())
    save('provenance.json',dict(inputs=hashes,source_files_verified=len(p['source_hashes']),source_snapshot_sha256=p['source_snapshot_sha256'],script_sha256=sha(Path(__file__)),wall_seconds=time.time()-start,simulation_invoked=False,algorithms_modified=False))
    print('DONE',OUT,flush=True)
if __name__=='__main__':main()
