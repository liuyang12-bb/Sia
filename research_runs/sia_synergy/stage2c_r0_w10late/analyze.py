"""Read-only raw-run revalidation and new paired four-cell statistics."""
import collections,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]; WORK=ROOT/'sia-joint-stage2c-r0/sia-simulator'
sys.path.insert(0,str(WORK))
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
    with p.open('x',encoding='utf-8') as f: json.dump(v,f,indent=2)
def geo(c): return (c['config']['config_id'],c['placement'],c['cpu_per_gpu'],c['ram_gb_per_gpu'])

def main(label):
    d=OUT/label; prereg=json.loads((d/'preregistration.json').read_text(encoding='utf-8'))
    suffix=sys.argv[2]+'_' if len(sys.argv)>2 else ''
    config=json.loads(Path(prereg['config']).read_text(encoding='utf-8'))
    import pandas as pd
    w=pd.read_csv(prereg['workload']); apps=dict(zip(w.name,w.application))
    audits={}; summaries={}; traces={}
    for response,cell in prereg['runs']:
        name=f'{response}_{cell}'; run=d/name
        s=json.loads((run/'summary.json').read_text()); summaries[name]=s
        oracle=R0Oracle(prereg['registry'],prereg['models'],apps,response)
        previous={}; previous_round=None; counters=collections.Counter(); totals=collections.defaultdict(lambda:collections.defaultdict(float)); progress={}; clipped=collections.Counter()
        tail_trace=[]; tail_configs=collections.defaultdict(lambda:collections.defaultdict(float))
        with gzip.open(run/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line); counters['rounds']+=1
                dec=r['decision_for_next_interval']; res=dec.get('result',{}); audit=res.get('audit',{}); selected=res.get('selected',{})
                fresh=dec.get('round')!=previous_round
                if fresh:
                    assert audit['valid']; assert audit['precommit_independent_audit']['valid']
                    events=dec.get('liveness_events',[]); renewed={e['job'] for e in events if e['event']=='ISSUE_SERVICE'}
                    for j in audit['checkpoint_locks']:
                        assert j in previous and j in selected and geo(previous[j])==geo(selected[j]); counters['locked_geometry_checks']+=1
                    for j in set(audit['service_commitments'])-renewed:
                        assert j in previous and geo(previous[j])==geo(selected[j]); counters['held_geometry_checks']+=1
                    reconstructed=collections.defaultdict(lambda:collections.defaultdict(float))
                    for j,col in selected.items():
                        ref=oracle.reference(j,col['config']['config_id']); cfg=col['config']
                        assert (cfg['cpu_demand_scale'],cfg['memory_demand_scale'])==(ref['phi_c'],ref['phi_m'])
                        assert math.isclose(col['rho'],independent_rho(oracle,j,cfg['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu']),rel_tol=1e-12)
                        counters['fixed_phi_checks']+=1
                        clipped['selected_numerator']+=int(col['source_coordinates']['clipped'])
                        clipped['selected_denominator']+=int(col['baseline_source_coordinates']['clipped'])
                        for side,key in [('numerator','source_coordinates'),('denominator','baseline_source_coordinates')]:
                            for axis in ('cpu','ram_gb'):
                                clipped[f'{side}_{axis}']+=int(col[key]['requested'][axis]!=col[key]['used'][axis])
                        assert sum(col['placement'].values())==cfg['gpus']
                        for node,g in col['placement'].items():
                            reconstructed[node]['gpus']+=g
                            reconstructed[node]['cpu']+=g*col['cpu_per_gpu']
                            reconstructed[node]['ram_gb']+=g*col['ram_gb_per_gpu']
                    for node,usage in reconstructed.items():
                        typ=node.rsplit(':',1)[0]; cap=config['cluster']['capacity_scenarios']['cpu_constrained'][typ]
                        assert usage['gpus']<=config['cluster']['ngpus_per_node'][typ]
                        assert usage['cpu']<=cap['cpu_cores']+1e-7 and usage['ram_gb']<=cap['ram_gb']+1e-7
                        assert all(math.isclose(v,audit['node_totals'][node][k],abs_tol=1e-7) for k,v in usage.items())
                        counters['independently_reconstructed_node_checks']+=1
                    previous=selected; previous_round=dec.get('round')
                for e in r['execution']:
                    j=e['job_id']; assert e['progress_after']>=e['progress_before']-1e-9
                    if j in progress: assert math.isclose(e['progress_before'],progress[j],abs_tol=1e-8)
                    progress[j]=e['progress_after']; assert e['independent_execution_audit']['valid']
                    for key in ('waiting_seconds','checkpoint_delay_consumed','training_seconds','allocated_seconds'): totals[j][key]+=e[key]
                    assert math.isclose(e['waiting_seconds']+e['checkpoint_delay_consumed']+e['training_seconds'],e['observed_active_seconds'],abs_tol=1e-7)
                    if e['training_seconds']==0: assert abs(e['progress_delta'])<1e-8
                    for seg in e['training_segments']:
                        assert seg['raw_rate_before_rho']>0
                        assert math.isclose(seg['effective_rate'],seg['raw_rate_before_rho']*seg['rho'],rel_tol=1e-12)
                        counters['direct_rate_checks']+=1
                    if response=='neutral': assert e['rho']==1
                    if j=='deepspeech2-159' and e['observed_active_seconds']>0:
                        a=e.get('resource_allocation'); cfg=a['config_id'] if a else 'WAIT'
                        tail_trace.append({k:e[k] for k in ('interval_start','interval_end','waiting_seconds','checkpoint_delay_consumed','training_seconds','progress_before','progress_after','raw_goodput','effective_goodput','rho','resource_allocation')})
                        for key in ('waiting_seconds','checkpoint_delay_consumed','training_seconds','allocated_seconds','progress_delta'): tail_configs[cfg][key]+=e[key]
                        tail_configs[cfg]['raw_integral']+=e['raw_goodput']*e['training_seconds']
                        tail_configs[cfg]['rho_training_integral']+=e['rho']*e['training_seconds']
        assert counters['rounds']==s['rounds']; assert s['violations']==0
        for j,row in s['jobs'].items():
            for left,right in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:
                assert math.isclose(row[left],totals[j][right],abs_tol=1e-6)
        audits[name]=dict(counters=counters,clipping_counts=clipped,violations=0,hashes={p.name:sha(p) for p in run.iterdir() if p.is_file()})
        traces[name]=dict(per_config=dict(tail_configs),rounds=tail_trace)
    effects={}
    for response in ('neutral','medium'):
        cells={c:summaries[f'{response}_{c}'] for c in ('u_fp','u_fo','u_jp','u_jo')}
        complete=all(s['completed']==20 for s in cells.values())
        metric='mean_jct' if complete else 'restricted_mean_jct'; y={c:s[metric] for c,s in cells.items()}
        effect=dict(metric=metric,complete_pair=complete,
            aux_given_F=y['u_fp']-y['u_fo'],gpu_given_P=y['u_fp']-y['u_jp'],
            gpu_given_O=y['u_fo']-y['u_jo'],aux_given_J=y['u_jp']-y['u_jo'],
            interaction_benefit=y['u_fo']+y['u_jp']-y['u_fp']-y['u_jo'],
            gpu_main_effect=(y['u_fp']+y['u_fo']-y['u_jp']-y['u_jo'])/2,
            aux_main_effect=(y['u_fp']+y['u_jp']-y['u_fo']-y['u_jo'])/2)
        comparisons={}
        for baseline in ('u_fp','u_fo','u_jp'):
            key='jct' if complete else 'restricted_jct'
            diffs={j:cells[baseline]['jobs'][j][key]-cells['u_jo']['jobs'][j][key] for j in cells['u_jo']['jobs']}
            total=sum(diffs.values()); tail='deepspeech2-159'; largest=max(diffs,key=lambda j:abs(diffs[j]))
            comparisons[baseline]=dict(improvement_seconds=y[baseline]-y['u_jo'],improvement_percent=100*(y[baseline]-y['u_jo'])/y[baseline],
                paired_improvement_seconds=diffs,better=sum(v>0 for v in diffs.values()),worse=sum(v<0 for v in diffs.values()),ties=sum(v==0 for v in diffs.values()),
                ds159_contribution_seconds=diffs[tail],ds159_share_of_net_benefit=diffs[tail]/total if total else None,
                diagnostic_mean_improvement_without_ds159=(total-diffs[tail])/19,
                largest_absolute_contributor=largest,diagnostic_mean_improvement_without_largest=(total-diffs[largest])/19)
        effect['JO_comparisons']=comparisons; effects[response]=effect
    neutral_pairs={f'{p}_{o}':all(summaries[f'neutral_{p}']['jobs'][j]['jct']==summaries[f'neutral_{o}']['jobs'][j]['jct']
                       for j in apps) for p,o in [('u_fp','u_fo'),('u_jp','u_jo')]}
    save(d/(suffix+'raw_validation.json'),audits)
    save(d/(suffix+'ds159_trace.json'),traces)
    save(d/(suffix+'aggregate_summary.json'),dict(commit=prereg['commit'],results=summaries,effects=effects,
        neutral_job_jct_pair_identity=neutral_pairs,
        interpretation='Positive effects mean lower mean JCT (or registered restricted mean if censored). No independent replication across jobs. Synthetic transfer only.'))
    print(json.dumps(dict(audited_cells=len(audits),effects=effects),indent=2))

if __name__=='__main__': main(sys.argv[1])
