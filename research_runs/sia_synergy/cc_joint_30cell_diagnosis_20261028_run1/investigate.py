"""Static recorded-column comparisons and restriction census. No simulation."""
import collections
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

from scan import BASE, HERE, rows, save, sha


def usage(c):
    return {n:(g,g*c['query']['cpu_per_gpu'],g*c['query']['ram_gb_per_gpu'])
            for n,g in c['placement']['node_gpus']}


def feasible(columns,cpu,ram):
    total=collections.defaultdict(lambda:[0.,0.,0.])
    for c in columns:
        for n,v in usage(c).items():
            for k,x in enumerate(v):total[n][k]+=x
    return all(all(x<=cap+1e-7 for x,cap in zip(v,[2,cpu,ram])) for v in total.values())


def supported(c):
    return not(c['query']['source_coordinate_clipped'] or c['query']['baseline_coordinate_clipped'])


def info(c,scores):
    v=scores[c['column_id']];q=c['query']
    return dict(column_id=c['column_id'],job=c['job_id'],config=c['config']['config_id'],
        cpu=q['cpu_per_gpu'],ram=q['ram_gb_per_gpu'],rho=q['rho'],
        placement=dict(c['placement']['node_gpus']),raw_rate=v['raw_rate'],
        effective_rate=v['raw_rate']*q['rho'] if v['raw_rate'] is not None else None,
        eta=v['estimated_completion'],priority=v.get('completion_priority',0.),
        utility=v['utility'],delay=v['delay'],remaining_work=v['remaining_work'],
        coordinates_supported=supported(c))


def evaluate(columns,scores):
    values=[info(c,scores) for c in columns]
    known=[v for v in values if v['eta'] is not None]
    return dict(selected={v['job']:v for v in values},
        objective=sum(v['utility'] for v in values),
        sum_reciprocal_eta=sum(v['priority'] for v in known),
        sum_known_eta=sum(v['eta'] for v in known),all_etas_known=len(known)==len(values),
        all_coordinates_supported=all(v['coordinates_supported'] for v in values))


def static_quotas(snapshot):
    d=snapshot['raw_round']['decision_for_next_interval'];selected=d['result']['selected']
    scores={x['column_id']:x for x in d.get('continuity_scores',[])}
    if not scores:return {'status':'NO_COMPARABLE_ETA_SCORES_IN_SIA_P_LOG'}
    allcols=d['probe']['joint_columns'];byid={c['column_id']:c for c in allcols}
    actual=[byid[c['column_id']] for c in selected.values()]
    scenario=snapshot['cell'].split('_')[0]
    cpu,ram=json.loads((BASE/'preregistration.json').read_text())['capacities'][scenario]
    pools={}
    for name,current in selected.items():
        raw=[c for c in allcols if c['job_id']==name
             and c['config']['config_id']==current['config']['config_id']
             and dict(c['placement']['node_gpus'])==current['placement']]
        if scores[current['column_id']]['estimated_completion'] is None:
            raw=[c for c in raw if c['column_id']==current['column_id']]
        spec=next(v for v in d['probe']['specs'] if v['job_id']==name)
        if spec['checkpoint_locked']:
            raw=[c for c in raw if c['query']['cpu_per_gpu']==current['cpu_per_gpu'] and c['query']['ram_gb_per_gpu']==current['ram_gb_per_gpu']]
        pools[name]=raw
    grouping=collections.defaultdict(list)
    for name,current in selected.items():
        assert len(current['placement'])==1
        grouping[next(iter(current['placement']))].append(name)
    modes={}
    for mode in ['PROPORTIONAL','CPU_ONLY','MEM_ONLY','FULL','FULL_SUPPORTED','MIN_SUM_ETA_SUPPORTED']:
        chosen=[];enumerated=0;possible=True
        for node,names in grouping.items():
            ps=[]
            for name in names:
                candidates=[]
                for c in pools[name]:
                    q=c['query']
                    if mode in ['PROPORTIONAL','MEM_ONLY'] and q['cpu_per_gpu']!=cpu/2:continue
                    if mode in ['PROPORTIONAL','CPU_ONLY'] and q['ram_gb_per_gpu']!=ram/2:continue
                    if mode in ['FULL_SUPPORTED','MIN_SUM_ETA_SUPPORTED'] and not supported(c):continue
                    candidates.append(c)
                ps.append(candidates)
            best=None;bestkey=None
            for combo in itertools.product(*ps):
                enumerated+=1
                if not feasible(combo,cpu,ram):continue
                v=evaluate(combo,scores)
                key=((v['sum_known_eta'],-v['objective']) if mode=='MIN_SUM_ETA_SUPPORTED'
                     else (-v['objective'],v['sum_known_eta']))
                if bestkey is None or key<bestkey:bestkey=key;best=combo
            if best is None:possible=False;break
            chosen.extend(best)
        modes[mode]=dict(status='FEASIBLE' if possible else 'NO_MATCHING_RECORDED_CANDIDATE',enumerated=enumerated)
        if possible:
            assert feasible(chosen,cpu,ram)
            modes[mode].update(evaluate(chosen,scores))
    result=dict(status='FIXED_SELECTED_JOBS_GPU_AND_PLACEMENT',cell=snapshot['cell'],time=snapshot['time'],
                actual=evaluate(actual,scores),modes=modes,
                limitations='No new GPU proposal, no policy reconstruction, no future observations, no JCT counterfactual. Unknown ETA tasks keep their actual bootstrap point; ETA totals exclude unknown tasks.')
    for baseline in ['PROPORTIONAL','CPU_ONLY','MEM_ONLY']:
        if modes[baseline]['status']!='FEASIBLE':continue
        left=result['actual'];right=modes[baseline]
        result['actual_vs_'+baseline]=dict(
            utility_delta=left['objective']-right['objective'],
            known_eta_delta=left['sum_known_eta']-right['sum_known_eta'],
            per_job={j:dict(eta_delta=v['eta']-right['selected'][j]['eta'] if v['eta'] is not None and right['selected'][j]['eta'] is not None else None,
                            effective_rate_change=v['effective_rate']/right['selected'][j]['effective_rate']-1 if v['effective_rate'] is not None and right['selected'][j]['effective_rate'] is not None else None)
                     for j,v in left['selected'].items()})
    return result


def census():
    pre=json.loads((BASE/'preregistration.json').read_text());result={}
    for scenario in ['R00','R10','R01','R11']:
        name=scenario+'_medium_CC_FULL';cpu,ram=pre['capacities'][scenario]
        count=collections.Counter();first=None;first_two=None;tail=[];positive_examples=[]
        for line_no,r in rows(name):
            d=r['decision_for_next_interval'];a=d['result']['audit'];selected=d['result']['selected']
            allcols=d['probe']['joint_columns'];byid={c['column_id']:c for c in allcols}
            scores={x['column_id']:x for x in d['continuity_scores']}
            actual=[byid[c['column_id']] for c in selected.values()]
            required=set(a.get('required_jobs',[]))
            count['decisions']+=1;count['selected_job_rounds']+=len(selected)
            count['required_selected_job_rounds']+=len(required & set(selected))
            count['optional_selected_job_rounds']+=len(set(selected)-required)
            count['all_selected_required_rounds']+=set(selected)<=required
            if a.get('ready'):count['queue_nonempty_rounds']+=1
            for j,fil in a.get('adaptive_candidate_filters',{}).items():
                old=selected[j];oldscore=scores[old['column_id']]
                if fil.get('tail_rule') not in [None,'NOT_ELIGIBLE']:
                    tail.append(dict(time=r['interval_end'],line=line_no,job=j,reason=fil.get('tail_rule'),selected=info(byid[old['column_id']],scores),tail_offer=fil.get('tail_offer')))
                for rejected in fil.get('rejected',[]):
                    if rejected['reason']!='SERVICE_DELAY_BUDGET':continue
                    count['budget_rejected_columns']+=1
                    c=byid.get(rejected['column_id'])
                    if not c:continue
                    others=[x for x in actual if x['job_id']!=j]
                    if not feasible(others+[c],cpu,ram):continue
                    count['budget_rejected_but_others_unchanged_capacity_feasible_columns']+=1
                    if not supported(c):continue
                    count['supported_others_unchanged_capacity_feasible_columns']+=1
                    newscore=scores[c['column_id']]
                    if newscore['estimated_completion'] is None or oldscore['estimated_completion'] is None:continue
                    delta=oldscore['estimated_completion']-newscore['estimated_completion']
                    if delta<=1e-9:
                        count['non_improving_others_unchanged_columns']+=1;continue
                    count['strict_eta_improving_others_unchanged_columns']+=1
                    current=info(byid[old['column_id']],scores);new=info(c,scores)
                    witness=dict(cell=name,time=r['interval_end'],line=line_no,job=j,current=current,target=new,
                        predicted_eta_saving=delta,other_jobs_unchanged={x['job_id']:info(x,scores) for x in others},
                        queue_empty=not a.get('ready'),active_jobs=len(d['probe']['specs']),
                        geometry_events=a.get('geometry_event_reasons'),
                        checkpoint_locked=next(s['checkpoint_locked'] for s in d['probe']['specs'] if s['job_id']==j),
                        delay_ledger=a.get('service_delay_ledgers',{}).get(j),
                        resource_capacity=[2,cpu,ram],
                        caution='Feasible instantaneous prediction only, not a simulated completion/JCT result. Budget or non-event restriction can still forbid it; no checkpoint or other service is changed.')
                    if first is None:first=witness
                    if not a.get('ready') and len(d['probe']['specs'])==2:
                        count['empty_queue_two_job_positive_columns']+=1
                        if first_two is None:first_two=witness
                    if len(positive_examples)<3:positive_examples.append(witness)
            # Lock/failure must stay enforced; this census never selects/commits.
        result[scenario]=dict(counts=dict(count),required_share=count['required_selected_job_rounds']/count['selected_job_rounds'],
            first_capacity_feasible_supported_eta_improvement=first,
            first_empty_queue_two_job_improvement=first_two,
            first_three_positive_columns=positive_examples,tail_decisions=tail)
        print('CENSUS',scenario,dict(count),flush=True)
    return result


def phases(cell,jobs):
    result={j:[] for j in jobs}
    with gzip.open(HERE/'slim_timelines'/f'{cell}.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            for e in r['execution']:
                j=e['job'];q=e['resource']
                if j not in result or not q or e['training']+e['checkpoint']<=0:continue
                key=(q['config'],q['cpu'],q['ram'],tuple(sorted(q['placement'].items())))
                end=r['time'] if e['completion'] is None else min(r['time'],e['completion'])
                phase=result[j][-1] if result[j] and tuple(result[j][-1]['key'])==key else None
                # In-memory keys remain tuples; JSON export converts them later.
                if phase is None:
                    phase=dict(key=key,start=r['start'],end=end,config=q['config'],cpu=q['cpu'],ram=q['ram'],rho=q['rho'],
                               training=0.,checkpoint=0.,progress=0.,raw_integral=0.,effective_integral=0.)
                    result[j].append(phase)
                phase['end']=end;phase['training']+=e['training'];phase['checkpoint']+=e['checkpoint'];phase['progress']+=e['progress_after']-e['progress_before']
                phase['raw_integral']+=(e['mean_raw_rate'] or 0.)*e['training']
                phase['effective_integral']+=(e['mean_effective_rate'] or 0.)*e['training']
    for values in result.values():
        for v in values:v.pop('key')
    return result


def main():
    save('restriction_census_rules.json',dict(
        timing='Written before census calculation; supplements aggregate budget-count interpretation, not a JCT outcome search.',
        scan='Every Medium CC_FULL decision in all four capacities, every SERVICE_DELAY_BUDGET rejected recorded column.',
        tests=['replace only that job column; keep all other selected jobs/GPU/quota/placement unchanged',
               'node GPU/CPU/RAM capacity remains legal','target numerator and baseline coordinates both supported',
               'current and target ETA known; compare signed saving, count non-improving too'],
        chronological_witness='First positive supported feasible column, separately first empty-queue two-active-job state; do not choose maximal gain.',
        not_guaranteed='Does not waive checkpoint, budget or event restrictions; no simulation or counterfactual JCT.'))
    manifest=json.loads((HERE/'selected_states.json').read_text());comparisons={}
    for key in manifest:
        snapshot=json.loads((HERE/'states'/f'{key}.json').read_text());comparisons[key]=static_quotas(snapshot)
    save('same_state_quota_comparisons.json',comparisons)
    save('restriction_census.json',census())
    selected_phases={}
    for cell in ['R00_medium_SIA_P','R00_medium_CC_FULL','R11_medium_SIA_P','R11_medium_CC_CPU','R11_medium_CC_FULL']:
        selected_phases[cell]=phases(cell,['deepspeech2-8','deepspeech2-9','cifar10-13'])
    save('selected_job_phases.json',selected_phases)
    print('STATIC_DIAGNOSIS_COMPLETE_NO_SIMULATION',flush=True)


if __name__=='__main__':main()
