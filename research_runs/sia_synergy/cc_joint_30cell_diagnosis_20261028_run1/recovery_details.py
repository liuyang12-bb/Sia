"""Audit all targets at the first chronological non-stealing recovery states."""
import json
from pathlib import Path
from investigate import BASE,HERE,rows,save,info,feasible,supported,usage


def main():
    census=json.loads((HERE/'restriction_census.json').read_text())
    pre=json.loads((BASE/'preregistration.json').read_text())
    output={}
    for scenario,value in census.items():
        witness=value['first_empty_queue_two_job_improvement']
        if witness is None:
            output[scenario]={'status':'NO_SUPPORTED_POSITIVE_NONSTEALING_BUDGET_WITNESS'};continue
        name=witness['cell'];line=witness['line'];_,r=next(x for x in rows(name) if x[0]==line)
        d=r['decision_for_next_interval'];a=d['result']['audit'];j=witness['job']
        byid={c['column_id']:c for c in d['probe']['joint_columns']};scores={x['column_id']:x for x in d['continuity_scores']}
        selected=[byid[c['column_id']] for c in d['result']['selected'].values()]
        old=byid[d['result']['selected'][j]['column_id']];others=[c for c in selected if c['job_id']!=j]
        cpu,ram=pre['capacities'][scenario];candidates=[]
        for item in a['adaptive_candidate_filters'][j]['rejected']:
            if item['reason']!='SERVICE_DELAY_BUDGET':continue
            c=byid[item['column_id']]
            if feasible(others+[c],cpu,ram) and supported(c) and scores[c['column_id']]['estimated_completion'] is not None:
                candidates.append(info(c,scores))
        best=min(candidates,key=lambda c:(c['eta'],c['column_id']))
        spec=next(x for x in d['probe']['specs'] if x['job_id']==j)
        assert not spec['checkpoint_locked'] and not a['ready'] and 'COMPLETION' in a['geometry_event_reasons']
        assert best['eta']<scores[old['column_id']]['estimated_completion']
        target=byid[best['column_id']]
        def totals(cols):
            total={}
            for c in cols:
                for n,v in usage(c).items():
                    if n not in total:total[n]=[0.,0.,0.]
                    total[n]=[x+y for x,y in zip(total[n],v)]
            return total
        output[scenario]=dict(status='CURRENT_STATE_SUPPORTED_CAPACITY_FEASIBLE_OTHER_JOBS_UNCHANGED',
            cell=name,line=line,time=r['interval_end'],job=j,
            current=info(old,scores),best_target_at_first_chronological_state=best,
            predicted_eta_saving=scores[old['column_id']]['estimated_completion']-best['eta'],
            current_node_usage=totals(selected),target_node_usage=totals(others+[target]),
            node_capacities=[2,cpu,ram],other_jobs={c['job_id']:info(c,scores) for c in others},
            budget=a['service_delay_ledgers'][j],productive_service=a['productive_service'][j],
            all_target_candidates_at_this_state=candidates,
            future_information_used=False,simulated_counterfactual_JCT=None,
            proof_scope='Only instantaneous feasibility and strict predicted ETA improvement including logged startup; does not prove future completion-time improvement.')
    save('nonstealing_recovery_witnesses.json',output)
    for key,v in output.items():
        print(key,v['status'],v.get('time'),v.get('predicted_eta_saving'),flush=True)


if __name__=='__main__':main()
