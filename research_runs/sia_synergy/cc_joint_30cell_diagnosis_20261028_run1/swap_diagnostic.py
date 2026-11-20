"""Explain the logged CF13 GPU path at S5; no scheduler or training execution."""
import itertools,json
from investigate import HERE,save,feasible,supported,evaluate


def main():
    s=json.loads((HERE/'states/S5.json').read_text());d=s['raw_round']['decision_for_next_interval']
    a=d['result']['audit'];cols=d['probe']['joint_columns'];scores={v['column_id']:v for v in d['continuity_scores']}
    byid={c['column_id']:c for c in cols};actual=[byid[c['column_id']] for c in d['result']['selected'].values()]
    target={j:c['config']['config_id'] for j,c in d['result']['selected'].items()}
    target['cifar10-13']='dgx-ext:1:1';target['deepspeech2-8']='aws:1:1'
    results={}
    for support in [False,True]:
        chosen=[];count=0;eligible=True
        for gpu_type in ['aws','dgx-ext']:
            names=[j for j,cfg in target.items() if cfg.startswith(gpu_type+':')]
            pools=[[c for c in cols if c['job_id']==j and c['config']['config_id']==target[j]
                    and (not support or supported(c))] for j in names]
            best=None;key=None
            for combination in itertools.product(*pools):
                count+=1
                if not feasible(combination,6,187.5):continue
                v=evaluate(combination,scores);k=(-v['objective'],v['sum_known_eta'])
                if key is None or k<key:key=k;best=combination
            if best is None:eligible=False;break
            chosen+=list(best)
        result=dict(status='CAPACITY_FEASIBLE_POLICY_BUDGET_RELAXED_ONLY_FOR_DIAGNOSIS' if eligible else 'NO_RECORDED_COLUMN_COMBINATION',
                    enumerated=count,all_points_supported=support)
        if eligible:
            assert feasible(chosen,6,187.5)
            result.update(evaluate(chosen,scores))
            result['primary_gain_vs_actual']=result['objective']-sum(c['scheduling_utility'] for c in actual)
            violations={}
            for c in chosen:
                j=c['job_id'];rejected={v['column_id']:v['reason'] for v in a['adaptive_candidate_filters'].get(j,{}).get('rejected',[])}
                if c['column_id'] in rejected:violations[j]=rejected[c['column_id']]
            result['actual_policy_rejection_reasons']=violations
        results['SUPPORTED_ONLY' if support else 'LEGACY_ALL']=result
    save('S5_GPU_swap_diagnostic.json',dict(cell=s['cell'],line=s['line'],time=s['time'],actual=evaluate(actual,scores),alternatives=results,
         limitation='An instantaneous capacity-feasible but policy-ineligible hypothetical action; does not prove a better JCT or justify general budget relaxation. Source candidates and score use only logged information.'))
    print('SWAP_DIAGNOSTIC',[(k,v.get('primary_gain_vs_actual'),v.get('actual_policy_rejection_reasons')) for k,v in results.items()])


if __name__=='__main__':main()
