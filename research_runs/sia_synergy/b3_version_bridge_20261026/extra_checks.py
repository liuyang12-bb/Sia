"""Recorded end-state counters and first-decision/tie diagnostics."""
import collections,gzip,json,math
from pathlib import Path
OUT=Path(__file__).resolve().parent;p=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
def main():
    rows=[];first={};allcells=p['reference_cells']+[dict(c,folder=str(OUT/c['id'])) for c in p['runs']]
    for c in allcells:
        folder=Path(c['folder']);s=json.loads((folder/'summary.json').read_text());final=None;firstd=None
        since=collections.defaultdict(float);last_round=None;transitions=collections.Counter();examples=[]
        with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);final=r
                if firstd is None:firstd=r['decision_for_next_interval']
                for e in r['execution']:since[e['job_id']]+=e['training_seconds']
                d=r['decision_for_next_interval']
                if d['round']==last_round:continue
                last_round=d['round'];sel=d['result']['selected'];pb=d['probe']
                for spec in pb['specs']:
                    j=spec['job_id'];new=sel.get(j)
                    old=(spec['current_config_id'],spec['current_placement']) if spec['current_config_id'] else None
                    fresh=(new['config']['config_id'],list(map(list,new['placement'].items()))) if new else None
                    if old and fresh!=old:
                        label='switch' if fresh else 'preempt'
                        transitions[label]+=1
                        if since[j]<=1e-9:
                            transitions[label+'_before_productive_since_geometry']+=1
                            if len(examples)<5:examples.append(dict(time=r['interval_end'],job=j,old=old,new=fresh,productive_since_geometry=since[j],action=label))
                        if fresh:
                            source=pb['provenance'][j][fresh[0]]['goodput_source']
                            transitions['switch_target_'+source]+=1
                    if fresh!=old:since[j]=0.
        for j,end in final['job_end_state'].items():
            if j not in s['jobs']:continue
            assert end['restarts']==s['jobs'][j]['restarts']
            assert math.isclose(end['wasted_gpu_seconds'],s['jobs'][j]['wasted_gpu_seconds'],abs_tol=1e-7)
            assert end['completion']==s['jobs'][j]['completion']
        key=(c['dataset'],c['response'],c['method']);first[key]=firstd
        rows.append(dict(dataset=c['dataset'],response=c['response'],method=c['method'],end_state_counters_valid=True,
            first_objective=firstd['result']['scheduling_objective'],first_selected=len(firstd['result']['selected']),
            gpu_transition_counts=dict(transitions),no_productive_transition_examples=examples))
    diagnostics=[]
    for ds in ['original8','original20','holdout20']:
      for r in ['neutral','medium']:
        free=first[ds,r,'b3_free'];g=first[ds,r,'joint_g'];a=free['result']['selected'];b=g['result']['selected']
        common=set(a)&set(b)
        diagnostics.append(dict(dataset=ds,response=r,free_objective=free['result']['scheduling_objective'],g_objective=g['result']['scheduling_objective'],
            free_selected=len(a),g_selected=len(b),different_selected_jobs=sorted(set(a)^set(b)),
            different_gpu_geometry=[j for j in sorted(common) if (a[j]['config']['config_id'],a[j]['placement'])!=(b[j]['config']['config_id'],b[j]['placement'])],
            different_aux=[j for j in sorted(common) if (a[j]['cpu_per_gpu'],a[j]['ram_gb_per_gpu'])!=(b[j]['cpu_per_gpu'],b[j]['ram_gb_per_gpu'])],
            free_goodput_sources=free['candidate_summary'],g_goodput_sources=g['candidate_summary'],
            note='Free runtime retains legacy Pareto/tie rules, G exact primary then startup/geometry. Equal objectives do not imply equal placement or isolate admission alone.'))
    with (OUT/'extra_validation.json').open('x') as f:json.dump(dict(valid=True,cells=rows,initial_decision_comparison=diagnostics),f,indent=2)
    print('EXTRA_COUNTERS_FIRST_DECISIONS',len(rows))
if __name__=='__main__':main()
