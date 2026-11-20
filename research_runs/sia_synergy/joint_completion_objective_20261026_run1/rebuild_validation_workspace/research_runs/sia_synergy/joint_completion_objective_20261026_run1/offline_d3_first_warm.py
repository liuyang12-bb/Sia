"""One structurally selected D3 warm state; all other allocations stay fixed."""
import itertools
import json
from offline_gate import OUT,RUNS,SCENARIOS,CPU_GRID,RAM_GRID,first_decision,query,sha,save

rule=dict(selection="First D3 CPU-symmetric SIA-P logged decision with a served CIFAR and served DeepSpeech whose every candidate has direct_profile; choose lowest source-row index within each app",budget="Hold all other selected jobs exactly fixed; use residual per-node budget; preserve selected pair membership; no candidate removal, clean-support optima reported separately",maximum_extra_states=1,training_executed=False)
save('offline_d3_selection_rules.json',rule)
path=RUNS/'cells/D3_queued_training_proposal_cpu_symmetric_sia_p/rounds.jsonl.gz'
def eligible(d):
    selected=d['result']['selected']
    good=[s for s in d['probe']['specs'] if s['job_id'] in selected and all(p['goodput_source']=='direct_profile' for p in d['probe']['provenance'][s['job_id']].values())]
    return good
state=first_decision(path,lambda d:all(any(s['job_id'].startswith(app+'-') for s in eligible(d)) for app in ['cifar10','deepspeech2']))
d=state['decision']; good=eligible(d)
specs=[min([s for s in good if s['job_id'].startswith(app+'-')],key=lambda s:s['source_row_index']) for app in ['cifar10','deepspeech2']]
chosen={s['job_id'] for s in specs}; nodes=SCENARIOS['D3_existing_cpu_symmetric']
residual={gpu:dict(node) for gpu,node in nodes.items()}
peers={j:c for j,c in d['result']['selected'].items() if j not in chosen}
for peer in peers.values():
    for node,g in peer['placement'].items():
        cap=residual[node.rsplit(':',1)[0]];cap['gpus']-=g;cap['cpu']-=g*peer['cpu_per_gpu'];cap['ram']-=g*peer['ram_gb_per_gpu']
points={}
for s in specs:
    job=s['job_id'];app=job.rsplit('-',1)[0];options=[]
    for config in s['configs']:
        for cpu,ram in itertools.product(CPU_GRID,RAM_GRID):
            q=query(app,config['config_id'],cpu,ram,config['sia_goodput'],job)
            q['placement']={config['gpu_type']+':0':config['gpus']}
            q['residual_individually_feasible']=all(config['gpus']*v <= residual[config['gpu_type']][resource]+1e-9 for resource,v in [('gpus',1),('cpu',cpu),('ram',ram)])
            current=dict(s['current_placement'])
            q['lock_eligible']=not s['checkpoint_locked'] or (config['config_id']==s['current_config_id'] and q['placement']==current and cpu==s['current_cpu_per_gpu'] and ram==s['current_ram_gb_per_gpu'])
            options.append(q)
    points[job]=options
def feasible(pair):
    if not all(p['lock_eligible'] and p['residual_individually_feasible'] for p in pair):return False
    for gpu,cap in residual.items():
        ps=[p for p in pair if p['gpu_type']==gpu]
        if sum(p['gpus'] for p in ps)>cap['gpus'] or sum(p['gpus']*p['cpu_per_gpu'] for p in ps)>cap['cpu']+1e-9 or sum(p['gpus']*p['ram_gb_per_gpu'] for p in ps)>cap['ram']+1e-9:return False
    return True
allpairs=list(itertools.product(*(points[s['job_id']] for s in specs)))
feasiblepairs=[p for p in allpairs if feasible(p)]
cleanpairs=[p for p in feasiblepairs if all(q['in_support'] for q in p)]
original=[d['result']['selected'][s['job_id']] for s in specs]
fixedpairs=[p for p in cleanpairs if all(q['config_id']==o['config']['config_id'] and q['placement']==o['placement'] for q,o in zip(p,original))]
score=lambda ps:sum(p['effective_goodput'] for p in ps)
best=max(cleanpairs,key=score);fixed=max(fixedpairs,key=score)
original_clean=all(not c['source_coordinates']['clipped'] and not c['baseline_source_coordinates']['clipped'] for c in original)
payload=dict(selection=rule,state=state,selected_pair=[s['job_id'] for s in specs],held_peers=peers,residual_budget=residual,
    all_candidates=points,config_counts={j:{c:dict(total=sum(p['config_id']==c for p in ps),clean=sum(p['config_id']==c and p['in_support'] for p in ps),residual_feasible=sum(p['config_id']==c and p['residual_individually_feasible'] for p in ps)) for c in sorted({p['config_id'] for p in ps})} for j,ps in points.items()},
    pair_counts=dict(all=len(allpairs),capacity_and_lock_feasible=len(feasiblepairs),clean=len(cleanpairs),fixed_gpu_clean=len(fixedpairs)),original_pair=original,original_pair_clean=original_clean,
    fixed_gpu_aux_best=fixed,joint_best=best,fixed_score=score(fixed),joint_score=score(best),gain=score(best)/score(fixed)-1,
    opportunity_supported=original_clean and score(best)>score(fixed)+1e-9,
    restrictions='Other jobs held exactly fixed; no admission difference. Zero switching costs make this an optimistic speed-only bound, not a JCT predictor. Configs with no residual room and out-of-support points remain recorded.',script_sha256=sha(__file__))
save('offline_d3_first_warm.json',payload)
print(json.dumps({k:payload[k] for k in ['selected_pair','residual_budget','pair_counts','original_pair_clean','fixed_score','joint_score','gain','opportunity_supported']},indent=2))
