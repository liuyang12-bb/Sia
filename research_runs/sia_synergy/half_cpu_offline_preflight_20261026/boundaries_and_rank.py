"""No further solve: resource-release certificate and logged-online rank tests."""
import collections,json,math
from pathlib import Path
import preflight as pf
OUT=Path(__file__).resolve().parent/'attempt2'
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
    states=json.loads((OUT/'selected_state_inputs.json').read_text(encoding='utf-8'))
    points=json.loads((OUT/'all_response_points.json').read_text(encoding='utf-8'))
    witnesses=json.loads((OUT/'same_geometry_resource_witnesses.json').read_text(encoding='utf-8'))
    certificate=[];rankrows=[]
    for state in states:
        if state['status']!='FOUND':continue
        r=state['round'];d=r['decision_for_next_interval'];pb=d['probe'];chosen=d['result']['selected']
        for level in ['main','half']:
            pc,pm=pf.prop(level);nodes={t:dict(gpu=0,cpu=0.,ram=0.) for t in pf.TYPES};releases=[]
            for name,j in chosen.items():
                app=name.rsplit('-',1)[0];cid=j['config']['config_id'];g=j['config']['gpus'];typ=j['config']['gpu_type']
                base=pf.query(app,cid,pc,pm,level);nodes[typ]['gpu']+=g;nodes[typ]['cpu']+=g*pc;nodes[typ]['ram']+=g*pm
                interior=[]
                for q in points:
                    if q['application']!=app or q['config']!=cid or q['capacity']!=level or not q['progress_floor_eligible']:continue
                    if not pf.inside(base) or not all(v=='inside' for v in q['clipping'].values()):continue
                    if q['cpu']<pc-1e-9 or q['ram']<pm-1e-9:interior.append({k:q[k] for k in ['cpu','ram','rho']})
                releases.append(dict(job=name,config=cid,baseline_clipping=pf.clip_flags(base),unclipped_resource_release_options=interior))
            saturated=all(pf.close(u['cpu'],pf.caps(level)[t]['cpu_cores']) and pf.close(u['ram'],pf.caps(level)[t]['ram_gb']) for t,u in nodes.items())
            zero=all(not x['unclipped_resource_release_options'] for x in releases)
            certificate.append(dict(dataset=state['dataset'],criterion=state['criterion'],time=r['interval_end'],capacity=level,
                baseline_usage=nodes,all_node_cpu_and_ram_saturated=saturated,no_interior_resource_releasing_move=zero,
                no_gain_if_clipped_jobs_frozen_and_only_interior_moves_allowed=saturated and zero,
                explanation='If all capacity rows are tight and no allowed changed point can release either resource, no task can increase either resource. This does not ban legacy clipping or alter its response; it diagnoses dependence of no-loss redistribution on out-of-grid moves.',jobs=releases))
        for spec in pb['specs']:
            name=spec['job_id'];app=name.rsplit('-',1)[0];options=[]
            for c in spec['configs']:
                if spec['checkpoint_locked'] and c['config_id']!=spec['current_config_id']:continue
                cid=c['config_id'];f=pb['restart_factors'][name][cid]
                main=pf.query(app,cid,3.,62.5,'main',job_id=name,raw=c['sia_goodput'])
                half=pf.query(app,cid,1.5,62.5,'half',job_id=name,raw=c['sia_goodput'])
                options.append(dict(config=cid,gpus=c['gpus'],gpu_type=c['gpu_type'],normalized_online_raw_goodput=c['sia_goodput'],restart_factor=f,
                    source=pb['provenance'][name][cid]['goodput_source'],main_score=main.effective_goodput*f,half_score=half.effective_goodput*f,half_rho=half.rho,
                    main_clipping=pf.clip_flags(main),half_clipping=pf.clip_flags(half)))
            assert options
            a=max(options,key=lambda x:(x['main_score'],x['config']));b=max(options,key=lambda x:(x['half_score'],x['config']))
            flips=[]
            for i,x in enumerate(options):
                for y in options[i+1:]:
                    before=x['main_score']-y['main_score'];after=x['half_score']-y['half_score']
                    if before*after<0 and abs(before)>1e-8 and abs(after)>1e-8:
                        flips.append(dict(first=x['config'],second=y['config'],same_gpu_count=x['gpus']==y['gpus'],main_difference=before,half_difference=after))
            rankrows.append(dict(dataset=state['dataset'],criterion=state['criterion'],time=r['interval_end'],job=name,application=app,
                main_best=a['config'],half_best=b['config'],best_changed_strictly=b['half_score']>a['half_score']+1e-8,
                half_gain_over_main_best=(b['half_score']/a['half_score']-1) if a['half_score']>0 else None,
                strict_pair_rank_reversals=flips,options=options,
                note='Uses only this saved current state online GPU model and its unchanged restart factors. Not a new trajectory, absolute progress/second, global scheduler solution, or forecast of JCT.'))
    summaries=[]
    for state in states:
        rows=[r for r in rankrows if r['dataset']==state['dataset'] and r['criterion']==state['criterion']]
        summaries.append(dict(dataset=state['dataset'],criterion=state['criterion'],time=state['round']['interval_end'] if state['round'] else None,jobs=len(rows),
            best_changed_jobs=sum(r['best_changed_strictly'] for r in rows),any_rank_flip_jobs=sum(bool(r['strict_pair_rank_reversals']) for r in rows),
            same_gpu_count_flip_jobs=sum(any(p['same_gpu_count'] for p in r['strict_pair_rank_reversals']) for r in rows)))
    save('boundary_release_certificates.json',certificate);save('online_gpu_rank_evidence.json',rankrows);save('online_gpu_rank_summary.json',summaries)
    print('BOUNDARY_CERTIFICATES',[(r['dataset'],r['criterion'],r['capacity'],r['no_gain_if_clipped_jobs_frozen_and_only_interior_moves_allowed']) for r in certificate])
    print('GPU_RANKS',summaries)
if __name__=='__main__':main()
