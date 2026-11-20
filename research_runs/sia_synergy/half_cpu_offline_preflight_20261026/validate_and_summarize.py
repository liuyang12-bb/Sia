"""Independent witness arithmetic, rank tie classification, immutable seal."""
import collections,datetime,gzip,hashlib,itertools,json,math,sys
from pathlib import Path
import preflight as pf
BASE=Path(__file__).resolve().parent;OUT=BASE/'attempt2'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
    witnesses=read('same_geometry_resource_witnesses.json');rank=read('online_gpu_rank_evidence.json');states=read('selected_state_inputs.json')
    checks=collections.Counter();cpu_evidence=[]
    for w in witnesses:
        assert w['diagnostics']['optimal'] and w['witness']['valid'];usage={t:[0.,0.,0.] for t in pf.TYPES};relative=[]
        for row in w['witness']['job_changes']:
            app=row['application'];cid=row['config'];c=row['allocation'];b=row['baseline'];g=row['gpus'];typ=cid.split(':')[0]
            qb=pf.query(app,cid,b['cpu'],b['ram'],w['capacity']);qa=pf.query(app,cid,c['cpu'],c['ram'],w['capacity'])
            assert pf.close(qb.rho,b['rho']) and pf.close(qa.rho,c['rho']) and qa.rho+1e-9>=qb.rho
            assert c['cpu'] in set(pf.GRID['aux_cpu_points']+[pf.prop(w['capacity'])[0]])
            assert c['ram'] in set(pf.GRID['aux_ram_points']+[62.5])
            usage[typ][0]+=g;usage[typ][1]+=g*c['cpu'];usage[typ][2]+=g*c['ram'];relative.append(qa.rho/qb.rho);checks['job_allocations']+=1
        assert len(relative)==w['jobs'] and pf.close(sum(relative),w['relative_speed_sum'])
        for t,u in usage.items():
            assert u[0]<=pf.TYPES[t] and u[1]<=pf.caps(w['capacity'])[t]['cpu_cores']+1e-8 and u[2]<=pf.caps(w['capacity'])[t]['ram_gb']+1e-8
            checks['node_capacity']+=1
        checks['witnesses']+=1
    classifications=[]
    for ds in ['original20','holdout20']:
      for criterion in ['FIRST_DECISION','FIRST_WARM_NO_CHECKPOINT']:
        rs=[x for x in rank if x['dataset']==ds and x['criterion']==criterion];strict=[];tie=[];rows=[]
        for x in rs:
            bestmain=max(v['main_score'] for v in x['options']);besthalf=max(v['half_score'] for v in x['options'])
            mainset={v['config'] for v in x['options'] if abs(v['main_score']-bestmain)<1e-8}
            halfset={v['config'] for v in x['options'] if abs(v['half_score']-besthalf)<1e-8}
            if not mainset&halfset:strict.append(x['job'])
            elif x['main_best']!=x['half_best']:tie.append(x['job'])
            rows.append(dict(job=x['job'],main_best_set=sorted(mainset),half_best_set=sorted(halfset),strict_displacement=not bool(mainset&halfset)))
        classifications.append(dict(dataset=ds,criterion=criterion,jobs=len(rs),strict_best_set_displacement_jobs=strict,tie_selection_changed_jobs=tie,
            same_gpu_count_strict_rank_flip_jobs=[x['job'] for x in rs if any(v['same_gpu_count'] for v in x['strict_pair_rank_reversals'])],rows=rows))
    save('gpu_rank_classification.json',classifications)
    # Feasible GPU-type pair swaps at the same proportional point. These are
    # static scheduling scores with saved restart factors, NOT JCT forecasts.
    swaps=[]
    for state in states:
        d=state['round']['decision_for_next_interval'];selected=d['result']['selected'];rs={r['job']:r for r in rank if r['dataset']==state['dataset'] and r['criterion']==state['criterion']}
        candidates=[]
        for j,k in itertools.combinations(sorted(selected),2):
            a,b=selected[j],selected[k]
            if a['config']['gpus']!=b['config']['gpus'] or a['config']['gpu_type']==b['config']['gpu_type']:continue
            g=a['config']['gpus'];newj=f"{b['config']['gpu_type']}:1:{g}";newk=f"{a['config']['gpu_type']}:1:{g}"
            jopts={v['config']:v for v in rs[j]['options']};kopts={v['config']:v for v in rs[k]['options']}
            if newj not in jopts or newk not in kopts:continue
            prevj=jopts[a['config']['config_id']];prevk=kopts[b['config']['config_id']];nextj=jopts[newj];nextk=kopts[newk]
            before=prevj['half_score']+prevk['half_score'];after=nextj['half_score']+nextk['half_score']
            if after>before+1e-8:
                candidates.append(dict(jobs=[j,k],before_configs=[a['config']['config_id'],b['config']['config_id']],after_configs=[newj,newk],
                    before_score=before,after_score=after,relative_score_gain=after/before-1,
                    no_individual_score_loss=nextj['half_score']+1e-8>=prevj['half_score'] and nextk['half_score']+1e-8>=prevk['half_score'],
                    after_clipping=[nextj['half_clipping'],nextk['half_clipping']],
                    warning='GPU counts and all node loads preserved by pair swap. Startup costs represented only by logged restart-factor proxy; actual migrations/leases/commitments not replayed.'))
        swaps.append(dict(dataset=state['dataset'],criterion=state['criterion'],time=state['round']['interval_end'],improving_pair_count=len(candidates),
            pareto_pair_count=sum(c['no_individual_score_loss'] for c in candidates),first_lexicographic_witness=next((c for c in candidates if c['no_individual_score_loss']),candidates[0] if candidates else None),all_candidates=candidates))
    save('feasible_gpu_type_swap_evidence.json',swaps)
    sources=read('source_validation.json');summary=read('summary.json');boundary=read('boundary_release_certificates.json')
    for n,h in pf.P['hashes'].items():assert pf.sha(n)==h
    old=json.loads((pf.BRIDGE/'DELIVERY_MANIFEST.json').read_text(encoding='utf-8'))
    for name,h in old['files'].items():assert pf.sha(pf.BRIDGE/name)==h
    for r in json.loads((pf.BRIDGE/'completion_record.json').read_text(encoding='utf-8'))['report_hashes'].items():assert pf.sha(r[0])==r[1]
    manifest=read('run_manifest.json')
    assert manifest['simulation_step_calls']==0 and not manifest['training_simulator_imported']
    assert not any(m in sys.modules for m in ['job','cluster','joint_prototype.full160'])
    assert summary['physical_proportional_feasible']
    assert all(t['observed_configs']==11 for t in read('task_coverage.json'))
    validation=dict(valid=True,checks=dict(checks),independent_formula_queries=manifest['query_count'],old_bridge_seal_files_unchanged=len(old['files']),
        original_hashes_unchanged=len(pf.P['hashes']),full_task_configuration_records=len(read('task_configuration_evidence.json')),
        actual_simulation_step_calls=0,rank_summary=[{k:v for k,v in c.items() if k!='rows'} for c in classifications],
        decision='CONDITIONAL_GO_FOR_FROZEN_SYNTHETIC_FOUR_STRATEGY_MECHANISM_COMPARISON_ONLY',
        boundary_gate='UNRESOLVED: all four fixed-GPU half-CPU states require at least one clipped resource-releasing move for no-loss improvement. Not target-hardware validation.',
        no_permission_to_run_simulation=True)
    save('validation.json',validation)
    print('VALIDATION',checks)
    print('STRICT_BEST',[(x['dataset'],x['criterion'],len(x['strict_best_set_displacement_jobs']),len(x['tie_selection_changed_jobs'])) for x in classifications])
    print('GPU_PAIR_SWAPS',[(x['dataset'],x['criterion'],x['improving_pair_count'],x['pareto_pair_count']) for x in swaps])
if __name__=='__main__':main()
