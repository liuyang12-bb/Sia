"""Read-only result reconciliation. No simulator imports or model mutations."""
import collections, hashlib, json, math
from pathlib import Path
O=Path(__file__).resolve().parent
def read(name):return json.loads((O/name).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def write(name,value):
    with (O/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False)
p=read('preregistration.json');c=read('coverage.json');refs=read('reference_models.json')
summary={'classification':{},'coverage':{},'static':{},'one_shot':{}}
for app in ('cifar10','deepspeech2','bert'):
    rows=[x for x in c['rows'] if x['application']==app]
    byconfig={x['config']:x for x in rows}
    summary['classification'][app]=[dict(config=k,phi_c=x['proportional']['cpu_demand_scale'],
        phi_m=x['proportional']['memory_demand_scale'],proportional_rho=x['proportional']['rho'],
        best_rho=x['max_rho'],cpu_only_best=x['cpu_only_max_rho'],memory_only_best=x['memory_only_max_rho'],
        columns=x['columns'],supported=x['fully_supported'],num_clipped=x['numerator_clipped'],den_clipped=x['denominator_clipped'])
        for k,x in byconfig.items()]
summary['coverage']=dict(job_config_pairs=len(c['rows']),missing_pairs=72-len(c['rows']),
    full12_candidate_rows=sum(x['columns'] for x in c['rows']),
    numerator_clipped_rows=sum(x['numerator_clipped'] for x in c['rows']),
    denominator_clipped_rows=sum(x['denominator_clipped'] for x in c['rows']),
    fully_supported_rows=sum(x['fully_supported'] for x in c['rows']),
    denominator_clipped_configs=[dict(job=x['job'],config=x['config']) for x in c['rows'] if x['denominator_clipped']],
    proportional_num_clipped_configs=[dict(job=x['job'],config=x['config']) for x in c['rows'] if x['proportional']['source_coordinate_clipped']],
    proportional_den_clipped_configs=[dict(job=x['job'],config=x['config']) for x in c['rows'] if x['proportional']['baseline_coordinate_clipped']])
for state in ('E0','E25'):
    summary['static'][state]={}
    for label in ('PROPORTIONAL_GPU_PROPOSAL','FO_AUX_OPTIMUM','FULL_RESOURCE_OPTIMUM'):
        x=read(state+'_'+label+'.json');assert x['status']==0 and x['audit']['valid'] and len(x['selected'])==6
        summary['static'][state][label]=dict(objective=x['objective'],sum_eta=x['sum_eta'],selected=x['selected'],nodes=x['audit']['node_totals'],
            earliest_completion=min(v['eta'] for v in x['selected']),all_supported=all(not(v['source_num_clipped'] or v['source_den_clipped']) for v in x['selected']))
for state in ('cold','reference_E0'):
    out={}
    for method in ('FO','FULL'):
        x=read('oneshot_v2_'+state+'_'+method+'.json');assert x['valid'] and x['training_steps']==0 and x['time_advanced']==0
        rows=[]
        for name,y in x['selected'].items():
            row=dict(job=name,config=y['config']['config_id'],gpus=y['config']['gpus'],cpu_per_gpu=y['cpu_per_gpu'],ram_per_gpu=y['ram_gb_per_gpu'],rho=y['rho'],
                num_clipped=y['source_coordinates']['clipped'],den_clipped=y['baseline_source_coordinates']['clipped'],placement=y['placement'])
            if state=='reference_E0':
                v=refs['E0'][name]['predictions'][row['config']]
                row.update(raw=v['raw_rate'],rate=v['raw_rate']*row['rho'],remaining=v['remaining'],total_work=v['total_work'],startup=v['delay'])
                row['constant_rate_eta']=v['delay']+v['remaining']/row['rate']
            rows.append(row)
        out[method]=dict(objective=x['objective'],selected=rows,unallocated=x['unallocated'],nodes=x['node_totals'],required=x['required'],optional=x['optional'])
    out['same_selected_set']=sorted(x['job'] for x in out['FO']['selected'])==sorted(x['job'] for x in out['FULL']['selected'])
    a={x['job']:x for x in out['FO']['selected']};b={x['job']:x for x in out['FULL']['selected']}
    out['gpu_changed']=[j for j in a.keys()&b.keys() if a[j]['config']!=b[j]['config']]
    if state=='reference_E0' and out['same_selected_set']:
        out['objective_gain_pct']=100*(out['FULL']['objective']/out['FO']['objective']-1)
        windows={}
        for h in (360,600):
            av={j:min(v['remaining'],max(0,h-v['startup'])*v['rate'])/v['total_work'] for j,v in a.items()}
            bv={j:min(v['remaining'],max(0,h-v['startup'])*v['rate'])/v['total_work'] for j,v in b.items()}
            diff={j:bv[j]-av[j] for j in a}
            windows[str(h)]=dict(net=sum(diff.values()),fo_total=sum(av.values()),relative_pct=100*sum(diff.values())/sum(av.values()),per_job=diff,
                receivers=sum(max(0,v) for v in diff.values()),donor_loss=-sum(min(0,v) for v in diff.values()))
        out['held_bundle_conditional_windows']=windows
        out['constant_rate_sum_eta']={m:sum(v['constant_rate_eta'] for v in out[m]['selected']) for m in ('FO','FULL')}
        out['earliest_constant_rate_completion']={m:min(v['constant_rate_eta'] for v in out[m]['selected']) for m in ('FO','FULL')}
    summary['one_shot'][state]=out
write('reconciled_preflight.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
