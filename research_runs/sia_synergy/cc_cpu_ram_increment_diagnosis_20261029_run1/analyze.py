"""Read-only diagnosis of sealed results; never constructs or steps a simulator."""
import ast, collections, gzip, hashlib, json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / 'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt3'
SOURCE = RUN / 'source'
def read(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def matrix(name):
    tree = ast.parse((SOURCE/'inputs/models'/name).read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='tput' for t in n.targets):
            return np.array(ast.literal_eval(n.value.args[0]),dtype=float)
    raise ValueError(name)
def main():
    seal=read(RUN/'delivery_hashes_final.json')
    assert all(sha(RUN/n)==h for n,h in seal.items())
    audit=read(RUN/'analysis/raw_audit.json')
    response=read(RUN/'analysis/execution_response_accounting.json')
    reg={r['key']:r for r in read(SOURCE/'inputs/registry.json')}
    result={'scope':'OFFLINE ONLY; no simulations, policy edits, or historical writes',
            'warning':'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE',
            'sealed_files_verified':len(seal),'source_checks':{},'comparisons':{},'response_domain':[], 'boundary_case':{}}
    for n in ['breakpoint_study.py','resource_study.py','sia-simulator/joint_prototype/completion_continuity.py',
              'sia-simulator/joint_prototype/joint_profile.py','sia-simulator/joint_prototype/joint_scheduler.py']:
        current=ROOT/'cc-joint-breakpoint-precision'/n
        result['source_checks'][n]={'executed_sha256':sha(SOURCE/n),'current_sha256':sha(current),'identical':sha(current)==sha(SOURCE/n)}
    for mix,cap in [('M1','R11'),('M2','R11'),('M3','R11'),('M2','R01')]:
        runs={m:audit[f'{mix}_{cap}_medium_{m}'] for m in ['CC_P','CC_CPU','CC_MEM','CC_FULL']}
        s={m:r['summary'] for m,r in runs.items()}
        jobs={n:{k:runs['CC_FULL']['jobs'][n][k]-v[k] for k in ['jct','wait','training','checkpoint','gpu_hours','cpu_core_hours','ram_gb_hours','restarts']} for n,v in runs['CC_CPU']['jobs'].items()}
        apps=collections.defaultdict(collections.Counter)
        for n,d in jobs.items():apps[n.rsplit('-',1)[0]].update(d)
        result['comparisons'][mix+'/'+cap]={
            'mean_jct':{m:x['mean_jct'] for m,x in s.items()},
            'full_minus_cpu_totals':{k:s['CC_FULL']['totals'][k]-s['CC_CPU']['totals'][k] for k in s['CC_CPU']['totals']},
            'interaction_seconds_full_minus_cpu_minus_mem_plus_p':s['CC_FULL']['mean_jct']-s['CC_CPU']['mean_jct']-s['CC_MEM']['mean_jct']+s['CC_P']['mean_jct'],
            'job_deltas':jobs,'application_deltas':{k:dict(v) for k,v in apps.items()},
            'execution_response':{m:response[f'{mix}_{cap}_medium_{m}']['totals'] for m in runs}}
    for app in ['cifar10','deepspeech2']:
        for gpu in ['aws','dgx-ext']:
            for g in [1,2]:
                cid=f'{gpu}:1:{g}';r=reg[app+'/'+cid];a=matrix(r['template'])
                def speed(cpu,ram):
                    x,y=cpu/r['phi_c'],ram/r['phi_m']
                    return float(np.interp(x,[1,2,3,4,5,6,9,12,24],[np.interp(y,[20.83,62.5,125,187.5,250],row) for row in a]))
                # In the tested capacity domain, source CPU <= 9, where RAM plateaus at source62.5.
                assert 12/g/r['phi_c']<9
                sat=62.5*r['phi_m'];prop=93.75;upper=187.5/g
                result['response_domain'].append({'application':app,'gpu_config':cid,'template':r['template'],
                    'phi_c':r['phi_c'],'phi_m':r['phi_m'],'ram_plateau_per_gpu':sat,'proportional_ram_per_gpu':prop,
                    'node_ram_upper_per_gpu':upper,'plateau_reachable':sat<=upper,
                    'ram_only_speedup_upper_at_cpu3':(speed(3,upper)/speed(3,prop))**.5-1,
                    'upper_bound_caveat':'Gives all node RAM to this job, ignores peers/checkpoint; not a feasible whole-state counterfactual.'})
    targets={'cifar10-3','cifar10-4','cifar10-10'}
    for method in ['CC_CPU','CC_FULL']:
        rows=[]
        with gzip.open(RUN/'cells'/f'M1_R11_medium_{method}'/'rounds.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                r=json.loads(line)
                if r['interval_end']>2160:break
                if r['interval_end'] not in [1980,2040,2100,2160]:continue
                d=r['decision_for_next_interval']
                rows.append({'interval_start':r['interval_start'],'interval_end':r['interval_end'],
                    'events':d['liveness_events'] if d else None,
                    'execution':[{k:e[k] for k in ['job_id','waiting_seconds','training_seconds','first_allocation_time','first_productive_training_time','progress_before','progress_after','rho','completion_time','resource_allocation']} for e in r['execution'] if e['job_id'] in targets]})
        result['boundary_case'][method]=rows
    result['input_hashes']={n:sha(RUN/n) for n in ['analysis/raw_audit.json','analysis/execution_response_accounting.json','preregistration.json','inputs/registry.json'] if (RUN/n).exists()}
    result['input_hashes']['source/inputs/registry.json']=sha(SOURCE/'inputs/registry.json')
    with (OUT/'findings.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    assert all(sha(RUN/n)==h for n,h in seal.items())
    print('READ_ONLY_DIAGNOSIS_COMPLETE',len(result['response_domain']),len(result['comparisons']),result['source_checks'])
if __name__=='__main__':main()
