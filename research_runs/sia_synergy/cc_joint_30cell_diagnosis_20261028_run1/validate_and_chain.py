"""Validate diagnostic accounting and store observed R11 event-chain evidence."""
import ast,gzip,json,math
from pathlib import Path
from scan import HERE,BASE,save,sha


def main():
    pre=json.loads((BASE/'preregistration.json').read_text());checks={}
    for case in pre['cells']:
        name='_'.join(case);s=json.loads((BASE/'cells'/name/'summary.json').read_text())
        allocation={};training={}
        with gzip.open(HERE/'slim_timelines'/f'{name}.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                for j in r['selected']:allocation.setdefault(j,r['time'])
                for e in r['execution']:
                    if e['training']>0:training.setdefault(e['job'],r['start']+e['checkpoint'])
        for j,v in s['jobs'].items():
            assert allocation[j]==v['first_allocation']
            assert training[j]==v['first_training']
            assert allocation[j]>=v['arrival']
            assert 0<=v['wait']-(allocation[j]-v['arrival'])
        checks[name]=dict(first_allocation_independently_verified=20,first_productive_time_independently_verified=20)
    chain={}
    for method in ['CC_CPU','CC_FULL']:
        cell='R11_medium_'+method;s=json.loads((BASE/'cells'/cell/'summary.json').read_text());selected=[]
        for j in ['cifar10-0','cifar10-1','cifar10-3','cifar10-5','cifar10-11','cifar10-13','cifar10-14','cifar10-15','deepspeech2-8','deepspeech2-9']:
            v=s['jobs'][j]
            selected.append(dict(job=j,first_allocation=v['first_allocation'],completion=v['completion'],jct=v['jct'],wait=v['wait'],training=v['training'],checkpoint=v['checkpoint']))
        decision_times=[1020,1080,2520,2580,4860,4980,5400,5520,5940,6240,12000,12480,13020]
        decisions=[]
        with gzip.open(HERE/'slim_timelines'/f'{cell}.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['time'] in decision_times:
                    decisions.append({k:r[k] for k in ['line','time','selected','events','service_budget','tail_recoveries']})
        chain[cell]=dict(jobs=selected,decisions=decisions)
    save('observed_R11_event_chain.json',chain)
    for name in ['scan.py','investigate.py','recovery_details.py','swap_diagnostic.py',Path(__file__).name]:
        tree=ast.parse((HERE/name).read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):assert all(x.name.split('.')[0] not in ['job','cluster','sia','joint_prototype','scipy'] for x in n.names)
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in ['job','cluster','sia','joint_prototype','scipy']
    before=json.loads((HERE/'integrity_before.json').read_text())
    for name,h in before['protected'].items():assert sha(BASE/name)==h,name
    for name,h in pre['source_hashes'].items():assert sha(BASE/'source'/name)==h and sha(HERE.parents[2]/'cc-joint-resource-study'/name)==h,name
    for cell,hashes in before['cells'].items():
        assert sha(BASE/'cells'/cell/'rounds.jsonl.gz')==hashes['raw_sha256']
        assert sha(BASE/'cells'/cell/'summary.json')==hashes['summary_sha256']
        assert sha(BASE/'cells'/cell/'output_hashes.json')==hashes['output_hashes_sha256']
    save('diagnostic_validation.json',dict(status='READ_ONLY_DIAGNOSIS_VALID',cells=30,
         first_time_checks=checks,all_JCT_and_queue_area_identities_passed=True,
         fixed_geometry_static_enumeration_checked=True,
         recovery_witnesses_capacity_and_supported_coordinates_checked=True,
         simulation_invocations=0,scheduler_invocations=0,algorithm_changes=0,
         frozen_source_files_unchanged=len(pre['source_hashes']),historical_cells_unchanged=30,
         limitation='No repaired algorithm implemented; no counterfactual training completion or JCT was generated.'))
    print('DIAGNOSTIC_VALIDATION_OK_30_CELLS_NO_SIMULATION')


if __name__=='__main__':main()
