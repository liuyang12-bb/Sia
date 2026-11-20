"""Create a dependency-closed copy; never edits the frozen source."""
import hashlib,json,shutil,time
from pathlib import Path
from inventory import ROOT,OUT,SOURCE,sha,save

def main():
    plan=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
    clean=ROOT/'cc-joint';sim=clean/'sia-simulator'
    # README/front door can already exist, but no engine file may be replaced.
    paths=set(plan['keep_python'])|set(plan['keep_data'])|{'requirements.txt'}
    paths={p for p in paths if not p.startswith('utils/')}
    hashes={}
    for name in sorted(paths):
        src=SOURCE/name;dst=sim/name
        assert not dst.exists(),str(dst)
        if name in plan['source_hashes']:assert sha(src)==plan['source_hashes'][name]
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert sha(src)==sha(dst);hashes['sia-simulator/'+name]=sha(dst)
    historical=ROOT/'research_runs/sia_synergy/cc_joint_sustained_cpu_20job_20261027_run1'
    parent=json.loads((historical/'preregistration.json').read_text(encoding='utf-8'))
    inputs=['config_symmetric.json','registry.json','workload.csv','original_workload-1.csv']
    inputs += [p.relative_to(historical/'inputs').as_posix() for p in (historical/'inputs/models').rglob('*') if p.is_file()]
    for name in inputs:
        src=historical/'inputs'/name;dst=clean/'inputs'/name
        assert not dst.exists();assert sha(src)==parent['input_hashes']['inputs/'+name]
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);hashes['inputs/'+name]=sha(dst)
    shutil.copy2(historical/'dependencies.lock.txt',clean/'dependencies.lock.txt')
    hashes['dependencies.lock.txt']=sha(clean/'dependencies.lock.txt')
    save(clean/'source_manifest.json',dict(created_unix=time.time(),algorithm_changes=False,
        authority=str(historical),authority_preregistration_sha256=sha(historical/'preregistration.json'),
        original_full_source_fingerprint=parent['source_fingerprint'],files=hashes,
        runtime_modules=plan['runtime_python'],regression_support_modules=sorted(paths-set(plan['runtime_python'])),
        excluded_python=plan['exclude_python'],
        note='All retained engine/tests bytes identical; only dependency selection and new front door. Original404 snapshot and historical outputs stay frozen.'))
    save(OUT/'prepare_result.json',dict(valid=True,clean=str(clean),copied=len(hashes),
        engine_copied=len(paths),excluded_python=len(plan['exclude_python']),source_manifest_sha256=sha(clean/'source_manifest.json')))
    print('CLEAN_COPY_CREATED',len(hashes),'files; engine bytes unchanged',flush=True)

if __name__=='__main__':main()
