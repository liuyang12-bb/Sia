import json,hashlib,zipfile,subprocess,sys,platform,time,datetime,importlib.metadata
from pathlib import Path
from difflib import unified_diff
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];A=OUT/'attempt3';p=json.loads((A/'preregistration.json').read_text());TREE=Path(p['tree']);WORK=TREE/'sia-simulator'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def save(f,x):
    with (OUT/f).open('x',encoding='utf-8') as s:json.dump(x,s,indent=2)
def main():
    for path,h in p['hashes'].items():assert sha(path)==h,path
    for path,h in p['baseline_hashes'].items():assert sha(path)==h,path
    for n in [8,20]:
        v=json.loads((A/f'validation_{n}.json').read_text());assert v['valid'] and v['all_complete']
    assert json.loads((OUT/'baseline_reaudit.json').read_text())['valid']
    assert json.loads((OUT/'version3_all_result.json').read_text())['exit_code']==0
    assert json.loads((OUT/'numerical_evidence.json').read_text())['valid']
    dist=sorted([dict(name=d.metadata['Name'],version=d.version) for d in importlib.metadata.distributions()],key=lambda d:d['name'].lower())
    save('dependencies.json',dist)
    with (OUT/'requirements-lock.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(v['name']+'=='+v['version'] for v in dist)+'\n')
    with zipfile.ZipFile(OUT/'frozen_prerequisites.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in p['hashes']:z.write(path,Path(path).relative_to(ROOT))
        for path in [ROOT/'official-artifacts/sia-simulator/job.py',ROOT/'official-artifacts/sia-simulator/sia.py',ROOT/'research_runs/sia_synergy/small_mechanism_20261025/preregistration.json']:
            if str(path) not in p['hashes']:z.write(path,path.relative_to(ROOT))
    with zipfile.ZipFile(OUT/'baseline_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in p['baseline_hashes']:z.write(path,Path(path).relative_to(ROOT))
    old=ROOT/'sia-joint-scale-check/sia-simulator';changes=[]
    for rel in ['joint_prototype/adaptive_service.py','joint_prototype/execution_audit.py','joint_prototype/bundle_adaptation.py','run_bundle_cell.py','tests/test_bundle_adaptation.py','tests/test_bundle_integration.py','tests/test_bundle_solver_numerics.py','tests/test_checkpoint_rounding_audit.py']:
        prior=(old/rel).read_text().splitlines(True) if (old/rel).exists() else []
        changes.extend(unified_diff(prior,(WORK/rel).read_text().splitlines(True),fromfile='old/'+rel,tofile='new/'+rel))
    with (OUT/'final_code_diff.patch').open('x',encoding='utf-8') as f:f.write(''.join(changes))
    events=[]
    for base in [OUT,OUT/'attempt2',A]:
        events.extend([json.loads(line) for line in (base/'execution_manifest.jsonl').read_text().splitlines()])
    branch=subprocess.check_output(['git','-C',str(TREE),'branch','--show-current']).decode().strip()
    info=dict(status='COMPLETE',actual_tree=str(TREE),actual_branch=branch,base_commit=p['base_commit'],new_code_committed=False,
        branch_metadata_erratum='Frozen manifest branch label inherited old branch; actual tree/hashes/commands correct; actual_branch here authoritative',
        python=sys.version,platform=platform.platform(),cpu_logical=psutil.cpu_count(),cpu_physical=psutil.cpu_count(logical=False),
        installed_packages=p['packages'],memory_total_gib=psutil.virtual_memory().total/1024**3,
        finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        task_elapsed_from_windows_current_task_creation_seconds=time.time()-(OUT/'CURRENT_TASK.md').stat().st_ctime,
        strategy_process_wall_all_attempts_seconds=sum(v['wall_seconds'] for v in events),
        strategy_process_wall_final_seconds=sum(json.loads(v)['wall_seconds'] for v in (A/'execution_manifest.jsonl').read_text().splitlines()),
        execution_attempts=len(events),final_cells=4,prior_baseline_cells=12,tests=141,
        frozen_source_input_hashes_verified=len(p['hashes']),baseline_file_hashes_verified=len(p['baseline_hashes']),
        git_status=subprocess.check_output(['git','-C',str(TREE),'status','--short','--branch']).decode())
    save('source_version.json',info)
    save('delivery_manifest.json',dict(source_version=info,artifacts={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file()}))
    print('SEALED',info['final_cells'],'final cells;',info['tests'],'tests;',len(p['hashes']),'frozen inputs;',len(p['baseline_hashes']),'baselines unchanged')
if __name__=='__main__':main()
