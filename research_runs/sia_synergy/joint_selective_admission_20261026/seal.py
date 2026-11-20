import ast,datetime,difflib,hashlib,importlib.metadata,json,platform,subprocess,sys,zipfile
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];p=json.loads((OUT/'preregistration.json').read_text());TREE=Path(p['tree']);WORK=TREE/'sia-simulator'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def save(f,v):
    with (OUT/f).open('x',encoding='utf-8') as out:json.dump(v,out,indent=2)
def main():
    assert json.loads((OUT/'green_final.json').read_text())['exit_code']==0
    for f,h in p['hashes'].items():assert sha(f)==h,f
    for f,h in p['baseline_hashes'].items():assert sha(f)==h,f
    for f in OUT.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
    old=json.loads((OUT/'restored_source_hashes.json').read_text());changes=[];parent=ROOT/'sia-joint-budgeted-adaptation'
    for f in WORK.rglob('*.py'):
        rel=f.relative_to(TREE);prior=parent/rel
        if prior.exists() and sha(prior)==sha(f):continue
        changes.extend(difflib.unified_diff(prior.read_text(encoding='utf-8').splitlines(True) if prior.exists() else [],f.read_text(encoding='utf-8').splitlines(True),fromfile='parent/'+rel.as_posix(),tofile='new/'+rel.as_posix()))
    with (OUT/'selective_diff.patch').open('x',encoding='utf-8') as f:f.write(''.join(changes))
    with zipfile.ZipFile(OUT/'frozen_prerequisites.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in p['hashes']:z.write(f,Path(f).relative_to(ROOT))
    packages=sorted([dict(name=d.metadata['Name'],version=d.version) for d in importlib.metadata.distributions()],key=lambda v:v['name'].lower())
    save('dependencies.json',packages)
    with (OUT/'requirements-lock.txt').open('x') as f:f.write('\n'.join(v['name']+'=='+v['version'] for v in packages)+'\n')
    registry=json.loads(Path(p['registry']).read_text());hold=json.loads((OUT/'holdout_preregistration.json').read_text());apps=sorted({v['application'] for v in hold['records']})
    domain={a:sorted(v['config_id'] for v in registry if v['application']==a) for a in apps};assert all(len(v)==11 for v in domain.values())
    save('holdout_offline_coverage.json',dict(applications=apps,configs=domain,policy_independent_registry=True,
        no_simulation=True,no_new_phi=True,csv_hash=hold['sha256'],registry_hash=sha(p['registry']),source_trace_hash=hold['source_sha256']))
    active=[]
    for proc in psutil.process_iter(['name','cmdline']):
        cmd=' '.join(proc.info['cmdline'] or [])
        if proc.pid!=psutil.Process().pid and any(x in cmd for x in ['run_selective_cell.py','run_full160_cell.py','run_bundle_cell.py']):active.append(dict(pid=proc.pid,command=cmd))
    assert not active,active
    pause=ROOT/'research_runs/sia_synergy/sia_r0_full160_comparison/attempt2/USER_PAUSE_20261025.md'
    info=dict(status='ENGINEERING_VALIDATED_EXPERIMENT_BLOCKED_MEMORY',tree=str(TREE),branch=p['branch'],base_commit=p['base_commit'],new_code_committed=False,
        tests=154,original_tests=141,new_tests=13,static_matches=3,static_no_match=1,planned_cells=10,started_training_cells=0,
        last_memory_gib=1.800018310546875,last_disk_free_gib=22.53000259399414,
        current_memory_gib=psutil.virtual_memory().available/1024**3,active_experiment_processes=active,
        frozen_hashes_verified=len(p['hashes']),historical_files_verified=len(p['baseline_hashes']),
        python=sys.version,platform=platform.platform(),finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        formal104='PAUSED_NOT_RESUMED',pause_record_hash=sha(pause),
        git_status=subprocess.check_output(['git','-C',str(TREE),'status','--short','--branch']).decode())
    save('source_version.json',info)
    save('delivery_manifest.json',dict(source_version=info,artifacts={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file()}))
    print('SEALED engineering154, runtime0/10 memory-blocked;',len(p['hashes']),'input/source hashes verified; history',len(p['baseline_hashes']),'unchanged',flush=True)
if __name__=='__main__':main()
