"""Seal completed frozen execution without altering parent evidence."""
import datetime,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;PARENT=OUT.parent;ROOT=OUT.parents[3]
p=json.loads((OUT/'preregistration.json').read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def main():
    validation=json.loads((OUT/'validation_summary.json').read_text());extra=json.loads((OUT/'extra_validation.json').read_text())
    assert validation['valid'] and extra['valid'] and validation['complete_cells']==validation['planned_cells']==10
    for f,h in p['hashes'].items():assert sha(f)==h,f
    for f,h in p['baseline_hashes'].items():assert sha(f)==h,f
    for f,h in json.loads((PARENT/'delivery_manifest.json').read_text())['artifacts'].items():assert sha(PARENT/f)==h,f
    assert sha(OUT/'preregistration.json')==sha(PARENT/'preregistration.json')
    active=[]
    for proc in psutil.process_iter(['name','cmdline']):
        if 'python' not in (proc.info['name'] or '').lower():continue
        args=proc.info['cmdline'] or []
        if any(Path(arg).name in ['run_selective_cell.py','run_bundle_cell.py','run_full160_cell.py'] for arg in args):active.append(dict(pid=proc.pid,args=args))
    assert not active,active
    events=[json.loads(v) for v in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
    assert len(events)==10 and all(v['exit_code']==0 for v in events)
    source=p['tree'];actual_branch=subprocess.check_output(['git','-C',source,'branch','--show-current']).decode().strip();assert actual_branch==p['branch']
    # Actual local GPU CSVs are tracked pristine baseline copies, not only hashes of old paths.
    prior=ROOT/'sia-joint-budgeted-adaptation/sia-simulator';work=Path(source)/'sia-simulator';profile_hashes={}
    for f in (work/'traces').rglob('*.csv'):
        rel=f.relative_to(work);assert sha(f)==sha(prior/rel),str(rel)
        profile_hashes[rel.as_posix()]=sha(f)
    with (OUT/'actual_gpu_profile_hashes.json').open('x') as f:json.dump(profile_hashes,f,indent=2)
    now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(json.loads((OUT/'resume_manifest.json').read_text())['resume_utc'])
    info=dict(status='COMPLETE_FROZEN_TEN_CELLS',finished_utc=now.isoformat(),resume_elapsed_seconds=(now-start).total_seconds(),
        strategy_process_wall_seconds=sum(e['seconds'] for e in events),cells=10,tests_reused_from_identical_frozen_source=154,
        actual_tree=source,actual_branch=actual_branch,base_commit=p['base_commit'],new_code_committed=False,
        execution_source_zip_sha256=sha(PARENT/'execution_source.zip'),preregistration_sha256=sha(OUT/'preregistration.json'),
        source_input_hashes_verified=len(p['hashes']),history_files_verified=len(p['baseline_hashes']),
        local_gpu_csvs_equal_parent=len(profile_hashes),active_experiment_processes=active,
        python=sys.version,packages=p['packages'],formal104='PAUSED_NOT_RESUMED',
        pause_sha256=sha(ROOT/'research_runs/sia_synergy/sia_r0_full160_comparison/attempt2/USER_PAUSE_20261025.md'),
        git_status=subprocess.check_output(['git','-C',source,'status','--short','--branch']).decode())
    with (OUT/'source_version.json').open('x') as f:json.dump(info,f,indent=2)
    with zipfile.ZipFile(OUT/'resume_tools_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in OUT.glob('*.py'):z.write(f,f.name)
    with (OUT/'delivery_manifest.json').open('x') as f:json.dump(dict(source_version=info,artifacts={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file()}),f,indent=2)
    print('SEALED_TEN_CELLS',round(info['strategy_process_wall_seconds'],2),'process seconds;',info['local_gpu_csvs_equal_parent'],'local profiles verified',flush=True)
if __name__=='__main__':main()
