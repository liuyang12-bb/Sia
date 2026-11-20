"""Resume only the frozen ten bounded cells into a fresh attempt; never 104."""
import datetime,hashlib,importlib.metadata,json,shutil,sys
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def main():
    p=json.loads((OUT/'preregistration.json').read_text())
    sealed=json.loads((OUT/'delivery_manifest.json').read_text())
    for name,h in sealed['artifacts'].items():assert sha(OUT/name)==h,name
    for name,h in p['hashes'].items():assert sha(name)==h,name
    for name,h in p['baseline_hashes'].items():assert sha(name)==h,name
    for pkg,version in p['packages'].items():assert importlib.metadata.version(pkg)==version,pkg
    active=[]
    for proc in psutil.process_iter(['name','cmdline']):
        if 'python' not in (proc.info['name'] or '').lower():continue
        args=proc.info['cmdline'] or []
        if any(Path(arg).name in ['run_selective_cell.py','run_bundle_cell.py','run_full160_cell.py'] for arg in args):
            active.append(dict(pid=proc.pid,args=args))
    assert not active,active
    memory=psutil.virtual_memory().available/1024**3;disk=psutil.disk_usage(str(OUT)).free/1024**3
    if memory<2:raise RuntimeError('MEMORY_GATE_2GiB')
    if disk<5:raise RuntimeError('DISK_GATE_5GiB')
    dest=OUT/'attempt2';dest.mkdir(exist_ok=False)
    for name in ['preregistration.json','run.py','audit.py']:shutil.copyfile(OUT/name,dest/name)
    assert sha(dest/'preregistration.json')==sha(OUT/'preregistration.json')
    receipt=dict(resume_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,
        memory_gib=memory,disk_gib=disk,source_input_hashes_verified=len(p['hashes']),
        history_hashes_verified=len(p['baseline_hashes']),prior_delivery_artifacts_verified=len(sealed['artifacts']),
        identical_preregistration_sha256=sha(OUT/'preregistration.json'),frozen_runs=p['runs'],active_processes=active,
        formal104='PAUSED_NOT_RESUMED',execution_source_zip=sha(OUT/'execution_source.zip'),
        note='Prior memory gate occurred before cluster creation, no training prefix exists; new attempt preserves root log and all old products')
    with (dest/'resume_manifest.json').open('x') as f:json.dump(receipt,f,indent=2)
    print('RESUME_VERIFIED',memory,'GiB;',len(p['runs']),'unchanged bounded cells;',len(p['hashes']),'hashes',flush=True)
if __name__=='__main__':main()
