"""Rebuild the actual source, not just its bare Git HEAD; no source edits."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PARENT=OUT.parent/'joint_exchange_realization_20261026_run1'
TARGET=ROOT/'sia-joint-completion-time/sia-simulator'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
    assert not TARGET.parent.exists(),'refuse existing source tree'
    prior=json.loads((PARENT/'preregistration.json').read_text(encoding='utf-8'))
    for rel,h in prior['source_hashes'].items():assert sha(PARENT/'source'/rel)==h,rel
    live=ROOT/'sia-joint-service-horizon/sia-simulator'
    comparisons={str(p.relative_to(live)):sha(p)==sha(PARENT/'source'/p.relative_to(live))
                 for p in live.rglob('*.py')}
    assert all(comparisons.values()),comparisons
    shutil.copytree(PARENT/'source',TARGET,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(PARENT/'inputs',OUT/'inputs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['metrics_legacy_adapter.py','audit_experiment.py']:
        shutil.copy2(PARENT/name,OUT/name)
    (OUT/'baseline_tracked.patch').write_bytes(subprocess.check_output(['git','-C',str(live),'diff','--binary','HEAD']))
    shutil.make_archive(str(OUT/'baseline_actual_source'),'zip',TARGET.parent)
    save(OUT/'freeze.json',dict(created_unix=time.time(),source=str(TARGET),parent=str(PARENT),
        bare_head=subprocess.check_output(['git','-C',str(live),'rev-parse','HEAD'],text=True).strip(),
        parent_status=subprocess.check_output(['git','-C',str(live),'status','--short'],text=True),
        tracked_patch_sha256=sha(OUT/'baseline_tracked.patch'),
        snapshot_sha256=sha(OUT/'baseline_actual_source.zip'),live_py_count=len(comparisons),
        live_all_match=all(comparisons.values()),
        source_hashes={str(p.relative_to(TARGET)):sha(p) for p in TARGET.rglob('*') if p.is_file()},
        input_hashes={str(p.relative_to(OUT)):sha(p) for p in (OUT/'inputs').rglob('*') if p.is_file()}))
    with (OUT/'dependencies.lock.txt').open('x',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    print('FROZEN',TARGET,len(comparisons))
if __name__=='__main__':main()
