"""Final immutable source bundle/hash index and protected-state recheck."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]; TREE=ROOT/'sia-joint-stage2c-r0'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
    with p.open('x',encoding='utf-8') as f: json.dump(v,f,indent=2)

def main():
    baseline=json.loads((OUT/'snapshot_manifest.json').read_text(encoding='utf-8'))
    changed=[name for name,digest in baseline['protected_hashes'].items() if not (ROOT/name).is_file() or sha(ROOT/name)!=digest]
    names=subprocess.check_output(['git','-C',str(TREE),'ls-files','-z']).decode().split('\0')[:-1]
    sources={n:sha(TREE/n) for n in names}
    changed_sources=[n for n,h in baseline['source_hashes'].items() if sources.get(n)!=h]
    assert changed_sources==['sia-simulator/joint_prototype/neutral_liveness.py'] or set(changed_sources)=={'sia-simulator/joint_prototype/neutral_liveness.py'}
    assert not changed,changed
    with zipfile.ZipFile(OUT/'final_full_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in names: z.write(TREE/n,n)
    save(OUT/'source_manifest.json',dict(commit=subprocess.check_output(['git','-C',str(TREE),'rev-parse','HEAD']).decode().strip(),
        source_hashes=sources,protected_checked=len(baseline['protected_hashes']),protected_changed=changed,
        modified_existing_source=changed_sources,baseline_files_unchanged=len(baseline['source_hashes'])-len(changed_sources)))
    report=ROOT/'research_notes/synergy/B3_R0_W10_LATE_EIGHT_CELL_REPORT.md'
    save(OUT/'delivery_manifest.json',dict(report=str(report),report_sha256=sha(report),
        artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps(dict(files=len(sources),protected=len(baseline['protected_hashes']),changed=changed)))

if __name__=='__main__': main()
