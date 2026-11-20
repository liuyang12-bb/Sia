"""Byte-preserving accepted-source isolation, without modifying old trees."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]
SRC=ROOT/'sia-joint-stage2c-r0'; DST=ROOT/'sia-joint-full160'
BASE='141138d95fc51755e2f71c15f01964b4c20757da'
def git(*a): return subprocess.check_output(['git','-C',str(SRC),*a])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
assert not DST.exists() and not (OUT/'source_base.zip').exists()
assert git('rev-parse','HEAD').decode().strip()==BASE
expected=json.loads((OUT.parent/'stage2c_r0_w10late/source_manifest.json').read_text())['source_hashes']
assert all(sha(SRC/n)==h for n,h in expected.items()), 'ACCEPTED_SOURCE_HASH_MISMATCH'
untracked=git('ls-files','--others','--exclude-standard','-z').decode().split('\0')[:-1]
assert not untracked, untracked
assert not git('diff','--name-only','HEAD'), 'canonical content changed'
(OUT/'base.patch').write_bytes(git('diff','--binary','HEAD'))
(OUT/'base_index.patch').write_bytes(git('diff','--cached','--binary'))
(OUT/'base_status.txt').write_bytes(git('status','--short','--branch'))
subprocess.run(['git','-C',str(SRC),'bundle','create',str(OUT/'base.bundle'),'HEAD'],check=True)
with zipfile.ZipFile(OUT/'source_base.zip','x',zipfile.ZIP_DEFLATED) as z:
    for n in expected: z.write(SRC/n,n)
subprocess.run(['git','-C',str(SRC),'worktree','add','-b','exp/sia-r0-full160-comparison',str(DST),BASE],check=True)
with zipfile.ZipFile(OUT/'source_base.zip') as z:
    assert all((DST/n).resolve().is_relative_to(DST.resolve()) for n in z.namelist())
    z.extractall(DST)
assert {n:sha(DST/n) for n in expected}==expected
(OUT/'requirements.lock.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze','--all']))
protected={}
for folder in ['sia-joint-stage2c-r0','official-artifacts','research_sources/Synergy','research_notes',
               'research_runs/sia_synergy/stage2c_r0_w10late']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts: protected[str(p.relative_to(ROOT))]=sha(p)
(OUT/'snapshot_manifest.json').write_text(json.dumps(dict(base=BASE,source_hashes=expected,protected_hashes=protected,untracked=untracked),indent=2),encoding='utf-8')
print(json.dumps(dict(source_files=len(expected),protected=len(protected),worktree=str(DST))))
