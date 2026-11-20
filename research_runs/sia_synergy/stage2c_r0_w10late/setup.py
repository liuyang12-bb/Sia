"""Snapshot Stage2B and create the authorized isolated R0 experiment branch."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]
SRC=ROOT/'sia-joint-stage2b-neutral'; DST=ROOT/'sia-joint-stage2c-r0'
BASE='bd6105a37c6662336236a135eb7ad610ca6e16de'
def git(*a): return subprocess.check_output(['git','-C',str(SRC),*a])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
assert not DST.exists() and not (OUT/'source_base.zip').exists()
assert git('rev-parse','HEAD').decode().strip()==BASE
assert not git('diff','--name-only','HEAD')
names=git('ls-files','-z').decode().split('\0')[:-1]
hashes={n:sha(SRC/n) for n in names if (SRC/n).is_file()}
(OUT/'base.patch').write_bytes(git('diff','--binary','HEAD'))
(OUT/'base_index.patch').write_bytes(git('diff','--cached','--binary'))
subprocess.run(['git','-C',str(SRC),'bundle','create',str(OUT/'base.bundle'),'HEAD'],check=True)
with zipfile.ZipFile(OUT/'source_base.zip','x',zipfile.ZIP_DEFLATED) as z:
    for n in hashes: z.write(SRC/n,n)
subprocess.run(['git','-C',str(SRC),'worktree','add','-b','exp/b3-stage2c-r0-w10late',str(DST),BASE],check=True)
with zipfile.ZipFile(OUT/'source_base.zip') as z:
    assert all((DST/n).resolve().is_relative_to(DST.resolve()) for n in z.namelist())
    z.extractall(DST)
assert {n:sha(DST/n) for n in hashes}==hashes
(OUT/'requirements.lock.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze','--all']))
protected={}
for folder in ['sia-joint-stage2b-neutral','sia-joint-correctness-stage1p1','official-artifacts',
               'research_sources/Synergy','research_notes','research_runs/sia_synergy/stage2b_neutral']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts: protected[str(p.relative_to(ROOT))]=sha(p)
(OUT/'snapshot_manifest.json').write_text(json.dumps(dict(base=BASE,source_hashes=hashes,protected_hashes=protected),indent=2),encoding='utf-8')
print(json.dumps(dict(source_files=len(hashes),protected=len(protected),worktree=str(DST))))
