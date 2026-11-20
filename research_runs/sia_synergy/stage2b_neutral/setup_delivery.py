"""Isolate the explicitly approved Stage1.1 commit; preserve execution bytes."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SRC = ROOT / 'sia-joint-correctness-stage1p1'
DST = ROOT / 'sia-joint-stage2b-neutral'
BASE = '216573cb26f3136520ce9d4f609623b4c2c75df1'
BRANCH = 'exp/b3-stage2b-neutral'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()

def git(*args):
    return subprocess.check_output(['git', '-C', str(SRC), *args])

if __name__ == '__main__':
    assert not DST.exists() and not (OUT/'base_snapshot.zip').exists()
    assert git('rev-parse', 'HEAD').decode().strip() == BASE
    assert not git('diff', '--name-only', 'HEAD')
    names = git('ls-files', '-z').decode().split('\0')[:-1]
    extra = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')[:-1]
    assert not [n for n in extra if '__pycache__' not in n and not n.endswith('.pyc')]
    hashes = {n:sha(SRC/n) for n in names if (SRC/n).is_file()}
    protected = {}
    for folder in ['sia-joint-correctness-stage1p1', 'sia-joint-medium-frozen',
                   'official-artifacts', 'research_sources/Synergy', 'research_notes',
                   'research_runs/sia_synergy/correctness_stage1p1',
                   'research_runs/sia_synergy/b3_w10_late_ablation_2x2']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts:
                protected[str(p.relative_to(ROOT))] = sha(p)
    for name, args in [('tracked.patch',('diff','--binary','HEAD')),
                       ('index.patch',('diff','--cached','--binary')),
                       ('git_status_before.txt',('status','--short','--branch'))]:
        (OUT/name).write_bytes(git(*args))
    subprocess.run(['git','-C',str(SRC),'bundle','create',str(OUT/'base.bundle'),'HEAD'],check=True)
    with zipfile.ZipFile(OUT/'base_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in hashes:
            z.write(SRC/n,n)
    subprocess.run(['git','-C',str(SRC),'worktree','add','-b',BRANCH,str(DST),BASE],check=True)
    with zipfile.ZipFile(OUT/'base_snapshot.zip') as z:
        assert all((DST/n).resolve().is_relative_to(DST.resolve()) for n in z.namelist())
        z.extractall(DST)
    assert {n:sha(DST/n) for n in hashes} == hashes
    (OUT/'requirements.lock.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze','--all']))
    (OUT/'snapshot_manifest.json').write_text(json.dumps(dict(base_commit=BASE,branch=BRANCH,
        source=str(SRC),destination=str(DST),source_hashes=hashes,protected_hashes=protected,
        necessary_untracked=[],python=sys.version,byte_identical=True),indent=2),encoding='utf-8')
    print(json.dumps(dict(source_files=len(hashes),protected_files=len(protected),branch=BRANCH)))
