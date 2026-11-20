"""Archive the approved Stage1 state and create an isolated Stage1.1 checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SRC = ROOT / 'sia-joint-correctness-stage1'
DST = ROOT / 'sia-joint-correctness-stage1p1'
BASE = '0461dcdf061197fdfffe23b2b3d07e46e0d9efb2'
BRANCH = 'exp/b3-correctness-stage1p1'

def git(*args):
    return subprocess.check_output(['git', '-C', str(SRC), *args])

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest().upper()

if __name__ == '__main__':
    assert git('rev-parse', 'HEAD').decode().strip() == BASE
    if DST.exists() or (OUT / 'source_snapshot.zip').exists():
        raise RuntimeError('refuse to overwrite a checkout or snapshot')
    tracked = git('ls-files', '-z').decode().split('\0')[:-1]
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')[:-1]
    needed = [p for p in untracked if '__pycache__' not in p and not p.endswith('.pyc')]
    assert not git('diff', '--name-only', 'HEAD'), 'unexpected tracked changes: stop for review'
    assert not needed, 'unexpected source additions: stop for review'
    hashes = {p: sha(SRC/p) for p in tracked if (SRC/p).is_file()}
    (OUT/'tracked.patch').write_bytes(git('diff', '--binary', 'HEAD'))
    (OUT/'index.patch').write_bytes(git('diff', '--cached', '--binary'))
    (OUT/'git_status_before.txt').write_bytes(git('status', '--short', '--branch'))
    subprocess.run(['git', '-C', str(SRC), 'bundle', 'create', str(OUT/'base.bundle'), 'HEAD'], check=True)
    with zipfile.ZipFile(OUT/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(SRC/name, name)
    protected = {}
    for folder in ['sia-joint-correctness-stage1', 'sia-joint-medium-frozen', 'official-artifacts',
                   'research_sources/Synergy', 'research_notes',
                   'research_runs/sia_synergy/correctness_stage1',
                   'research_runs/sia_synergy/b3_w10_late_ablation_2x2']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts:
                protected[str(p.relative_to(ROOT))] = sha(p)
    subprocess.run(['git', '-C', str(SRC), 'worktree', 'add', '-b', BRANCH, str(DST), BASE], check=True)
    # Preserve exact Windows execution bytes, not only Git's normalized blob bytes.
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:
        assert all((DST/n).resolve().is_relative_to(DST.resolve()) for n in z.namelist())
        z.extractall(DST)
    assert {p: sha(DST/p) for p in hashes} == hashes
    (OUT/'requirements.lock.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all']))
    (OUT/'snapshot_manifest.json').write_text(json.dumps(dict(base_commit=BASE, branch=BRANCH,
        source=str(SRC), destination=str(DST), source_hashes=hashes, protected_hashes=protected,
        necessary_untracked=needed, excluded_generated=untracked, python=sys.version,
        byte_identical=True), indent=2), encoding='utf-8')
    print(json.dumps(dict(files=len(hashes), protected=len(protected), destination=str(DST))))
