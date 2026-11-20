"""Local commit and reproducible package; no experiments and no remote writes."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
WORK = ROOT/'sia-joint-correctness-stage1p1'
BASE = '0461dcdf061197fdfffe23b2b3d07e46e0d9efb2'
EXPECTED = ['sia-simulator/joint_prototype/joint_integration.py',
            'sia-simulator/joint_prototype/joint_scheduler.py',
            'sia-simulator/joint_prototype/synergy_adapter.py',
            'sia-simulator/tests/test_stage1p1_correctness.py']

def git(*args):
    return subprocess.check_output(['git','-C',str(WORK),*args])

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest().upper()

def function(text, name):
    return ast.dump(next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name))

def main():
    assert git('rev-parse','HEAD').decode().strip()==BASE
    assert not (OUT/'repair_manifest.json').exists()
    snapshot = json.loads((OUT/'snapshot_manifest.json').read_text())
    current = {p:sha(WORK/p) for p in snapshot['source_hashes']}
    changed = [p for p in current if current[p]!=snapshot['source_hashes'][p]]
    assert sorted(changed)==sorted(EXPECTED[:-1]), changed
    assert not git('diff','--cached','--name-only')
    # Index stat entries may report LF byte-restored files as dirty although
    # Git's canonical diff is empty. Explicit paths only, byte-verified first.
    stat_only = []
    for line in git('status','--porcelain').decode().splitlines():
        p = line[3:]
        if p not in EXPECTED and line[:2]==' M':
            assert current[p]==snapshot['source_hashes'][p]
            assert not git('diff','--name-only','--',p)
            stat_only.append(p)
    git('add','--',*EXPECTED,*stat_only)
    assert sorted(git('diff','--cached','--name-only').decode().splitlines())==sorted(EXPECTED)
    git('diff','--cached','--check')
    subprocess.run(['git','-C',str(WORK),'commit','-m',
        'fix: align projected Sia candidates and requalify unlocked fallback columns'],check=True)
    commit = git('rev-parse','HEAD').decode().strip()
    assert not git('status','--porcelain')
    (OUT/'repair.patch').write_bytes(git('diff','--binary',BASE,commit))
    (OUT/'repair_diff_stat.txt').write_bytes(git('diff','--stat',BASE,commit))
    subprocess.run(['git','-C',str(WORK),'bundle','create',str(OUT/'repair.bundle'),'HEAD'],check=True)
    names = git('ls-files','-z').decode().split('\0')[:-1]
    hashes = {p:sha(WORK/p) for p in names if (WORK/p).is_file()}
    with zipfile.ZipFile(OUT/'repair_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in hashes:
            z.write(WORK/p,p)
    # Freeze external official response templates used by short tests.
    with zipfile.ZipFile(OUT/'external_response_templates.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT/'research_sources/Synergy/simulator/models').glob('*.py'):
            z.write(p,p.name)
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:
        before = z.read('sia-simulator/joint_prototype/joint_integration.py').decode('utf-8-sig')
    after = (WORK/'sia-simulator/joint_prototype/joint_integration.py').read_text(encoding='utf-8-sig')
    checks = {name:function(before,name)==function(after,name) for name in ('_b3_result','_restart_factor')}
    assert all(checks.values())
    untouched = ['sia-simulator/'+p for p in ('sia.py','job.py','cluster.py','goodput.py','simulator_config.py',
                 'joint_prototype/joint_profile.py','joint_prototype/execution_audit.py')]
    assert all(hashes[p]==snapshot['source_hashes'][p] for p in untouched)
    protected_changes = [p for p,h in snapshot['protected_hashes'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not protected_changes, protected_changes
    dep = dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','cvxpy','cylp','pandas','autograd')},
        engine_versions='CBC/HiGHS independently unqueried; no guessed engine versions',
        solver_parameters='byte-identical sia.py; JointScheduler.solve AST unchanged')
    (OUT/'dependencies.json').write_text(json.dumps(dep,indent=2),encoding='utf-8')
    manifest = dict(base_commit=BASE,repair_commit=commit,branch=git('branch','--show-current').decode().strip(),
        changed_files=EXPECTED,source_hashes=hashes,source_files=len(hashes),untracked_source=[],
        index_stat_only_refresh=stat_only,protected_files_verified=len(snapshot['protected_hashes']),
        protected_changes=protected_changes,unchanged_functions=checks,byte_identical_core=untouched,
        experiments_run=False,unresolved=['B3-PROP empty OPTIMAL absorbing state','liveness semantics','response boundary model'])
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:
        original = z.read('sia-simulator/joint_prototype/joint_scheduler.py').decode('utf-8-sig')
    new = (WORK/'sia-simulator/joint_prototype/joint_scheduler.py').read_text(encoding='utf-8-sig')
    manifest['solver_function_unchanged'] = function(original,'solve')==function(new,'solve')
    assert manifest['solver_function_unchanged']
    (OUT/'repair_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(commit=commit,changed=EXPECTED,protected_verified=manifest['protected_files_verified'])))

if __name__=='__main__':
    main()
