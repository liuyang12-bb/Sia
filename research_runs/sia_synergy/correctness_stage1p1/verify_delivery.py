"""Read-only package/source verification and evidence index. No simulation."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
WORK = ROOT/'sia-joint-correctness-stage1p1'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest().upper()

if __name__=='__main__':
    assert not (OUT/'delivery_manifest.json').exists()
    base=json.loads((OUT/'snapshot_manifest.json').read_text())
    repair=json.loads((OUT/'repair_manifest.json').read_text())
    zip_checks={}
    for name, manifest in [('source_snapshot.zip',base),('repair_source_snapshot.zip',repair)]:
        with zipfile.ZipFile(OUT/name) as z:
            expected=manifest['source_hashes']
            assert set(z.namelist())==set(expected)
            assert all(hashlib.sha256(z.read(n)).hexdigest().upper()==h for n,h in expected.items())
            zip_checks[name]=dict(files=len(expected),all_bytes_verified=True)
    verified=[]
    for name in ('base.bundle','repair.bundle'):
        r=subprocess.run(['git','-C',str(WORK),'bundle','verify',str(OUT/name)],capture_output=True)
        (OUT/(name+'.verify.log')).write_bytes(r.stdout+r.stderr)
        assert r.returncode==0
        verified.append(name)
    assert all(sha(ROOT/p)==h for p,h in base['protected_hashes'].items())
    assert all(sha(WORK/p)==h for p,h in repair['source_hashes'].items())
    assert not subprocess.check_output(['git','-C',str(WORK),'status','--porcelain'])
    red=json.loads((OUT/'red_evidence/bert_mid_full_chain.json').read_text())
    green=json.loads((OUT/'final_tests_evidence/bert_mid_full_chain.json').read_text())
    a=[x for x in red['trace'] if x['event']=='enumeration'][-1]
    b=[x for x in green['trace'] if x['event']=='enumeration'][-1]
    root_cause=dict(red_last_jobinfo=a, green_last_jobinfo=b,
        added_configs=sorted(set(b['sources'])-set(a['sources'])),
        status='LEGAL_PROJECTED_CONFIG_OMITTED_BY_LOCAL_INSTEAD_OF_SHARED_BOUNDS',
        same_local_bounds=a['max']==b['max'],green_states=green['states'],
        response_note='phi formula unchanged; repairing the candidate set can change its existing per-GPU reference minimum',
        old_min_per_gpu=min(v['pre_normalization_goodput']/int(k.split(':')[-1]) for k,v in a['sources'].items()),
        new_min_per_gpu=min(v['pre_normalization_goodput']/int(k.split(':')[-1]) for k,v in b['sources'].items()))
    (OUT/'root_cause.json').write_text(json.dumps(root_cause,indent=2),encoding='utf-8')
    tests={}
    for p in OUT.glob('*.log'):
        txt=p.read_text(encoding='utf-8',errors='replace')
        matches=re.findall(r'Ran (\d+) tests in ([\d.]+)s',txt)
        if matches:
            summary=json.loads(p.with_suffix('.json').read_text())
            tests[p.name]=dict(count=int(matches[-1][0]),test_seconds=float(matches[-1][1]),
                              exit_code=summary['exit_code'],wall_seconds=summary['wall_seconds'])
    result=dict(base_commit=repair['base_commit'],repair_commit=repair['repair_commit'],
        branch=repair['branch'],zip_checks=zip_checks,bundles_verified=verified,
        protected_files_verified=len(base['protected_hashes']),protected_unchanged=True,
        repaired_source_verified=True,worktree_clean=True,tests=tests,
        original_sia_regression=json.loads((OUT/'legacy_regression/comparison.json').read_text()),
        unresolved_b3_prop_empty=True,formal_experiments_run=False)
    (OUT/'validation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    report=ROOT/'research_notes/synergy/B3_CORRECTNESS_STAGE1P1_REPORT.md'
    files={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files[str(report.relative_to(ROOT))]=sha(report)
    (OUT/'delivery_manifest.json').write_text(json.dumps(dict(repair_commit=repair['repair_commit'],files=files,
        self_hash_excluded=True),indent=2),encoding='utf-8')
    print(json.dumps(dict(commit=repair['repair_commit'],protected_unchanged=True,tests=tests,
                         snapshot_sha256=sha(OUT/'repair_source_snapshot.zip'),patch_sha256=sha(OUT/'repair.patch'))))
