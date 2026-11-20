"""Independently recompute R0 selected rows and freeze all delivered evidence."""
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
WORK=ROOT/'sia-joint-stage2b-neutral'
REPORT=ROOT/'research_notes/synergy/B3_STAGE2B_NEUTRAL_LIVENESS_REPORT.md'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def dump(p,v):
    with p.open('x',encoding='utf-8') as f: json.dump(v,f,indent=2,allow_nan=False)

registry=json.loads((OUT/'g0_verified/registry.json').read_text())
rates={}
for item in registry:
    times=[]
    for row in item['source_rows']:
        path=Path(row['file'])
        assert sha(path)==row['sha256']
        with path.open(newline='',encoding='utf-8') as f:
            source=list(csv.DictReader(f))[row['line']-2]
        assert float(source['local_bsz'])==item['b_ref']
        assert float(source['step_time'])==row['step_time']
        times.append(float(source['step_time']))
    rate=item['b_ref']/statistics.median(times)
    assert math.isclose(rate,item['per_gpu_T0'],rel_tol=1e-14)
    rates[item['key']]=rate
for item in reversed(registry):
    denom=min(rates[key] for key in item['denominator_domain'])
    ratio=rates[item['key']]/denom
    assert math.isclose(item['phi_c'],min(4,max(.5,ratio)),rel_tol=1e-14)
    assert math.isclose(item['phi_m'],min(2,max(.5,math.sqrt(ratio))),rel_tol=1e-14)
    assert item['denominator']==denom

base=json.loads((OUT/'snapshot_manifest.json').read_text())
changed=[n for n,h in base['protected_hashes'].items() if sha(ROOT/n)!=h]
assert not changed
repair=json.loads((OUT/'repair_manifest.json').read_text())
assert all(sha(WORK/n)==h for n,h in repair['source_hashes'].items())
acceptance=json.loads((OUT/'acceptance_tests.json').read_text())
assert acceptance['exit_code']==0 and acceptance['commit']==repair['commit']
assert all(sha(WORK/'sia-simulator'/n)==h for n,h in acceptance['source_hashes'].items())
log=(OUT/'acceptance_tests.log').read_text(encoding='utf-8')
assert re.search(r'Ran 88 tests',log) and re.search(r'^OK$',log,re.M)
legacy=json.loads((OUT/'legacy_regression/comparison.json').read_text())
assert legacy['exact_state_match']
metrics=json.loads((OUT/'mechanism_summary.json').read_text())
assert metrics['all_runtime_audits_valid']
versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','cvxpy','cylp','pandas','autograd','psutil')}
dump(OUT/'validation_summary.json',dict(status='PASS_STAGE2B_NEUTRAL_SMALL_ONLY',
    base_commit=base['base_commit'],commit=repair['commit'],tests=88,historical_tests=71,new_tests=17,
    protected_files_checked=len(base['protected_hashes']),protected_changed=changed,
    unchanged_historical_source_files=len(base['source_hashes']),
    registry_recomputed_entries=len(registry),registry_source_row_hashes_valid=True,
    phi_recomputed_in_reverse_iteration_order=True,original_sia_bounded_exact=True,
    runtime_capacity_lock_rho_time_violations=0,
    historical_empty_optimal_preserved=True,nonneutral_registry_connected=False,
    pending='E-strict removes all proportional configs; non-Neutral boundary choice requires review. Scope guard prohibits >3 active jobs.',
    no_formal_experiments=True,python=platform.python_version(),platform=platform.platform(),dependencies=versions,
    CBC_engine_version='not independently queried',HiGHS_engine_version='not independently queried; SciPy package version recorded'))
dump(OUT/'run_manifest.json',dict(base_commit=base['base_commit'],commit=repair['commit'],
    scope='OFFLINE_G0_AND_NEUTRAL_MAX_3_JOB_MECHANISMS',
    worktree=str(WORK),branch='exp/b3-stage2b-neutral',report=str(REPORT),
    commands={p.stem:json.loads(p.read_text()) for p in sorted(OUT.glob('*.json'))
              if p.stem in ('red','green1','green2','boundary_red','green_full','final_tests','acceptance_tests')},
    offline_command=['.venv-sia-baseline/Scripts/python.exe','-B',
                     'sia-joint-stage2b-neutral/sia-simulator/stage2b_preflight.py','--output',
                     'research_runs/sia_synergy/stage2b_neutral/g0_verified','--models',
                     'research_sources/Synergy/simulator/models'],
    original_sia_regression=legacy['commands'],
    g0_manifest_hash=sha(OUT/'g0_verified/preflight_summary.json'),
    registry_hash=sha(OUT/'g0_verified/registry.json'),
    neutral_config_hash=sha(WORK/'sia-simulator/joint_prototype/stage2b_neutral_config.json'),
    dependency_lock_hash=sha(OUT/'requirements.lock.txt'),observations_extended=False))
files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
dump(OUT/'delivery_manifest.json',dict(files=files,report=str(REPORT),report_sha256=sha(REPORT)))
print(json.dumps(dict(status='PASS',registry_entries=len(registry),protected_changed=changed,tests=88,artifacts=len(files))))
