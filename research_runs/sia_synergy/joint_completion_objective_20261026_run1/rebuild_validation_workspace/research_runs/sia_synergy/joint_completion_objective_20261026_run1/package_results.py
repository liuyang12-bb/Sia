"""Read-only source verification and exclusive-create reproducibility packaging."""
from pathlib import Path
import difflib
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'sia-joint-completion-time/sia-simulator'
PRIOR = OUT.parent / 'joint_exchange_realization_20261026_run1'
RUNS = OUT.parent / 'joint_opportunity_experiments_20261026_run1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()


def files(root):
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix != '.pyc')


def save(name, value):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def main():
    assert not (OUT / 'delivery_manifest.json').exists(), 'NO_OVERWRITE'
    assert not (OUT / 'rebuildable_source_inputs.zip').exists(), 'NO_OVERWRITE'
    freeze = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    unchanged = {rel: sha(SOURCE / rel) == expected
                 for rel, expected in freeze['source_hashes'].items()}
    assert all(unchanged.values()), 'LEGACY_SOURCE_CHANGED'
    inputs = {rel: sha(OUT / rel) == expected
              for rel, expected in freeze['input_hashes'].items()}
    assert all(inputs.values()), 'FROZEN_INPUT_CHANGED'
    live = ROOT / 'sia-joint-service-horizon/sia-simulator'
    live_matches = {str(p.relative_to(live)): sha(p) == sha(PRIOR / 'source' / p.relative_to(live))
                    for p in live.rglob('*.py')}
    assert all(live_matches.values()), 'PARENT_ACTUAL_SOURCE_CHANGED'
    legacy_names = {Path(rel).as_posix() for rel in freeze['source_hashes']}
    new_names = {p.relative_to(SOURCE).as_posix() for p in files(SOURCE)} - legacy_names
    assert new_names == {'joint_prototype/completion_time.py', 'tests/test_completion_time.py'}, new_names
    diff = []
    for rel in sorted(new_names):
        diff.extend(difflib.unified_diff([], (SOURCE / rel).read_text(encoding='utf-8').splitlines(True),
                    fromfile='/dev/null', tofile='b/sia-simulator/' + rel))
    with (OUT / 'minimal_change.diff').open('x', encoding='utf-8', newline='') as stream:
        stream.writelines(diff)

    raw = (OUT / 'test_final_full_suite.log').read_bytes()
    log = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8', errors='replace')
    match = re.search(r'Ran 188 tests in ([0-9.]+)s', log)
    assert match and re.search(r'^OK\s*$', log, re.M), 'FINAL_SUITE_NOT_PASSED'
    audit = json.loads((OUT / 'controlled_independent_audit.json').read_text())
    assert audit['valid'] and audit['capacity_violations'] == 0
    source_hashes = {p.relative_to(SOURCE).as_posix(): sha(p) for p in files(SOURCE)}
    source_fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest().upper()
    final_test = dict(tests=188, passed=188, legacy=173, new=15,
                     test_seconds=float(match.group(1)), process_seconds=59.07515,
                     command='python -X utf8 -B -m unittest discover -s tests -p test_*.py -v',
                     cwd=str(SOURCE), exit_code=0, source_fingerprint=source_fingerprint,
                     log='test_final_full_suite.log', log_sha256=sha(OUT / 'test_final_full_suite.log'),
                     scope='Bounded correctness tests; no full160 or complete JCT performance run')
    save('test_final_results.json', final_test)
    save('performance_gate.json', dict(status='PERFORMANCE_GATE_NOT_PASSED',
        checks=dict(engineering='PASS', clean_fixed_gpu_cpu_exchange='PASS',
                    sustained_actual_online_gpu_coupling='NOT_ESTABLISHED',
                    complete_clean_response_domain='FAIL',
                    small_net_completion_prediction_robust_to_observed_error='FAIL'),
        full_training_comparisons=[dict(scenario=scenario, policy=policy, status='NOT_STARTED_GATE',
            average_jct=None) for scenario in ['D2_existing', 'D3_existing_cpu_symmetric']
            for policy in ['SIA-P', 'SIA-ECT-AUX', 'JOINT-ECT', 'JOINT-ECT-P']],
        no_new_jct_claim=True, no_imputation=True, no_parameter_search=True))

    # Include precisely the external evidence read by static diagnostics, under
    # original project-relative paths. Never alter or repair historical logs.
    external = {
        PRIOR / 'training_probe_result.json', PRIOR / 'same_state_diagnostic.json',
        RUNS / 'cells/D2_gpu_auxiliary_coupling_medium_sia_p/rounds.jsonl.gz',
        RUNS / 'cells/D3_queued_training_proposal_cpu_symmetric_sia_p/rounds.jsonl.gz',
        ROOT / 'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv',
    }
    provenance = json.loads((OUT / 'offline_provenance_verification.json').read_text(encoding='utf-8'))
    for row in provenance['checks']:
        if 'path' in row:
            path = Path(row['path'])
            expected = row.get('expected', row.get('official_sha256'))
            if expected:
                assert sha(path) == expected, ('EXTERNAL_SOURCE_CHANGED', str(path))
            external.add(path)
    external_hashes = {str(p): sha(p) for p in sorted(external)}
    save('external_evidence_manifest.json', dict(hashes=external_hashes,
        note='Original project-relative paths archived; source absolute provenance strings intentionally retained.'))
    archive_files = set(files(SOURCE)) | set(files(OUT)) | external
    # Baseline zip already lives in OUT and is included; never include this new zip itself.
    archive = OUT / 'rebuildable_source_inputs.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(archive_files):
            zipped.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None, 'ARCHIVE_CRC_FAILURE'
    artifact_hashes = {p.relative_to(OUT).as_posix(): sha(p) for p in files(OUT)}
    finished = time.time()
    status = subprocess.check_output(['git', '-C', str(live), 'status', '--short'], text=True)
    save('delivery_manifest.json', dict(
        status='COMPLETE_ENGINEERING_VALID_PERFORMANCE_GATE_NOT_PASSED',
        warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE',
        source=str(SOURCE), source_fingerprint=source_fingerprint, source_hashes=source_hashes,
        unchanged_baseline_file_count=len(unchanged), all_baseline_files_unchanged=all(unchanged.values()),
        parent_live_python_count=len(live_matches), parent_live_all_unchanged=all(live_matches.values()),
        new_source_files=sorted(new_names), frozen_input_hashes=freeze['input_hashes'],
        baseline_bare_head=freeze['bare_head'], bare_head_not_actual_source=True,
        full_snapshot_not_git_commit=True, pushed=False, baseline_parent_status_at_delivery=status,
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        final_tests=final_test, controlled_training_seconds=600, common_warmup_seconds=240,
        controlled_complete_jobs=0, jct_performance_started=False,
        archive=str(archive), archive_sha256=sha(archive), artifact_hashes=artifact_hashes,
        external_evidence_hashes=external_hashes, started_unix=freeze['created_unix'], finished_unix=finished,
        elapsed_wall_seconds_since_baseline_freeze=finished-freeze['created_unix'],
        elapsed_definition='Whole engineering/review interval, not simulation compute time',
        stop_reason='No established supported online GPU coupling; rate error reverses tiny predicted net ETA gain',
        historical_full160_paused=True))
    print(json.dumps(dict(status='PACKAGED', new_source=sorted(new_names),
        source_fingerprint=source_fingerprint, archive_bytes=archive.stat().st_size,
        archive_sha256=sha(archive), tests=188, full_performance_started=False)))


if __name__ == '__main__':
    main()
