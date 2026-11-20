"""Append missing historical test fixtures; restore into a new directory and test.
No production source, original fixture or old archive is edited.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import zipfile
from package_results import OUT, ROOT, SOURCE, files, sha, save


def main():
    import psutil
    assert psutil.virtual_memory().available >= 2 * 2**30, 'MEMORY_GATE'
    assert psutil.disk_usage(str(OUT)).free >= 5 * 2**30, 'DISK_GATE'
    first = json.loads((OUT / 'delivery_manifest.json').read_text(encoding='utf-8'))
    archive = OUT / 'rebuildable_source_inputs_complete.zip'
    recovered = OUT / 'rebuild_validation_workspace'
    assert not archive.exists() and not recovered.exists(), 'NO_OVERWRITE'
    extra = set(files(ROOT / 'official-artifacts/sia-simulator'))
    extra |= set(files(ROOT / 'research_sources/Synergy/simulator/models'))
    extra |= set(files(OUT.parent / 'joint_opportunity_experiments_20261026_run1/inputs'))
    extra |= {
        OUT.parent / 'small_mechanism_20261025/preregistration.json',
        OUT.parent / 'small_mechanism_20261025/workload.csv',
        OUT.parent / 'sia_r0_full160_comparison/inputs/config.json',
        OUT.parent / 'sia_r0_full160_comparison/inputs/registry.json',
        OUT.parent / 'stage2b_neutral/g0_verified/registry.json',
        OUT.parent / 'b3_multiworkload_medium/followup_w10_late_6000/frozen_config_6000.json',
        OUT / 'REPRODUCE_APPENDIX.md', Path(__file__),
    }
    assert all(p.is_file() for p in extra)
    archive_extra = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(extra)}
    with zipfile.ZipFile(OUT / 'rebuildable_source_inputs.zip') as initial:
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as complete:
            names = set(initial.namelist())
            for entry in initial.infolist():
                with initial.open(entry) as inp, complete.open(entry.filename, 'w') as output:
                    shutil.copyfileobj(inp, output, 1024 * 1024)
            for path in sorted(extra):
                name = path.relative_to(ROOT).as_posix()
                if name in names:
                    import hashlib
                    assert hashlib.sha256(initial.read(name)).hexdigest().upper() == sha(path)
                else:
                    complete.write(path, name)
    recovered.mkdir()
    with zipfile.ZipFile(archive) as complete:
        assert complete.testzip() is None
        for member in complete.infolist():
            resolved = (recovered / member.filename).resolve()
            assert recovered.resolve() in resolved.parents, 'UNSAFE_ARCHIVE_PATH'
        complete.extractall(recovered)
    restored_source = recovered / 'sia-joint-completion-time/sia-simulator'
    assert all(sha(restored_source / rel) == expected for rel, expected in first['source_hashes'].items())
    fixture = recovered / 'research_runs/sia_synergy/small_mechanism_20261025/preregistration.json'
    original_hash = sha(fixture)
    data = json.loads(fixture.read_text(encoding='utf-8'))
    remapped = {}
    for key in ['config', 'registry', 'models', 'workload']:
        old = Path(data[key])
        new = recovered / old.relative_to(ROOT)
        assert new.exists(), ('MISSING_RECOVERY_FIXTURE', key, str(new))
        remapped[key] = dict(original=str(old), restored=str(new))
        data[key] = str(new)
    # This is a generated recovery copy, never the original historical fixture.
    fixture.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    start = time.perf_counter()
    log = OUT / 'test_rebuilt_full_suite.log'
    with log.open('xb') as stream:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest',
            'discover', '-s', 'tests', '-p', 'test_*.py', '-v'], cwd=restored_source,
            stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    elapsed = time.perf_counter() - start
    text = log.read_text(encoding='utf-8', errors='replace')
    import re
    passed = result.returncode == 0 and bool(re.search(r'^OK\s*$', text, re.M)) and 'Ran 188 tests' in text
    # Check copies and originals, not only test status.
    unchanged = all(sha(SOURCE / rel) == expected for rel, expected in first['source_hashes'].items())
    original_fixture = ROOT / fixture.relative_to(recovered)
    assert sha(original_fixture) == original_hash, 'ORIGINAL_FIXTURE_CHANGED'
    final = dict(status=first['status'], source_fingerprint=first['source_fingerprint'],
        baseline_manifest='delivery_manifest.json', baseline_manifest_sha256=sha(OUT / 'delivery_manifest.json'),
        complete_archive=str(archive), complete_archive_sha256=sha(archive),
        complete_archive_bytes=archive.stat().st_size, added_fixture_hashes=archive_extra,
        recovered_source_all_match=True, live_source_unchanged=unchanged,
        recovered_fixture_path_mapping=remapped, original_fixture_unchanged=True,
        original_fixture_sha256=original_hash, recovered_fixture_sha256=sha(fixture),
        recovered_tests=dict(passed=passed, exit_code=result.returncode, expected_count=188,
            wall_seconds=elapsed, command='python -X utf8 -B -m unittest discover -s tests -p test_*.py -v',
            cwd=str(restored_source), log=str(log), log_sha256=sha(log)),
        report='REPORT.md', report_sha256=sha(OUT / 'REPORT.md'),
        appendix='REPRODUCE_APPENDIX.md', appendix_sha256=sha(OUT / 'REPRODUCE_APPENDIX.md'),
        no_full_training_experiment=True, all_historical_outputs_preserved=True)
    save('final_delivery_manifest.json', final)
    print(json.dumps(dict(recovered_tests_passed=passed, exit_code=result.returncode,
        wall_seconds=elapsed, archive_sha256=sha(archive))))
    if not passed or not unchanged:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
