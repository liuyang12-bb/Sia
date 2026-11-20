"""Resume only the frozen, never-started suffix; preserve all previous evidence."""
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'
RESUME = OUT / 'resume_20261028_run1'
RUNNER = ROOT / 'maintenance/run_cc_resource_factorial_20261027.py'
AUDITOR = ROOT / 'maintenance/audit_cc_resource_factorial_20261027.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def processes():
    import psutil
    result = []
    names = {RUNNER.name, Path(__file__).name}
    own_chain = {os.getpid()} | {p.pid for p in psutil.Process().parents()}
    for proc in psutil.process_iter(['cmdline']):
        try:
            args = proc.info['cmdline'] or []
            # Match actual script arguments, not strings inside a diagnostic -c.
            if proc.pid not in own_chain and '-c' not in args and any(Path(a).name in names for a in args[1:]):
                result.append({'pid': proc.pid, 'cmdline': args})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def main():
    runner = module('frozen_resource_runner', RUNNER)
    assert not RESUME.exists(), 'REFUSE_EXISTING_RESUME_DIRECTORY'
    conflicts = processes()
    assert not conflicts, ('DUPLICATE_PROCESS', conflicts)
    p = runner.setup()
    assert runner.allfiles(runner.TREE) == p['source_hashes'], 'WORKING_TREE_DIFFERS_FROM_FROZEN_SOURCE'
    prefix = json.loads((OUT / 'completed_prefix_hashes.json').read_text(encoding='utf-8'))
    records = [json.loads(x) for x in (OUT / 'execution_manifest.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    assert records == prefix['records'], 'EXECUTION_PREFIX_CHANGED'
    assert len(records) == 9
    assert p['cells'] == [list(x) for x in runner.CELLS], 'MATRIX_CHANGED'
    for name, files in prefix['cells'].items():
        for relative, expected in files.items():
            assert runner.sha(OUT / relative) == expected, relative
        for relative, expected in json.loads((OUT / 'cells' / name / 'output_hashes.json').read_text()).items():
            assert runner.sha(OUT / 'cells' / name / relative) == expected, relative
    for i, record in enumerate(records):
        assert record['index'] == i and record['exit_code'] == 0
        assert record['cell'] == runner.label(*runner.CELLS[i])
    for cell in runner.CELLS[len(records):]:
        name = runner.label(*cell)
        assert not (OUT / 'cells' / name).exists(), ('REFUSE_PARTIAL_OR_EXISTING_CELL', name)
        assert not (OUT / (name + '.log')).exists(), ('REFUSE_EXISTING_LOG', name)
    resources = runner.resources()
    RESUME.mkdir()
    protected = ['preregistration.json', 'completed_prefix_hashes.json', 'REPORT.md',
                 'independent_audit_and_comparison.json', 'USER_PAUSE_20261027.md', 'pause_summary.json', 'safe_stop.json']
    save(RESUME / 'preflight.json', {
        'time_utc': stamp(), 'status': 'VERIFIED', 'authorization': 'User explicitly requested completing the unfinished experiment.',
        'completed_prefix_cells': len(records), 'remaining_cells': p['cells'][len(records):],
        'resources': resources, 'duplicate_processes': conflicts,
        'protected_artifact_hashes': {n: runner.sha(OUT / n) for n in protected},
        'preregistration_sha256': runner.sha(OUT / 'preregistration.json'),
        'frozen_runner_sha256': runner.sha(RUNNER), 'auditor_sha256': runner.sha(AUDITOR),
        'resume_script_sha256': runner.sha(Path(__file__)), 'source_and_inputs_verified': True,
        'prefix_all_hashes_verified': True, 'algorithm_grid_response_parameters_changed': False,
    })
    current = OUT / 'CURRENT_TASK.md'
    previous = current.read_text(encoding='utf-8')
    current.write_text('最新状态：RESUMING_USER_AUTHORIZED。用户已明确授权续跑未完成21格。'
                       '原9格与冻结源码/输入哈希已核验，原暂停记录保留为历史。'
                       '原顺序、独立进程串行、2GiB/5GiB门槛不变。\n'
                       '恢复入口：maintenance/resume_cc_resource_factorial_20261028.py；'
                       '恢复证据：resume_20261028_run1/。不得重复启动。\n\n' + previous,
                       encoding='utf-8')
    auditor = module('frozen_resource_auditor', AUDITOR)
    started = time.perf_counter()
    print('VERIFIED_PREFIX', len(records), 'REMAINING', 30-len(records), resources, flush=True)
    for i, (scenario, response, method) in enumerate(runner.CELLS[len(records):], start=len(records)):
        name = runner.label(scenario, response, method)
        try:
            resources = runner.resources()
        except Exception as exc:
            save(RESUME / 'safe_stop.json', {'time_utc': stamp(), 'next_index': i, 'next_cell': name,
                                           'error': repr(exc), 'no_cell_started': True})
            print('SAFE_STOP', name, repr(exc), flush=True)
            return
        assert not (OUT / 'cells' / name).exists()
        command = [sys.executable, '-X', 'utf8', '-B', str(RUNNER), 'run',
                   '--scenario', scenario, '--response', response, '--method', method]
        save(RESUME / (name + '_launch.json'), {'time_utc': stamp(), 'index': i,
                                               'command': command, 'resources': resources})
        start = time.perf_counter()
        print('START', i+1, '/30', name, resources, flush=True)
        with (OUT / (name + '.log')).open('x', encoding='utf-8') as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = {'index': i, 'cell': name, 'exit_code': child.returncode,
                  'wall_seconds': time.perf_counter()-start, 'resources': resources,
                  'resume_id': RESUME.name}
        with (OUT / 'execution_manifest.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        save(RESUME / (name + '_exit.json'), record)
        if child.returncode:
            save(RESUME / 'failure.json', {'time_utc': stamp(), 'record': record,
                                          'reason': 'CHILD_FAILED_BATCH_STOPPED_NO_RETRY'})
            print('FAILED', name, child.returncode, flush=True)
            return
        try:
            result = auditor.audit(OUT / 'cells' / name, p)
            save(RESUME / (name + '_independent_audit.json'), result)
        except Exception as exc:
            save(RESUME / 'failure.json', {'time_utc': stamp(), 'record': record,
                                          'reason': 'INDEPENDENT_AUDIT_FAILED', 'error': repr(exc),
                                          'traceback': traceback.format_exc()})
            print('AUDIT_FAILED', name, repr(exc), flush=True)
            return
        summary = result['summary']
        print('CELL_END', i+1, '/30', name, 'COMPLETED', summary['completed'],
              'MEAN', summary['mean_jct'], 'AUDIT_VALID', result['valid'], flush=True)
    # Original partial report and audit remain unchanged. A separate final analysis follows.
    for name, expected in json.loads((RESUME / 'preflight.json').read_text())['protected_artifact_hashes'].items():
        assert runner.sha(OUT / name) == expected, ('PROTECTED_ARTIFACT_CHANGED', name)
    for files in prefix['cells'].values():
        for relative, expected in files.items():
            assert runner.sha(OUT / relative) == expected, relative
    save(RESUME / 'execution_complete.json', {'time_utc': stamp(), 'completed_new_cells': 21,
                                            'matrix_cells_with_summary': 30,
                                            'resume_wall_seconds': time.perf_counter()-started,
                                            'frozen_prefix_preserved': True,
                                            'status': 'EXECUTION_AND_PER_CELL_AUDIT_COMPLETE'})
    print('EXECUTION_COMPLETE_30', flush=True)


if __name__ == '__main__':
    main()
