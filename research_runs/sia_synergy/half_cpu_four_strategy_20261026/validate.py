"""Read only raw round records; independently check all completed cells."""
import collections
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PLAN = json.loads((OUT / 'preregistration.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(Path(PLAN['tree']) / 'sia-simulator'))
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def check(cell, folder=None):
    folder = Path(folder) if folder is not None else OUT / cell['id']
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    oracle = R0Oracle(PLAN['registry'], PLAN['models'],
                      {job: job.rsplit('-', 1)[0] for job in summary['jobs']}, 'medium')
    caps = {'aws': (4, 6., 250.), 'dgx-ext': (8, 12., 500.), 'rtx': (8, 12., 500.)}
    time = collections.defaultdict(lambda: dict(wait=0., checkpoint=0., training=0.,
                                    gpu_hours=0., cpu_core_hours=0., ram_gb_hours=0.,
                                    completion=None, progress=0.))
    status = collections.Counter()
    clipping = collections.Counter()
    counts = collections.Counter()
    last_round = None
    with gzip.open(folder / 'rounds.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            decision = record['decision_for_next_interval']
            selected = decision['result']['selected']
            for event in record['execution']:
                assert audit_execution(event)['valid']
                job = event['job_id']
                row = time[job]
                assert close(row['progress'], event['progress_before'])
                assert event['progress_after'] + 1e-8 >= event['progress_before']
                row['progress'] = event['progress_after']
                row['completion'] = event['completion_time']
                row['wait'] += event['waiting_seconds']
                row['checkpoint'] += event['checkpoint_delay_consumed']
                row['training'] += event['training_seconds']
                resource = event['resource_allocation']
                if resource:
                    rho = independent_rho(oracle, job, resource['config_id'],
                                          resource['cpu_per_gpu'], resource['ram_gb_per_gpu'])
                    assert close(event['rho'], rho)
                    for key, multiplier in [('gpu_hours', 1.),
                                            ('cpu_core_hours', resource['cpu_per_gpu']),
                                            ('ram_gb_hours', resource['ram_gb_per_gpu'])]:
                        row[key] += resource['gpus'] * multiplier * event['allocated_seconds'] / 3600
                counts['execution_records'] += 1
            if decision['round'] == last_round:
                continue
            last_round = decision['round']
            assert decision['result']['audit']['valid']
            status[decision['result']['status']] += 1
            counts['decisions'] += 1
            usage = {name: [0., 0., 0.] for name in caps}
            for job, column in selected.items():
                config = column['config']
                gpu_type = config['gpu_type']
                rho = independent_rho(oracle, job, config['config_id'],
                                      column['cpu_per_gpu'], column['ram_gb_per_gpu'])
                assert close(column['rho'], rho)
                if cell['method'] == 'sia_p':
                    assert close(column['cpu_per_gpu'], 1.5)
                    assert close(column['ram_gb_per_gpu'], 62.5)
                for node, gpus in column['placement'].items():
                    assert node == gpu_type + ':0'
                    usage[gpu_type][0] += gpus
                    usage[gpu_type][1] += gpus * column['cpu_per_gpu']
                    usage[gpu_type][2] += gpus * column['ram_gb_per_gpu']
                source = column['source_coordinates']['requested']
                base = column['baseline_source_coordinates']['requested']
                for label, point in [('numerator', source), ('denominator', base)]:
                    clipping[label + '_cpu'] += int(not 1 <= point['cpu'] <= 24)
                    clipping[label + '_ram'] += int(not 20.83 <= point['ram_gb'] <= 250)
                counts['selected_columns'] += 1
            for name, allocated in usage.items():
                limits = caps[name]
                assert all(used <= limit + 1e-7 for used, limit in zip(allocated, limits))
                counts['node_capacity_checks'] += 1
                counts['cpu_saturated_gpu_idle'] += int(
                    allocated[1] >= limits[1] - 1e-7 and allocated[0] < limits[0])
            for spec in decision['probe']['specs']:
                if not spec['checkpoint_locked'] or not spec['current_config_id']:
                    continue
                col = selected[spec['job_id']]
                assert col['config']['config_id'] == spec['current_config_id']
                assert close(col['cpu_per_gpu'], spec['current_cpu_per_gpu'])
                assert close(col['ram_gb_per_gpu'], spec['current_ram_gb_per_gpu'])
                assert list(map(list, col['placement'].items())) == spec['current_placement']
                counts['checkpoint_lock_checks'] += 1
    completed_jct = []
    assert len(time) == summary['total']
    for job, recorded in time.items():
        expected = summary['jobs'][job]
        for key in ('wait', 'checkpoint', 'training', 'gpu_hours', 'cpu_core_hours', 'ram_gb_hours'):
            assert close(recorded[key], expected[key]), (job, key)
        assert recorded['completion'] == expected['completion']
        if recorded['completion'] is not None:
            jct = recorded['completion'] - expected['arrival']
            assert close(jct, recorded['wait'] + recorded['checkpoint'] + recorded['training'])
            completed_jct.append(jct)
    assert len(completed_jct) == summary['completed']
    if len(completed_jct) == summary['total']:
        mean = sum(completed_jct) / len(completed_jct)
        assert close(mean, summary['mean_jct'])
    else:
        assert summary['mean_jct'] is None
        mean = None
    result = dict(valid=True, cell=cell, completed=len(completed_jct), total=summary['total'],
                  independent_mean_jct=mean, counts=dict(counts), statuses=dict(status),
                  clipping=dict(clipping), raw_rounds_sha256=sha(folder / 'rounds.jsonl.gz'))
    with (folder / 'independent_validation.json').open('x', encoding='utf-8') as target:
        json.dump(result, target, indent=2)
    print('VALID', cell['id'], result['completed'], result['total'])
    return result


def main():
    for cell in PLAN['run_order']:
        check(cell)


if __name__ == '__main__':
    main()
