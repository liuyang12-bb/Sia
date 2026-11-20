"""One preregistered half-CPU cell; run in its own Python process."""
import argparse
import dataclasses
import gzip
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def default(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, default=default, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    cell = next(x for x in manifest['run_order'] if x['id'] == args.cell)
    sys.path.insert(0, str(Path(manifest['tree']) / 'sia-simulator'))
    import pandas as pd
    import psutil
    from joint_prototype.full160 import make_cluster
    from joint_prototype.small_mechanism import ProbeSia, ProbeAux
    from joint_prototype.version_bridge import install as install_free
    from joint_prototype.bundle_adaptation import install as install_g
    from run_full160_cell import FullMetrics

    if psutil.virtual_memory().available < manifest['memory_gate_gib'] * 2**30:
        raise RuntimeError('MEMORY_GATE_2GiB')
    if psutil.disk_usage(str(args.output.parent)).free < manifest['disk_gate_gib'] * 2**30:
        raise RuntimeError('DISK_GATE_5GiB')
    if sha(__file__) != manifest['runner_sha256']:
        raise RuntimeError('RUNNER_CHANGED')
    for path, digest in manifest['hashes'].items():
        if sha(path) != digest:
            raise RuntimeError(f'FROZEN_INPUT_OR_SOURCE_CHANGED:{path}')
    if sha(Path(args.manifest).parent.parent / 'b3_version_bridge_20261026/execution_source.zip') != manifest['source_snapshot_hash']:
        raise RuntimeError('SOURCE_SNAPSHOT_CHANGED')
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    config = json.loads(Path(manifest['config']).read_text(encoding='utf-8'))
    workload = pd.read_csv(manifest['datasets'][cell['dataset']])
    mode = 'joint_o' if cell['method'] == 'joint_g' else 'sia_p'
    cluster = make_cluster(config, workload, manifest['registry'], manifest['models'],
                           mode, cell['response'], 'half', True)
    if cell['method'] == 'sia_p':
        cluster.policy.__class__ = ProbeSia
    elif cell['method'] == 'sia_aux':
        cluster.policy.__class__ = ProbeAux
        cluster.policy.mode = 'b4'
    elif cell['method'] == 'b3_free':
        install_free(cluster)
    else:
        install_g(cluster)
    metrics = FullMetrics(cluster)
    save(args.output / 'start.json', dict(command=sys.argv, cell=cell,
         manifest_sha256=sha(args.manifest), available_memory_gib=psutil.virtual_memory().available / 2**30,
         source_snapshot_sha256=manifest['source_snapshot_hash']))
    try:
        with gzip.open(args.output / 'rounds.jsonl.gz', 'xt', encoding='utf-8') as handle:
            for index in range(manifest['max_rounds']):
                if cluster.all_complete():
                    break
                tick = time.perf_counter()
                cluster.step(manifest['interval'])
                record = cluster.joint_round_logs[-1]
                record['round_wall_seconds'] = time.perf_counter() - tick
                record['job_end_state'] = {job.name: dict(restarts=job.num_restarts or 0,
                    migrations=job.num_migrations or 0, wasted_gpu_seconds=job.wasted_gpu_seconds,
                    completion=job.completion_time) for job in cluster.jobs}
                handle.write(json.dumps(record, default=default, separators=(',', ':')) + '\n')
                handle.flush()
                metrics.consume(cluster, record)
                cluster.joint_round_logs.clear()
                cluster.logs.clear()
                cluster.rescale_events.clear()
                cluster.migrate_events.clear()
                if (index + 1) % 50 == 0:
                    print('PROGRESS', index + 1,
                          sum(job.completion_time is not None for job in cluster.jobs), flush=True)
        result = metrics.summary(cluster, dict(rmst_tau_seconds=manifest['max_rounds'] * manifest['interval']))
        result.update(method=cell['method'], response=cell['response'], capacity='half',
                      dataset=cell['dataset'], total_wall_seconds=time.perf_counter() - started)
        save(args.output / 'summary.json', result)
        print('RESULT', result['completed'], result['mean_jct'], flush=True)
    except Exception as exc:
        save(args.output / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
             time=cluster.current_time, last_decision=cluster.policy.last_decision,
             solver_diagnostics=getattr(exc, 'solver_diagnostics', None)))
        raise


if __name__ == '__main__':
    main()
