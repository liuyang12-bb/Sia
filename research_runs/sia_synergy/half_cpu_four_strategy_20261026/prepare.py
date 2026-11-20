"""Freeze only the already approved half-CPU comparison inputs."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BRIDGE = ROOT / 'research_runs/sia_synergy/b3_version_bridge_20261026'
PREFLIGHT = ROOT / 'research_runs/sia_synergy/half_cpu_offline_preflight_20261026/attempt2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    target = OUT / 'preregistration.json'
    if target.exists():
        raise RuntimeError('FROZEN_MANIFEST_EXISTS')
    original = json.loads((BRIDGE / 'preregistration.json').read_text(encoding='utf-8'))
    for path, expected in original['hashes'].items():
        if sha(path) != expected:
            raise RuntimeError(f'BRIDGE_HASH_CHANGED:{path}')
    validation = json.loads((PREFLIGHT / 'validation.json').read_text(encoding='utf-8'))
    if not validation['valid'] or validation['actual_simulation_step_calls'] != 0:
        raise RuntimeError('HALF_CPU_PREFLIGHT_NOT_VALID')
    expected_source = 'F64B64EA06474A39AE1EC28B3E73ADCEC7ABAE439820F5F04EA5CBF798B82951'
    if sha(BRIDGE / 'execution_source.zip') != expected_source:
        raise RuntimeError('BRIDGE_SOURCE_SNAPSHOT_CHANGED')
    cells = [dict(id=f'{dataset}_half_medium_{method}', dataset=dataset,
                  capacity='half', response='medium', method=method)
             for dataset in ('original20', 'holdout20')
             for method in ('sia_p', 'sia_aux', 'b3_free', 'joint_g')]
    frozen = {k: original[k] for k in ('tree', 'config', 'registry', 'models', 'datasets',
                                       'hashes', 'packages', 'interval', 'max_rounds')}
    if frozen['interval'] != 60 or frozen['max_rounds'] != 600:
        raise RuntimeError('BRIDGE_OBSERVATION_WINDOW_CHANGED')
    frozen.update(scope='FROZEN_HALF_CPU_FOUR_STRATEGY_SYNTHETIC_MECHANISM',
                  source_snapshot_hash=expected_source,
                  original_bridge_preregistration_sha256=sha(BRIDGE / 'preregistration.json'),
                  offline_preflight_validation_sha256=sha(PREFLIGHT / 'validation.json'),
                  offline_preflight_summary_sha256=sha(PREFLIGHT / 'summary.json'),
                  runner_sha256=sha(OUT / 'run_cell.py'),
                  controller_sha256=sha(OUT / 'run_all.py'),
                  run_order=cells, memory_gate_gib=2, disk_gate_gib=5,
                  response_note='R0/E-legacy fixed-reference 3 CPU and 62.5 GB per GPU; alpha 0.5',
                  censoring='600 rounds; incomplete jobs retain censoring, no JCT imputation',
                  warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
    with target.open('x', encoding='utf-8') as handle:
        json.dump(frozen, handle, indent=2, ensure_ascii=False)
    print('FROZEN', len(cells), sha(target))


if __name__ == '__main__':
    main()
