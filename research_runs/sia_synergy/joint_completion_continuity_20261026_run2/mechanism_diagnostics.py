"""Read-only same-geometry quota diagnostics; no scheduling or simulation."""
import collections
import gzip
import json
import sys
from experiment import OUT, METHODS, load, save


def main():
    load()
    sys.path.insert(0, str(OUT / 'source'))
    from joint_prototype.r0_liveness import R0Oracle
    from run_stage2c_cell import independent_rho
    import csv
    apps = {r['name']: r['application'] for r in csv.DictReader(
        (OUT / 'inputs/workload.csv').open(encoding='utf-8'))}
    oracle = R0Oracle(OUT / 'inputs/registry.json', OUT / 'inputs/models', apps, 'medium')
    output = {}
    for method in METHODS:
        first = {}
        quotas = collections.defaultdict(collections.Counter)
        score_delta = []
        examples = []
        with gzip.open(OUT / 'cells' / method / 'rounds.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                d = r['decision_for_next_interval']
                selected = d['result']['selected']
                specs = {s['job_id']: s for s in d['probe']['specs']}
                scores = {s['column_id']: s for s in d.get('completion_scores', [])}
                actual_priority = 0.
                prop_priority = 0.
                all_available = bool(selected)
                components = []
                for job, col in selected.items():
                    cfg = col['config']['config_id']
                    cpu, ram = col['cpu_per_gpu'], col['ram_gb_per_gpu']
                    first.setdefault(job, dict(round=d['round'], config=cfg, cpu=cpu, ram=ram,
                                               placement=col['placement'], rho=col['rho']))
                    key = json.dumps([cfg, cpu, ram])
                    quotas[job][key] += 1
                    sc = scores.get(col['column_id'])
                    if not sc or sc['raw_rate'] is None or specs[job]['checkpoint_locked']:
                        all_available = False
                        continue
                    prop_rho = independent_rho(oracle, job, cfg, 6., 125.)
                    row = oracle.reference(job, cfg)
                    supported = all(1 <= c / row['phi_c'] <= 24 and
                                    20.83 <= m / row['phi_m'] <= 250
                                    for c, m in [(cpu, ram), (6., 125.), (3., 62.5)])
                    ta = sc['delay'] + sc['remaining_work'] / (sc['raw_rate'] * col['rho'])
                    tp = sc['delay'] + sc['remaining_work'] / (sc['raw_rate'] * prop_rho)
                    actual_priority += 1 / ta
                    prop_priority += 1 / tp
                    components.append(dict(job=job, config=cfg, cpu=cpu, ram=ram,
                        actual_rho=col['rho'], prop_rho=prop_rho,
                        rate_ratio=col['rho']/prop_rho,
                        predicted_remaining_seconds_change=ta-tp,
                        all_coordinates_supported=supported))
                if all_available:
                    score_delta.append(dict(round=d['round'],
                        reciprocal_eta_delta=actual_priority-prop_priority,
                        components=components))
                    # First occurrence only, not selected by benefit magnitude.
                    if len(examples) < 3 and any(abs(x['rate_ratio']-1) > 1e-8 for x in components):
                        examples.append(score_delta[-1])
        output[method] = dict(first_allocation=first,
            selected_quota_round_counts={j:dict(v) for j,v in quotas.items()},
            static_comparison_rounds=len(score_delta),
            positive_priority_rounds=sum(s['reciprocal_eta_delta'] > 1e-10 for s in score_delta),
            negative_priority_rounds=sum(s['reciprocal_eta_delta'] < -1e-10 for s in score_delta),
            first_differing_examples=examples)
    save(OUT / 'mechanism_diagnostics.json', dict(results=output,
        definition='Same selected jobs, GPU geometry, batch/raw prediction and startup delay; replace only auxiliary quota with feasible 6 CPU/125GB per GPU. Locked/unknown-score rounds excluded.',
        limits='Static online prediction, not replayed progress or JCT. Quota switching may affect later state; clipping is flagged. Raw rates across applications are not summed.',
        warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
    print('MECHANISM_DIAGNOSTICS_COMPLETE')


if __name__ == '__main__':
    main()
