"""Supplement frozen auditor; recompute both sides of NC reference cost."""
import collections, csv, gzip, json, math, sys
from experiment import OUT, load, save, sha


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6), (a, b)


def main():
    load()
    sys.path.insert(0, str(OUT / 'source'))
    from joint_prototype.r0_liveness import R0Oracle
    from run_stage2c_cell import independent_rho
    with (OUT / 'inputs/workload.csv').open(encoding='utf-8') as f:
        apps = {r['name']: r['application'] for r in csv.DictReader(f)}
    oracle = R0Oracle(OUT / 'inputs/registry.json', OUT / 'inputs/models', apps, 'medium')
    result = {}
    for method in ['JOINT_NC_P', 'JOINT_NC']:
        counter = collections.Counter()
        rows = []
        last = None
        with gzip.open(OUT / 'cells' / method / 'rounds.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)['decision_for_next_interval']
                if d['round'] == last:
                    continue
                last = d['round']
                net = d['result']['audit']['net_completion']
                selected = d['result']['selected']
                scores = {v['column_id']: v for v in d['completion_scores']}
                columns = {v['column_id']: v for v in d['probe']['joint_columns']}
                assert set(net['reference_columns']) == set(net['reference_jobs']) == set(selected)
                reference_sum = chosen_sum = reciprocal = 0.
                differences = {}
                unknown = {}
                reference_usage = collections.defaultdict(lambda: [0., 0., 0.])
                for job, cid in net['reference_columns'].items():
                    ref = scores[cid]
                    chosen = scores[selected[job]['column_id']]
                    assert ref['job'] == chosen['job'] == job
                    col = columns[cid]
                    for node, gpu, cpu, ram in col['node_usage']:
                        for n, value in enumerate([gpu, cpu, ram]):
                            reference_usage[node][n] += value
                    for key in [cid, selected[job]['column_id']]:
                        q = columns[key]['query']
                        rho = independent_rho(oracle, job, q['config_id'], q['cpu_per_gpu'], q['ram_gb_per_gpu'])
                        close(scores[key]['rho'], rho)
                        counter['independent_rho_calculations'] += 1
                    reciprocal += ref['admission_utility']
                    if ref['estimated_completion'] is None:
                        assert selected[job]['column_id'] == cid
                        assert chosen['estimated_completion'] is None
                        unknown[job] = cid
                        counter['unknown_frozen_checks'] += 1
                        continue
                    assert chosen['estimated_completion'] is not None
                    costs = []
                    for row in [ref, chosen]:
                        assert row['raw_rate'] > 0 and row['rho'] > 0 and row['remaining_work'] > 0
                        eta = row['delay'] + row['remaining_work'] / (row['raw_rate'] * row['rho'])
                        close(eta, row['estimated_completion'])
                        costs.append(eta)
                        counter['independent_eta_calculations'] += 1
                    reference_sum += costs[0]
                    chosen_sum += costs[1]
                    differences[job] = dict(reference_eta=costs[0], chosen_eta=costs[1],
                        delta=costs[1]-costs[0], reference_column=cid,
                        chosen_column=selected[job]['column_id'],
                        reference_config=col['config']['config_id'],
                        reference_quota=[col['query']['cpu_per_gpu'], col['query']['ram_gb_per_gpu']],
                        chosen_config=selected[job]['config']['config_id'],
                        chosen_quota=[selected[job]['cpu_per_gpu'], selected[job]['ram_gb_per_gpu']],
                        reference_score=ref, chosen_score=chosen)
                for node, used in reference_usage.items():
                    assert all(v <= cap + 1e-7 for v, cap in zip(used, [2, 12, 250])), (node, used)
                    counter['reference_node_capacity_checks'] += 1
                assert unknown == net['unknown_frozen']
                close(reference_sum, net['reference_eta_sum'])
                close(chosen_sum, net['chosen_eta_sum'])
                close(reciprocal, net['reference_reciprocal_objective'])
                assert chosen_sum <= reference_sum + 1e-5
                counter['rounds'] += 1
                changed = any(v['reference_column'] != v['chosen_column'] for v in differences.values())
                counter['changed_reference_columns_rounds'] += changed
                counter['strict_predicted_cost_improvement_rounds'] += chosen_sum < reference_sum-1e-5
                if changed:
                    rows.append(dict(round=d['round'], reference_sum=reference_sum, chosen_sum=chosen_sum,
                        receiver_gain=sum(-v['delta'] for v in differences.values() if v['delta'] < 0),
                        donor_loss=sum(v['delta'] for v in differences.values() if v['delta'] > 0),
                        jobs=differences))
        result[method] = dict(checks=dict(counter), first_structural_difference=rows[0] if rows else None,
            changed_rounds=rows,
            warning='Repeated current-state ETA differences are not additive realized JCT savings.')
    save(OUT / 'reference_audit.json', dict(valid=True, results=result,
        original_frozen_auditor_sha256=sha(OUT / 'audit_results.py'),
        supplementary_auditor_sha256=sha(__file__),
        purpose='Reference and chosen ETA independently recomputed from logged current candidate primitives; no simulation'))
    print('REFERENCE_AUDIT_PASS', {m:r['checks'] for m,r in result.items()})


if __name__ == '__main__':
    main()
