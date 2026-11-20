"""Replay one historical same-geometry opportunity; no Job.step or future data.

State selection was frozen before new training: first previous CC round with
nonproportional quotas and all selected ETA known, previously identified as4.
Independent node-local exhaustive enumeration checks both fixed-set objectives.
"""
import csv
import gzip
import itertools
import json
import math
import sys
from dataclasses import replace
from prepare import OUT, TREE, OLD, save, sha

sys.path.insert(0, str(TREE))
from joint_prototype.joint_profile import JobSpec, GpuConfig, NodeSpec
from joint_prototype.neutral_liveness import FactorScheduler
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.small_mechanism import fixed_set_solve


def main():
    source = OLD/'cells/JOINT_CC/rounds.jsonl.gz'
    with gzip.open(source, 'rt', encoding='utf-8') as f:
        r = next(json.loads(line) for line in f
                 if json.loads(line)['decision_for_next_interval']['round'] == 4)
    d = r['decision_for_next_interval']
    selected = d['result']['selected']
    scores = {v['column_id']:v for v in d['completion_scores']}
    specs = []
    for row in d['probe']['specs']:
        row = row.copy()
        row['configs'] = tuple(GpuConfig(**c) for c in row['configs'])
        row['current_nodes'] = tuple(row['current_nodes'])
        row['current_placement'] = tuple(tuple(x) for x in row['current_placement'])
        specs.append(JobSpec(**row))
    apps = {v['name']:v['application'] for v in csv.DictReader(
        (OLD/'inputs/workload.csv').open(encoding='utf-8'))}
    oracle = R0Oracle(OLD/'inputs/registry.json', OLD/'inputs/models', apps, 'medium')
    nodes = [NodeSpec(t+':0', t, 2, 12., 250.) for t in ['aws','dgx-ext']]
    s = FactorScheduler(nodes, specs, oracle, [1.,2.,3.,4.,6.,8.,9.,12.,24.],
        [20.83,62.5,125.,187.5,250.], minimum_cpu_per_gpu=1.,
        minimum_ram_gb_per_gpu=20.83, verify_exact=False, candidate_policy='b3')
    s.enforce_floor = False
    columns = [c for c in s.build_joint_columns() if c.job_id in selected
        and c.config.config_id == selected[c.job_id]['config']['config_id']
        and dict(c.placement.node_gpus) == selected[c.job_id]['placement']
        and c.column_id in scores and scores[c.column_id]['estimated_completion'] is not None]
    assert all(not j.checkpoint_locked for j in specs if j.job_id in selected)
    costs = {c.column_id: scores[c.column_id]['estimated_completion'] for c in columns}
    scale = max(costs.values())
    old = fixed_set_solve(s, [replace(c, scheduling_utility=1./costs[c.column_id]) for c in columns], selected)
    new = fixed_set_solve(s, [replace(c, scheduling_utility=-costs[c.column_id]/scale) for c in columns], selected)
    props = [c for c in columns if (c.query.cpu_per_gpu,c.query.ram_gb_per_gpu)==(6.,125.)]
    assert len(props)==len(selected) and s.audit(props)['valid']

    exact_sum = 0.; exact_reciprocal = 0.; count = 0
    for node in nodes:
        jobs = [j for j,v in selected.items() if node.node_id in v['placement']]
        lists = [[c for c in columns if c.job_id==j] for j in jobs]
        feasible = []
        for combo in itertools.product(*lists):
            count += 1
            gpu = sum(c.config.gpus for c in combo)
            cpu = sum(c.config.gpus*c.query.cpu_per_gpu for c in combo)
            ram = sum(c.config.gpus*c.query.ram_gb_per_gpu for c in combo)
            if gpu<=2 and cpu<=12+1e-9 and ram<=250+1e-9:
                feasible.append((sum(costs[c.column_id] for c in combo),
                                 sum(1./costs[c.column_id] for c in combo)))
        assert feasible
        exact_sum += min(x[0] for x in feasible)
        exact_reciprocal += max(x[1] for x in feasible)

    def describe(cols):
        return dict(sum_eta=sum(costs[c.column_id] for c in cols),
            reciprocal_sum=sum(1./costs[c.column_id] for c in cols),
            rows=[dict(job=c.job_id, config=c.config.config_id,
                cpu=c.query.cpu_per_gpu, ram=c.query.ram_gb_per_gpu,
                rho=c.query.rho, eta=costs[c.column_id],
                numerator_clipped=c.query.source_coordinate_clipped,
                denominator_clipped=c.query.baseline_coordinate_clipped) for c in cols])
    oldv = describe(list(old.selected.values()));newv=describe(list(new.selected.values()))
    propv = describe(props)
    assert math.isclose(newv['sum_eta'],exact_sum,rel_tol=1e-9)
    assert math.isclose(oldv['reciprocal_sum'],exact_reciprocal,rel_tol=1e-9)
    assert newv['sum_eta'] <= propv['sum_eta']+1e-5 < oldv['sum_eta']
    save(OUT/'offline_gate.json', dict(valid=True, historical_round=4,
        raw_log_sha256=sha(source), source='Frozen prior current-state candidates and ETA only',
        candidate_count=len(columns), enumerated_node_bundles=count,
        unchanged_gpu_geometry=True, same_selected_jobs=sorted(selected),
        old_reciprocal=oldv,new_sum_eta=newv,proportional=propv,
        exact_min_sum_eta=exact_sum,exact_max_reciprocal=exact_reciprocal,
        limits='Static fixed geometry and current model; does not predict actual global JCT benefit.',
        warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
    print('OFFLINE_GATE_PASS',oldv['sum_eta'],propv['sum_eta'],newv['sum_eta'],count)


if __name__=='__main__':main()
