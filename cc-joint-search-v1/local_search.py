"""A small, time-bounded search around a same-round feasible FO bundle."""

from itertools import product
from math import fsum, isfinite
from time import perf_counter

from joint_prototype.joint_scheduler import CandidateLimitReached


MAX_JOBS = 3
MAX_COLUMNS_PER_JOB = 8
MAX_EXPANDED_COLUMNS = 2400
RESOURCE_TOLERANCE = 1e-8
UTILITY_TOLERANCE = 1e-9
ETA_TOLERANCE = 1e-8


def _finish(stats, status, columns):
    stats["local_status"] = status
    return columns


def _resource_use(columns, node_ids):
    return {
        node_id: tuple(fsum(c.usage_for(node_id)[d] for c in columns) for d in range(3))
        for node_id in node_ids
    }


def _eta(provider, column):
    value = provider.eta(column)
    return value if value is not None and isfinite(value) and value > 0 else None


def _pressure(table, capacity, node_ids):
    return max(
        (table[node][d] / capacity[node][d] for node in node_ids for d in (1, 2)
         if capacity[node][d] > 0),
        default=0.0,
    )


def _alternatives(provider, baseline, editable, capacity, use, deadline):
    """Rank alternate GPU configs without enumerating their auxiliary grids."""
    proportional = {}
    for column in provider.proportional:
        if perf_counter() >= deadline:
            return []
        if column.job_id in editable:
            proportional.setdefault(column.job_id, []).append(column)

    possible = []
    for name in sorted(editable):
        existing = baseline[name]
        current_eta = _eta(provider, existing)
        if current_eta is None:
            continue
        for column in proportional.get(name, ()):
            if perf_counter() >= deadline:
                return []
            alt_eta = _eta(provider, column)
            if column.config.config_id == existing.config.config_id or alt_eta is None:
                continue
            linked_nodes = set(existing.placement.node_ids) | set(column.placement.node_ids)
            pressure = _pressure(use, capacity, linked_nodes)
            current_gpu = existing.config.sia_goodput
            alt_gpu = column.config.sia_goodput
            rank_reversal = (alt_gpu > current_gpu and alt_eta > current_eta) or (
                alt_gpu < current_gpu and alt_eta < current_eta
            )
            response_gap = abs(column.query.rho - existing.query.rho)
            demand_gap = max(
                abs(column.config.cpu_demand_scale - existing.config.cpu_demand_scale),
                abs(column.config.memory_demand_scale - existing.config.memory_demand_scale),
            )
            if not (rank_reversal or response_gap > 1e-6 or pressure >= 0.7):
                continue
            rank = (
                not rank_reversal,
                -(response_gap + 0.25 * demand_gap),
                -pressure,
                alt_eta / current_eta,
                name,
                column.config.config_id,
                column.column_id,
            )
            possible.append((rank, column, linked_nodes))
    return sorted(possible, key=lambda option: option[0])


def _usage_fraction(column, residual, dimension):
    return fsum(amounts[dimension] / max(residual[node][dimension], 1e-12)
                for node, *amounts in column.node_usage)


def _choices(incumbent, candidates, provider, residual):
    """Keep the FO witness, fast points, and low-CPU/RAM alternatives."""
    result = {incumbent.column_id: (incumbent, _eta(provider, incumbent))}
    viable = []
    for column in candidates:
        if column.column_id == incumbent.column_id:
            continue
        if any(any(amount > residual[node][d] + RESOURCE_TOLERANCE
                   for d, amount in enumerate(amounts))
               for node, *amounts in column.node_usage):
            continue
        eta = _eta(provider, column)
        if eta is not None and isfinite(column.scheduling_utility):
            viable.append((column, eta))

    def rank(item):
        column, eta = item
        return (eta, -column.scheduling_utility, column.column_id)

    groups = {}
    for item in viable:
        groups.setdefault(item[0].config.config_id, []).append(item)
    for config_id in sorted(groups):
        group = groups[config_id]
        fast = min(group, key=rank)
        low_cpu = min(group, key=lambda item: (_usage_fraction(item[0], residual, 1), rank(item)))
        low_ram = min(group, key=lambda item: (_usage_fraction(item[0], residual, 2), rank(item)))
        for item in (fast, low_cpu, low_ram):
            result[item[0].column_id] = item
    for item in sorted(viable, key=rank):
        if len(result) >= MAX_COLUMNS_PER_JOB:
            break
        result[item[0].column_id] = item
    return list(result.values())[:MAX_COLUMNS_PER_JOB]


def improve(admission, scheduler, chosen, provider, deadline, now, stats):
    """Retain FO on timeout, failed audit, or insufficient local improvement.

    `deadline` uses perf_counter wall time. `expand` and `legal_local` derive
    columns from this decision's shared eligibility and enforce their own time
    check. This function never commits a service, budget, or recovery permit.
    """
    if perf_counter() >= deadline:
        return _finish(stats, "timeout", chosen)
    if not chosen:
        return _finish(stats, "no_bundle", chosen)

    baseline = {column.job_id: column for column in chosen}
    if len(baseline) != len(chosen):
        return _finish(stats, "invalid_fo_bundle", chosen)
    editable = {
        name for name, column in baseline.items()
        if not scheduler.lock_active(scheduler.job_map[name]) and _eta(provider, column) is not None
    }
    if not editable:
        return _finish(stats, "no_editable_job", chosen)

    capacity = {n.node_id: (n.gpus, n.cpu_cores, n.ram_gb) for n in scheduler.nodes}
    use = _resource_use(chosen, capacity)
    alternatives = _alternatives(provider, baseline, editable, capacity, use, deadline)
    if perf_counter() >= deadline:
        return _finish(stats, "timeout", chosen)
    stats["local_ranked_alternatives"] = len(alternatives)
    if not alternatives:
        return _finish(stats, "no_linked_alternative", chosen)

    anchor_rank, alternate, linked_nodes = alternatives[0]
    anchor = alternate.job_id
    peers = []
    for name in sorted(editable - {anchor}):
        column = baseline[name]
        shared_nodes = linked_nodes.intersection(column.placement.node_ids)
        if not shared_nodes:
            continue
        load = fsum(column.usage_for(node)[d] / capacity[node][d]
                    for node in shared_nodes for d in (1, 2) if capacity[node][d] > 0)
        peers.append((-load, name))
    local = [anchor, *(name for _, name in sorted(peers)[:MAX_JOBS - 1])]
    scope = {name: {baseline[name].config.config_id} for name in local}
    scope[anchor].add(alternate.config.config_id)
    for name in local[1:]:
        for _, column, nodes in alternatives:
            if column.job_id == name and nodes.intersection(linked_nodes):
                scope[name].add(column.config.config_id)
                break
    stats["local_jobs"] = local
    stats["local_configs"] = sum(len(config_ids) for config_ids in scope.values())
    stats["local_anchor_gpu"] = alternate.config.config_id
    stats["local_anchor_rank_reversal"] = not anchor_rank[0]
    stats["local_anchor_aux_pressure"] = -anchor_rank[2]

    external = [c for c in chosen if c.job_id not in scope]
    external_use = _resource_use(external, capacity)
    residual = {node: tuple(cap - used for cap, used in zip(capacity[node], external_use[node]))
                for node in capacity}
    if perf_counter() >= deadline:
        return _finish(stats, "timeout", chosen)
    try:
        explored = list(provider.expand(scope, deadline=deadline, max_columns=MAX_EXPANDED_COLUMNS))
    except CandidateLimitReached:
        stats["local_expanded_columns"] = MAX_EXPANDED_COLUMNS
        return _finish(stats, "expansion_limit", chosen)
    stats["local_expanded_columns"] = len(explored)
    if perf_counter() >= deadline:
        return _finish(stats, "timeout", chosen)
    legal = list(provider.legal_local(explored))
    if perf_counter() >= deadline:
        return _finish(stats, "timeout", chosen)
    buckets = {name: [] for name in local}
    for column in legal:
        if column.job_id in scope and column.config.config_id in scope[column.job_id]:
            buckets[column.job_id].append(column)
    choices = [_choices(baseline[name], buckets[name], provider, residual) for name in local]
    stats["local_candidate_columns"] = sum(map(len, choices))

    initial_eta = fsum(_eta(provider, baseline[name]) for name in local)
    initial_score = fsum(baseline[name].scheduling_utility for name in local)
    best_key, best = None, None
    examined = 0
    for combination in product(*choices):
        if perf_counter() >= deadline:
            stats["local_combinations"] = examined
            return _finish(stats, "timeout", chosen)
        examined += 1
        columns, etas = zip(*combination)
        if all(c.config.config_id == baseline[c.job_id].config.config_id for c in columns):
            continue
        score = fsum(c.scheduling_utility for c in columns)
        eta = fsum(etas)
        if score + UTILITY_TOLERANCE < initial_score or eta >= initial_eta - ETA_TOLERANCE:
            continue
        local_use = _resource_use(columns, residual)
        if any(used > cap + RESOURCE_TOLERANCE for node in residual
               for cap, used in zip(residual[node], local_use[node])):
            continue
        key = (eta, -score, tuple(c.column_id for c in columns))
        if best_key is None or key < best_key:
            best_key, best = key, columns

    stats["local_combinations"] = examined
    if best is None:
        return _finish(stats, "no_safe_eta_improvement", chosen)
    replacements = {column.job_id: column for column in best}
    result = [replacements.get(column.job_id, column) for column in chosen]
    if perf_counter() >= deadline:
        return _finish(stats, "timeout", chosen)
    if not scheduler.audit(result)["valid"]:
        return _finish(stats, "audit_rejected", chosen)
    if perf_counter() >= deadline:
        return _finish(stats, "timeout", chosen)
    stats["local_eta_saved"] = initial_eta - best_key[0]
    stats["local_utility_delta"] = -best_key[1] - initial_score
    return _finish(stats, "accepted", result)
