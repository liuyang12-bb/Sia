"""Conservative certificate for bypassing the existing recovery-only search."""

from math import isfinite


RECOVERY_ETA_TOLERANCE = 1e-9


def proves_no_recovery(witness_eta, alternative_etas):
    """Compare a legal incumbent ETA upper bound to all alternative lower bounds."""
    if witness_eta is None or not isfinite(witness_eta) or witness_eta <= 0:
        return False
    cutoff = witness_eta - RECOVERY_ETA_TOLERANCE
    return all(eta is not None and isfinite(eta) and eta > 0 and eta >= cutoff
               for eta in alternative_etas)


def potential_recovery_jobs(policy, scheduler):
    live = policy.liveness
    history = live.services
    jobs = scheduler.job_map
    if not set(jobs) <= set(history):
        if len(jobs) != 1 or not policy._recovery_precheck:
            return ()
        # For a lone requeued job, the original solo rule can read history
        # even outside the held service set. Held service overrides history.
        history = {**getattr(live, "_lifetime_history", {}), **history}
    return tuple(
        name for name, job in jobs.items()
        if name in history and not scheduler.lock_active(job)
        and not history[name].get("tail_recovery_used", False)
        and job.current_config_id is not None
        and policy.productive.get(name, 0.0) > history[name]["last_geometry_productive"] + 1e-9
    )


def _fixed_peer_gpu_capacity(scheduler, name):
    """Bound possible multi-job recovery despite auxiliary-quota tolerance."""
    remaining = {node.node_id: node.gpus for node in scheduler.nodes}
    for peer_name, peer in scheduler.job_map.items():
        if peer_name == name:
            continue
        if peer.current_config_id is None or not peer.current_placement:
            # Lack of a complete live witness cannot justify pruning here.
            return None
        for node, gpus in peer.current_placement:
            if node not in remaining:
                return None
            remaining[node] -= gpus
    return remaining


def certify_job(policy, scheduler, name, geometry_bounds):
    live = policy.liveness
    service = live.services.get(name, getattr(live, "_lifetime_history", {}).get(name))
    if service is None:
        return False
    job = scheduler.job_map[name]
    column = service["column"]
    current = (column.config.config_id, column.placement.node_gpus)
    # Same geometry and same live placement incur no startup under the
    # ordinary eligibility rule. The lifetime budget still applies.
    charged = service["delay_charged"]
    if (current != (job.current_config_id, job.current_placement)
            or not isfinite(charged) or charged > live.delay_limit(name, service) + 1e-9):
        return False
    witness_eta = geometry_bounds.get(current, (None, None))[1]
    residual = _fixed_peer_gpu_capacity(scheduler, name) if len(scheduler.job_map) > 1 else None
    alternatives = []
    for (config_id, placement), (optimistic_eta, _) in geometry_bounds.items():
        if (config_id, placement) == current:
            continue
        # A recovery has to coexist with every peer on its exact live GPU
        # placement. All auxiliary points share this GPU requirement.
        if residual is not None and any(gpus > residual[node] for node, gpus in placement):
            continue
        alternatives.append(optimistic_eta)
    return proves_no_recovery(witness_eta, alternatives)
