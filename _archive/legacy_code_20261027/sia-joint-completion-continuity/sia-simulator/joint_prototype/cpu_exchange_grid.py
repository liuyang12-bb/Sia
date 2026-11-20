"""Opt-in G112 candidate repair: express the verified 6+6 -> 8+4 exchange.

No objective, admission, response, locks, costs, or default policy changes.
The Oracle already defines CPU8 by the existing interpolation. Candidate and
fallback audits receive the same extended grid through the existing scheduler.
"""
from .service_horizon import install as install_service_horizon


def install(cluster):
    install_service_horizon(cluster)
    cluster.policy.aux_cpu_points = tuple(sorted(set(cluster.policy.aux_cpu_points) | {8.0}))
    return cluster
