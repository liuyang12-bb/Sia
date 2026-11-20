"""Red-test adapter: unchanged historical behavior before the new mechanism."""
from .completion_time import install as install_legacy

def completion_priority(work, raw_rate, rho, delay):
    # Reproduce the existing best-proportional wait-reference cancellation.
    eta = delay + work / (raw_rate * rho)
    return 300. + eta - eta

def install(cluster, proportional=False, new_score=True, continuity=True):
    return install_legacy(cluster, proportional=proportional)
