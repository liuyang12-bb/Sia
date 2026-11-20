"""Bounded opt-in scale adapter; unchanged FIFO/adaptation/utility rules."""
from .adaptive_service import AdaptiveAdmission
from .adaptive_probe import AdaptiveJoint, install as install_three


class ScaledAdmission(AdaptiveAdmission):
    MAX_ACTIVE_JOBS = 20

    def _decide(self, scheduler, columns, now, productive):
        if len(scheduler.jobs) > self.MAX_ACTIVE_JOBS:
            raise RuntimeError('SCALE_SCOPE_MAX_TWENTY')
        return super()._decide(scheduler, columns, now, productive)


class ScaledJoint(AdaptiveJoint):
    def optimize(self, *args):
        result = super().optimize(*args)
        self.last_decision['shared_contract']['max_active_jobs'] = 20
        return result


def install(cluster):
    if len(cluster.jobs) > 20:
        raise ValueError('SCALE_SCOPE_MAX_TWENTY')
    install_three(cluster)
    cluster.policy.__class__ = ScaledJoint
    cluster.policy.liveness.__class__ = ScaledAdmission
    return cluster
