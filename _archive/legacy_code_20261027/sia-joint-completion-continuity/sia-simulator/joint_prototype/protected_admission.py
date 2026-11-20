"""Required checkpoint/service/head, optional remaining jobs. Old modes opt out."""
from .bundle_adaptation import BundleAdmission
from .exchange_netprogress import ExchangePolicy,install as install_exchange


class ProtectedAdmission(BundleAdmission):
    _optional_job_selection = True

    def _decide(self,*args):
        result=super()._decide(*args)
        result.policy='JOINT_PROTECTED_OPTIONAL'
        result.status='DRAIN_FOR_HEAD' if result.audit['drain'] else 'OPTIMAL_PROTECTED_OPTIONAL'
        return result


class ProtectedPolicy(ExchangePolicy):
    def optimize(self,*args):
        result=super().optimize(*args)
        self.last_decision['shared_contract'].update(
            policy_version='JOINT_PROTECTED_OPTIONAL_V1',admission_unchanged=False,
            admission='LOCKS_SERVICE_HEAD_REQUIRED_OTHER_JOBS_OPTIONAL',
            drain='NO_NEW_COMMITMENTS_WHILE_HEAD_BLOCKED')
        return result


def install(cluster,allow_exchange=True,net_progress=True):
    install_exchange(cluster,allow_exchange,net_progress)
    cluster.policy.__class__=ProtectedPolicy
    cluster.policy.liveness.__class__=ProtectedAdmission
    return cluster
