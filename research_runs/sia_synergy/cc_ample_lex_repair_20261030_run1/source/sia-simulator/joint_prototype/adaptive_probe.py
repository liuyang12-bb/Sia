"""Opt-in adaptive-service JO; preserve historical strategy classes."""
from .full160 import FullJointPolicy
from .small_mechanism import EvidenceMixin
from .adaptive_service import AdaptiveAdmission

class AdaptiveBase(FullJointPolicy):
    ADMISSION_CLASS=AdaptiveAdmission

    def factor_columns(self,jobs,specs,s):
        self.liveness.profile_sources={j:{c:v['goodput_source'] for c,v in rows.items()}
                                       for j,rows in self.probe_provenance.items()}
        return super().factor_columns(jobs,specs,s)

    def optimize(self,*args):
        result=super().optimize(*args)
        self.last_decision['shared_contract'].update(
            max_active_jobs=3,admission='FIFO_PRODUCTIVE_ADAPTIVE_DELAY_BUDGET',
            service_geometry='ADAPT_AFTER_CHECKPOINT_WITH_PAYBACK',
            total_delay_budget=self.liveness.delay_bound)
        return result

class AdaptiveJoint(EvidenceMixin,AdaptiveBase):
    pass

def install(cluster):
    # This adapter is validated only for the frozen one-node-per-type topology.
    if any(n!=1 for n in cluster.num_nodes.values()):
        raise ValueError('ADAPTIVE_SINGLE_NODE_PER_TYPE_ONLY')
    old=cluster.policy.liveness
    if old.services or old.seen:raise RuntimeError('INSTALL_BEFORE_FIRST_DECISION')
    cluster.policy.__class__=AdaptiveJoint
    cluster.policy.liveness=AdaptiveAdmission(old.quantum,old.delay_bound,old.interval)
    cluster.policy.liveness.delay_costs={j.name:{t:float(a.rescale_time or 30)
        for t,a in j.applications.items()} for j in cluster.jobs}
    return cluster
