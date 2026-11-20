"""JOINT-S: bounded FIFO head guarantee, utility-selected optional admissions.

Keep all real locks and unfulfilled productive-service commitments mandatory.
Only NEW commitments become selective; no mid-service evictions or weights.
"""
from .bundle_adaptation import BundleAdmission,BundleJoint,install as install_bundle

class SelectiveAdmission(BundleAdmission):
    def _admission_sets(self,held,solve,now):
        held=set(held)
        if not self.ready:return held,set(),False
        head=self.ready[0]
        if solve({head}) is None:
            raise RuntimeError(f'NO_SOLO_FEASIBLE_CANDIDATE: {head}')
        required=held|{head}
        if solve(required) is None:
            self.events.append(dict(time=now,event='DRAIN_FOR_HEAD',job=head,held=sorted(held)))
            return held,set(),True
        return required,set(self.ready[1:]),False

    def _decide(self,*args):
        result=super()._decide(*args)
        result.policy='JOINT_S'
        result.status='DRAIN_FOR_HEAD' if result.audit['drain'] else 'OPTIMAL_SELECTIVE_ADMISSION'
        result.audit['admission_rule']='HELD_AND_FIFO_HEAD_MANDATORY_OTHER_READY_OPTIONAL'
        return result

class SelectiveJoint(BundleJoint):
    def optimize(self,*args):
        result=super().optimize(*args)
        self.last_decision['shared_contract']['admission']='FIFO_HEAD_AND_COMMITMENTS_PLUS_FAIR_OPTIONAL'
        return result

def install(cluster):
    install_bundle(cluster)
    cluster.policy.__class__=SelectiveJoint
    cluster.policy.liveness.__class__=SelectiveAdmission
    return cluster
