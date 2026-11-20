"""Old free B3 decision core, current correctness and fixed R0 execution.

No FIFO, productive lease, delay budget, bonus or tie objective is added.
The inherited Stage1.1 free objective/fallback/lock audit is used literally.
Only the already-tested G numerical guard wraps the identical legacy MILP.
"""
from unittest.mock import patch
from .joint_integration import JointSiaPolicy
from .joint_scheduler import JointScheduler
from .full160 import SiaDecisionR0Policy
from .small_mechanism import EvidenceMixin
from .bundle_adaptation import BundleAdmission


class BridgeScheduler(JointScheduler):
    def solve(self, policy, columns, dimensions):
        guard=BundleAdmission(300.,250.,60.)
        guard.numerical_solver_retries=[]
        guard._diagnostic_job_order=[j.job_id for j in self.jobs]
        traces=[]
        def guarded(**kwargs):
            cs=kwargs['constraints']
            constraints=list(cs) if isinstance(cs,(list,tuple)) else [cs]
            result=guard._milp(kwargs['c'],columns,constraints)
            traces.append(dict(status=int(result.status),message=str(result.message),
                success=bool(result.success),mip_gap=getattr(result,'mip_gap',None)))
            return result
        with patch('joint_prototype.joint_scheduler.milp',side_effect=guarded):
            result=super().solve(policy,columns,dimensions)
        result.audit.update(bridge_solver_trace=traces,
            numerical_solver_retries=guard.numerical_solver_retries,
            bridge_admission='FREE_OPTIONAL_JOBS_EXACT_CHECKPOINT_LOCKS_ONLY',
            bridge_candidate_counts={j.job_id:sum(c.job_id==j.job_id for c in columns) for j in self.jobs})
        return result


class FreeB3Base(SiaDecisionR0Policy):
    MODES={'b3'}
    optimize=JointSiaPolicy.optimize

    def _scheduler(self,nodes,specs):
        return BridgeScheduler(nodes,specs,self.oracle,self.aux_cpu_points,self.aux_ram_points,
            allocation_bonus=0.,verify_exact=False,minimum_cpu_per_gpu=self.minimum_cpu_per_gpu,
            minimum_ram_gb_per_gpu=self.minimum_ram_gb_per_gpu,candidate_policy='b3')


class FreeB3(EvidenceMixin,FreeB3Base):
    pass


def install(cluster):
    if cluster.policy.round_index!=0 or cluster.policy.current_resources:
        raise RuntimeError('BRIDGE_INSTALL_BEFORE_FIRST_DECISION_ONLY')
    if not 0<len(cluster.jobs)<=20:raise ValueError('BRIDGE_SCOPE_MAX_TWENTY')
    cluster.policy.__class__=FreeB3
    cluster.policy.mode='b3'
    return cluster
