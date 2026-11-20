"""Full-trace experiment adapters. No change to official Sia or old policies.

SIA modes call the byte-identical official SiaPolicy.optimize, with no second
admission optimizer. TUNE is the existing demand/fair/peer port. An unprovable
full-intent fallback stops explicitly; it never drops a job or a checkpoint.
"""
import copy
import time
from unittest.mock import patch
import cvxpy as cp
import simulator_config as sim_config
from sia import SiaPolicy
from .joint_integration import JointSiaPolicy
from .r0_liveness import R0Oracle,R0Policy,R0Cluster,R0Admission,APPLICATION_TEMPLATES
from .synergy_adapter import SynergyTuneAdapter

METHODS=('sia_p','sia_tune','joint_p','joint_o')
def capacities(level):
    factor={'half':.5,'main':1.,'double':2.}[level]
    return {t:dict(cpu_cores=g*3.*factor,ram_gb=g*62.5) for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]}

class FullAdmission(R0Admission):
    # Same FIFO, service quantum, drain and equality constraints. Tests exercise
    # all 160 arrivals, current services, locks and requeue, not just this guard.
    MAX_ACTIVE_JOBS=160

class FullJointPolicy(R0Policy):
    ADMISSION_CLASS=FullAdmission
    def optimize(self,*args):
        result=super().optimize(*args)
        self.last_decision['shared_contract']['max_active_jobs']=160
        return result

class FullCluster(R0Cluster):
    def optimize_policy(self):
        # Observation before a new decision can issue another checkpoint.
        self.checkpoint_before_decision={j.name:j.rescale_time for j in self.jobs}
        return super().optimize_policy()

    def step(self,seconds=60):
        super().step(seconds)
        for e in self.joint_round_logs[-1]['execution']:
            e['checkpoint_remaining_before_decision']=self.checkpoint_before_decision[e['job_id']]

class SiaDecisionR0Policy(R0Policy):
    MODES={'b1','b5'}
    def __init__(self,**kwargs):
        # Deliberately bypass NeutralL2Policy.__init__: NO L2 queue or services.
        JointSiaPolicy.__init__(self,**kwargs)

    @staticmethod
    def canonical(allocations,jobs):
        return {j:(a[0],tuple(a[1])) if a and a[0] is not None and a[1] else (None,())
                for j in jobs for a in [allocations.get(j)]}

    def optimize(self,jobs,nodes,base_allocations):
        started=time.perf_counter()
        specs,provenance=self._job_specs(jobs,base_allocations)
        scheduler=JointSiaPolicy._scheduler(self,self._nodes(nodes),specs)
        scheduler.locked_columns()
        # Actual official objective, candidates, CBC solve and placement.
        solver_records=[]; original_solve=cp.Problem.solve
        def observe(problem,*args,**kwargs):
            tick=time.perf_counter();value=original_solve(problem,*args,**kwargs)
            solver_records.append(dict(status=problem.status,objective=problem.value,
                wall_seconds=time.perf_counter()-tick,parameters=kwargs))
            return value
        with patch.object(cp.Problem,'solve',observe):
            intent=SiaPolicy.optimize(self,jobs,nodes,base_allocations)
        proportional=self._b1_result(intent,specs,scheduler)
        intent_canonical=self.canonical(intent,jobs)
        if self.canonical(self._allocations_from_result(proportional,jobs),jobs)!=intent_canonical:
            raise RuntimeError('OFFICIAL_SIA_INTENT_CHECKPOINT_CONFLICT')
        fixed={j:c.config for j,c in proportional.selected.items()}
        if self.mode=='b1':
            result=proportional
        else:
            cause=None; tuned=None
            try:
                tuned=SynergyTuneAdapter(scheduler).run('SIA_TUNE_FIXED_OFFICIAL_GPU',fixed,
                    dict(source='OFFICIAL_SIA_OPTIMIZE',uses_aux_for_gpu_choice=False))
                if self.canonical(self._allocations_from_result(tuned,jobs),jobs)!=intent_canonical:
                    cause='TUNE_DROPPED_OR_CHANGED_OFFICIAL_INTENT'
                elif not scheduler.audit(list(tuned.selected.values()))['valid']:
                    cause='TUNE_ATOMIC_AUDIT_FAILED'
            except (RuntimeError,ValueError) as exc:
                cause=repr(exc)
            if cause is not None:
                # Proportional for unlocked jobs, exact original quotas for
                # real checkpoint locks. Never assume this is still feasible.
                fallback=scheduler.audit(list(proportional.selected.values()))
                if not fallback['valid']:
                    self.last_failure=dict(reason='NO_SAFE_FULL_INTENT_FALLBACK',cause=cause,
                        original_sia_allocations=intent,proportional=proportional.to_dict(),
                        tune=tuned.to_dict() if tuned else None,independent_audit=fallback)
                    raise RuntimeError(f'NO_SAFE_FULL_INTENT_FALLBACK: {fallback}')
                result=proportional;result.status='FALLBACK_PROPORTIONAL'
                result.audit.update(fallback_reason=cause,tune_attempt=tuned.to_dict() if tuned else None)
            else: result=tuned
        independent=scheduler.audit(list(result.selected.values()))
        if not independent['valid']:
            self.last_failure=dict(original_sia_allocations=intent,result=result.to_dict(),audit=independent)
            raise RuntimeError(f'SIA_FULL_INTENT_ATOMIC_AUDIT: {independent}')
        if self.canonical(self._allocations_from_result(result,jobs),jobs)!=intent_canonical:
            raise RuntimeError('SIA_GPU_INTENT_CHANGED')
        result.audit['precommit_independent_audit']=independent
        result.solve_seconds=time.perf_counter()-started
        self._commit_metadata(result,provenance,{j:c.config_id for j,c in fixed.items()})
        self.last_decision.update(original_sia_allocations=copy.deepcopy(intent),
            admission='OFFICIAL_SIA_NO_L2_NO_SECOND_ADMISSION',gpu_optimizer='sia.SiaPolicy.optimize',
            strict_full_intent=True,registry_sha256=self.oracle.registry_sha256)
        self.last_decision['official_solver_records']=solver_records
        # Preserve original list/tuple representation too: Cluster compares it.
        return intent

def make_cluster(config,workload,registry,models,method,response,capacity,seed=True):
    if method not in METHODS: raise ValueError('unregistered method')
    if len(workload)>160 or len(workload)==0: raise ValueError('FULL160_SCOPE')
    sim_config.num_parallel_jobs=1;sim_config.num_parallel_traces=1
    sim_config.sia_solver_verbose=False;sim_config.preserve_ckpt_allocs=True
    cc=config['cluster'];rc=config['response'];sc=config['simulation']
    apps=dict(zip(workload.name,workload.application))
    oracle=R0Oracle(registry,models,apps,response)
    cls=SiaDecisionR0Policy if method.startswith('sia_') else FullJointPolicy
    mode={'sia_p':'b1','sia_tune':'b5','joint_p':'u_jp','joint_o':'u_jo'}[method]
    p=cls(mode=mode,oracle=oracle,node_capacities=capacities(capacity),
        template_map={j:APPLICATION_TEMPLATES[a] for j,a in apps.items()},
        aux_cpu_points=rc['aux_cpu_points'],aux_ram_points=rc['aux_ram_points'],
        minimum_cpu_per_gpu=rc['minimum_legal_profile_point']['cpu_per_gpu'],
        minimum_ram_gb_per_gpu=rc['minimum_legal_profile_point']['ram_gb_per_gpu'],
        p_fairness=sc['p_fairness'],restart_penalty=30,lambda_a=sc['lambda_a'],lambda_n=sc['lambda_n'],
        project_throughputs=sc['project_throughputs'],share_max_replicas=sc['share_max_replicas'],
        sched_interval=sc['interval_seconds'],enable_bsz_tuning=not sc['disable_batch_size_tuning'])
    p.populate_valid_configs(cc['nnodes'],cc['ngpus_per_node'],cc['max_physical_nodes'])
    c=FullCluster(workload.copy(),p,cc['nnodes'],cc['ngpus_per_node'],cc['max_physical_nodes'])
    if seed:
        for j in c.jobs: j.seed_profiles(sc['seed_profiles_num_nodes'],sc['seed_profiles_num_replicas'])
    return c
