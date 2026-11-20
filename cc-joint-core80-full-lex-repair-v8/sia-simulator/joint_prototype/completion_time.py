"""Opt-in completion-time opportunity cost. Historical policies stay intact.

This is a constant-current-rate ETA surrogate, not a global JCT optimizer.
Only declared work and already-observed performance/gradient state are read.
Unknown ETA is never obtained from the Sia cold-start raw=1 placeholder.
"""
import copy
import math
from dataclasses import asdict, replace

from sia import SiaPolicy
from .exchange_netprogress import ExchangeCluster
from .protected_admission import ProtectedPolicy, install as install_protected
from .small_mechanism import EvidenceMixin, AuxBase, fixed_set_solve


def completion_cost(work, raw_rate, rho, delay):
    values = (work, raw_rate, rho, delay)
    if (not all(math.isfinite(v) for v in values) or work < 0 or
            raw_rate <= 0 or rho <= 0 or delay < 0):
        raise ValueError('INVALID_COMPLETION_ESTIMATE')
    return delay + work / (raw_rate * rho)


def completion_saving(work, raw_rate, rho, delay, wait_cost):
    if not math.isfinite(wait_cost) or wait_cost < 0:
        raise ValueError('INVALID_WAIT_REFERENCE')
    return wait_cost - completion_cost(work, raw_rate, rho, delay)


def predict_job(job, config):
    """No Job.step, true throughput/gradient lookup, or future completion data.

    A shallow copy runs the actual batch-selection code on CURRENT learned
    parameters; the live Job, profile and progress are not modified.
    """
    model = job.get_goodput_fn(config.gpu_type)
    if model is None:
        return dict(raw_rate=None, atomic_bsz=None, accum_steps=None,
                    source='ETA_UNKNOWN_UNOBSERVED_MODEL')
    app = job.applications[config.gpu_type]
    if app.num_stages != 1:
        # Keep the legal GPU candidate, but don't invent DP ETA units for PMP.
        return dict(raw_rate=None, atomic_bsz=None, accum_steps=None,
                    source='ETA_UNKNOWN_PMP_UNVALIDATED')
    probe = copy.copy(job)
    probe.current_cluster = config.gpu_type
    # Sia legal distributed shapes are whole nodes; small tested topologies
    # have one node/type, but the estimate preserves the node count explicitly.
    if config.gpus % config.num_nodes:
        raise ValueError('NONUNIFORM_LEGAL_CONFIG')
    placement = (config.gpus // config.num_nodes,) * config.num_nodes
    probe.update_local_bsz(placement)
    goodput = float(model.evaluate(config.num_nodes, config.gpus,
                                  probe.atomic_bsz, probe.accum_steps))
    raw = goodput * job.calibration_factor / app.init_batch_size
    if not math.isfinite(raw) or raw <= 0:
        raise ValueError('NONPOSITIVE_CURRENT_MODEL_RATE')
    return dict(raw_rate=raw, atomic_bsz=int(probe.atomic_bsz),
                accum_steps=int(probe.accum_steps), source='CURRENT_MODEL_ACTUAL_BATCH',
                model_goodput=goodput, calibration=float(job.calibration_factor),
                init_batch=int(app.init_batch_size))


def declare_work(job):
    app = job.reference_app
    if app.target_metric is not None:
        raise ValueError('UNDECLARED_CONVERGENCE_WORK_BUDGET')
    # This is a workload declaration, not a future performance observation.
    budget = float(app.get_progress(app.max_epochs))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('INVALID_DECLARED_WORK_BUDGET')
    return dict(total_progress=budget, epochs=int(app.max_epochs),
                source='ARTIFACT_FIXED_EPOCH_PUBLIC_WORK_DECLARATION')


def score_columns(policy, specs, scheduler, columns, horizon=300.):
    """Shared JO/JP score and wait-reference domain, before quota ablation."""
    by_job = {s.job_id: s for s in specs}
    predictions = {}
    for spec in specs:
        for config in spec.configs:
            predictions[spec.job_id, config.config_id] = predict_job(
                policy.completion_jobs[spec.job_id], config)
    proportional = scheduler.build_proportional_columns()
    keys = {(c.job_id, c.config.config_id, c.placement.node_gpus,
             c.query.cpu_per_gpu, c.query.ram_gb_per_gpu) for c in proportional}
    wait = {}
    remaining = {}
    for spec in specs:
        job = policy.completion_jobs[spec.job_id]
        remaining[spec.job_id] = max(0., policy.declared_work[spec.job_id]['total_progress'] - job.progress)
        estimates = []
        for col in proportional:
            if col.job_id != spec.job_id:
                continue
            pred = predictions[col.job_id, col.config.config_id]
            if pred['raw_rate'] is not None:
                resume = policy.completion_delay_costs[col.job_id][col.config.gpu_type]
                estimates.append(completion_cost(remaining[col.job_id], pred['raw_rate'], col.query.rho, resume))
        wait[spec.job_id] = horizon + min(estimates) if estimates else None
    scored, evidence = [], []
    for col in columns:
        spec = by_job[col.job_id]
        job = policy.completion_jobs[col.job_id]
        pred = predictions[col.job_id, col.config.config_id]
        same = (spec.current_config_id == col.config.config_id and
                spec.current_placement == col.placement.node_gpus)
        delay = float(job.rescale_time) if same else policy.completion_delay_costs[col.job_id][col.config.gpu_type]
        key = (col.job_id, col.config.config_id, col.placement.node_gpus,
               col.query.cpu_per_gpu, col.query.ram_gb_per_gpu)
        if pred['raw_rate'] is None or wait[col.job_id] is None:
            # Real locks are always exact. Newly unobserved tasks bootstrap at
            # identical proportional points in JO and JP, never a fake ETA.
            if key not in keys and not scheduler.lock_active(spec):
                continue
            cost, value = None, 0.
        else:
            cost = completion_cost(remaining[col.job_id], pred['raw_rate'], col.query.rho, delay)
            value = wait[col.job_id] - cost
        scored.append(replace(col, scheduling_utility=value))
        evidence.append(dict(job=col.job_id, column_id=col.column_id,
            raw_rate=pred['raw_rate'], rho=col.query.rho, delay=delay,
            remaining_work=remaining[col.job_id], estimated_completion=cost,
            wait_cost=wait[col.job_id], utility=value, prediction_source=pred['source'],
            atomic_bsz=pred['atomic_bsz'], accum_steps=pred['accum_steps']))
    return scored, evidence


class CompletionCluster(ExchangeCluster):
    def optimize_policy(self):
        self.policy.completion_jobs = {j.name: j for j in self.jobs
            if j.submission_time <= self.current_time and j.completion_time is None}
        return super().optimize_policy()


class CompletionPolicy(ProtectedPolicy):
    def factor_columns(self, jobs, specs, scheduler):
        old, proposals = super().factor_columns(jobs, specs, scheduler)
        scored, self.completion_scores = score_columns(self, specs, scheduler, old)
        # EvidenceMixin/ExchangePolicy serialize the actual final candidate set.
        self.exchange_columns = scored
        return scored, proposals

    def optimize(self, *args):
        result = super().optimize(*args)
        self.last_decision['shared_contract'].update(
            policy_version='JOINT_ECT_PROP_V1' if self.mode.endswith('p') else 'JOINT_ECT_V1',
            objective='SUM_EXPECTED_REMAINING_COMPLETION_SECONDS_WITH_WAIT_REFERENCE',
            scoring='PUBLIC_WORK_OVER_CURRENT_PHYSICAL_RATE_PLUS_ACTUAL_STARTUP',
            p_used=False, lambda_n_used=False, waiting_reference_horizon=300.,
            net_progress=False, restart_scoring='ACTUAL_DELAY_IN_ETA_NO_HISTORIC_FACTOR',
            cold_start='UNKNOWN_ETA_PROPORTIONAL_BOOTSTRAP',
            admission='UNCHANGED_PROTECTED_FIFO_PRODUCTIVE_SERVICE',
            auxiliary_optimized=not self.mode.endswith('p'))
        self.last_decision['completion_scores'] = self.completion_scores
        self.last_decision['declared_work'] = self.declared_work
        self.last_decision['probe']['contract'].update(
            objective='COMPLETION_SECONDS', p_used=False, lambda_n_used=False,
            net_progress=False, progress_floor=False)
        self.last_decision['probe'].pop('floor_rejected', None)
        self.last_decision.pop('net_progress_scores', None)
        return result


class CompletionAuxBase(AuxBase):
    """Original full Sia intent; ECT allocation only, no L2 or progress floor."""
    def optimize(self, jobs, nodes, base):
        specs, provenance = self._job_specs(jobs, base)
        scheduler = self._scheduler(self._nodes(nodes), specs)
        scheduler.enforce_floor = False
        scheduler.locked_columns()
        intent = SiaPolicy.optimize(self, jobs, nodes, base)
        prop = self._b1_result(intent, specs, scheduler)
        if self.canonical(self._allocations_from_result(prop, jobs), jobs) != self.canonical(intent, jobs):
            raise RuntimeError('OFFICIAL_INTENT_LOCK_CONFLICT')
        fixed = {j: c.config.config_id for j, c in prop.selected.items()}
        geometry = {j: c.placement.node_gpus for j, c in prop.selected.items()}
        allowed = {j.job_id: ((fixed[j.job_id],) if j.job_id in fixed else ()) for j in specs}
        columns = [c for c in scheduler.build_joint_columns(allowed_configs=allowed)
                   if c.config.config_id == fixed.get(c.job_id) and c.placement.node_gpus == geometry[c.job_id]]
        columns, self.completion_scores = score_columns(self, specs, scheduler, columns)
        self.actual_completion_columns = columns
        result = fixed_set_solve(scheduler, columns, fixed)
        if self.canonical(self._allocations_from_result(result, jobs), jobs) != self.canonical(intent, jobs):
            raise RuntimeError('FIXED_GPU_INTENT_CHANGED')
        self._commit_metadata(result, provenance, fixed)
        self.last_decision.update(original_sia_allocations=copy.deepcopy(intent),
            strict_full_intent=True, gpu_optimizer='sia.SiaPolicy.optimize',
            admission='OFFICIAL_SIA_NO_L2_NO_SECOND_ADMISSION',
            completion_scores=self.completion_scores, declared_work=self.declared_work)
        return intent


class CompletionAux(EvidenceMixin, CompletionAuxBase):
    def optimize(self, *args):
        result = super().optimize(*args)
        self.last_decision['probe']['joint_columns'] = [asdict(c) for c in self.actual_completion_columns]
        self.last_decision['probe']['contract'].update(
            objective='COMPLETION_SECONDS_FIXED_OFFICIAL_GPU', p_used=False,
            lambda_n_used=False, progress_floor=False)
        self.last_decision['probe'].pop('floor_rejected', None)
        return result


def attach_state(cluster):
    cluster.__class__ = CompletionCluster
    cluster.policy.completion_jobs = {}
    cluster.policy.declared_work = {j.name: declare_work(j) for j in cluster.jobs}
    cluster.policy.completion_delay_costs = {j.name: {t: float(a.rescale_time or 30)
        for t, a in j.applications.items()} for j in cluster.jobs}


def install(cluster, proportional=False):
    install_protected(cluster, allow_exchange=True, net_progress=True)
    cluster.policy.__class__ = CompletionPolicy
    cluster.policy.mode = 'u_jp' if proportional else 'u_jo'
    # Reuse the already-tested optional grid8 repair; no further grid search.
    cluster.policy.aux_cpu_points = tuple(sorted(set(cluster.policy.aux_cpu_points) | {8.}))
    attach_state(cluster)
    return cluster


def install_aux(cluster):
    cluster.policy.__class__ = CompletionAux
    cluster.policy.mode = 'b4'
    cluster.policy.aux_cpu_points = tuple(sorted(set(cluster.policy.aux_cpu_points) | {8.}))
    attach_state(cluster)
    return cluster
