"""Opt-in completion priority and non-preemptive service continuity.

This is a current-rate completion-priority surrogate, not globally optimal JCT.
Historical policies and the execution/response models are unchanged.
"""
import copy
import math
from dataclasses import replace

from .completion_time import (CompletionPolicy, completion_cost,
                              install as install_legacy)
from .protected_admission import ProtectedAdmission
from .adaptive_service import gpu_geometry


def completion_priority(work, raw_rate, rho, delay):
    estimate = completion_cost(work, raw_rate, rho, delay)
    if estimate <= 0:
        raise ValueError('NONPOSITIVE_COMPLETION_ESTIMATE')
    return 1. / estimate


class ContinuityAdmission(ProtectedAdmission):
    """Finite-work FIFO liveness without a mandatory periodic eviction.

    Existing service remains required until actual completion. The inherited
    delay_charged budget is lifetime because service records never expire.
    Admission state still commits only through L2Admission.decide's trial copy.
    """
    def service_expired(self, name, service, productive):
        return False

    def check_service_clock(self, name, service, now, productive):
        value = float(productive.get(name, 0.))
        previous = service.get('last_observed_productive', service['productive_base'])
        last_time = service.get('last_productive_observation_time', service['issued_time'])
        if not math.isfinite(value) or value < previous - 1e-8:
            raise RuntimeError('NON_MONOTONIC_CONTINUITY_SERVICE')
        if now < service['issued_time'] or now < last_time:
            raise RuntimeError('NON_MONOTONIC_CONTINUITY_TIME')
        if value > previous + 1e-9:
            last_time = now
        if now - last_time > self.delay_bound + self.interval + 1e-8:
            raise RuntimeError(f'CONTINUITY_NO_PRODUCTIVE_PROGRESS: {name}')
        service['last_observed_productive'] = value
        service['last_productive_observation_time'] = last_time

    def eligible_held_columns(self, scheduler, job, service, columns, productive, now):
        eligible, evidence = super().eligible_held_columns(
            scheduler, job, service, columns, productive, now)
        # Real checkpoint, provenance, last-productive and actual startup
        # budget were already enforced by the inherited eligibility checks.
        rejected = []
        if not self.geometry_event_reasons:
            current = gpu_geometry(service['column'])
            rejected = [c.column_id for c in eligible if gpu_geometry(c) != current]
            eligible = [c for c in eligible if gpu_geometry(c) == current]
        evidence.update(eligibility_rule='COMPLETION_CONTINUITY_EVENT_GEOMETRY_LIFETIME_DELAY',
            geometry_event_reasons=self.geometry_event_reasons,
            non_event_geometry_rejected=rejected, eligible_count=len(eligible),
            productive_remaining=None, service_end='ACTUAL_COMPLETION')
        return eligible, evidence

    def _decide(self, scheduler, columns, now, productive):
        active = set(scheduler.job_map)
        previous = getattr(self, 'previous_active', set())
        known = set(getattr(scheduler, 'current_known_configs', set()))
        old_known = getattr(self, 'previous_known_configs', set())
        reasons = []
        if active - previous:
            reasons.append('ARRIVAL')
        if previous - active:
            reasons.append('COMPLETION')
        if known - old_known:
            reasons.append('FIRST_VALID_CURRENT_MODEL')
        self.geometry_event_reasons = reasons
        result = super()._decide(scheduler, columns, now, productive)
        self.previous_active = active
        # Candidate contraction during checkpoint does not erase observations.
        self.previous_known_configs = old_known | known
        result.audit.update(geometry_event_reasons=reasons,
            completion_continuity=True, periodic_service_expiry=False,
            lifetime_startup_budget=self.delay_bound)
        result.policy = 'COMPLETION_CONTINUITY'
        result.status = 'DRAIN_FOR_HEAD' if result.audit['drain'] else 'OPTIMAL_COMPLETION_CONTINUITY'
        return result


class CompletionContinuityPolicy(CompletionPolicy):
    def factor_columns(self, jobs, specs, scheduler):
        columns, proposals = super().factor_columns(jobs, specs, scheduler)
        scores = self.completion_scores
        scheduler.current_known_configs = {
            (c.job_id, c.config.config_id) for c, v in zip(columns, scores)
            if v['raw_rate'] is not None}
        if self.use_completion_priority:
            priorities = [0. if v['estimated_completion'] is None else completion_priority(
                v['remaining_work'], v['raw_rate'], v['rho'], v['delay']) for v in scores]
            # Shared positive numerical scale, no relative weight tuning.
            scale = max(priorities, default=0.) or 1.
            revised = []
            for column, old, priority in zip(columns, scores, priorities):
                value = priority / scale
                revised.append(replace(column, scheduling_utility=value))
                old.update(historical_wait_reference_ignored=old['wait_cost'], wait_cost=None,
                    completion_priority=priority, common_numerical_scale=scale, utility=value)
            columns = revised
        self.exchange_columns = columns
        self.continuity_scores = scores
        return columns, proposals

    def optimize(self, *args):
        previous_controller = copy.deepcopy(self.liveness)
        try:
            result = super().optimize(*args)
        except Exception:
            # Parent factor construction updates profile-source diagnostics
            # before its trial copy. Failed calls must not retain that update.
            self.liveness = previous_controller
            raise
        contract = self.last_decision['shared_contract']
        contract.update(policy_version='JOINT_COMPLETION_CONTINUITY_V1',
            objective=('SUM_RECIPROCAL_CURRENT_COMPLETION_ETA' if self.use_completion_priority
                       else 'HISTORICAL_ETA_SAVING'),
            admission=('FIFO_HEAD_NONPREEMPTIVE_COMPLETION_CONTINUITY' if self.use_continuity
                       else 'HISTORICAL_PERIODIC_PRODUCTIVE_SERVICE'),
            geometry_events=('ARRIVAL_COMPLETION_FIRST_MODEL' if self.use_continuity else 'HISTORICAL'),
            startup_budget_scope='JOB_LIFETIME' if self.use_continuity else 'SERVICE_QUANTUM',
            nonpreemptive_completion=self.use_continuity,
            reciprocal_eta_priority=self.use_completion_priority)
        self.last_decision['probe']['contract']['objective'] = contract['objective']
        self.last_decision['continuity_scores'] = self.continuity_scores
        return result


def install(cluster, proportional=False, new_score=True, continuity=True):
    install_legacy(cluster, proportional=proportional)
    cluster.policy.__class__ = CompletionContinuityPolicy
    cluster.policy.use_completion_priority = bool(new_score)
    cluster.policy.use_continuity = bool(continuity)
    if continuity:
        cluster.policy.liveness.__class__ = ContinuityAdmission
    return cluster
