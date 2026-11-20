"""Opt-in completion priority and non-preemptive service continuity.

This is a current-rate completion-priority surrogate, not globally optimal JCT.
Historical policies and the execution/response models are unchanged.
"""
import copy
import math
from dataclasses import replace
import numpy as np
from scipy.optimize import OptimizeResult

from .completion_time import (CompletionPolicy, completion_cost,
                              install as install_legacy)
from .protected_admission import ProtectedAdmission
from .adaptive_service import gpu_geometry


def completion_priority(work, raw_rate, rho, delay):
    estimate = completion_cost(work, raw_rate, rho, delay)
    if estimate <= 0:
        raise ValueError('NONPOSITIVE_COMPLETION_ESTIMATE')
    return 1. / estimate


def canonical_integral_solution(raw, constraints):
    """Integerize solver roundoff only if the exact binary vector is legal.

    The downstream selector already accepts <=1e-6 integrality error. Resource
    and primary/secondary rows are rechecked, not relaxed to that tolerance.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or not np.all(np.isfinite(raw)):
        raise ValueError('INVALID_NUMERICAL_VECTOR')
    rounded = np.rint(raw)
    if (np.any(abs(raw-rounded) > 1e-6) or np.any(rounded < 0) or np.any(rounded > 1)):
        raise ValueError('NOT_NEAR_BINARY')
    for constraint in constraints:
        values = np.asarray(constraint.A @ rounded).reshape(-1)
        lower = np.broadcast_to(constraint.lb, values.shape)
        upper = np.broadcast_to(constraint.ub, values.shape)
        narrow = np.isfinite(lower) & np.isfinite(upper) & (upper-lower < 1e-6)
        tolerance = np.where(narrow, 1e-10, 1e-7)
        if np.any(values < lower-tolerance) or np.any(values > upper+tolerance):
            raise ValueError('ROUNDED_VECTOR_VIOLATES_ORIGINAL_ROWS')
    return rounded


def condition_constraint_bands(constraints):
    """Algebraically identical power-of-two scaling, never widen a band.

    A2**k/b2**k round-trip bit-for-bit to A/b. Narrow nonzero objective
    bands are represented at width>=1 for HiGHS; all returned solutions
    must still pass canonical_integral_solution on the ORIGINAL rows.
    """
    from scipy.optimize import LinearConstraint
    conditioned, exponents = [], []
    for constraint in constraints:
        matrix = np.atleast_2d(np.asarray(constraint.A, dtype=float))
        lower = np.broadcast_to(constraint.lb, (matrix.shape[0],)).copy()
        upper = np.broadcast_to(constraint.ub, (matrix.shape[0],)).copy()
        powers = np.zeros(matrix.shape[0], dtype=int)
        for row in range(matrix.shape[0]):
            if np.isfinite(lower[row]) and np.isfinite(upper[row]):
                width = upper[row] - lower[row]
                if 0 < width < 1e-6:
                    powers[row] = math.ceil(-math.log2(width))
        a = np.ldexp(matrix, powers[:, None])
        lo, hi = np.ldexp(lower, powers), np.ldexp(upper, powers)
        if (not np.array_equal(np.ldexp(a, -powers[:, None]), matrix)
                or not np.array_equal(np.ldexp(lo, -powers), lower)
                or not np.array_equal(np.ldexp(hi, -powers), upper)):
            raise RuntimeError('NONREVERSIBLE_NUMERICAL_CONDITIONING')
        conditioned.append(LinearConstraint(a, lo, hi))
        exponents.append(powers.tolist())
    return conditioned, exponents


def objective_solver_units(cost):
    """Common positive units, derived from the existing primary half-band.

    Only fractional coefficient vectors need this unit conversion. Every
    relative coefficient and feasible allocation ordering is unchanged.
    Solver options are not changed. Returned fun is restored to original units.
    """
    original = np.asarray(cost, dtype=float)
    if original.ndim != 1 or not np.all(np.isfinite(original)):
        raise ValueError('INVALID_OBJECTIVE_COEFFICIENTS')
    power = 0 if np.array_equal(original, np.rint(original)) else math.ceil(-math.log2(1e-9))
    converted = np.ldexp(original, power)
    if not np.all(np.isfinite(converted)) or not np.array_equal(np.ldexp(converted, -power), original):
        raise ValueError('NONREVERSIBLE_OBJECTIVE_UNITS')
    return converted, float(2. ** power)


class NumericalAdmission(ProtectedAdmission):
    def _decide(self, *args):
        self.objective_unit_certifications = []
        result = super()._decide(*args)
        result.audit['objective_unit_certifications'] = self.objective_unit_certifications
        return result

    def _milp(self, cost, columns, constraints):
        backend_cost, cost_scale = objective_solver_units(cost)
        if cost_scale > 1:
            if not hasattr(self, 'objective_unit_certifications'):
                self.objective_unit_certifications = []
            self.objective_unit_certifications.append(dict(scale=cost_scale,
                source='EXISTING_PRIMARY_HALF_BAND_1E_MINUS9', reversible=True,
                relative_weights_unchanged=True, solver_options_unchanged=True))
        diagnostic = None
        try:
            result = super()._milp(backend_cost, columns, constraints)
        except RuntimeError as exc:
            d = getattr(exc, 'solver_diagnostics', None)
            if (str(exc) != 'INVALID_SOLVER_RESULT_AFTER_NUMERICAL_RETRY'
                    or not d or d.get('retry_status') != 0 or d.get('retry_x') is None):
                raise
            diagnostic = d
            result = OptimizeResult(x=np.asarray(d['retry_x']), status=0,
                                    success=True, message=d['retry_message'])
        if result.status != 0 or not result.success or result.x is None:
            return result
        try:
            rounded = canonical_integral_solution(result.x, constraints)
        except ValueError:
            # Never accept a secondary solution outside the original primary
            # band, even when a broad resource tolerance would permit it.
            from scipy.optimize import Bounds
            from . import adaptive_service as solver_backend
            conditioned, powers = condition_constraint_bands(constraints)
            if not any(any(k > 0 for k in row) for row in powers):
                raise RuntimeError('INVALID_SOLVER_RESULT_WITHOUT_CONDITIONABLE_BAND')
            options = dict(time_limit=30., mip_rel_gap=0., presolve=False)
            retried = solver_backend.milp(c=backend_cost, integrality=np.ones(len(columns)),
                bounds=Bounds(np.zeros(len(columns)), np.ones(len(columns))),
                constraints=conditioned, options=options)
            record = dict(reformulation='REVERSIBLE_POWER_OF_TWO_ROW_SCALING',
                row_exponents=powers, solver_options=options,
                objective_unit_scale=cost_scale,
                status=int(retried.status), message=str(retried.message),
                unscaled_original_rows_validated=False)
            self.numerical_solver_retries.append(record)
            if retried.status != 0 or not retried.success or retried.x is None:
                raise RuntimeError('CONDITIONED_SOLVER_NO_OPTIMAL_SOLUTION')
            try:
                rounded = canonical_integral_solution(retried.x, constraints)
            except ValueError as exc:
                raise RuntimeError('CONDITIONED_SOLUTION_FAILS_ORIGINAL_ROWS') from exc
            record['unscaled_original_rows_validated'] = True
            result = retried
        if diagnostic is not None:
            diagnostic.update(canonical_binary_accepted=True,
                canonical_x=rounded.tolist(), primary_and_secondary_rows_preserved=True)
        return OptimizeResult(x=rounded, status=0, success=True,
            fun=float(np.dot(cost, rounded)),
            message=str(result.message)+'; integer vector validated on original rows')


class ContinuityAdmission(NumericalAdmission):
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
    elif new_score:
        cluster.policy.liveness.__class__ = NumericalAdmission
    return cluster
