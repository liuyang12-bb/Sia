"""Bounded adaptive service; historical L2 policies remain unchanged.

FIFO, equality admission, drain, expiry and lexicographic MIP solving below
are retained from neutral_liveness.L2Admission. The new eligibility contract
separates a guaranteed productive service from a fixed resource geometry.
Actual target application startup costs are supplied by the caller, never
inferred from offline throughput or the historical restart-factor proxy.
"""
import math
import time
from collections import Counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .neutral_liveness import L2Admission, geometry
from .joint_scheduler import PolicyResult


def gpu_geometry(column):
    return column.config.config_id, column.placement.node_gpus


class AdaptiveAdmission(L2Admission):
    MAX_ACTIVE_JOBS = 3
    REQUIRE_NEUTRAL = False

    def __init__(self, quantum, delay_bound, interval):
        super().__init__(quantum, delay_bound, interval)
        self.delay_costs = {}
        self.profile_sources = {}

    def _delay(self, job_id, column):
        try:
            cost = float(self.delay_costs[job_id][column.config.gpu_type])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f'MISSING_ACTUAL_STARTUP_COST: {job_id}') from exc
        if not math.isfinite(cost) or cost <= 0 or cost > self.delay_bound + 1e-9:
            raise RuntimeError(f'INVALID_ACTUAL_STARTUP_COST: {job_id}: {cost}')
        return cost

    @staticmethod
    def _rate(column):
        value = column.query.raw_sia_goodput * column.query.rho
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f'INVALID_PREDICTED_RATE: {column.job_id}')
        return value

    def _secondary_tie(self, scheduler, columns, required, constraints, call, chosen):
        """Historical modes retain their original lexicographic tie rule."""
        return chosen

    def _milp(self, cost, columns, constraints):
        return milp(c=cost, integrality=np.ones(len(columns)),
                    bounds=Bounds(np.zeros(len(columns)), np.ones(len(columns))),
                    constraints=constraints,
                    options={'time_limit': 30., 'mip_rel_gap': 0.})

    def eligible_held_columns(self, scheduler, job, service, columns, productive, now):
        """Return current-round eligible columns plus explicit filter evidence.

        This method is pure: no queue, service ledger or supplied column is
        mutated. The unchanged current point is retained even when another
        configuration has a higher estimated rate. Quota-only adaptations
        have zero startup cost under the inherited execution model.
        """
        name = job.job_id
        old = service['column']
        fresh = scheduler.refresh_column(old)
        if (job.current_config_id != fresh.config.config_id
                or job.current_placement != fresh.placement.node_gpus):
            raise RuntimeError(f'INFEASIBLE_COMMITMENT: {name}')
        current = next((c for c in columns if c.job_id == name
                        and geometry(c) == geometry(fresh)), None)
        if current is None:
            raise RuntimeError(f'INFEASIBLE_COMMITMENT_CANDIDATE: {name}')
        done = float(productive.get(name, 0.)) - service['productive_base']
        spent = float(service['delay_charged'])
        if not math.isfinite(spent) or spent < 0 or spent > self.delay_bound + 1e-9:
            raise RuntimeError(f'INVALID_SERVICE_DELAY_LEDGER: {name}')
        if done < -1e-8 or now < service['issued_time']:
            raise RuntimeError(f'NON_MONOTONIC_SERVICE_COUNTER: {name}')
        remaining = self.quantum - done
        base = self._rate(current)
        eligible = [current]
        rejected = []
        for candidate in columns:
            if candidate.job_id != name or geometry(candidate) == geometry(current):
                continue
            reason = None
            if scheduler.lock_active(job):
                reason = 'REAL_CHECKPOINT_LOCK'
            elif gpu_geometry(candidate) == gpu_geometry(current):
                if self._rate(candidate) + 1e-12 < base:
                    reason = 'AUX_REDUCES_CURRENT_RATE'
            elif (productive.get(name, 0.)
                  <= service['last_geometry_productive'] + 1e-9):
                reason = 'NO_PRODUCTIVE_SERVICE_SINCE_GEOMETRY'
            elif self.profile_sources.get(name, {}).get(candidate.config.config_id) not in {
                    'direct_profile', 'projected_profile'}:
                reason = 'UNPROFILED_TARGET'
            else:
                delay = self._delay(name, candidate)
                if spent + delay > self.delay_bound + 1e-9:
                    reason = 'SERVICE_DELAY_BUDGET'
                elif ((remaining - delay) * self._rate(candidate)
                      <= remaining * base + 1e-12):
                    reason = 'NO_REMAINING_SERVICE_PAYBACK'
            if reason:
                rejected.append(dict(column_id=candidate.column_id, reason=reason))
            else:
                eligible.append(candidate)
        return eligible, dict(
            time=now, job=name, current_column_id=current.column_id,
            current_effective_goodput=base, productive_done=done,
            productive_remaining=remaining, delay_charged=spent,
            delay_budget=self.delay_bound, eligible_count=len(eligible),
            rejection_counts=dict(Counter(r['reason'] for r in rejected)),
            rejected=rejected)

    def _decide(self, s, columns, now, productive):
        start = time.perf_counter()
        jobs = s.job_map
        if len(jobs) > self.MAX_ACTIVE_JOBS:
            raise RuntimeError('ADAPTIVE_SCOPE_MAX_THREE_ACTIVE_JOBS')
        if any(j.arrival_time > now for j in s.jobs):
            raise RuntimeError('FUTURE_JOB_INPUT')
        if any(not math.isfinite(c.query.rho) or c.query.rho <= 0 for c in columns):
            raise RuntimeError('INVALID_RESPONSE_COLUMN')
        self.ready = [j for j in self.ready if j in jobs]
        for name in list(self.services):
            if name not in jobs:
                self.events.append(dict(time=now, event='COMPLETED_SERVICE', job=name))
                del self.services[name]
        new = sorted((j for j in s.jobs if j.job_id not in self.seen),
                     key=lambda j: (j.arrival_time, j.source_row_index, j.job_id))
        for j in new:
            self.ready.append(j.job_id)
            self.seen.add(j.job_id)
            self.events.append(dict(time=now, event='ARRIVE_READY', job=j.job_id))
        expired = []
        for name, service in self.services.items():
            done = productive.get(name, 0.) - service['productive_base']
            if done < -1e-8:
                raise RuntimeError('NON_MONOTONIC_SERVICE_COUNTER')
            if done >= self.quantum - 1e-9:
                if s.lock_active(jobs[name]):
                    raise RuntimeError('CHECKPOINT_AFTER_SERVICE_WITHOUT_RECONFIGURATION')
                expired.append(name)
            elif now - service['issued_time'] > self.delay_bound + self.quantum + self.interval + 1e-8:
                raise RuntimeError(f'SERVICE_BOUND_EXCEEDED: {name}')
        for name in sorted(expired, key=lambda n: self.services[n]['sequence']):
            self.ready.append(name)
            previous = self.services.pop(name)
            self.events.append(dict(time=now, event='REQUEUE_PRODUCTIVE_QUANTUM', job=name,
                                    delay_charged=previous['delay_charged'],
                                    geometry_changes=previous['geometry_changes']))
        locked = {j.job_id for j in s.jobs if s.lock_active(j)}
        if locked - set(self.services):
            raise RuntimeError('LOCK_WITHOUT_SERVICE_EVIDENCE')
        held = set(self.services)
        filters = {}
        for name in held:
            eligible, filters[name] = self.eligible_held_columns(
                s, jobs[name], self.services[name], columns, productive, now)
            columns = [c for c in columns if c.job_id != name] + eligible
        columns = sorted(columns, key=geometry)
        traces = []

        # Original L2 equality admission, capacity matrix and exact-utility
        # lexicographic tie solving. No allocation bonus or utility change.
        def solve(required, objective=False, tie=False, optional=False):
            required = set(required)
            if not required and not optional:
                if locked:
                    raise RuntimeError('INVALID_EMPTY_LOCK')
                return []
            if not columns or any(not any(c.job_id == j for c in columns) for j in required):
                return None
            matrix, upper = s._constraint_matrix(columns, ('gpu', 'cpu', 'ram'))
            lower = np.full(len(upper), -np.inf)
            for i, j in enumerate(s.jobs):
                lower[i] = float(j.job_id in required)
                upper[i] = 1. if optional else lower[i]
            weights = np.asarray([c.scheduling_utility for c in columns])
            constraints = [LinearConstraint(matrix, lower, upper)]

            def call(cost):
                result = self._milp(cost, columns, constraints)
                traces.append(dict(required=sorted(required), status=int(result.status),
                                   message=str(result.message),
                                   phase='objective' if objective else 'feasibility'))
                if result.status == 2:
                    return None
                if result.x is None or not result.success or result.status != 0:
                    raise RuntimeError(f'SOLVER_UNDECIDED: {traces[-1]}')
                x = np.asarray(result.x)
                if (len(x) != len(columns) or not np.all(np.isfinite(x))
                        or np.any(abs(x - np.round(x)) > 1e-6)):
                    raise RuntimeError('INVALID_SOLVER_ALLOCATION integrality')
                chosen = [c for c, v in zip(columns, x) if v > .5]
                audit = s.audit(chosen, require_locks=locked <= required)
                selected_jobs = {c.job_id for c in chosen}
                set_valid = required <= selected_jobs if optional else selected_jobs == required
                if (not audit['valid'] or not set_valid
                        or len(chosen) != len(selected_jobs)):
                    raise RuntimeError(f'INVALID_SOLVER_ALLOCATION: {audit}')
                return chosen

            chosen = call(-weights if objective else np.zeros(len(columns)))
            if chosen is None:
                return None
            if tie:
                optimum = sum(c.scheduling_utility for c in chosen)
                constraints.append(LinearConstraint(weights, optimum - 1e-9, optimum + 1e-9))
                chosen = self._secondary_tie(s, columns, required, constraints, call, chosen)
                order = ([j.job_id for j in sorted(s.jobs,key=lambda j:(j.arrival_time,j.source_row_index,j.job_id))]
                         if optional else sorted(required))
                for name in order:
                    if optional and name not in required:
                        presence = np.asarray([float(c.job_id == name) for c in columns])
                        candidate = call(-presence)
                        if candidate is None:
                            raise RuntimeError('OPTIONAL_TIE_INCONSISTENT')
                        present = float(any(c.job_id == name for c in candidate))
                        constraints.append(LinearConstraint(presence,present,present))
                        if not present:
                            chosen = candidate
                            continue
                    cost = np.zeros(len(columns))
                    candidates = [i for i, c in enumerate(columns) if c.job_id == name]
                    for rank, i in enumerate(candidates):
                        cost[i] = rank
                    chosen = call(cost)
                    if chosen is None:
                        raise RuntimeError('LEXICOGRAPHIC_OBJECTIVE_INCONSISTENT')
                    c = next(c for c in chosen if c.job_id == name)
                    mask = np.asarray([float(v is c) for v in columns])
                    constraints.append(LinearConstraint(mask, 1., 1.))
            return chosen

        admitted = set(held)
        drain = False
        if held and solve(held) is None:
            raise RuntimeError('INFEASIBLE_COMMITMENT')
        if self.ready:
            head = self.ready[0]
            if solve({head}) is None:
                raise RuntimeError(f'NO_SOLO_FEASIBLE_CANDIDATE: {head}')
            if solve(admitted | {head}) is None:
                drain = True
                self.events.append(dict(time=now, event='DRAIN_FOR_HEAD', job=head, held=sorted(held)))
            elif getattr(self, '_optional_job_selection', False):
                admitted.add(head)
            else:
                for name in self.ready:
                    if solve(admitted | {name}) is not None:
                        admitted.add(name)
        required_for_audit = set(admitted)
        optional = getattr(self, '_optional_job_selection', False) and not drain
        chosen = solve(admitted, objective=True, tie=True, optional=optional)
        if chosen is None:
            raise RuntimeError('INFEASIBLE_FINAL_ADMISSION')
        independent = s.audit(chosen)
        if (not independent['valid'] or any(
                geometry(c) not in {geometry(v) for v in columns} for c in chosen)):
            raise RuntimeError(f'INVALID_FINAL_COMMIT: {independent}')
        selected = {c.job_id: c for c in chosen}
        if getattr(self, '_optional_job_selection', False):
            independent.update(required_jobs=sorted(required_for_audit),
                optional_jobs=sorted(set(jobs)-required_for_audit),
                optional_selected=sorted(set(selected)-required_for_audit),
                ready_before_commit=list(self.ready),optional_admission_enabled=optional)
        admitted = set(selected)
        budget_changes = {}
        for name in sorted(admitted - held, key=lambda n: self.ready.index(n)):
            column, job = selected[name], jobs[name]
            same_gpu = (job.current_config_id == column.config.config_id
                        and job.current_placement == column.placement.node_gpus)
            charged = 0. if same_gpu else self._delay(name, column)
            self.sequence += 1
            self.services[name] = dict(
                column=column, issued_time=now, productive_base=productive.get(name, 0.),
                sequence=self.sequence, delay_charged=charged, geometry_changes=0,
                last_geometry_productive=productive.get(name, 0.))
            budget_changes[name] = dict(before=None, charged_this_round=charged,
                                        change='INITIAL_STARTUP' if charged else 'CONTINUE_EXISTING_GPU')
            self.events.append(dict(time=now, event='ISSUE_SERVICE', job=name,
                                    sequence=self.sequence, delay_charged=charged))
        self.ready = [j for j in self.ready if j not in admitted]
        for name in held:
            service, column = self.services[name], selected[name]
            old = service['column']
            before = {key: service[key] for key in (
                'issued_time', 'productive_base', 'delay_charged',
                'last_geometry_productive', 'geometry_changes')}
            cost = 0.
            change = 'UNCHANGED'
            if gpu_geometry(old) != gpu_geometry(column):
                cost = self._delay(name, column)
                change = 'GPU_GEOMETRY'
                service['delay_charged'] += cost
                service['geometry_changes'] += 1
                service['last_geometry_productive'] = productive.get(name, 0.)
                self.events.append(dict(time=now, event='ADAPT_GEOMETRY', job=name,
                                        old_config=old.config.config_id, new_config=column.config.config_id,
                                        delay_cost=cost, delay_charged=service['delay_charged']))
            elif geometry(old) != geometry(column):
                change = 'AUX_QUOTA'
                self.events.append(dict(time=now, event='ADAPT_AUX', job=name,
                                        old_cpu=old.query.cpu_per_gpu, new_cpu=column.query.cpu_per_gpu,
                                        old_ram=old.query.ram_gb_per_gpu, new_ram=column.query.ram_gb_per_gpu))
            if service['delay_charged'] > self.delay_bound + 1e-9:
                raise RuntimeError(f'INVALID_SERVICE_DELAY_LEDGER: {name}')
            service['column'] = column
            budget_changes[name] = dict(before=before, charged_this_round=cost,
                                        change=change)
        ledgers = {name: {key: service[key] for key in (
            'issued_time', 'productive_base', 'delay_charged',
            'last_geometry_productive', 'geometry_changes')}
                   for name, service in self.services.items()}
        for name, values in budget_changes.items():
            values['after'] = ledgers[name]
        independent.update(
            ready=list(self.ready), checkpoint_locks=sorted(locked),
            service_commitments=sorted(self.services), drain=drain,
            solver_trace=traces, service_delay_ledgers=ledgers,
            service_budget=budget_changes,
            adaptive_candidate_filters=filters,
            productive_service={j: productive.get(j, 0.) - v['productive_base']
                                for j, v in self.services.items()})
        return PolicyResult(
            'L2_ADAPTIVE', 'DRAIN_FOR_HEAD' if drain else 'OPTIMAL_ADAPTIVE_L2',
            selected, sorted(set(jobs) - admitted),
            sum(c.scheduling_utility for c in chosen),
            sum(c.query.effective_goodput for c in chosen),
            time.perf_counter() - start, audit=independent)
