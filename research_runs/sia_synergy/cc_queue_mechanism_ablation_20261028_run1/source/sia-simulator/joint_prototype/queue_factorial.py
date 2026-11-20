"""Opt-in 2^3 queue ablation. Historical controllers stay unchanged.

A: reciprocal physical ETA vs existing Sia normalized-goodput fairness score.
B: forced FIFO head/drain vs objective-driven optional admission.
C: completion-continuity vs existing 300 productive-second service segments.
All share exact locks, source candidates, physical execution, finite lifetime
startup cap, once-only legacy tail recovery, and nonempty-start safety.
No job-level starvation guarantee is claimed for B=0.
"""
import copy,math,time
from collections import Counter
from dataclasses import replace
import numpy as np
from scipy.optimize import LinearConstraint
from .adaptive_service import AdaptiveAdmission,gpu_geometry
from .neutral_liveness import geometry
from .joint_scheduler import PolicyResult
from .tail_recovery import TailRecoveryAdmission,TailRecoveryPolicy
from .small_mechanism import utility

class FactorCore(AdaptiveAdmission):
    def _decide(self, s, columns, now, productive):
        if self.factors[1:] == (1, 1):
            return super()._decide(s, columns, now, productive)
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
            if self.service_expired(name, service, productive):
                if s.lock_active(jobs[name]):
                    raise RuntimeError('CHECKPOINT_AFTER_SERVICE_WITHOUT_RECONFIGURATION')
                expired.append(name)
            else:
                self.check_service_clock(name, service, now, productive)
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
        # The common lifetime budget survives expiry, waiting, and reissue.
        filtered = []
        for column in columns:
            if column.job_id in held:
                filtered.append(column)
                continue
            job = jobs[column.job_id]
            history = self._lifetime_history.get(column.job_id, {})
            same = (job.current_config_id == column.config.config_id
                    and job.current_placement == column.placement.node_gpus)
            charge = 0. if same else self._delay(column.job_id, column)
            limit = self.delay_bound + history.get('tail_extra_budget', 0.)
            if history.get('delay_charged', 0.) + charge <= limit + 1e-9:
                filtered.append(column)
        columns = sorted(filtered, key=geometry)
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
            if optional and not required:
                # Shared nonempty-start safety. No allocation bonus, no FIFO
                # head requirement, no maximum-cardinality surrogate.
                constraints.append(LinearConstraint(np.ones(len(columns)), 1., np.inf))

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
        if self.ready and self.factors[1]:
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
            history = self._lifetime_history.get(name, {})
            spent = float(history.get('delay_charged', 0.))
            self.sequence += 1
            self.services[name] = dict(
                column=column, issued_time=now, productive_base=productive.get(name, 0.),
                sequence=self.sequence, delay_charged=spent+charged, geometry_changes=history.get('geometry_changes', 0),
                last_geometry_productive=productive.get(name, 0.))
            for key in ('tail_extra_budget', 'tail_recovery_used'):
                if key in history:
                    self.services[name][key] = history[key]
            if spent + charged > self.delay_limit(name, self.services[name]) + 1e-9:
                raise RuntimeError('LIFETIME_REISSUE_BUDGET_VIOLATION')
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
            if service['delay_charged'] > self.delay_limit(name, service) + 1e-9:
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


class QueueFactorAdmission(TailRecoveryAdmission, FactorCore):
    def service_expired(self, name, service, productive):
        if self.factors[2]:
            return False
        if productive.get(name, 0.)-service['productive_base'] < self.quantum-1e-9:
            return False
        # Expiry must not destroy a service which cannot ever restart under
        # the SHARED lifetime budget. This is a safety floor, not a fresh cap.
        cost = self._resume_lower_bound.get(name, math.inf)
        limit = self.delay_bound + service.get('tail_extra_budget', 0.)
        if service['delay_charged'] + cost > limit + 1e-9:
            self._budget_hold.add(name)
            return False
        return True

    def _decide(self, scheduler, columns, now, productive):
        self._lifetime_history = getattr(self, '_lifetime_history', {})
        self._lifetime_history.update(copy.deepcopy(self.services))
        self._resume_lower_bound = {
            name: min((self._delay(name,c) for c in columns if c.job_id==name), default=math.inf)
            for name in scheduler.job_map}
        self._budget_hold = set()
        result = super()._decide(scheduler, columns, now, productive)
        for name in sorted(self._budget_hold):
            self.events.append(dict(time=now,event='QUANTUM_HELD_NO_LIFETIME_RESUME_BUDGET',job=name))
        self._lifetime_history.update(copy.deepcopy(self.services))
        result.audit.update(queue_factors=dict(zip('ABC',self.factors)),
            lifetime_budget_preserved_across_reissue=True,
            budget_safety_holds=sorted(self._budget_hold),
            nonempty_start_safety=True,individual_starvation_guarantee=bool(self.factors[1]),
            service_rule='UNTIL_COMPLETION' if self.factors[2] else '300_PRODUCTIVE_SECONDS_WITH_NONDESTRUCTIVE_BUDGET_SAFETY')
        return result

class QueueFactorPolicy(TailRecoveryPolicy):
    def factor_columns(self,jobs,specs,scheduler):
        columns,proposals=super().factor_columns(jobs,specs,scheduler)
        if not self.queue_factors[0]:
            columns=utility(self,jobs,specs,columns)
            by={c.column_id:c.scheduling_utility for c in columns}
            for row in self.continuity_scores:
                row['completion_counterfactual_utility']=row['utility']
                row['utility']=by[row['column_id']]
        self.exchange_columns=columns
        return columns,proposals

    def optimize(self,*args):
        result=super().optimize(*args)
        d=self.last_decision;d['queue_factors']=dict(zip('ABC',self.queue_factors))
        d['shared_contract'].update(policy_version='QUEUE_2X2X2_V1',
            objective='SUM_RECIPROCAL_CURRENT_COMPLETION_ETA' if self.queue_factors[0] else 'SIA_NORMALIZED_GOODPUT_FAIR_UTILITY_AND_ORIGINAL_RESTART_PROXY',
            admission='FIFO_HEAD_AND_DRAIN' if self.queue_factors[1] else 'OPTIONAL_OBJECTIVE_WITH_NONEMPTY_START',
            service='UNTIL_COMPLETION' if self.queue_factors[2] else 'PRODUCTIVE_300S_WITH_SHARED_LIFETIME_CAP',
            source_candidates_shared=True,response_and_execution_unchanged=True,
            external_official_sia=False)
        return result

def install(cluster,a,b,c):
    factors=tuple(int(x) for x in (a,b,c))
    if any(x not in (0,1) for x in factors):raise ValueError('UNREGISTERED_FACTOR')
    if cluster.policy.liveness.seen:raise RuntimeError('INSTALL_BEFORE_FIRST_DECISION')
    cluster.policy.__class__=QueueFactorPolicy
    cluster.policy.queue_factors=factors
    cluster.policy.liveness.__class__=QueueFactorAdmission
    cluster.policy.liveness.factors=factors
    return cluster
