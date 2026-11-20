"""Conditional sum-ETA allocation after unchanged reference service selection.

No global JCT optimality claim: selection, future rate changes and resource
release are still approximate. Historical modes do not enter this module.
"""
import math
from dataclasses import replace

import numpy as np
from scipy.optimize import LinearConstraint

from .completion_continuity import (
    CompletionContinuityPolicy, ContinuityAdmission, install as install_previous)
from .neutral_liveness import geometry


class NetCompletionAdmission(ContinuityAdmission):
    def _primary_solution(self, s, columns, required, constraints, call,
                          weights, objective, optional):
        if not objective:
            return super()._primary_solution(s, columns, required, constraints,
                                             call, weights, objective, optional)
        records = self.completion_records
        reference_weights = np.asarray([
            records[c.column_id]['admission_utility'] for c in columns])
        base_count = len(constraints)
        reference = call(-reference_weights)
        if reference is None:
            return None, weights

        # Reproduce the entire previous selection, including startup and
        # canonical ties. Comparing only the first MILP's membership could
        # silently change admission on equal primary utilities.
        optimum = sum(records[c.column_id]['admission_utility'] for c in reference)
        constraints.append(LinearConstraint(reference_weights, optimum-1e-9, optimum+1e-9))
        reference = self._secondary_tie(s, columns, required, constraints, call, reference)
        order = ([j.job_id for j in sorted(s.jobs, key=lambda j:
            (j.arrival_time, j.source_row_index, j.job_id))]
            if optional else sorted(required))
        for name in order:
            if optional and name not in required:
                mask = np.asarray([float(c.job_id == name) for c in columns])
                reference = call(-mask)
                if reference is None:
                    raise RuntimeError('NET_REFERENCE_PRESENCE_INCONSISTENT')
                present = float(any(c.job_id == name for c in reference))
                constraints.append(LinearConstraint(mask, present, present))
                if not present:
                    continue
            cost = np.zeros(len(columns))
            for rank, i in enumerate(i for i,c in enumerate(columns) if c.job_id == name):
                cost[i] = rank
            reference = call(cost)
            if reference is None:
                raise RuntimeError('NET_REFERENCE_CANONICAL_INCONSISTENT')
            chosen = next(c for c in reference if c.job_id == name)
            mask = np.asarray([float(c is chosen) for c in columns])
            constraints.append(LinearConstraint(mask, 1., 1.))

        # Remove only temporary objective/tie rows. Capacity/required-job/lock
        # rows remain intact. Freeze membership, not the chosen GPU geometry.
        del constraints[base_count:]
        selected = {c.job_id: c for c in reference}
        unknown = {}
        for job in s.jobs:
            present = float(job.job_id in selected)
            mask = np.asarray([float(c.job_id == job.job_id) for c in columns])
            constraints.append(LinearConstraint(mask, present, present))
            if not present:
                continue
            old = selected[job.job_id]
            if records[old.column_id]['estimated_completion'] is None:
                unknown[job.job_id] = old.column_id
                mask = np.asarray([float(c is old) for c in columns])
                constraints.append(LinearConstraint(mask, 1., 1.))
            else:
                mask = np.asarray([float(c.job_id == job.job_id and
                    records[c.column_id]['estimated_completion'] is None) for c in columns])
                constraints.append(LinearConstraint(mask, 0., 0.))
        answer = call(-weights)
        if answer is None:
            raise RuntimeError('NET_FIXED_REFERENCE_SET_INFEASIBLE')
        self.net_phase = dict(reference_jobs=sorted(selected),
            reference_columns={j:c.column_id for j,c in selected.items()},
            reference_eta_sum=sum(records[c.column_id]['estimated_completion'] or 0.
                                  for c in reference),
            reference_reciprocal_objective=optimum,
            unknown_frozen=unknown,
            unknown_cost='OMITTED_FIXED_COLUMN_CONSTANT_NOT_ZERO_ETA',
            second_stage='MIN_SUM_CURRENT_ETA_WITH_EXACT_REFERENCE_MEMBERSHIP')
        return answer, weights

    def _baseline_comparable(self, baseline, selected):
        chosen = {c.job_id: c for c in selected}
        for old in baseline:
            new = chosen[old.job_id]
            a = self.completion_records[old.column_id]['estimated_completion']
            b = self.completion_records[new.column_id]['estimated_completion']
            if a is None or b is None:
                if a is not None or b is not None or geometry(old) != geometry(new):
                    return False
        return True

    def _decide(self, scheduler, columns, now, productive):
        self.net_phase = None
        for c in columns:
            row = self.completion_records[c.column_id]
            eta = row['estimated_completion']
            if eta is not None and (not math.isfinite(eta) or eta <= 0):
                raise RuntimeError('INVALID_NET_COMPLETION_ESTIMATE')
        result = super()._decide(scheduler, columns, now, productive)
        if self.net_phase is None:
            raise RuntimeError('MISSING_NET_COMPLETION_PHASE')
        chosen_cost = sum(self.completion_records[c.column_id]['estimated_completion'] or 0.
                          for c in result.selected.values())
        if (set(result.selected) != set(self.net_phase['reference_jobs']) or
                chosen_cost > self.net_phase['reference_eta_sum'] + 1e-5):
            raise RuntimeError('NET_REFERENCE_COST_OR_MEMBERSHIP_REGRESSION')
        self.net_phase['chosen_eta_sum'] = chosen_cost
        result.audit['net_completion'] = self.net_phase
        result.policy = 'JOINT_NET_COMPLETION'
        result.status = 'DRAIN_FOR_HEAD' if result.audit['drain'] else 'OPTIMAL_NET_COMPLETION'
        return result


class NetCompletionPolicy(CompletionContinuityPolicy):
    def factor_columns(self, jobs, specs, scheduler):
        columns, proposals = super().factor_columns(jobs, specs, scheduler)
        scale = max((r['estimated_completion'] for r in self.completion_scores
                     if r['estimated_completion'] is not None), default=1.)
        records = {}
        revised = []
        for col, row in zip(columns, self.completion_scores):
            eta = row['estimated_completion']
            value = 0. if eta is None else -eta / scale
            row.update(admission_utility=row['utility'], net_completion_cost=eta,
                       net_completion_scale=scale, utility=value)
            records[col.column_id] = row.copy()
            revised.append(replace(col, scheduling_utility=value))
        self.liveness.completion_records = records
        self.exchange_columns = revised
        return revised, proposals

    def optimize(self, *args):
        result = super().optimize(*args)
        self.last_decision['shared_contract'].update(
            policy_version='JOINT_NET_COMPLETION_V1',
            objective='MIN_SUM_CURRENT_NET_COMPLETION_ETA_FIXED_REFERENCE_SET',
            admission='UNCHANGED_RECIPROCAL_REFERENCE_WITH_COMPLETION_CONTINUITY',
            cold_start='EXACT_REFERENCE_BOOTSTRAP_NO_UNKNOWN_ETA_SUBSTITUTION',
            reciprocal_eta_priority=False, reciprocal_eta_for_admission_only=True,
            global_jct_optimal=False)
        self.last_decision['probe']['contract']['objective'] = (
            'MIN_SUM_CURRENT_NET_COMPLETION_ETA_FIXED_REFERENCE_SET')
        return result


def install(cluster, proportional=False):
    install_previous(cluster, proportional=proportional)
    cluster.policy.__class__ = NetCompletionPolicy
    cluster.policy.liveness.__class__ = NetCompletionAdmission
    return cluster
