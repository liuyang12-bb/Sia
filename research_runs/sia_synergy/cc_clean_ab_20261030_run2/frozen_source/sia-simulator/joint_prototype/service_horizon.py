"""Opt-in service-aware scoring for protected joint admission.

The baseline policy scores every candidate over 300 wall seconds.  A held
service can have less productive time left; this policy scores its candidates
over that remaining commitment.  Job selection and capacity constraints are
still solved jointly, including resource donations by slower jobs.
"""
import math
from dataclasses import replace

from .exchange_netprogress import fair_utility
from .protected_admission import ProtectedPolicy, install as install_protected


class ServiceHorizonPolicy(ProtectedPolicy):
    def factor_columns(self, jobs, specs, scheduler):
        columns, proposals = super().factor_columns(jobs, specs, scheduler)
        services = self.liveness.services
        productive = self.productive
        revised = []
        evidence = []
        for column, old in zip(columns, self.exchange_scores):
            service = services.get(column.job_id)
            if service is None:
                horizon = self.liveness.quantum
            else:
                remaining = self.liveness.quantum - (
                    productive.get(column.job_id, 0.) - service['productive_base'])
                if not math.isfinite(remaining):
                    raise RuntimeError('INVALID_REMAINING_PRODUCTIVE_SERVICE')
                # Scoring precedes the admission state machine.  An expired
                # service will be requeued there and, if selected, issued a
                # fresh quantum rather than continuing the old commitment.
                horizon = (self.liveness.quantum if remaining <= 1e-9 else
                           max(self.liveness.interval, remaining))
            net_rate = (old['normalized_rate_before_delay']
                        * max(horizon - old['candidate_delay'], 0.) / horizon)
            score = fair_utility(net_rate, self.p_fairness, self.lambda_n)
            revised.append(replace(column, scheduling_utility=score))
            evidence.append(dict(old, service_horizon=horizon,
                                 net_normalized_rate=net_rate, utility=score))
        self.exchange_columns = revised
        self.exchange_scores = evidence
        return revised, proposals

    def optimize(self, *args):
        result = super().optimize(*args)
        self.last_decision['shared_contract'].update(
            policy_version='JOINT_SERVICE_HORIZON_V1',
            scoring='REMAINING_PRODUCTIVE_COMMITMENT_FOR_HELD_Q_FOR_NEW')
        return result


def install(cluster):
    install_protected(cluster, allow_exchange=True, net_progress=True)
    cluster.policy.__class__ = ServiceHorizonPolicy
    return cluster
