"""Write an isolated opt-in controller derived mechanically from frozen core."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SIM=ROOT/'cc-joint-queue-ablation/sia-simulator'
original=(SIM/'joint_prototype/adaptive_service.py').read_text(encoding='utf-8')
core=original[original.index('    def _decide(self, s, columns, now, productive):'):]
core=core.replace('        start = time.perf_counter()','''        if self.factors[1:] == (1, 1):
            return super()._decide(s, columns, now, productive)
        start = time.perf_counter()''',1)
core=core.replace('        columns = sorted(columns, key=geometry)','''        # The common lifetime budget survives expiry, waiting, and reissue.
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
        columns = sorted(filtered, key=geometry)''',1)
core=core.replace("            constraints = [LinearConstraint(matrix, lower, upper)]",'''            constraints = [LinearConstraint(matrix, lower, upper)]
            if optional and not required:
                # Shared nonempty-start safety. No allocation bonus, no FIFO
                # head requirement, no maximum-cardinality surrogate.
                constraints.append(LinearConstraint(np.ones(len(columns)), 1., np.inf))''',1)
core=core.replace('        if self.ready:\n            head = self.ready[0]', '        if self.ready and self.factors[1]:\n            head = self.ready[0]',1)
core=core.replace("            charged = 0. if same_gpu else self._delay(name, column)\n            self.sequence += 1",'''            charged = 0. if same_gpu else self._delay(name, column)
            history = self._lifetime_history.get(name, {})
            spent = float(history.get('delay_charged', 0.))
            self.sequence += 1''',1)
core=core.replace('sequence=self.sequence, delay_charged=charged, geometry_changes=0,','sequence=self.sequence, delay_charged=spent+charged, geometry_changes=history.get(\'geometry_changes\', 0),',1)
core=core.replace("            budget_changes[name] = dict(before=None, charged_this_round=charged,",'''            for key in ('tail_extra_budget', 'tail_recovery_used'):
                if key in history:
                    self.services[name][key] = history[key]
            if spent + charged > self.delay_limit(name, self.services[name]) + 1e-9:
                raise RuntimeError('LIFETIME_REISSUE_BUDGET_VIOLATION')
            budget_changes[name] = dict(before=None, charged_this_round=charged,''',1)
header='''"""Opt-in 2^3 queue ablation. Historical controllers stay unchanged.

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
'''
footer='''

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
'''
target=SIM/'joint_prototype/queue_factorial.py'
with target.open('x',encoding='utf-8') as f:f.write(header+core+footer)
compile(target.read_text(encoding='utf-8'),str(target),'exec')
print('ISOLATED_MODULE_WRITTEN',target)
