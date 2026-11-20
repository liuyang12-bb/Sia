"""Net completion cost contracts, with an executable previous-policy red mode.

Controller witnesses use production candidate/capacity/lock/solver paths and
explicit synthetic ETA tables. They are not measurements or fake application
profiles. The bounded integration tests use the genuine unseeded BERT fixture,
its real optimize_policy path, and at most 12 scheduling periods.

NET_COMPLETION_TEST_LEGACY=1 runs the same contracts against the previous
reciprocal-ETA controller, preserving proof of the harmful exchange choice.
"""
import contextlib
import copy
import importlib
import io
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from test_adaptive_service import admission, cfg, current, scheduler, spec
import test_exchange_integration as real_fixture
from joint_prototype.completion_continuity import (
    ContinuityAdmission, install as install_previous)
from joint_prototype.execution_audit import audit_execution
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.neutral_liveness import geometry


class LegacyReciprocalAdapter(ContinuityAdmission):
    """Red-only adapter: retain the real previous MILP and admission semantics."""
    def decide(self, scheduler, columns, now, productive):
        previous = [replace(c, scheduling_utility=
            self.completion_records[c.column_id]['admission_utility'])
            for c in columns]
        return super().decide(scheduler, previous, now, productive)


def implementation():
    if os.environ.get('NET_COMPLETION_TEST_LEGACY') == '1':
        return SimpleNamespace(NetCompletionAdmission=LegacyReciprocalAdapter,
                               install=install_previous)
    return importlib.import_module('joint_prototype.net_completion')


def scoring(columns, eta):
    """Independent ETA inputs, not derived from the production objective."""
    known = [float(eta[c.column_id]) for c in columns
             if eta[c.column_id] is not None]
    cost_scale = max(known, default=1.)
    priority_scale = max((1. / v for v in known), default=1.)
    records = {}
    result = []
    for c in columns:
        value = eta[c.column_id]
        records[c.column_id] = dict(estimated_completion=value,
            admission_utility=0. if value is None else 1. / value / priority_scale)
        result.append(replace(c, scheduling_utility=
            0. if value is None else -value / cost_scale))
    return result, records


def controller(jobs, records, previous=False):
    cls = LegacyReciprocalAdapter if previous else implementation().NetCompletionAdmission
    a = cls(300., 250., 60.)
    baseline = admission(jobs, 50.)
    a.delay_costs = baseline.delay_costs
    a.profile_sources = baseline.profile_sources
    a.completion_records = copy.deepcopy(records)
    return a


def mark_known(s, records, columns):
    s.current_known_configs = {(c.job_id, c.config.config_id) for c in columns
        if records[c.column_id]['estimated_completion'] is not None}


def exchange_fixture(donor_alternative=1200.):
    jobs = [spec('receiver'), spec('donor', index=1)]
    s = scheduler(jobs, [NodeSpec('aws:0', 'aws', 2, 12., 250.)])
    costs = {'receiver': {6.: 100., 9.: 80.},
             'donor': {6.: 1000., 3.: float(donor_alternative)}}
    columns = [c for c in s.build_joint_columns()
        if c.query.ram_gb_per_gpu == 125.
        and c.query.cpu_per_gpu in costs[c.job_id]]
    eta = {c.column_id: costs[c.job_id][c.query.cpu_per_gpu] for c in columns}
    columns, records = scoring(columns, eta)
    mark_known(s, records, columns)
    return jobs, s, columns, records


def selected_eta(result, records):
    return sum(records[c.column_id]['estimated_completion']
               for c in result.selected.values())


class NetCompletionTests(unittest.TestCase):
    def test_previous_solver_reproduces_harmful_donor_exchange(self):
        jobs, s, columns, records = exchange_fixture()
        old = controller(jobs, records, previous=True)
        result = old.decide(s, columns, 0., {})
        self.assertEqual(set(result.selected), {'receiver', 'donor'})
        self.assertEqual(result.selected['receiver'].query.cpu_per_gpu, 9.)
        self.assertEqual(result.selected['donor'].query.cpu_per_gpu, 3.)
        self.assertEqual(selected_eta(result, records), 1280.)
        self.assertTrue(result.audit['valid'])

    def test_harmful_exchange_is_rejected_by_sum_completion_cost(self):
        jobs, s, columns, records = exchange_fixture()
        result = controller(jobs, records).decide(s, columns, 0., {})
        self.assertEqual(set(result.selected), {'receiver', 'donor'})
        self.assertEqual(result.selected['receiver'].query.cpu_per_gpu, 6.)
        self.assertEqual(result.selected['donor'].query.cpu_per_gpu, 6.)
        self.assertEqual(selected_eta(result, records), 1100.)
        self.assertTrue(result.audit['valid'])

    def test_beneficial_exchange_can_slow_donor(self):
        jobs, s, columns, records = exchange_fixture(1010.)
        result = controller(jobs, records).decide(s, columns, 0., {})
        self.assertEqual(result.selected['receiver'].query.cpu_per_gpu, 9.)
        self.assertEqual(result.selected['donor'].query.cpu_per_gpu, 3.)
        self.assertEqual(selected_eta(result, records), 1090.)
        self.assertGreater(records[result.selected['donor'].column_id]
                           ['estimated_completion'], 1000.)
        self.assertTrue(result.audit['valid'])

    def test_negative_cost_does_not_drop_optional_selected_job(self):
        jobs, s, columns, records = exchange_fixture()
        columns = [c for c in columns if c.query.cpu_per_gpu == 6.]
        result = controller(jobs, records).decide(s, columns, 0., {})
        self.assertEqual(set(result.selected), {'receiver', 'donor'})
        self.assertEqual(selected_eta(result, records), 1100.)
        self.assertLess(result.scheduling_objective, 0.)

    def test_stage_two_preserves_reciprocal_selected_task_set(self):
        for donor_eta in (1010., 1200.):
            with self.subTest(donor_eta=donor_eta):
                jobs, s, columns, records = exchange_fixture(donor_eta)
                old = controller(jobs, records, previous=True).decide(s, columns, 0., {})
                new = controller(jobs, records).decide(s, columns, 0., {})
                self.assertEqual(set(old.selected), set(new.selected))
                self.assertEqual(old.audit['required_jobs'], new.audit['required_jobs'])

    def test_reference_optional_tie_uses_source_order_not_job_id(self):
        jobs = [spec('head', index=0), spec('a_later', index=2),
                spec('z_earlier', index=1)]
        s = scheduler(jobs, [NodeSpec('aws:0', 'aws', 2, 12., 250.)])
        columns = s.build_proportional_columns()
        eta = {c.column_id: 100. if c.job_id == 'head' else 1000.
               for c in columns}
        columns, records = scoring(columns, eta)
        mark_known(s, records, columns)
        old = controller(jobs, records, previous=True).decide(s, columns, 0., {})
        new = controller(jobs, records).decide(s, columns, 0., {})
        self.assertEqual(set(old.selected), {'head', 'z_earlier'})
        self.assertEqual(set(new.selected), set(old.selected))
        self.assertEqual(new.unallocated_jobs, ['a_later'])
        self.assertTrue(new.audit['valid'])

    def test_known_job_cannot_escape_eta_cost_via_unknown_configuration(self):
        jobs = [spec(configs=[cfg('aws'), cfg('aws', 2)])]
        s = scheduler(jobs, [NodeSpec('aws:0', 'aws', 2, 12., 250.)])
        columns = s.build_proportional_columns()
        eta = {c.column_id: 100. if c.config.gpus == 1 else None for c in columns}
        columns, records = scoring(columns, eta)
        mark_known(s, records, columns)
        result = controller(jobs, records).decide(s, columns, 0., {})
        self.assertEqual(result.selected['j'].config.gpus, 1)
        self.assertEqual(records[result.selected['j'].column_id]['estimated_completion'], 100.)

    def test_unknown_selected_job_keeps_exact_admission_bootstrap_column(self):
        jobs = [spec(configs=[cfg('aws'), cfg('aws', 2)])]
        s = scheduler(jobs, [NodeSpec('aws:0', 'aws', 2, 12., 250.)])
        columns = s.build_proportional_columns()
        columns, records = scoring(columns, {c.column_id: None for c in columns})
        mark_known(s, records, columns)
        old = controller(jobs, records, previous=True).decide(s, columns, 0., {})
        new = controller(jobs, records).decide(s, columns, 0., {})
        self.assertEqual(geometry(new.selected['j']), geometry(old.selected['j']))
        self.assertIsNone(records[new.selected['j'].column_id]['estimated_completion'])
        self.assertTrue(new.audit['valid'])

    def test_current_checkpoint_locks_remain_exact_despite_alternative_costs(self):
        jobs, s, columns, records = exchange_fixture(1010.)
        a = controller(jobs, records)
        first = a.decide(s, [c for c in columns if c.query.cpu_per_gpu == 6.], 0., {})
        locked_jobs = [current(j, first.selected, locked=True) for j in jobs]
        s = scheduler(locked_jobs, [NodeSpec('aws:0', 'aws', 2, 12., 250.)])
        locked_columns = s.build_joint_columns()
        locked_columns, records = scoring(locked_columns,
            {c.column_id: 10000. for c in locked_columns})
        mark_known(s, records, locked_columns)
        a.completion_records = records
        result = a.decide(s, locked_columns, 60., {})
        for name in first.selected:
            self.assertEqual(geometry(first.selected[name]), geometry(result.selected[name]))
        self.assertEqual(set(result.audit['checkpoint_locks']), set(first.selected))
        self.assertTrue(result.audit['valid'])

    def test_solver_failure_does_not_commit_controller_or_resource_state(self):
        jobs, s, columns, records = exchange_fixture()
        a = controller(jobs, records)
        before = copy.deepcopy(a.__dict__)
        with patch('joint_prototype.adaptive_service.milp', return_value=
                   SimpleNamespace(x=None, status=1, success=False,
                                   message='fixed injected timeout')):
            with self.assertRaisesRegex(RuntimeError, 'SOLVER_UNDECIDED'):
                a.decide(s, columns, 0., {})
        self.assertEqual(a.__dict__, before)

    def test_second_stage_failure_after_reference_solve_is_atomic(self):
        jobs, s, columns, records = exchange_fixture()
        a = controller(jobs, records)
        before = copy.deepcopy(a.__dict__)
        real_solver = type(a)._milp
        calls = {'reference': 0, 'second_stage': 0}

        def fail_only_second_stage(owner, cost, actual_columns, constraints):
            reference = [-records[c.column_id]['admission_utility']
                         for c in actual_columns]
            completion = [-c.scheduling_utility for c in actual_columns]
            if all(abs(float(v) - expected) < 1e-12
                   for v, expected in zip(cost, reference)):
                calls['reference'] += 1
            if all(abs(float(v) - expected) < 1e-12
                   for v, expected in zip(cost, completion)):
                calls['second_stage'] += 1
                # The second pass must already retain node/required rows and
                # add explicit membership for every active job.
                self.assertGreaterEqual(len(constraints), 1 + len(jobs))
                return SimpleNamespace(x=None, status=1, success=False,
                                       message='injected after complete reference solve')
            return real_solver(owner, cost, actual_columns, constraints)

        with patch.object(type(a), '_milp', new=fail_only_second_stage):
            with self.assertRaisesRegex(RuntimeError, 'SOLVER_UNDECIDED'):
                a.decide(s, columns, 0., {})
        self.assertEqual(calls, {'reference': 1, 'second_stage': 1})
        self.assertEqual(a.__dict__, before)

    def test_every_selected_exchange_is_jointly_within_capacity(self):
        for donor_eta in (1010., 1200.):
            jobs, s, columns, records = exchange_fixture(donor_eta)
            result = controller(jobs, records).decide(s, columns, 0., {})
            self.assertTrue(s.audit(list(result.selected.values()))['valid'])
            used = [c.usage_for('aws:0') for c in result.selected.values()]
            self.assertEqual(sum(v[0] for v in used), 2.)
            self.assertLessEqual(sum(v[1] for v in used), 12.)
            self.assertLessEqual(sum(v[2] for v in used), 250.)

    def build(self, proportional=False):
        return implementation().install(real_fixture.ExchangeIntegration().build(), proportional=proportional)

    def test_real_unseeded_bert_full_chain_and_nonduplicated_progress(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            prior_progress = 0.
            training = checkpoint = 0.
            for _ in range(12):
                c.step(60)
                decision = c.policy.last_decision
                self.assertTrue(decision['result']['audit']['valid'])
                self.assertGreaterEqual(c.jobs[0].progress, prior_progress)
                prior_progress = c.jobs[0].progress
                for execution in c.joint_round_logs[-1]['execution']:
                    self.assertTrue(audit_execution(execution)['valid'])
                    training += execution['training_seconds']
                    checkpoint += execution['checkpoint_delay_consumed']
                    interval = max(0., execution['interval_end'] - max(
                        execution['interval_start'], execution['arrival_time']))
                    if execution['checkpoint_delay_consumed'] >= interval > 0:
                        self.assertEqual(execution['progress_before'], execution['progress_after'])
            self.assertGreater(training, 300.)
            self.assertGreater(checkpoint, 0.)
            self.assertGreater(c.jobs[0].progress, 0.)
            self.assertIsNotNone(c.jobs[0].first_productive_training_time)

    def test_real_checkpoint_protects_full_allocation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            c.step(60)
            self.assertGreater(c.jobs[0].rescale_time, 60.)
            allocation = copy.deepcopy(c.allocations)
            resources = copy.deepcopy(c.policy.current_resources)
            progress = c.jobs[0].progress
            c.step(60)
            self.assertEqual(c.allocations, allocation)
            self.assertEqual(c.jobs[0].progress, progress)
            for key in ('config_id', 'cpu_per_gpu', 'ram_gb_per_gpu', 'placement'):
                self.assertEqual(c.policy.current_resources['bert-mid'][key],
                                 resources['bert-mid'][key])

    def test_real_proportional_ablation_obeys_node_proportions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build(proportional=True)
            checked = 0
            for _ in range(8):
                c.step(60)
                for record in c.policy.last_decision['result']['selected'].values():
                    typ = record['config']['gpu_type']
                    capacity = c.policy.node_capacities[typ]
                    self.assertAlmostEqual(record['cpu_per_gpu'],
                        capacity['cpu_cores'] / c.ngpus_per_node[typ])
                    self.assertAlmostEqual(record['ram_gb_per_gpu'],
                        capacity['ram_gb'] / c.ngpus_per_node[typ])
                    checked += 1
            self.assertGreater(checked, 0)

    def test_prop_and_optimized_share_objective_admission_and_response(self):
        with contextlib.redirect_stdout(io.StringIO()):
            jo, jp = self.build(), self.build(proportional=True)
            self.assertIs(type(jo.policy.liveness), type(jp.policy.liveness))
            self.assertEqual(jo.policy.oracle.provenance_digest(), jp.policy.oracle.provenance_digest())
            for c in (jo, jp):
                c.step(60)
                self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
            first, second = (c.policy.last_decision['shared_contract'] for c in (jo, jp))
            self.assertEqual(first['objective'], second['objective'])
            self.assertEqual(first['admission'], second['admission'])
            self.assertNotEqual(first['objective'], 'SUM_RECIPROCAL_CURRENT_COMPLETION_ETA')

    def test_real_solver_failure_is_atomic(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            c.step(60)
            allocations = copy.deepcopy(c.allocations)
            resources = copy.deepcopy(c.policy.current_resources)
            c.policy.liveness.profile_sources = {'failure_sentinel': {}}
            controller_before = copy.deepcopy(c.policy.liveness.__dict__)
            with patch('joint_prototype.adaptive_service.milp', return_value=
                       SimpleNamespace(x=None, status=1, success=False,
                                       message='fixed integration timeout')):
                with self.assertRaises(RuntimeError):
                    c.optimize_policy()
            self.assertEqual(c.allocations, allocations)
            self.assertEqual(c.policy.current_resources, resources)
            self.assertEqual(c.policy.liveness.__dict__, controller_before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
