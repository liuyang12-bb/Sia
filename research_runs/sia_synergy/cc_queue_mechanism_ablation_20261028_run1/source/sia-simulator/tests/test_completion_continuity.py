"""Bounded regression contracts for opt-in completion continuity.

The implementation is imported lazily so this suite records the absent/new
mechanism as red before implementation. Real integration cases use the original
unseeded BERT fixture and genuine Cluster.optimize_policy calls. No observation
is extended beyond the fixed 12 x 60 second limit.
"""
import contextlib
import copy
import importlib
import io
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_exchange_integration as exchange_fixture
from joint_prototype.execution_audit import audit_execution


def implementation():
    return importlib.import_module('joint_prototype.completion_continuity')


def selected_geometry(record):
    return (record['config']['config_id'],
            tuple(sorted(record['placement'].items())))


class CompletionContinuityTests(unittest.TestCase):
    def build(self, proportional=False, new_score=True, continuity=True):
        return implementation().install(
            exchange_fixture.ExchangeIntegration().build(),
            proportional=proportional, new_score=new_score,
            continuity=continuity)

    def test_remaining_work_does_not_cancel_completion_priority(self):
        priority = implementation().completion_priority
        # Two jobs competing for the same one-GPU slot at the same rate.
        # The old best-proportional wait-minus-ETA score erased this difference.
        self.assertAlmostEqual(priority(100., 2., 1., 0.), .02)
        self.assertAlmostEqual(priority(1000., 2., 1., 0.), .002)
        self.assertGreater(priority(100., 2., 1., 0.),
                           priority(1000., 2., 1., 0.))

    def test_startup_cost_can_reverse_nominal_speed_ranking(self):
        priority = implementation().completion_priority
        # Doubling speed cannot repay a 60-second restart for 100 seconds of
        # remaining work, but it can repay it for 1,000 seconds of work.
        self.assertGreater(priority(100., 1., 1., 0.),
                           priority(100., 2., 1., 60.))
        self.assertLess(priority(1000., 1., 1., 0.),
                        priority(1000., 2., 1., 60.))
        self.assertAlmostEqual(priority(120., 2., 3., 10.), 1. / 30.)

    def test_invalid_priority_inputs_are_rejected(self):
        priority = implementation().completion_priority
        for values in ((1., 0., 1., 0.), (1., 1., 0., 0.),
                       (-1., 1., 1., 0.), (1., 1., 1., -1.),
                       (1., math.nan, 1., 0.),
                       (1., 1., math.inf, 0.)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    priority(*values)

    def test_unseeded_unknown_eta_bootstraps_without_fabricated_rate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            c.step(60)
            d = c.policy.last_decision
            self.assertTrue(d['result']['audit']['valid'])
            self.assertIn('bert-mid', d['result']['selected'])
            self.assertTrue(d['continuity_scores'])
            for score in d['continuity_scores']:
                self.assertIsNone(score['raw_rate'])
                self.assertIsNone(score['estimated_completion'])
                self.assertEqual(score['utility'], 0.)

    def test_real_unseeded_bert_full_scheduling_execution_chain(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            productive = startup = 0.
            previous = 0.
            known_scores = []
            for _ in range(12):
                c.step(60)
                d = c.policy.last_decision
                self.assertTrue(d['result']['audit']['valid'])
                self.assertGreaterEqual(c.jobs[0].progress, previous)
                previous = c.jobs[0].progress
                known_scores.extend(v for v in d['continuity_scores']
                                    if v['raw_rate'] is not None)
                for execution in c.joint_round_logs[-1]['execution']:
                    self.assertTrue(audit_execution(execution)['valid'])
                    productive += execution['training_seconds']
                    startup += execution['checkpoint_delay_consumed']
                    span = max(0., execution['interval_end'] - max(
                        execution['interval_start'], execution['arrival_time']))
                    if execution['checkpoint_delay_consumed'] >= span > 0:
                        self.assertEqual(execution['progress_before'],
                                         execution['progress_after'])
            self.assertGreater(productive, 0.)
            self.assertGreater(startup, 0.)
            self.assertGreater(c.jobs[0].progress, 0.)
            self.assertIsNotNone(c.jobs[0].first_productive_training_time)
            self.assertTrue(known_scores)
            self.assertTrue(all(v['utility'] > 0. for v in known_scores))

    def test_real_checkpoint_preserves_full_gpu_and_aux_allocation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            c.step(60)
            self.assertGreater(c.jobs[0].rescale_time, 60.)
            allocation = copy.deepcopy(c.allocations)
            resources = copy.deepcopy(c.policy.current_resources)
            before = c.jobs[0].progress
            c.step(60)
            self.assertEqual(c.allocations, allocation)
            self.assertEqual(c.jobs[0].progress, before)
            for key in ('config_id', 'cpu_per_gpu', 'ram_gb_per_gpu', 'placement'):
                self.assertEqual(c.policy.current_resources['bert-mid'][key],
                                 resources['bert-mid'][key])
            self.assertIn('bert-mid',
                          c.policy.last_decision['result']['audit']['checkpoint_locks'])

    def test_crossing_300_productive_seconds_does_not_force_requeue(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            events = []
            for _ in range(12):
                c.step(60)
                self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
                events.extend(c.policy.last_decision['liveness_events'])
            self.assertGreater(c.jobs[0].execution_time, 300.)
            self.assertFalse(any(e['event'] == 'REQUEUE_PRODUCTIVE_QUANTUM'
                                 for e in events))
            self.assertIn('bert-mid', c.policy.last_decision['result']['selected'])

    def test_no_event_round_cannot_change_incumbent_gpu_geometry(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            previous = {}
            checked = 0
            for _ in range(12):
                c.step(60)
                d = c.policy.last_decision
                reasons = d['result']['audit']['geometry_event_reasons']
                self.assertIsInstance(reasons, list)
                current = {name: selected_geometry(v)
                           for name, v in d['result']['selected'].items()}
                if not reasons:
                    for name in previous.keys() & current.keys():
                        self.assertEqual(previous[name], current[name])
                        checked += 1
                previous = current
            self.assertGreater(checked, 0)

    def test_proportional_ablation_cannot_choose_free_aux_quotas(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build(proportional=True)
            count = 0
            for _ in range(8):
                c.step(60)
                d = c.policy.last_decision
                self.assertTrue(d['result']['audit']['valid'])
                for record in d['result']['selected'].values():
                    typ = record['config']['gpu_type']
                    capacity = c.policy.node_capacities[typ]
                    self.assertAlmostEqual(record['cpu_per_gpu'],
                        capacity['cpu_cores'] / c.ngpus_per_node[typ])
                    self.assertAlmostEqual(record['ram_gb_per_gpu'],
                        capacity['ram_gb'] / c.ngpus_per_node[typ])
                    count += 1
            self.assertGreater(count, 0)

    def test_joint_and_prop_share_objective_liveness_and_response(self):
        with contextlib.redirect_stdout(io.StringIO()):
            jo = self.build()
            jp = self.build(proportional=True)
            self.assertIs(type(jo.policy.liveness), type(jp.policy.liveness))
            self.assertEqual(jo.policy.oracle.provenance_digest(),
                             jp.policy.oracle.provenance_digest())
            self.assertEqual(jo.policy.aux_cpu_points, jp.policy.aux_cpu_points)
            self.assertEqual(jo.policy.aux_ram_points, jp.policy.aux_ram_points)
            for c in (jo, jp):
                c.step(60)
                self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
            jo_contract = jo.policy.last_decision['shared_contract']
            jp_contract = jp.policy.last_decision['shared_contract']
            self.assertEqual(jo_contract['objective'], jp_contract['objective'])
            self.assertEqual(jo_contract['admission'], jp_contract['admission'])

    def test_solver_failure_leaves_allocation_resources_and_service_unchanged(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            c.step(60)
            before_alloc = copy.deepcopy(c.allocations)
            before_resources = copy.deepcopy(c.policy.current_resources)
            before_service = copy.deepcopy(c.policy.liveness.__dict__)
            # Force an observable source-metadata difference during factor
            # construction; it too must be rolled back on solver failure.
            c.policy.liveness.profile_sources = {'failure_sentinel': {}}
            before_service = copy.deepcopy(c.policy.liveness.__dict__)
            with patch('joint_prototype.adaptive_service.milp',
                       return_value=SimpleNamespace(x=None, status=1,
                           success=False, message='fixed unit timeout')):
                with self.assertRaises(RuntimeError):
                    c.optimize_policy()
            self.assertEqual(c.allocations, before_alloc)
            self.assertEqual(c.policy.current_resources, before_resources)
            self.assertEqual(c.policy.liveness.__dict__, before_service)


if __name__ == '__main__':
    unittest.main(verbosity=2)
