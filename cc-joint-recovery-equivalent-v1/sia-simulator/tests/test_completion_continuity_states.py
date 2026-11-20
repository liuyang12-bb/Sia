"""Independent bounded controller sequences; no Job.step or train simulation.

Supplied progress counters and model-availability events are explicit unit
inputs. Capacity, candidate validation, checkpoint locks and the MILP use the
production implementations. The analytic exchange response is reused only as
a labelled synthetic correctness witness, never as application performance.
"""
import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from test_adaptive_service import (admission, cfg, columns, current,
                                   scheduler, spec)
from test_exchange_netprogress import witness
from joint_prototype.completion_continuity import ContinuityAdmission
from joint_prototype.joint_profile import NodeSpec


def controller(jobs, delay=50.):
    result = ContinuityAdmission(300., 250., 60.)
    baseline = admission(jobs, delay)
    result.delay_costs = baseline.delay_costs
    result.profile_sources = baseline.profile_sources
    return result


def state(jobs, nodes=None, known=None):
    result = scheduler(jobs, nodes)
    result.current_known_configs = (set(known) if known is not None else {
        (j.job_id, c.config_id) for j in jobs for c in j.configs})
    return result


def start(job, delay=50., known=None):
    a = controller([job], delay)
    s = state([job], known=known)
    first_config = job.configs[0].config_id
    allowed = [c for c in columns(s) if c.config.config_id == first_config]
    return a, a.decide(s, allowed, 0., {job.job_id: 0.})


class CompletionContinuityStateTests(unittest.TestCase):
    def test_gang_waits_beyond_quantum_then_starts_after_incumbent_completion(self):
        small = spec('small')
        gang = spec('gang', [cfg('aws', 2)], arrival=60., index=1)
        peer = spec('peer', arrival=60., index=2)
        nodes = [NodeSpec('aws:0', 'aws', 2, 6., 125.)]
        a = controller([small, gang, peer])
        s = state([small], nodes)
        first = a.decide(s, columns(s), 0., {})
        self.assertEqual(set(first.selected), {'small'})

        active = [current(small, first.selected), gang, peer]
        s = state(active, nodes)
        waiting = a.decide(s, columns(s), 60., {'small': 10.})
        self.assertTrue(waiting.audit['drain'])
        self.assertEqual(set(waiting.selected), {'small'})
        self.assertEqual(a.ready, ['gang', 'peer'])
        # Peer fits the spare GPU but cannot extend a head-blocking drain.
        self.assertNotIn('peer', a.services)

        s = state([current(small, waiting.selected, 2), gang, peer], nodes)
        beyond_quantum = a.decide(s, columns(s), 360., {'small': 310.})
        self.assertEqual(set(beyond_quantum.selected), {'small'})
        self.assertEqual(a.services['small']['productive_base'], 0.)
        self.assertFalse(any(e['event'] == 'REQUEUE_PRODUCTIVE_QUANTUM'
                             for e in a.events))

        # The enclosing simulator removes an actually completed job from its
        # active JobSpecs; this unit input models that exact boundary.
        s = state([gang, peer], nodes)
        after_completion = a.decide(s, columns(s), 420., {'small': 370.})
        self.assertEqual(set(after_completion.selected), {'gang'})
        self.assertEqual(a.ready, ['peer'])
        self.assertIn('COMPLETION', after_completion.audit['geometry_event_reasons'])
        self.assertTrue(after_completion.audit['valid'])

    def test_known_model_event_permits_legal_gpu_resize(self):
        job = spec(configs=[cfg('aws'), cfg('aws', 2, rate=100.)])
        one = {(job.job_id, 'aws:1:1')}
        a, first = start(job, known=one)
        s = state([current(job, first.selected)], known=one | {(job.job_id, 'aws:1:2')})
        second = a.decide(s, columns(s), 180., {'j': 130.})
        self.assertIn('FIRST_VALID_CURRENT_MODEL', second.audit['geometry_event_reasons'])
        self.assertEqual(second.selected['j'].config.gpus, 2)
        self.assertEqual(a.services['j']['delay_charged'], 100.)
        self.assertTrue(second.audit['valid'])

    def test_no_event_excludes_otherwise_beneficial_gpu_resize(self):
        job = spec(configs=[cfg('aws'), cfg('aws', 2, rate=100.)])
        all_known = {(job.job_id, c.config_id) for c in job.configs}
        a, first = start(job, known=all_known)
        s = state([current(job, first.selected)], known=all_known)
        second = a.decide(s, columns(s), 180., {'j': 130.})
        self.assertEqual(second.audit['geometry_event_reasons'], [])
        self.assertEqual(second.selected['j'].config.gpus, 1)
        rejected = second.audit['adaptive_candidate_filters']['j']['non_event_geometry_rejected']
        self.assertTrue(rejected)
        self.assertFalse(any(e['event'] == 'ADAPT_GEOMETRY' for e in a.events))

    def test_checkpoint_candidate_contraction_does_not_rediscover_old_model(self):
        job = spec(configs=[cfg('aws'), cfg('aws', 2, rate=100.)])
        all_known = {('j', 'aws:1:1'), ('j', 'aws:1:2')}
        a, first = start(job, delay=120., known=all_known)
        # A real checkpoint exposes only its exact selected geometry to the
        # candidate scorer. Temporarily absent candidates are not new models
        # when the same checkpoint unlocks.
        locked_state = state([current(job, first.selected, locked=True)],
                             known={('j', 'aws:1:1')})
        locked = a.decide(locked_state, columns(locked_state), 60., {'j': 0.})
        self.assertEqual(locked.selected['j'].config.gpus, 1)
        unlocked_state = state([current(job, locked.selected, 2)], known=all_known)
        unlocked = a.decide(unlocked_state, columns(unlocked_state), 180., {'j': 60.})
        self.assertNotIn('FIRST_VALID_CURRENT_MODEL',
                         unlocked.audit['geometry_event_reasons'])
        self.assertEqual(unlocked.audit['geometry_event_reasons'], [])
        self.assertEqual(unlocked.selected['j'].config.gpus, 1)
        self.assertTrue(all_known <= a.previous_known_configs)

    def test_lifetime_budget_is_not_reset_after_300_productive_seconds(self):
        job = spec(configs=[cfg('aws'), cfg('dgx-ext', rate=10.),
                            cfg('rtx', rate=100.)])
        known = {('j', 'aws:1:1')}
        a, first = start(job, delay=120., known=known)
        known.add(('j', 'dgx-ext:1:1'))
        s = state([current(job, first.selected)], known=known)
        without_rtx = [c for c in columns(s) if c.config.gpu_type != 'rtx']
        second = a.decide(s, without_rtx, 180., {'j': 60.})
        self.assertEqual(second.selected['j'].config.gpu_type, 'dgx-ext')
        self.assertEqual(a.services['j']['delay_charged'], 240.)

        known.add(('j', 'rtx:1:1'))
        s = state([current(job, second.selected, 2)], known=known)
        third = a.decide(s, columns(s), 480., {'j': 360.})
        self.assertIn('FIRST_VALID_CURRENT_MODEL', third.audit['geometry_event_reasons'])
        self.assertEqual(third.selected['j'].config.gpu_type, 'dgx-ext')
        self.assertEqual(a.services['j']['delay_charged'], 240.)
        self.assertEqual(a.services['j']['productive_base'], 0.)
        self.assertEqual(a.services['j']['issued_time'], 0.)
        self.assertGreater(third.audit['adaptive_candidate_filters']['j']
                           ['rejection_counts']['SERVICE_DELAY_BUDGET'], 0)

    def test_auxiliary_exchange_can_slow_donor_without_geometry_event(self):
        jobs, s = witness(exchange=True)
        s.current_known_configs = {(j.job_id, c.config_id) for j in jobs for c in j.configs}
        a = controller(jobs)
        initial_columns = [replace(c, scheduling_utility=c.query.effective_goodput)
            for c in s.build_joint_columns()
            if c.query.cpu_per_gpu == 3. and c.query.ram_gb_per_gpu == 62.5]
        first = a.decide(s, initial_columns, 0., {})
        self.assertEqual(set(first.selected), {'receiver', 'donor'})
        updated = [current(j, first.selected) for j in jobs]
        _, s = witness(exchange=True)
        s.jobs = updated
        s.job_map = {j.job_id: j for j in updated}
        s.current_known_configs = {(j.job_id, c.config_id) for j in updated for c in j.configs}
        cs = [replace(c, scheduling_utility=c.query.effective_goodput)
              for c in s.build_joint_columns()]
        second = a.decide(s, cs, 60., {'receiver': 10., 'donor': 10.})
        self.assertEqual(second.audit['geometry_event_reasons'], [])
        self.assertEqual(second.selected['receiver'].query.cpu_per_gpu, 4.)
        self.assertEqual(second.selected['donor'].query.cpu_per_gpu, 2.)
        self.assertLess(second.selected['donor'].query.effective_goodput,
                        first.selected['donor'].query.effective_goodput)
        self.assertGreater(sum(c.query.effective_goodput for c in second.selected.values()),
                           sum(c.query.effective_goodput for c in first.selected.values()))
        self.assertTrue(second.audit['valid'])
        for name in ('receiver', 'donor'):
            self.assertEqual(a.services[name]['delay_charged'], 50.)
            self.assertEqual(a.services[name]['geometry_changes'], 0)

    def test_missing_solo_candidate_is_explicit_failure_without_state_commit(self):
        job = spec()
        a = controller([job])
        before = copy.deepcopy(a.__dict__)
        s = state([job])
        with self.assertRaisesRegex(RuntimeError, 'NO_SOLO_FEASIBLE_CANDIDATE'):
            a.decide(s, [], 0., {})
        self.assertEqual(a.__dict__, before)

    def test_missing_solver_solution_is_not_committed(self):
        job = spec(configs=[cfg('aws'), cfg('aws', 2, rate=100.)])
        a, first = start(job, known={('j', 'aws:1:1')})
        before = copy.deepcopy(a.__dict__)
        s = state([current(job, first.selected)])
        failed = SimpleNamespace(x=None, status=1, success=False, message='injected timeout')
        with patch('joint_prototype.adaptive_service.milp', return_value=failed):
            with self.assertRaisesRegex(RuntimeError, 'SOLVER_UNDECIDED'):
                a.decide(s, columns(s), 180., {'j': 130.})
        self.assertEqual(a.__dict__, before)

    def test_invalid_successful_solver_vector_is_not_committed(self):
        job = spec()
        a, first = start(job)
        before = copy.deepcopy(a.__dict__)
        s = state([current(job, first.selected)])
        cs = columns(s)
        # A falsely successful all-zero vector violates required incumbent
        # service. Both the initial answer and its numerical retry are bad.
        def invalid(c, **kwargs):
            return SimpleNamespace(x=np.zeros(len(c)), status=0, success=True,
                                   message='injected invalid success')
        with patch('joint_prototype.adaptive_service.milp', side_effect=invalid):
            with self.assertRaisesRegex(RuntimeError, 'INVALID_SOLVER_RESULT'):
                a.decide(s, cs, 180., {'j': 130.})
        self.assertEqual(a.__dict__, before)

    def test_zero_productive_service_exceeding_watchdog_stops_atomically(self):
        job = spec()
        a, first = start(job)
        s = state([current(job, first.selected)])
        # 250-second startup bound + one 60-second sampling interval is the
        # existing protection limit, not a test-selected performance timeout.
        at_boundary = a.decide(s, columns(s), 310., {'j': 0.})
        self.assertTrue(at_boundary.audit['valid'])
        before = copy.deepcopy(a.__dict__)
        s = state([current(job, at_boundary.selected, 2)])
        with self.assertRaisesRegex(RuntimeError, 'CONTINUITY_NO_PRODUCTIVE_PROGRESS'):
            a.decide(s, columns(s), 311., {'j': 0.})
        self.assertEqual(a.__dict__, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
