"""Deterministic release-rollout tests; real columns/audits, explicit toy rates.

These are mechanism tests, not Sia prediction-accuracy or JCT experiments.
No future arrivals, application lookup or Job.step is supplied to the evaluator.
"""
import copy
import unittest

from test_adaptive_service import spec, current, columns, scheduler
from test_clean_ablation import state
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.release_evaluation import evaluate_plan


class ReleaseEvaluationTests(unittest.TestCase):
    def fixture(self, names, gpus=1):
        jobs = [spec(name, index=i) for i, name in enumerate(names)]
        scheduler = state(jobs, [NodeSpec('aws:0', 'aws', gpus,
                                        3. * gpus, 62.5 * gpus)])
        scheduler.enforce_floor = False
        all_columns = columns(scheduler)
        prop = {name: next(c for c in all_columns if c.job_id == name
                          and c.query.cpu_per_gpu == 3.
                          and c.query.ram_gb_per_gpu == 62.5)
                for name in names}
        return scheduler, all_columns, prop

    def scores(self, cols, work, raw=None, delays=None, rhos=None):
        raw, delays, rhos = raw or {}, delays or {}, rhos or {}
        rows = {}
        for c in cols:
            n = c.job_id
            rate, delay, rho = raw.get(n, 1.), delays.get(n, 0.), rhos.get(n, 1.)
            rows[c.column_id] = dict(raw_rate=rate, rho=rho,
                remaining_work=float(work[n]), delay=float(delay),
                estimated_completion=(None if rate is None
                                      else delay + work[n] / (rate * rho)))
        return rows

    def test_actual_startup_and_rho_applied_once(self):
        s = scheduler([spec('a',template='res18_1.py')],
                      [NodeSpec('aws:0','aws',1,6.,125.)],medium=True)
        s.enforce_floor=False
        cols=columns(s)
        c=next(c for c in cols if c.query.cpu_per_gpu==6. and c.query.ram_gb_per_gpu==125.)
        cols=[c];rho=c.query.rho
        self.assertNotEqual(rho,1.)
        scores = self.scores(cols, {'a': 100.}, raw={'a': 2.},
                             rhos={'a': rho}, delays={'a': 30.})
        result = evaluate_plan(s, cols, scores, [c], [])
        self.assertEqual(result['status'], 'OK')
        self.assertAlmostEqual(result['completion_times']['a'], 30.+100./(2.*rho))
        self.assertAlmostEqual(result['sum_completion'], 30.+100./(2.*rho))

    def test_completion_releases_resources_only_for_next_boundary_start(self):
        s, cols, p = self.fixture(['a', 'b'])
        scores = self.scores(cols, {'a': 75., 'b': 10.}, delays={'b': 5.})
        result = evaluate_plan(s, cols, scores, [p['a']], ['b'])
        self.assertEqual(result['status'], 'OK')
        self.assertAlmostEqual(result['completion_times']['a'], 75.)
        self.assertAlmostEqual(result['completion_times']['b'], 135.)
        self.assertAlmostEqual(result['sum_completion'], 210.)

    def test_fifo_queue_is_not_sorted_by_shortest_prediction(self):
        s, cols, p = self.fixture(['a', 'b', 'c'])
        scores = self.scores(cols, {'a': 10., 'b': 90., 'c': 1.})
        result = evaluate_plan(s, cols, scores, [p['a']], ['b', 'c'])
        self.assertEqual(result['status'], 'OK')
        self.assertEqual(result['completion_times'], {'a': 10., 'b': 150., 'c': 181.})
        self.assertAlmostEqual(result['sum_completion'], 341.)

    def test_uncommitted_waiter_cannot_start_at_time_zero_in_free_space(self):
        s,cols,p=self.fixture(['a','b'],gpus=2)
        scores=self.scores(cols,{'a':300.,'b':10.})
        result=evaluate_plan(s,cols,scores,[p['a']],['b'])
        self.assertEqual(result['completion_times'],{'a':300.,'b':70.})
        start=next(e for e in result['events'] if e['event']=='FIFO_START')
        self.assertEqual(start['time'],60.)

    def test_all_donor_delay_counts_not_only_receiver_gain(self):
        s, cols, p = self.fixture(['receiver', 'donor'], gpus=2)
        before = self.scores(cols, {'receiver': 60., 'donor': 100.})
        after = self.scores(cols, {'receiver': 60., 'donor': 100.},
                            raw={'receiver': 2., 'donor': .5})
        plan = [p['receiver'], p['donor']]
        old = evaluate_plan(s, cols, before, plan, [])
        new = evaluate_plan(s, cols, after, plan, [])
        self.assertAlmostEqual(old['sum_completion'], 160.)
        self.assertAlmostEqual(new['sum_completion'], 230.)
        self.assertGreater(new['sum_completion'], old['sum_completion'])

    def test_earlier_release_can_outweigh_larger_running_only_eta_sum(self):
        s, cols, p = self.fixture(['a', 'b', 'waiter'], gpus=2)
        work = {'a': 61., 'b': 61., 'waiter': 100.}
        first_scores = self.scores(cols, work)
        second_scores = self.scores(cols, work,
                                   raw={'a': 61./59., 'b': 61./70.})
        initial = [p['a'], p['b']]
        first = evaluate_plan(s, cols, first_scores, initial, ['waiter'])
        second = evaluate_plan(s, cols, second_scores, initial, ['waiter'])
        # Running-only sums prefer 61+61 to 59+70. Waiting task can start
        # one full 60-second boundary earlier in the second complete plan.
        self.assertGreater(59. + 70., 61. + 61.)
        self.assertAlmostEqual(first['sum_completion'], 342.)
        self.assertAlmostEqual(second['sum_completion'], 289.)
        self.assertLess(second['sum_completion'], first['sum_completion'])

    def test_unknown_waiter_abstains_instead_of_disappearing(self):
        s, cols, p = self.fixture(['a', 'unknown'])
        scores = self.scores(cols, {'a': 20., 'unknown': 100.}, raw={'unknown': None})
        result = evaluate_plan(s, cols, scores, [p['a']], ['unknown'])
        self.assertEqual(result['status'], 'UNKNOWN_MODEL')
        self.assertIsNone(result.get('sum_completion'))

    def test_invalid_initial_bundle_is_rejected(self):
        s, cols, p = self.fixture(['a', 'b'])
        scores = self.scores(cols, {'a': 10., 'b': 10.})
        with self.assertRaises((ValueError, RuntimeError)):
            evaluate_plan(s, cols, scores, [p['a'], p['b']], [])

    def test_checkpoint_lock_cannot_be_replaced_by_cheaper_quota(self):
        s, cols, p = self.fixture(['a'])
        locked = current(s.jobs[0], {'a': p['a']}, locked=True)
        locked_s = state([locked], s.nodes)
        locked_s.enforce_floor = False
        # This is a real normal column from the same legal grid but violates
        # the current lock's exact CPU/RAM contract.
        replacement = next(c for c in cols if c.job_id == 'a'
                           and c.query.cpu_per_gpu == 1.
                           and c.query.ram_gb_per_gpu == 20.83)
        scores = self.scores(cols, {'a': 10.})
        with self.assertRaises((ValueError, RuntimeError)):
            evaluate_plan(locked_s, cols, scores, [replacement], [])

    def test_evaluation_does_not_mutate_models_queue_or_initial(self):
        s, cols, p = self.fixture(['a', 'b'])
        scores = self.scores(cols, {'a': 20., 'b': 30.})
        frozen_scores = copy.deepcopy(scores)
        queue, initial = ['b'], [p['a']]
        evaluate_plan(s, cols, scores, initial, queue)
        self.assertEqual(scores, frozen_scores)
        self.assertEqual(queue, ['b'])
        self.assertEqual(initial, [p['a']])
        self.assertEqual(set(s.job_map), {'a', 'b'})


if __name__ == '__main__':
    unittest.main()
