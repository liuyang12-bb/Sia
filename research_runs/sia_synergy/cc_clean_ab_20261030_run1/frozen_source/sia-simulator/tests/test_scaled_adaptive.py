"""Real MILP state transitions at the two preregistered queue sizes."""
import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from test_adaptive_service import spec, cfg, current, scheduler, columns
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.scaled_adaptive import ScaledAdmission


class ScaleTests(unittest.TestCase):
    def simulate(self, n):
        jobs = [spec(f'j{i:02}', configs=[cfg('aws')], index=i) for i in range(n)]
        nodes = [NodeSpec('aws:0', 'aws', 4, 12., 250.)]
        a = ScaledAdmission(300., 250., 60.)
        a.delay_costs = {j.job_id: {'aws': 50.} for j in jobs}
        selected, totals, delays = {}, {j.job_id: 0. for j in jobs}, {}
        locked_checks = 0
        for step in range(90):
            if step:
                for name in selected:
                    delay = min(60., delays.get(name, 0.))
                    delays[name] = delays.get(name, 0.) - delay
                    totals[name] += 60. - delay
            active = [current(j, selected, step, delays.get(j.job_id, 0.) > 0) for j in jobs]
            s = scheduler(active, nodes)
            old = selected
            result = a.decide(s, columns(s, negative=True), step*60., totals)
            selected = result.selected
            self.assertTrue(result.audit['valid'])
            self.assertLessEqual(len(selected), 4)
            for name in old:
                if delays.get(name, 0.) > 0:
                    self.assertEqual(selected[name], old[name]); locked_checks += 1
            for name in selected:
                if name not in old: delays[name] = 50.
        self.assertTrue(all(t >= 600. for t in totals.values()), totals)
        self.assertEqual(len(a.seen), n)
        self.assertEqual(set(a.ready) | set(a.services), set(totals))
        self.assertTrue(any(e['event'] == 'DRAIN_FOR_HEAD' for e in a.events))
        # Full queue must survive a solver failure without partial service updates.
        before = copy.deepcopy(a.__dict__)
        s = scheduler([current(j, selected, 90) for j in jobs], nodes)
        fail = SimpleNamespace(x=None, success=False, status=1, message='injected')
        with patch('joint_prototype.adaptive_service.milp', return_value=fail):
            with self.assertRaisesRegex(RuntimeError, 'SOLVER_UNDECIDED'):
                a.decide(s, columns(s), 5400., totals)
        self.assertEqual(a.__dict__, before)

    def test_eight_fifo_repeated_service(self): self.simulate(8)
    def test_twenty_fifo_repeated_service(self): self.simulate(20)
    def test_no_silent_truncation(self):
        jobs = [spec(str(i)) for i in range(21)]
        a = ScaledAdmission(300., 250., 60.)
        s = scheduler(jobs)
        with self.assertRaisesRegex(RuntimeError, 'SCALE_SCOPE_MAX_TWENTY'):
            a.decide(s, columns(s), 0., {})
        self.assertFalse(a.seen)
