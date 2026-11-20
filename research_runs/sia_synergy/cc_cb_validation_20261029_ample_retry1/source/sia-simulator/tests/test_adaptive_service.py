"""Bounded state-machine checks; no Job.step or training simulation.

Toy GPU rates are explicit unit inputs. Response values are read from the
official matrices, and resource audits and MILP selection use real code.
"""
import copy
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from joint_prototype.adaptive_service import AdaptiveAdmission
from joint_prototype.joint_profile import GpuConfig, JobSpec, NodeSpec, JointProfileOracle
from joint_prototype.neutral_liveness import FactorScheduler, L2Admission, geometry


ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / 'research_sources/Synergy/simulator/models'
NODES = [NodeSpec(f'{t}:0', t, 4, 12., 250.) for t in ('aws', 'dgx-ext', 'rtx')]


def cfg(typ, gpus=1, rate=1.):
    return GpuConfig(f'{typ}:1:{gpus}', typ, gpus, 1, rate)


def spec(name='j', configs=None, template='bert_1.py', arrival=0., index=0):
    return JobSpec(name, template, tuple(configs or [cfg('aws')]),
                   arrival_time=arrival, source_row_index=index)


def current(job, selected, round_index=1, locked=False):
    col = selected.get(job.job_id)
    return replace(job, query_round=round_index, checkpoint_locked=bool(col and locked),
                   current_config_id=col.config.config_id if col else None,
                   current_nodes=col.placement.node_ids if col else (),
                   current_placement=col.placement.node_gpus if col else (),
                   current_cpu_per_gpu=col.query.cpu_per_gpu if col else None,
                   current_ram_gb_per_gpu=col.query.ram_gb_per_gpu if col else None)


def scheduler(jobs, nodes=None, medium=False):
    return FactorScheduler(nodes or NODES, jobs,
        JointProfileOracle(MODELS, 'fixed_reference', neutral_response=not medium,
                           response_exponent=.5 if medium else 0.),
        [1., 3., 4., 6., 9., 12.], [20.83, 62.5, 125.], verify_exact=False,
        minimum_cpu_per_gpu=1., minimum_ram_gb_per_gpu=20.83,
        candidate_policy='b3')


def columns(s, negative=False):
    # Use the production fairness transformation without an allocation bonus.
    return [replace(c, scheduling_utility=(-100. if negative else
                    1.1 - round(c.query.effective_goodput ** -.5, 2)))
            for c in s.build_joint_columns()]


def admission(jobs, delay=120.):
    a = AdaptiveAdmission(300., 250., 60.)
    a.delay_costs = {j.job_id: {t: delay for t in ('aws', 'dgx-ext', 'rtx')} for j in jobs}
    a.profile_sources = {j.job_id: {c.config_id: 'direct_profile' for c in j.configs} for j in jobs}
    return a


def start(job, delay=120., nodes=None, cpu=None, medium=False):
    a = admission([job], delay)
    s = scheduler([job], nodes, medium)
    cs = [c for c in columns(s) if c.config.config_id == job.configs[0].config_id]
    if cpu is not None:
        cs = [c for c in cs if c.query.cpu_per_gpu == cpu]
    r = a.decide(s, cs, 0., {job.job_id: 0.})
    return a, r


class AdaptiveServiceTests(unittest.TestCase):
    def test_profiled_adaptation_breaks_legacy_geometry_freeze(self):
        job = spec(configs=[cfg('aws'), cfg('dgx-ext', rate=100.)])
        a, first = start(job)
        old = L2Admission(300., 250., 60.)
        old.ready, old.seen, old.sequence = [], {'j'}, 1
        old.services = {'j': dict(column=first.selected['j'], issued_time=0.,
                                 productive_base=0., sequence=1)}
        s = scheduler([current(job, first.selected)])
        legacy = old.decide(s, columns(s), 180., {'j': 60.})
        new = a.decide(s, columns(s), 180., {'j': 60.})
        self.assertEqual(legacy.selected['j'].config.gpu_type, 'aws')
        self.assertEqual(new.selected['j'].config.gpu_type, 'dgx-ext')
        self.assertEqual(a.services['j']['delay_charged'], 240.)
        self.assertEqual(a.services['j']['issued_time'], 0.)
        self.assertEqual(a.services['j']['productive_base'], 0.)
        self.assertTrue(new.audit['valid'])

    def test_negative_utility_still_serves_true_fifo(self):
        jobs = [spec('a', arrival=40), spec('z', arrival=20)]
        s = scheduler(jobs, [NodeSpec('aws:0', 'aws', 1, 3., 62.5)])
        a = admission(jobs, 50.)
        r = a.decide(s, columns(s, negative=True), 60., {})
        self.assertEqual(set(r.selected), {'z'})
        self.assertLess(r.scheduling_objective, 0.)

    def test_real_checkpoint_preserves_complete_allocation(self):
        job = spec(configs=[cfg('aws'), cfg('dgx-ext', rate=100.)])
        a, first = start(job, cpu=3.)
        s = scheduler([current(job, first.selected, locked=True)])
        r = a.decide(s, columns(s), 60., {'j': 0.})
        self.assertEqual(geometry(first.selected['j']), geometry(r.selected['j']))
        self.assertEqual(r.selected['j'].candidate_kind, 'locked')
        self.assertEqual(a.services['j']['delay_charged'], 120.)

    def test_unprofiled_target_rejected(self):
        job = spec(configs=[cfg('aws'), cfg('dgx-ext', rate=100.)])
        a, first = start(job)
        a.profile_sources['j']['dgx-ext:1:1'] = 'unprofiled_sia_fallback'
        s = scheduler([current(job, first.selected)])
        r = a.decide(s, columns(s), 180., {'j': 60.})
        self.assertEqual(r.selected['j'].config.gpu_type, 'aws')
        self.assertGreater(r.audit['adaptive_candidate_filters']['j']['rejection_counts']['UNPROFILED_TARGET'], 0)

    def test_no_productive_training_rejects_gpu_switch(self):
        job = spec(configs=[cfg('aws'), cfg('dgx-ext', rate=100.)])
        a, first = start(job)
        s = scheduler([current(job, first.selected)])
        r = a.decide(s, columns(s), 120., {'j': 0.})
        self.assertEqual(r.selected['j'].config.gpu_type, 'aws')
        self.assertGreater(r.audit['adaptive_candidate_filters']['j']['rejection_counts']['NO_PRODUCTIVE_SERVICE_SINCE_GEOMETRY'], 0)

    def test_bert_two_120_second_charges_allowed_third_rejected(self):
        job = spec(configs=[cfg('aws'), cfg('dgx-ext', rate=10.), cfg('rtx', rate=100.)])
        a, first = start(job)
        s = scheduler([current(job, first.selected)])
        cs = [c for c in columns(s) if c.config.gpu_type != 'rtx']
        second = a.decide(s, cs, 180., {'j': 60.})
        self.assertEqual(second.selected['j'].config.gpu_type, 'dgx-ext')
        self.assertEqual(a.services['j']['delay_charged'], 240.)
        s = scheduler([current(job, second.selected, 2)])
        third = a.decide(s, columns(s), 360., {'j': 120.})
        self.assertEqual(third.selected['j'].config.gpu_type, 'dgx-ext')
        self.assertEqual(a.services['j']['delay_charged'], 240.)
        self.assertGreater(third.audit['adaptive_candidate_filters']['j']['rejection_counts']['SERVICE_DELAY_BUDGET'], 0)

    def test_cifar_actual_50_not_fairness_proxy_30_budget_boundary(self):
        configs = [cfg('aws'), cfg('dgx-ext', rate=3.), cfg('rtx', rate=9.),
                   cfg('aws', 2, 27.), cfg('dgx-ext', 2, 81.), cfg('rtx', 2, 243.)]
        job = spec(configs=configs, template='alexnet_1.py')
        a, r = start(job, delay=50.)
        for step in range(1, 6):
            s = scheduler([current(job, r.selected, step)])
            allowed = {c.config_id for c in configs[:step+1]}
            cs = [c for c in columns(s) if c.config.config_id in allowed]
            r = a.decide(s, cs, step*60., {'j': step*10.})
            if step <= 4:
                self.assertEqual(r.selected['j'].config.config_id, configs[step].config_id)
                self.assertEqual(a.services['j']['delay_charged'], 50.*(step+1))
        self.assertEqual(r.selected['j'].config.config_id, configs[4].config_id)
        self.assertEqual(a.services['j']['delay_charged'], 250.)
        self.assertEqual(a.services['j']['productive_base'], 0.)
        self.assertEqual(a.services['j']['issued_time'], 0.)

    def test_no_remaining_service_payback_rejects_slight_speed_gain(self):
        job = spec(configs=[cfg('aws'), cfg('dgx-ext', rate=1.1)])
        a, first = start(job)
        s = scheduler([current(job, first.selected)])
        r = a.decide(s, columns(s), 180., {'j': 60.})
        self.assertEqual(r.selected['j'].config.gpu_type, 'aws')
        self.assertGreater(r.audit['adaptive_candidate_filters']['j']['rejection_counts']['NO_REMAINING_SERVICE_PAYBACK'], 0)

    def test_flat_peer_quota_release_enables_sensitive_peer_without_delay(self):
        flat, sensitive = spec('flat'), spec('sensitive', template='alexnet_1.py')
        nodes = [NodeSpec('aws:0', 'aws', 2, 6., 125.)]
        a = admission([flat, sensitive], 50.)
        s = scheduler([flat, sensitive], nodes, medium=True)
        cs = [c for c in columns(s) if c.query.cpu_per_gpu == 3. and c.query.ram_gb_per_gpu == 62.5]
        first = a.decide(s, cs, 0., {})
        s = scheduler([current(j, first.selected) for j in (flat, sensitive)], nodes, medium=True)
        r = a.decide(s, columns(s), 60., {'flat': 10., 'sensitive': 10.})
        self.assertLess(r.selected['flat'].query.cpu_per_gpu, 3.)
        self.assertGreater(r.selected['sensitive'].query.effective_goodput,
                           first.selected['sensitive'].query.effective_goodput)
        self.assertEqual(r.selected['flat'].query.effective_goodput,
                         first.selected['flat'].query.effective_goodput)
        self.assertTrue(r.audit['valid'])
        for name in ('flat', 'sensitive'):
            self.assertEqual(a.services[name]['delay_charged'], 50.)
            self.assertEqual(a.services[name]['geometry_changes'], 0)
            self.assertEqual(a.services[name]['productive_base'], 0.)
            self.assertEqual(a.services[name]['last_geometry_productive'], 0.)

    def test_solver_failure_preserves_all_service_and_budget_state(self):
        job = spec(configs=[cfg('aws'), cfg('dgx-ext', rate=100.)])
        a, first = start(job)
        before = copy.deepcopy(a.__dict__)
        s = scheduler([current(job, first.selected)])
        failed = SimpleNamespace(x=None, status=1, success=False, message='injected timeout')
        with patch('joint_prototype.adaptive_service.milp', return_value=failed):
            with self.assertRaisesRegex(RuntimeError, 'SOLVER_UNDECIDED'):
                a.decide(s, columns(s), 180., {'j': 60.})
        self.assertEqual(a.__dict__, before)

    def test_audit_failure_preserves_all_service_and_budget_state(self):
        job = spec(configs=[cfg('aws'), cfg('dgx-ext', rate=100.)])
        a, first = start(job)
        before = copy.deepcopy(a.__dict__)
        s = scheduler([current(job, first.selected)])
        cs = columns(s)
        ordinary_audit = s.audit
        def fail_whole_commit(chosen, **kwargs):
            # Preserve refresh_column's single-column eligibility check;
            # inject the failure at the whole-allocation precommit audit.
            if kwargs.get('require_locks', True):
                return {'valid': False, 'violations': ['injected audit']}
            return ordinary_audit(chosen, **kwargs)
        with patch.object(s, 'audit', side_effect=fail_whole_commit):
            with self.assertRaisesRegex(RuntimeError, 'INVALID_SOLVER_ALLOCATION'):
                a.decide(s, cs, 180., {'j': 60.})
        self.assertEqual(a.__dict__, before)

    def test_expiry_same_existing_gpu_costs_zero_and_repeats_service(self):
        job = spec()
        a, first = start(job, 50.)
        s = scheduler([current(job, first.selected)])
        r = a.decide(s, columns(s), 360., {'j': 310.})
        self.assertEqual(a.services['j']['productive_base'], 310.)
        self.assertEqual(a.services['j']['delay_charged'], 0.)
        self.assertEqual(r.audit['service_budget']['j']['change'], 'CONTINUE_EXISTING_GPU')
        self.assertTrue(any(e['event'] == 'REQUEUE_PRODUCTIVE_QUANTUM' for e in a.events))

    def test_fifo_large_gang_drains_and_all_jobs_receive_repeated_service(self):
        jobs = [spec('small-a'), spec('small-b', index=1),
                spec('gang', configs=[cfg('aws', 4)], arrival=60., index=2)]
        nodes = [NodeSpec('aws:0', 'aws', 4, 12., 250.)]
        a = admission(jobs, 50.)
        chosen, totals, delay_left, records = {}, {j.job_id: 0. for j in jobs}, {}, []
        # Fixed 24 boundaries; deterministic physical delay accounting, no Job.step.
        for step in range(24):
            if step:
                for name in chosen:
                    delay = min(60., delay_left.get(name, 0.))
                    delay_left[name] = delay_left.get(name, 0.) - delay
                    totals[name] += 60. - delay
            active = [current(j, chosen, step, delay_left.get(j.job_id, 0.) > 0)
                      for j in jobs if j.arrival_time <= step*60.]
            s = scheduler(active, nodes)
            r = a.decide(s, columns(s, negative=True), step*60., totals)
            for name in r.selected:
                if name not in chosen:
                    delay_left[name] = 50.
            chosen = r.selected
            records.append((step*60., set(chosen), r.audit['drain']))
            self.assertTrue(r.audit['valid'])
        self.assertTrue(any(drain for _, _, drain in records))
        self.assertEqual(records[1][1], {'small-a', 'small-b'})
        self.assertTrue(any(selected == {'gang'} for _, selected, _ in records))
        self.assertTrue(all(value >= 300. for value in totals.values()))
        issued = [e['job'] for e in a.events if e['event'] == 'ISSUE_SERVICE']
        self.assertTrue(all(issued.count(j.job_id) >= 2 for j in jobs))


if __name__ == '__main__':
    unittest.main(verbosity=2)
