"""Stage1.1 red/green cases. Real unseeded scheduling plus bounded fault injection."""
import json
import os
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import test_joint_integration_coverage as fixtures
from joint_prototype.joint_integration import JointSiaPolicy
from joint_prototype.joint_profile import GpuConfig, JobSpec, JointProfileOracle, NodeSpec
from joint_prototype.joint_scheduler import JointScheduler


def evidence(name, data):
    if os.environ.get('STAGE1_EVIDENCE_DIR'):
        folder = Path(os.environ['STAGE1_EVIDENCE_DIR'])
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/(name+'.json')).open('x', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v))


class ObservedPolicy(JointSiaPolicy):
    """Observation only: every override calls the unmodified real implementation."""
    def _raw_candidates(self, info, max_gpu_override=None):
        self.trace = getattr(self, 'trace', [])
        configs, sources = super()._raw_candidates(info, max_gpu_override)
        self.trace.append(dict(event='enumeration', round=self.round_index,
            min=dict(info.min_replicas), max=dict(info.max_replicas),
            scale=dict(info.scale_unit), direct={k:v is not None for k,v in info.speedup_fn.items()},
            sources=sources))
        return configs, sources

    def _compute_goodputs_v2(self, info, cluster, nn, ng):
        values = super()._compute_goodputs_v2(info, cluster, nn, ng)
        self.trace = getattr(self, 'trace', [])
        self.trace.append(dict(event='goodput', round=self.round_index, cluster=cluster,
            nodes=nn.tolist(), gpus=ng.tolist(), values=np.asarray(values).tolist(),
            min=dict(info.min_replicas), max=dict(info.max_replicas)))
        return values

    def _b1_result(self, allocations, specs, scheduler):
        self.trace.append(dict(event='sia_intent', round=self.round_index, allocations=allocations,
            candidates={j.job_id:[c.config_id for c in j.configs] for j in specs}))
        return super()._b1_result(allocations, specs, scheduler)


class Stage1p1CorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.JointIntegrationCoverageTests()
        self.fixture.setUp()

    def test_unseeded_bert_mid_full_optimize_policy_chain(self):
        c = self.fixture._cluster([dict(name='bert-mid', time=17, application='bert',
            num_replicas=1, batch_size=384)], mode='b1', policy_class=ObservedPolicy)
        states = []
        try:
            for seconds in (60, 60, 60, 1):
                c.step(seconds)  # Includes the REAL next optimize_policy, no seed/mock.
                j = c.jobs[0]
                states.append(dict(time=c.current_time, progress=j.progress, rescale=j.rescale_time,
                    allocation=c.allocations, first_allocation=j.first_allocation_time,
                    first_productive=j.first_productive_training_time))
            self.assertEqual(states[0]['first_allocation'], 60)
            self.assertEqual(states[1]['progress'], 0)
            self.assertEqual(states[2]['progress'], 0)
            self.assertGreater(states[3]['progress'], 0)
            self.assertEqual(states[3]['first_productive'], 180)
            self.assertEqual(c.policy.current_resources['bert-mid']['config_id'], 'dgx-ext:1:2')
            last_enum = [t for t in c.policy.trace if t['event']=='enumeration'][-1]
            self.assertEqual(last_enum['max'], {'aws':1,'dgx-ext':1,'rtx':2})
            self.assertFalse(last_enum['direct']['dgx-ext'])
            self.assertEqual(last_enum['sources']['dgx-ext:1:2']['goodput_source'], 'projected_profile')
            self.assertTrue(c.policy.share_max_replicas)
            self.assertEqual(len(c.joint_round_logs), 4)
            self.assertTrue(c.policy.last_decision['result']['audit']['precommit_independent_audit']['valid'])
        finally:
            evidence('bert_mid_full_chain', dict(states=states, trace=c.policy.trace, rounds=c.joint_round_logs))

    def scheduler(self, *, locked=False, cpu=3., ram=62.5, neutral=True, policy='b3', query_round=0):
        cfg = GpuConfig('aws:1:1', 'aws', 1, 1, 1.)
        job = JobSpec('keep', 'res18_1.py', (cfg,), checkpoint_locked=locked,
            current_config_id=cfg.config_id, current_nodes=('aws:0',),
            current_placement=(('aws:0',1),), current_cpu_per_gpu=cpu,
            current_ram_gb_per_gpu=ram, query_round=query_round)
        s = JointScheduler([NodeSpec('aws:0','aws',4,12,500)], [job],
            JointProfileOracle(fixtures.MODELS, 'fixed_reference', neutral_response=neutral),
            [1,2,3,4,6,8], [20.83,62.5,125,250], verify_exact=False)
        # Authoritative caller context, deliberately NOT a field on a column.
        s.candidate_policy = policy
        return s

    def fallback(self, old, current):
        s = current
        with patch('joint_prototype.joint_scheduler.milp', return_value=SimpleNamespace(
                x=None, status=1, success=False, message='injected timeout')):
            cols = s.build_proportional_columns() if s.candidate_policy=='b3_prop' else s.build_joint_columns()
            failed = s.solve('B3_FAULT_INJECTION', cols, ('gpu','cpu','ram'))
        p = object.__new__(JointSiaPolicy)
        p._last_committed_columns = {'keep':old}
        p.mode = s.candidate_policy
        return p._b3_failure_fallback(failed, s.jobs, s), failed

    def test_locked_unlocked_solver_failure_valid_quota_requalifies(self):
        old = self.scheduler(locked=True).locked_columns()['keep']
        current = self.scheduler(query_round=1)
        r, failed = self.fallback(old, current)
        self.assertIsNotNone(r)
        self.assertEqual(r.selected['keep'].candidate_kind, 'joint_aux')
        self.assertEqual(r.selected['keep'].query_round, 1)
        self.assertEqual(r.selected['keep'].placement, old.placement)
        self.assertTrue(current.audit(list(r.selected.values()))['valid'])
        evidence('valid_unlock_fallback', r.to_dict())

    def test_unlocked_outside_grid_fallback_rejected(self):
        old = self.scheduler(locked=True, cpu=2.5).locked_columns()['keep']
        r, failed = self.fallback(old, self.scheduler(query_round=1))
        self.assertIsNone(r)
        self.assertFalse(failed.audit['safe_previous_allocation_available'])

    def test_unlocked_below_progress_floor_fallback_rejected(self):
        old = self.scheduler(locked=True, cpu=1, ram=20.83, neutral=False).locked_columns()['keep']
        s = self.scheduler(neutral=False, query_round=1)
        baseline = s.oracle.query(s.jobs[0],s.jobs[0].configs[0],3,125,(3,125))
        self.assertLess(old.query.effective_goodput, baseline.effective_goodput)
        r, failed = self.fallback(old,s)
        self.assertIsNone(r)

    def test_unknown_and_forged_locked_kinds_rejected(self):
        s = self.scheduler()
        col = s.build_proportional_columns()[0]
        for kind in ('unknown', 'legacy', 'locked', 'synergy_tune'):
            with self.subTest(kind=kind):
                self.assertFalse(s.audit([replace(col,candidate_kind=kind)])['valid'])

    def test_b3_prop_rejects_cross_policy_nonproportional_and_old_lock(self):
        s = self.scheduler(policy='b3_prop')
        aux = self.scheduler().build_joint_columns()[0]
        self.assertNotEqual((aux.query.cpu_per_gpu,aux.query.ram_gb_per_gpu),(3,125))
        for kind in ('joint_aux','proportional','synergy_tune','locked','unknown'):
            with self.subTest(kind=kind):
                self.assertFalse(s.audit([replace(aux,candidate_kind=kind)])['valid'])
        old = self.scheduler(locked=True, cpu=1,ram=20.83).locked_columns()['keep']
        r,_ = self.fallback(old,s)
        self.assertIsNone(r)

    def test_still_locked_non_grid_quota_fully_protected(self):
        s = self.scheduler(locked=True,cpu=2.5,policy='b3_prop')
        col = s.locked_columns()['keep']
        self.assertTrue(s.audit([col])['valid'])
        self.assertFalse(s.audit([])['valid'])
        changed = replace(col, placement=replace(col.placement,node_gpus=(('aws:0',2),)))
        self.assertFalse(s.audit([changed])['valid'])
        r,_ = self.fallback(col,s)
        self.assertIsNotNone(r)
        self.assertEqual(r.selected['keep'].candidate_kind,'locked')

    def test_real_cluster_unlock_failure_requalifies_b3_and_prop(self):
        for mode, kind in [('b3','joint_aux'),('b3_prop','proportional')]:
            with self.subTest(mode=mode):
                c = self.fixture._cluster([dict(name='bert-unlock',time=17,application='bert',
                    num_replicas=1,batch_size=384)],mode=mode)
                c.step(60)
                c.step(60)
                old = c.policy._last_committed_columns['bert-unlock']
                self.assertEqual(old.candidate_kind,'locked')
                with patch('joint_prototype.joint_scheduler.milp',return_value=SimpleNamespace(
                        x=None,status=1,success=False,message='injected timeout at real unlock')):
                    c.step(60)
                new = c.policy._last_committed_columns['bert-unlock']
                self.assertEqual(new.candidate_kind,kind)
                self.assertEqual(new.placement,old.placement)
                self.assertEqual(new.query.cpu_per_gpu,old.query.cpu_per_gpu)
                self.assertGreater(new.query_round,old.query_round)
                self.assertEqual(c.policy.last_decision['result']['status'],'FALLBACK_PRESERVE_PREVIOUS_VALID_ALLOCATION')
                evidence('real_unlock_'+mode,c.joint_round_logs)

    def test_refresh_unknown_tag_rejected_and_prop_valid_unlock_allowed(self):
        old = self.scheduler(locked=True,cpu=3,ram=125).locked_columns()['keep']
        s = self.scheduler(policy='b3_prop',query_round=1)
        with self.assertRaises(ValueError):
            s.refresh_column(replace(old,candidate_kind='unknown'))
        r,_ = self.fallback(old,s)
        self.assertIsNotNone(r)
        self.assertEqual(r.selected['keep'].candidate_kind,'proportional')
        good = s.build_proportional_columns()[0]
        self.assertFalse(s.audit([replace(good,candidate_kind='joint_aux')])['valid'])

    def test_b4_infeasible_proportional_intent_refuses_fallback_without_crash(self):
        from joint_prototype.joint_scheduler import PolicyResult, Placement
        cfg = GpuConfig('rtx:1:1','rtx',1,1,1.)
        job = JobSpec('oversized','res18_1.py',(cfg,))
        s = self.scheduler()
        s = JointScheduler([s.nodes[0],NodeSpec('rtx:0','rtx',1,.5,100)], [s.jobs[0],job],
            s.oracle,s.aux_cpu_points,s.aux_ram_points,verify_exact=False,minimum_cpu_per_gpu=1)
        normal = s.build_proportional_columns()[0]
        p = Placement((('rtx:0',1),))
        q = s.oracle.query(job,cfg,1,100,(.5,100))
        invalid = replace(normal,job_id=job.job_id,config=cfg,placement=p,query=q,
            node_usage=s._node_usage(p,1,100),candidate_kind='proportional')
        intent = PolicyResult('B1','INTENT',{'keep':normal,'oversized':invalid},[],0,0,0)
        result = s.run_b4(intent)
        self.assertTrue(result.audit['valid'])
        self.assertEqual(set(result.selected),{'keep'})
        self.assertIn('oversized',result.audit['fallback_reason'])


if __name__ == '__main__':
    unittest.main()
