"""Completion-oriented opt-in policy: fixed, bounded engineering regressions.

The synthetic scalar examples test seconds, not application throughput sums.
Real Job/Cluster tests use official BERT profiles and never mock scheduling.
Every integration observation is fixed at at most 12 x 60 seconds.
"""
import contextlib
import copy
import importlib
import io
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_exchange_integration as exchange_fixture
from joint_prototype.joint_profile import GpuConfig
from joint_prototype.execution_audit import audit_execution


def implementation():
    return importlib.import_module('joint_prototype.completion_time')


class CompletionTimeTests(unittest.TestCase):
    def build(self, proportional=False):
        return implementation().install(exchange_fixture.ExchangeIntegration().build(),
                                        proportional=proportional)

    def pair(self, proportional=False, aux=False):
        import pandas as pd
        from joint_prototype.full160 import make_cluster
        from joint_prototype.physical_domain import restrict_single_node_domain
        root=Path(__file__).resolve().parents[3]
        data=root/'research_runs/sia_synergy/joint_opportunity_experiments_20261026_run1/inputs'
        config=json.loads((data/'D3_queued_training_proposal_config.json').read_text())
        config['cluster'].update(nnodes={'aws':1},ngpus_per_node={'aws':2},max_physical_nodes={'aws':1})
        work=pd.read_csv(data/'D3_queued_training_proposal.csv')
        work=work[work.name.isin(['cifar10-1','deepspeech2-9'])]
        c=make_cluster(config,work,data/'registry.json',data/'models','joint_o','medium','main',True)
        c.policy.node_capacities={'aws':dict(cpu_cores=12,ram_gb=250)}
        restrict_single_node_domain(c.policy)
        return (implementation().install_aux(c) if aux else
                implementation().install(c,proportional=proportional))

    def test_completion_cost_has_seconds_and_applies_rho_once(self):
        m = implementation()
        self.assertEqual(m.completion_cost(120., 2., 3., 10.), 30.)
        self.assertEqual(m.completion_saving(120., 2., 3., 10., 500.), 470.)
        self.assertEqual(m.completion_cost(120., 2., 1., 10.), 70.)

    def test_invalid_nonfinite_and_nonpositive_rates_are_rejected(self):
        m = implementation()
        for work, raw, rho, delay in [(1,0,1,0),(1,1,0,0),
                (1,float('nan'),1,0),(1,1,float('inf'),0),
                (-1,1,1,0),(1,1,1,-1)]:
            with self.subTest(values=(work,raw,rho,delay)):
                with self.assertRaises(ValueError):
                    m.completion_cost(work,raw,rho,delay)

    def test_receiver_speed_gain_cannot_hide_large_donor_completion_loss(self):
        m = implementation()
        # Receiver +25%, donor -10%: total raw rate increases, sum ETC worsens.
        base = m.completion_cost(100,1,1,0)+m.completion_cost(10000,1,1,0)
        exchanged = m.completion_cost(100,1,1.25,0)+m.completion_cost(10000,1,.9,0)
        self.assertGreater(1.25+.9, 2.)
        self.assertGreater(exchanged, base)
        base_saving = m.completion_saving(100,1,1,0,1000)+m.completion_saving(10000,1,1,0,20000)
        exchanged_saving = m.completion_saving(100,1,1.25,0,1000)+m.completion_saving(10000,1,.9,0,20000)
        self.assertLess(exchanged_saving, base_saving)

    def test_startup_loss_can_reverse_nominal_speedup(self):
        m = implementation()
        self.assertLess(m.completion_cost(100,1,1,0),
                        m.completion_cost(100,2,1,60))
        self.assertGreater(m.completion_cost(1000,1,1,0),
                           m.completion_cost(1000,2,1,60))

    def test_unprofiled_prediction_is_explicit_unknown_not_raw_one(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = exchange_fixture.ExchangeIntegration().build()
            j = c.jobs[0]
            before = (j.current_cluster, j.placement, j.atomic_bsz, j.accum_steps)
            p = implementation().predict_job(j, GpuConfig('aws:1:1','aws',1,1,999.))
            self.assertIsNone(p['raw_rate'])
            self.assertEqual(before,(j.current_cluster,j.placement,j.atomic_bsz,j.accum_steps))

    def test_predictor_uses_observed_model_with_prospective_actual_batch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = exchange_fixture.ExchangeIntegration().build()
            j = c.jobs[0]
            j.seed_profiles(1,8)
            # Explicit current gradient estimate; no future get_grad_stats call.
            j.grad_params = (2., 8.)
            original = (j.current_cluster,j.placement,j.atomic_bsz,j.accum_steps)
            prospect = copy.copy(j)
            prospect.current_cluster = 'aws'
            prospect.update_local_bsz((1,))
            app = j.applications['aws']
            expected = float(j.get_goodput_fn('aws').evaluate(1,1,
                prospect.atomic_bsz,prospect.accum_steps))/app.init_batch_size*j.calibration_factor
            with patch.object(app,'get_throughput',side_effect=AssertionError('future throughput lookup')), \
                 patch.object(app,'get_grad_stats',side_effect=AssertionError('future gradient lookup')):
                actual = implementation().predict_job(j,GpuConfig('aws:1:1','aws',1,1,999.))
            self.assertAlmostEqual(actual['raw_rate'],expected,places=10)
            self.assertEqual(actual['atomic_bsz'],prospect.atomic_bsz)
            self.assertEqual(actual['accum_steps'],prospect.accum_steps)
            self.assertEqual(original,(j.current_cluster,j.placement,j.atomic_bsz,j.accum_steps))
            self.assertNotEqual(actual['raw_rate'],999.)

    def test_real_unseeded_bert_full_optimize_policy_chain(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build()
            totals = dict(training=0., startup=0.)
            previous = 0.
            for _ in range(12):
                c.step(60)
                d = c.policy.last_decision
                self.assertTrue(d['result']['audit']['valid'])
                self.assertGreaterEqual(c.jobs[0].progress,previous)
                previous = c.jobs[0].progress
                for e in c.joint_round_logs[-1]['execution']:
                    self.assertTrue(audit_execution(e)['valid'])
                    totals['training'] += e['training_seconds']
                    totals['startup'] += e['checkpoint_delay_consumed']
                    span=max(0.,e['interval_end']-max(e['interval_start'],e['arrival_time']))
                    if e['checkpoint_delay_consumed'] >= span > 0:
                        self.assertEqual(e['progress_after'],e['progress_before'])
            self.assertGreater(totals['training'],0.)
            self.assertGreater(totals['startup'],0.)
            self.assertGreater(c.jobs[0].progress,0.)
            self.assertIsNotNone(c.jobs[0].first_productive_training_time)

    def test_proportional_ablation_is_a_real_quota_contract(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c = self.build(proportional=True)
            observed=0
            for _ in range(8):
                c.step(60)
                d=c.policy.last_decision
                self.assertTrue(d['result']['audit']['valid'])
                for record in d['result']['selected'].values():
                    q=record
                    typ=record['config']['gpu_type']
                    capacity=c.policy.node_capacities[typ]
                    self.assertAlmostEqual(q['cpu_per_gpu'],capacity['cpu_cores']/c.ngpus_per_node[typ])
                    self.assertAlmostEqual(q['ram_gb_per_gpu'],capacity['ram_gb']/c.ngpus_per_node[typ])
                    observed+=1
            self.assertGreater(observed,0)

    def test_cold_candidates_are_proportional_and_zero_scored(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=self.build()
            c.step(60)
            d=c.policy.last_decision
            self.assertGreater(len(d['completion_scores']),0)
            self.assertGreater(len(d['result']['selected']),0)
            for score in d['completion_scores']:
                self.assertIsNone(score['raw_rate'])
                self.assertEqual(score['utility'],0.)
                self.assertIsNone(score['estimated_completion'])
            for column in d['probe']['joint_columns']:
                typ=column['config']['gpu_type']
                cap=c.policy.node_capacities[typ]
                self.assertAlmostEqual(column['query']['cpu_per_gpu'],cap['cpu_cores']/c.ngpus_per_node[typ])
                self.assertAlmostEqual(column['query']['ram_gb_per_gpu'],cap['ram_gb']/c.ngpus_per_node[typ])

    def test_joint_and_prop_share_objective_contract_and_liveness(self):
        with contextlib.redirect_stdout(io.StringIO()):
            a=self.build(False);b=self.build(True)
            self.assertIs(type(a.policy.liveness),type(b.policy.liveness))
            self.assertEqual(a.policy.liveness.quantum,b.policy.liveness.quantum)
            self.assertEqual(a.policy.aux_cpu_points,b.policy.aux_cpu_points)
            self.assertEqual(a.policy.aux_ram_points,b.policy.aux_ram_points)
            self.assertEqual(a.policy.oracle.provenance_digest(),b.policy.oracle.provenance_digest())
            for c in (a,b):
                c.step(60)
                self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
                self.assertTrue(c.policy.last_decision['completion_scores'])

    def test_real_observed_candidates_include_distinct_gpu_and_aux_choices(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=self.pair()
            known=[]
            for _ in range(8):
                c.step(60)
                d=c.policy.last_decision
                self.assertTrue(d['result']['audit']['valid'])
                known.extend(v for v in d['completion_scores'] if v['raw_rate'] is not None)
                for selected in d['result']['selected'].values():
                    self.assertGreater(selected['cpu_per_gpu'],0.)
                    self.assertGreater(selected['ram_gb_per_gpu'],0.)
            self.assertTrue(known)
            columns=c.policy.last_decision['probe']['joint_columns']
            self.assertGreater(len({v['config']['config_id'] for v in columns}),1)
            self.assertGreater(len({(v['query']['cpu_per_gpu'],v['query']['ram_gb_per_gpu']) for v in columns}),1)
            self.assertTrue(all(j.progress>0 for j in c.jobs))
            for score in known:
                self.assertAlmostEqual(score['estimated_completion'],
                    score['delay']+score['remaining_work']/(score['raw_rate']*score['rho']))
                self.assertAlmostEqual(score['utility'],score['wait_cost']-score['estimated_completion'])

    def test_same_warmed_state_prop_and_joint_use_identical_wait_reference(self):
        with contextlib.redirect_stdout(io.StringIO()):
            a=self.pair()
            for _ in range(4):
                a.step(60)
            module=implementation()
            # Capture the true next scheduling call after real training. Compare
            # the shared scoring function on its JO and JP candidate domains;
            # reinstalling a policy mid-service is intentionally forbidden.
            with patch.object(module,'score_columns',wraps=module.score_columns) as observe:
                a.optimize_policy()
            policy,specs,scheduler,all_columns=observe.call_args.args
            _,prop_scores=module.score_columns(policy,specs,scheduler,
                                              scheduler.build_proportional_columns())
            references=[]
            for scores in (a.policy.last_decision['completion_scores'],prop_scores):
                values={}
                for score in scores:
                    if score['raw_rate'] is not None:
                        values.setdefault(score['job'],set()).add(score['wait_cost'])
                self.assertTrue(values)
                self.assertTrue(all(len(v)==1 for v in values.values()))
                references.append(values)
            self.assertEqual(references[0],references[1])

    def test_aux_adapter_calls_official_sia_and_freezes_complete_gpu_intent(self):
        from sia import SiaPolicy
        original=SiaPolicy.optimize
        with contextlib.redirect_stdout(io.StringIO()):
            c=self.pair(aux=True)
            with patch.object(SiaPolicy,'optimize',autospec=True,side_effect=original) as observe:
                for _ in range(6):
                    c.step(60)
                    d=c.policy.last_decision
                    self.assertTrue(d['result']['audit']['valid'])
                    intent=d['original_sia_allocations']
                    for name, (typ,nodes) in intent.items():
                        if not nodes:
                            self.assertNotIn(name,d['result']['selected'])
                            continue
                        selected=d['result']['selected'][name]
                        self.assertEqual(selected['config']['gpu_type'],typ)
                        self.assertEqual(selected['config']['gpus'],len(nodes))
                        actual={f'{typ}:{node}':nodes.count(node) for node in set(nodes)}
                        self.assertEqual(selected['placement'],actual)
                    self.assertEqual(d['admission'],'OFFICIAL_SIA_NO_L2_NO_SECOND_ADMISSION')
            self.assertEqual(observe.call_count,6)

    def test_real_checkpoint_preserves_full_gpu_and_aux_geometry(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=self.build()
            c.step(60)
            self.assertGreater(c.jobs[0].rescale_time,60.)
            allocation=copy.deepcopy(c.allocations)
            resource=copy.deepcopy(c.policy.current_resources)
            before=c.jobs[0].progress
            c.step(60)
            self.assertEqual(c.allocations,allocation)
            for key in ('config_id','cpu_per_gpu','ram_gb_per_gpu','placement'):
                self.assertEqual(c.policy.current_resources['bert-mid'][key],resource['bert-mid'][key])
            self.assertEqual(c.jobs[0].progress,before)
            self.assertIn('bert-mid',c.policy.last_decision['result']['audit']['checkpoint_locks'])

    def test_solver_failure_does_not_partially_commit_resources_or_service(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=self.build()
            c.step(60)
            before_alloc=copy.deepcopy(c.allocations)
            before_resources=copy.deepcopy(c.policy.current_resources)
            before_services=copy.deepcopy(c.policy.liveness.services)
            with patch('joint_prototype.adaptive_service.milp',return_value=SimpleNamespace(
                    x=None,status=1,success=False,message='fixed unit timeout')):
                with self.assertRaises(RuntimeError):
                    c.optimize_policy()
            self.assertEqual(c.allocations,before_alloc)
            self.assertEqual(c.policy.current_resources,before_resources)
            self.assertEqual(c.policy.liveness.services,before_services)


if __name__=='__main__':
    unittest.main(verbosity=2)
