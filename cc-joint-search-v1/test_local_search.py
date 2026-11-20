"""Synthetic states for localized exchange and fallback, without a simulator."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import local_search
from joint_prototype.joint_scheduler import CandidateLimitReached


def make_column(job, config, node, cpu, eta, column_id):
    config_rank = 1.0 if config == "gpu-a" else 2.0
    usage = (1, cpu, 1)
    return SimpleNamespace(
        job_id=job,
        column_id=column_id,
        config=SimpleNamespace(
            config_id=config,
            sia_goodput=config_rank,
            cpu_demand_scale=config_rank,
            memory_demand_scale=1.0,
        ),
        query=SimpleNamespace(rho=1.0),
        placement=SimpleNamespace(node_ids=(node,)),
        node_usage=((node, *usage),),
        usage_for=lambda node_id: usage if node_id == node else (0, 0, 0),
        scheduling_utility=1.0 / eta,
        eta=eta,
    )


class Provider:
    def __init__(self, proportional, expanded):
        self.proportional = proportional
        self.expanded = expanded
        self.expansion_calls = []

    def expand(self, scope, *, deadline=None, max_columns=None):
        self.expansion_calls.append((scope, deadline, max_columns))
        if len(self.expanded) > max_columns:
            raise CandidateLimitReached(max_columns)
        return self.expanded

    @staticmethod
    def legal_local(columns):
        return columns

    @staticmethod
    def eta(column):
        return column.eta


class LocalSearchTest(TestCase):
    def setUp(self):
        self.anchor = make_column("anchor", "gpu-a", "n0", 10, 10, "anchor-fo")
        self.peer = make_column("peer", "gpu-b", "n1", 8, 5, "peer-fo")
        self.locked = make_column("locked", "gpu-a", "n0", 1, 8, "locked-fo")
        self.move = make_column("anchor", "gpu-b", "n1", 5, 7, "anchor-move")
        self.donation = make_column("peer", "gpu-b", "n1", 4, 6, "peer-donation")
        self.fo = [self.anchor, self.peer, self.locked]
        self.provider = Provider(
            [self.anchor, self.move, self.peer, self.locked], [self.move, self.donation]
        )
        self.nodes = [
            SimpleNamespace(node_id=node_id, gpus=2, cpu_cores=12, ram_gb=24)
            for node_id in ("n0", "n1")
        ]
        self.audited = []
        self.scheduler = SimpleNamespace(
            nodes=self.nodes,
            job_map={column.job_id: column for column in self.fo},
            lock_active=lambda job: job is self.locked,
            audit=self.audit,
        )

    def audit(self, columns):
        self.audited.append(columns)
        return {
            "valid": (
                len(columns) == 3
                and {column.job_id for column in columns} == set(self.scheduler.job_map)
                and columns[2] is self.locked
                and all(
                    all(sum(column.usage_for(node.node_id)[dimension] for column in columns) <= capacity
                        for dimension, capacity in enumerate((node.gpus, node.cpu_cores, node.ram_gb)))
                    for node in self.nodes
                )
            )
        }

    def improve(self, deadline=1.0):
        stats = {}
        result = local_search.improve(None, self.scheduler, self.fo, self.provider, deadline, 0, stats)
        self.assertIs(self.fo[0], self.anchor)
        self.assertIs(self.fo[1], self.peer)
        self.assertIs(self.fo[2], self.locked)
        return result, stats

    @patch.object(local_search, "perf_counter", return_value=0.0)
    def test_gpu_move_and_cpu_donation_must_both_be_selected(self, _clock):
        # The move in isolation uses 13 CPUs on n1, which only has 12.
        combined, stats = self.improve()
        self.assertEqual(stats["local_status"], "accepted")
        self.assertIsNot(combined, self.fo)
        self.assertIs(combined[0], self.move)
        self.assertIs(combined[1], self.donation)
        self.assertIs(combined[2], self.locked)
        self.assertAlmostEqual(stats["local_eta_saved"], 2.0)
        self.assertGreaterEqual(stats["local_utility_delta"], 0.0)
        self.assertEqual(stats["local_combinations"], 4)
        self.assertEqual(
            self.provider.expansion_calls,
            [({"anchor": {"gpu-a", "gpu-b"}, "peer": {"gpu-b"}},
              1.0, local_search.MAX_EXPANDED_COLUMNS)],
        )
        self.assertEqual(self.audited, [combined])

        # Removing either half makes the exchange infeasible or unhelpful.
        for only in (self.move, self.donation):
            with self.subTest(only=only.column_id):
                self.provider.expanded = [only]
                result, stats = self.improve()
                self.assertEqual(stats["local_status"], "no_safe_eta_improvement")
                self.assertIs(result, self.fo)

    @patch.object(local_search, "perf_counter", return_value=0.0)
    def test_failed_final_audit_returns_exact_fo_bundle(self, _clock):
        def reject(columns):
            self.audited.append(columns)
            return {"valid": False}

        self.scheduler.audit = reject
        result, stats = self.improve()
        self.assertEqual(stats["local_status"], "audit_rejected")
        self.assertIs(result, self.fo)
        self.assertEqual(len(self.audited), 1)
        self.assertIs(self.audited[0][2], self.locked)

    @patch.object(local_search, "perf_counter", return_value=0.0)
    def test_expired_deadline_returns_exact_fo_bundle_before_search(self, _clock):
        result, stats = self.improve(deadline=0.0)
        self.assertEqual(stats["local_status"], "timeout")
        self.assertIs(result, self.fo)
        self.assertEqual(self.provider.expansion_calls, [])
        self.assertEqual(self.audited, [])

    @patch.object(local_search, "MAX_EXPANDED_COLUMNS", 1)
    @patch.object(local_search, "perf_counter", return_value=0.0)
    def test_expansion_cap_returns_exact_fo_bundle(self, _clock):
        result, stats = self.improve()
        self.assertEqual(stats["local_status"], "expansion_limit")
        self.assertEqual(stats["local_expanded_columns"], 1)
        self.assertIs(result, self.fo)
        self.assertEqual(self.audited, [])


if __name__ == "__main__":
    main()
