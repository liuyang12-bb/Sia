import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np


SIMULATOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATOR_DIR not in sys.path:
    sys.path.insert(0, SIMULATOR_DIR)

import simulator_config as sim_config
from sia_afs import (
    SiaAFSPolicy,
    afs_equation_three,
    allocate_afs_p_homogeneous,
    processor_share_counter,
)
from utils import NodeInfo


class FakeSpeedup:
    def __init__(self, values):
        self.values = dict(values)

    def get_goodput(self, num_nodes, num_replicas):
        replicas = np.asarray(num_replicas, dtype=np.int64)
        return np.asarray(
            [self.values[int(replica)] for replica in replicas],
            dtype=np.float32,
        )


def adaptive_job(
    speedups,
    *,
    max_replicas=4,
    execution_time=0,
    scaling_underway=False,
    scale_unit=1,
):
    clusters = list(speedups)
    return SimpleNamespace(
        category=None,
        speedup_fn={name: FakeSpeedup(values) for name, values in speedups.items()},
        max_replicas={name: max_replicas for name in clusters},
        min_replicas={name: 0 for name in clusters},
        scale_unit={name: scale_unit for name in clusters},
        cluster_throughputs={name: values[1] for name, values in speedups.items()},
        cluster_throughput_ratios={name: {} for name in clusters},
        scaling_underway=scaling_underway,
        execution_time=execution_time,
        profile_counts={name: 1 for name in clusters},
        used_gpu_seconds=execution_time,
        wasted_gpu_seconds=0,
        attained_service=execution_time,
        age=execution_time,
        num_restarts=0,
        num_migrations=0,
    )


def nodes(capacities):
    return {
        cluster: {
            idx: NodeInfo(
                resources={"nvidia.com/gpu": gpus_per_node},
                preemptible=False,
            )
            for idx, gpus_per_node in enumerate(node_gpus)
        }
        for cluster, node_gpus in capacities.items()
    }


def make_policy(cluster_nodes, cluster_gpus, **kwargs):
    policy = SiaAFSPolicy(
        processor_share_unit_seconds=7200,
        afs_seed=0,
        project_throughputs=False,
        share_max_replicas=False,
        **kwargs,
    )
    policy.populate_valid_configs(
        cluster_nodes,
        cluster_gpus,
        cluster_nodes,
    )
    return policy


class EquationAndReferenceAllocatorTests(unittest.TestCase):
    def test_equation_three_uses_the_paper_asymmetric_denominators(self):
        # (15-10)/15 > (11-10)/10, so the longer job b wins.
        self.assertTrue(afs_equation_three(10.0, 11.0, 10.0, 15.0))
        # A symmetric marginal-gain comparison would tie; the paper does not.
        self.assertFalse(afs_equation_three(10.0, 15.0, 10.0, 11.0))

    def test_single_job_receives_all_useful_share(self):
        result = allocate_afs_p_homogeneous(
            {"job": [0.0, 1.0, 1.8, 2.4, 2.8]},
            total_gpus=4,
            scheduled_seconds={"job": 0},
            unit_seconds=7200,
            seed=0,
        )
        self.assertEqual(result.shares, {"job": 4})
        self.assertEqual(result.stage, "efficiency")

    def test_two_job_equation_three_decision(self):
        result = allocate_afs_p_homogeneous(
            {
                "shorter_like": [0.0, 10.0, 11.0],
                "efficient_like": [0.0, 10.0, 15.0],
            },
            total_gpus=3,
            scheduled_seconds={"shorter_like": 0, "efficient_like": 0},
            unit_seconds=7200,
            seed=0,
        )
        self.assertEqual(result.shares["shorter_like"], 1)
        self.assertEqual(result.shares["efficient_like"], 2)

    def test_high_contention_selects_lowest_processor_share_counters(self):
        curves = {name: [0.0, 1.0] for name in ("a", "b", "c")}
        result = allocate_afs_p_homogeneous(
            curves,
            total_gpus=2,
            scheduled_seconds={"a": 0, "b": 7200, "c": 14400},
            unit_seconds=7200,
            seed=0,
        )
        self.assertEqual(result.shares, {"a": 1, "b": 1, "c": 0})
        self.assertEqual(result.stage, "processor_sharing")

    def test_high_contention_is_non_preemptive_until_quantum_expires(self):
        curves = {name: [0.0, 1.0] for name in ("new", "incumbent")}
        protected = allocate_afs_p_homogeneous(
            curves,
            total_gpus=1,
            scheduled_seconds={"new": 0, "incumbent": 14400},
            unit_seconds=7200,
            seed=0,
            incumbents={"incumbent"},
            dispatch_counters={"incumbent": 2},
        )
        self.assertEqual(protected.shares["incumbent"], 1)
        expired = allocate_afs_p_homogeneous(
            curves,
            total_gpus=1,
            scheduled_seconds={"new": 0, "incumbent": 21600},
            unit_seconds=7200,
            seed=0,
            incumbents={"incumbent"},
            dispatch_counters={"incumbent": 2},
        )
        self.assertEqual(expired.shares["new"], 1)
        self.assertEqual(expired.shares["incumbent"], 0)

    def test_counter_is_scheduled_wall_time_not_gpu_seconds(self):
        self.assertEqual(processor_share_counter(14399, 7200), 1)
        self.assertEqual(processor_share_counter(14400, 7200), 2)


class HeterogeneousMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_verbose = sim_config.sia_solver_verbose
        cls.old_threads = sim_config.sia_solver_num_threads
        sim_config.sia_solver_verbose = False
        sim_config.sia_solver_num_threads = 1

    @classmethod
    def tearDownClass(cls):
        sim_config.sia_solver_verbose = cls.old_verbose
        sim_config.sia_solver_num_threads = cls.old_threads

    def test_single_job_maps_to_faster_legal_gpu_type(self):
        policy = make_policy(
            {"slow": 1, "fast": 1},
            {"slow": 4, "fast": 4},
        )
        job = adaptive_job(
            {
                "slow": {1: 1.0, 2: 1.8, 3: 2.4, 4: 2.8},
                "fast": {1: 2.0, 2: 3.6, 3: 4.8, 4: 5.6},
            }
        )
        result = policy.optimize(
            {"job": job}, nodes({"slow": [4], "fast": [4]}), {}
        )
        self.assertEqual(result["job"][0], "fast")
        self.assertIn(len(result["job"][1]), {1, 2, 4})

    def test_equal_share_mapping_prefers_reuse_before_gpu_type_gain(self):
        policy = make_policy(
            {"slow": 1, "fast": 1},
            {"slow": 4, "fast": 4},
        )
        job = adaptive_job(
            {
                "slow": {1: 1.0, 2: 1.8, 3: 2.4, 4: 2.8},
                "fast": {1: 2.0, 2: 3.6, 3: 4.8, 4: 5.6},
            }
        )
        previous = {"job": ("slow", (0, 0, 0, 0))}
        result = policy.optimize(
            {"job": job}, nodes({"slow": [4], "fast": [4]}), previous
        )
        self.assertEqual(result["job"], ("slow", [0, 0, 0, 0]))

    def test_cold_start_linear_curve_is_capped_at_fair_share(self):
        policy = make_policy({"gpu": 1}, {"gpu": 8})
        job = adaptive_job(
            {"gpu": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}},
            max_replicas=8,
        )
        job.speedup_fn["gpu"] = None
        curve = policy._virtual_curve(job, total_gpus=8, fair_share=4)
        self.assertEqual(len(curve) - 1, 4)
        self.assertEqual(curve, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_heterogeneous_legal_configs_and_capacity(self):
        policy = make_policy(
            {"a100": 1, "rtx": 1},
            {"a100": 4, "rtx": 4},
        )
        jobs = {
            f"job-{idx}": adaptive_job(
                {
                    "a100": {1: 2.0, 2: 3.0, 3: 3.6, 4: 4.0},
                    "rtx": {1: 1.0, 2: 1.7, 3: 2.2, 4: 2.5},
                }
            )
            for idx in range(3)
        }
        result = policy.optimize(
            jobs, nodes({"a100": [4], "rtx": [4]}), {}
        )
        totals = {"a100": 0, "rtx": 0}
        for cluster, placement in result.values():
            self.assertIn(len(placement), {1, 2, 4})
            totals[cluster] += len(placement)
        self.assertLessEqual(totals["a100"], 4)
        self.assertLessEqual(totals["rtx"], 4)

    def test_unmappable_scale_unit_is_left_unallocated(self):
        policy = make_policy({"gpu": 1}, {"gpu": 4})
        job = adaptive_job(
            {"gpu": {1: 1.0, 2: 1.8, 3: 2.4, 4: 2.8}},
            max_replicas=1,
            scale_unit=3,
        )
        result = policy.optimize({"job": job}, nodes({"gpu": [4]}), {})
        self.assertEqual(result["job"], (None, ()))
        self.assertIn("job", policy.last_decision["unmapped_jobs"])

    def test_checkpoint_allocation_is_preserved_and_cost_state_not_mutated(self):
        policy = make_policy({"gpu": 1}, {"gpu": 4})
        job = adaptive_job(
            {"gpu": {1: 1.0, 2: 1.8, 3: 2.4, 4: 2.8}},
            scaling_underway=True,
        )
        job.num_restarts = 7
        job.wasted_gpu_seconds = 123
        previous = {"job": ("gpu", (0, 0))}
        result = policy.optimize({"job": job}, nodes({"gpu": [4]}), previous)
        self.assertEqual(result["job"], ("gpu", [0, 0]))
        self.assertEqual(job.num_restarts, 7)
        self.assertEqual(job.wasted_gpu_seconds, 123)

    def test_no_recompute_event_reuses_previous_allocation(self):
        policy = make_policy({"gpu": 1}, {"gpu": 4})
        job = adaptive_job(
            {"gpu": {1: 1.0, 2: 1.8, 3: 2.4, 4: 2.8}}
        )
        first = policy.optimize({"job": job}, nodes({"gpu": [4]}), {})
        second = policy.optimize({"job": job}, nodes({"gpu": [4]}), first)
        self.assertEqual(second, first)
        self.assertEqual(policy.last_decision["reason"], "no_event")


if __name__ == "__main__":
    unittest.main()
