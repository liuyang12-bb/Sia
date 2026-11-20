import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np


SIMULATOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATOR_DIR not in sys.path:
    sys.path.insert(0, SIMULATOR_DIR)

import simulator_config as sim_config
from sia import SiaPolicy, ZERO_ALLOC_GAIN
from sia_las import (
    SiaLASPolicy,
    apply_las_priority_factor,
    get_las_attained_service,
    is_high_priority,
)
from utils import NodeInfo


def rigid_job(service, *, scaling_underway=False, restarts=0, age=600):
    return SimpleNamespace(
        category="rigid",
        max_replicas={"gpu": 1},
        cluster_throughput_ratios={"gpu": 1.0},
        num_restarts=restarts,
        age=age,
        attained_service=service,
        used_gpu_seconds=service,
        wasted_gpu_seconds=0,
        scaling_underway=scaling_underway,
    )


def one_gpu_nodes():
    return {
        "gpu": {
            0: NodeInfo(resources={"nvidia.com/gpu": 1}, preemptible=False),
        }
    }


def make_policy(cls, **kwargs):
    policy = cls(
        p_fairness=-0.5,
        lambda_a=0.01,
        lambda_n=1.1,
        project_throughputs=False,
        share_max_replicas=False,
        **kwargs,
    )
    policy.populate_valid_configs(
        {"gpu": 1}, {"gpu": 1}, {"gpu": 1}
    )
    return policy


class ServiceAccountingTests(unittest.TestCase):
    def test_uses_productive_gpu_seconds_not_age_or_restart_waste(self):
        job = SimpleNamespace(
            used_gpu_seconds=3599,
            attained_service=4199,
            wasted_gpu_seconds=600,
            age=10000,
        )
        self.assertEqual(get_las_attained_service(job), 3599)

    def test_threshold_boundary(self):
        self.assertTrue(is_high_priority(3599, 3600))
        self.assertFalse(is_high_priority(3600, 3600))


class ObjectiveTests(unittest.TestCase):
    def test_factor_one_is_bitwise_identity(self):
        matrix = np.asarray(
            [[ZERO_ALLOC_GAIN, 1.0, 4.0], [2.0, 3.0, 8.0]],
            dtype=np.float64,
        )
        actual = apply_las_priority_factor(
            matrix, [True, False], priority_factor=1.0
        )
        self.assertTrue(np.array_equal(actual, matrix))

    def test_factor_increases_preference_for_default_negative_p(self):
        matrix = np.asarray([[1.0, 4.0]], dtype=np.float64)
        boosted = apply_las_priority_factor(
            matrix, [True], priority_factor=2.0
        )
        base_cost = np.power(matrix, -0.5)
        boosted_cost = np.power(boosted, -0.5)
        self.assertTrue(np.all(boosted_cost < base_cost))

    def test_factor_increases_preference_for_positive_p_without_assuming_direction(self):
        matrix = np.asarray([[1.0, 4.0]], dtype=np.float64)
        boosted = apply_las_priority_factor(
            matrix, [True], priority_factor=2.0
        )
        base_value = np.power(matrix, 0.5)
        boosted_value = np.power(boosted, 0.5)
        self.assertTrue(np.all(boosted_value > base_value))

    def test_zero_alloc_floor_is_unchanged(self):
        matrix = np.asarray([[ZERO_ALLOC_GAIN, 1.0]], dtype=np.float64)
        boosted = apply_las_priority_factor(
            matrix, [True], priority_factor=2.0
        )
        self.assertEqual(boosted[0, 0], ZERO_ALLOC_GAIN)
        self.assertEqual(boosted[0, 1], 2.0)


class SolverIntegrationTests(unittest.TestCase):
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

    def test_factor_one_matches_sia_objective_allocation_and_unallocated_penalty(self):
        jobs = {
            "cifar10-new": rigid_job(0),
            "cifar10-old": rigid_job(7200),
        }
        base = make_policy(SiaPolicy)
        las = make_policy(
            SiaLASPolicy,
            las_threshold_gpu_seconds=3600,
            las_priority_factor=1.0,
        )
        base_alloc = base.optimize(jobs, one_gpu_nodes(), {})
        las_alloc = las.optimize(jobs, one_gpu_nodes(), {})
        self.assertEqual(base_alloc, las_alloc)
        self.assertEqual(base.objective_values, las.objective_values)
        self.assertEqual(
            sum(bool(alloc[1]) for alloc in las_alloc.values()), 1
        )

    def test_boosted_high_priority_job_wins_without_breaking_capacity(self):
        jobs = {
            "cifar10-new": rigid_job(0),
            "cifar10-old": rigid_job(7200),
        }
        las = make_policy(
            SiaLASPolicy,
            las_threshold_gpu_seconds=3600,
            las_priority_factor=2.0,
        )
        allocations = las.optimize(jobs, one_gpu_nodes(), {})
        self.assertTrue(allocations["cifar10-new"][1])
        self.assertFalse(allocations["cifar10-old"][1])
        self.assertLessEqual(
            sum(len(alloc[1]) for alloc in allocations.values()), 1
        )

    def test_checkpoint_preservation_and_reallocation_logic_match_sia(self):
        jobs = {
            "cifar10-checkpoint": rigid_job(
                7200, scaling_underway=True, restarts=2, age=9000
            ),
            "cifar10-new": rigid_job(0),
        }
        previous = {"cifar10-checkpoint": ("gpu", (0,))}
        base = make_policy(SiaPolicy)
        las = make_policy(
            SiaLASPolicy,
            las_threshold_gpu_seconds=3600,
            las_priority_factor=1.0,
        )
        base.prev_allocs_v2["rigid"]["cifar10-checkpoint"] = 0
        las.prev_allocs_v2["rigid"]["cifar10-checkpoint"] = 0
        base_alloc = base.optimize(jobs, one_gpu_nodes(), previous)
        las_alloc = las.optimize(jobs, one_gpu_nodes(), previous)
        self.assertEqual(base_alloc, las_alloc)
        self.assertEqual(
            las_alloc["cifar10-checkpoint"], ("gpu", [0])
        )
        self.assertEqual(base.objective_values, las.objective_values)


if __name__ == "__main__":
    unittest.main()
