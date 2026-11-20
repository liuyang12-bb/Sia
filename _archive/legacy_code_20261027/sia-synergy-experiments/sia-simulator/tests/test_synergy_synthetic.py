import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from job import Job
from synergy_synthetic import SyntheticAuxAllocator, SyntheticResourceModelError


HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent
CONFIG = SIMULATOR / "synthetic_resource_model.json"
NODES = {"aws": 6, "dgx-ext": 2, "rtx": 3}
GPUS = {"aws": 4, "dgx-ext": 8, "rtx": 8}


class FakeJob:
    def __init__(self, goodput=10.0):
        self.goodput = goodput

    def estimate_current_goodput(self):
        return self.goodput


def allocator(mode="s2", scenario="mixed"):
    return SyntheticAuxAllocator(
        str(CONFIG), mode, scenario, NODES, GPUS)


class SyntheticAuxAllocatorTests(unittest.TestCase):
    def test_config_is_explicitly_synthetic_and_baseline_is_one(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["resource_model_type"], "synthetic")
        self.assertTrue(config["source"]["not_real_hardware"])
        self.assertEqual(
            config["gpu_proportional_baseline"]["relative_progress_multiplier"], 1.0)

    def test_s1_gpu_proportional_allocation(self):
        model = allocator("s1", "mixed")
        result = model.allocate(
            {"job-0": ("aws", [0, 0, 1])},
            {"job-0": FakeJob()}, 60)
        job = result["jobs"]["job-0"]
        self.assertEqual(job["multiplier"], 1.0)
        self.assertEqual(job["nodes"]["0"], {"gpus": 2, "cpu": 6.0, "ram_gb": 125.0})
        self.assertEqual(job["nodes"]["1"], {"gpus": 1, "cpu": 3.0, "ram_gb": 62.5})

    def test_neutral_s1_and_s2_are_identical(self):
        allocations = {"job-0": ("aws", [0]), "job-1": ("rtx", [0, 0])}
        jobs = {name: FakeJob() for name in allocations}
        s1 = allocator("s1", "neutral").allocate(allocations, jobs, 60)
        s2 = allocator("s2", "neutral").allocate(allocations, jobs, 60)
        self.assertEqual(s1["jobs"], s2["jobs"])
        self.assertEqual(s1["node_usage"], s2["node_usage"])

    def test_cpu_sensitive_job_receives_extra_cpu_and_gain(self):
        result = allocator("s2", "cpu_sensitive").allocate(
            {"job-0": ("aws", [0])}, {"job-0": FakeJob(8.0)}, 60)
        job = result["jobs"]["job-0"]
        self.assertEqual(job["profile"], "cpu_sensitive")
        self.assertEqual(job["cpu_ratio"], 1.5)
        self.assertEqual(job["ram_ratio"], 1.0)
        self.assertAlmostEqual(job["multiplier"], 1.3)

    def test_memory_sensitive_job_receives_extra_ram_and_gain(self):
        result = allocator("s2", "memory_sensitive").allocate(
            {"job-1": ("aws", [0])}, {"job-1": FakeJob(8.0)}, 60)
        job = result["jobs"]["job-1"]
        self.assertEqual(job["profile"], "memory_sensitive")
        self.assertEqual(job["cpu_ratio"], 1.0)
        self.assertEqual(job["ram_ratio"], 1.5)
        self.assertAlmostEqual(job["multiplier"], 1.25)

    def test_insensitive_job_has_no_gain_or_extra_resources(self):
        result = allocator("s2", "mixed").allocate(
            {"job-2": ("aws", [0])}, {"job-2": FakeJob()}, 60)
        job = result["jobs"]["job-2"]
        self.assertEqual(job["profile"], "insensitive")
        self.assertEqual(job["cpu_ratio"], 1.0)
        self.assertEqual(job["ram_ratio"], 1.0)
        self.assertEqual(job["multiplier"], 1.0)

    def test_capacity_never_exceeded(self):
        allocations = {
            "job-0": ("aws", [0]),
            "job-1": ("aws", [0]),
            "job-2": ("aws", [0]),
            "job-3": ("aws", [0]),
        }
        result = allocator("s2", "mixed").allocate(
            allocations, {name: FakeJob() for name in allocations}, 60)
        self.assertEqual(result["capacity_violations"], [])
        self.assertLessEqual(result["node_usage"]["aws"]["0"]["cpu"], 12.0)
        self.assertLessEqual(result["node_usage"]["aws"]["0"]["ram_gb"], 250.0)

    def test_multi_node_ratio_is_consistent(self):
        result = allocator("s2", "cpu_sensitive").allocate(
            {"job-0": ("rtx", [0, 0, 0, 0, 1, 1])},
            {"job-0": FakeJob()}, 60)
        job = result["jobs"]["job-0"]
        node0 = job["nodes"]["0"]
        node1 = job["nodes"]["1"]
        self.assertAlmostEqual(node0["cpu"] / node0["gpus"],
                               node1["cpu"] / node1["gpus"])
        self.assertAlmostEqual(node0["ram_gb"] / node0["gpus"],
                               node1["ram_gb"] / node1["gpus"])

    def test_response_saturates(self):
        model = allocator("s2", "mixed")
        self.assertAlmostEqual(model.response_multiplier("cpu_sensitive", 1.5, 1.0), 1.3)
        self.assertAlmostEqual(model.response_multiplier("cpu_sensitive", 3.0, 1.0), 1.3)
        self.assertAlmostEqual(model.response_multiplier("memory_sensitive", 1.0, 1.5), 1.25)
        self.assertAlmostEqual(model.response_multiplier("memory_sensitive", 1.0, 3.0), 1.25)

    def test_effective_goodput_multiplies_sia_goodput_once(self):
        result = allocator("s2", "cpu_sensitive").allocate(
            {"job-0": ("aws", [0])}, {"job-0": FakeJob(10.0)}, 60)
        job = result["jobs"]["job-0"]
        self.assertEqual(job["sia_goodput"], 10.0)
        self.assertAlmostEqual(job["effective_goodput"], 13.0)

    def test_full_node_resource_shortage_falls_back_to_baseline(self):
        result = allocator("s2", "cpu_sensitive").allocate(
            {"job-0": ("aws", [0, 0, 0, 0])},
            {"job-0": FakeJob()}, 60)
        job = result["jobs"]["job-0"]
        self.assertEqual(job["choice"], "cpu=1,ram=1")
        self.assertEqual(job["multiplier"], 1.0)
        self.assertIsNone(result["fallback"])

    def test_profile_assignment_does_not_depend_on_application_label(self):
        model = allocator("s2", "mixed")
        self.assertEqual(model.profile_for("cifar10-0"), model.profile_for("unrelated-0"))
        self.assertEqual(model.profile_for("bert-1"), model.profile_for("another-1"))

    def test_invalid_cluster_schema_is_rejected(self):
        with self.assertRaises(SyntheticResourceModelError):
            SyntheticAuxAllocator(str(CONFIG), "s1", "neutral",
                                  {"aws": 5, "dgx-ext": 2, "rtx": 3}, GPUS)

    def test_auxiliary_multiplier_does_not_change_checkpoint_restart_semantics(self):
        job = Job.__new__(Job)
        job.completion_time = None
        job.placement = (2,)
        job.current_time = 100.0
        job.rescale_time = 30.0
        job.attained_service = 0.0
        job.wasted_gpu_seconds = 0.0
        job.used_gpu_seconds = 0.0
        job.num_restarts = 3
        job.step(20, auxiliary_multiplier=1.3)
        self.assertEqual(job.current_time, 120.0)
        self.assertEqual(job.rescale_time, 10.0)
        self.assertEqual(job.wasted_gpu_seconds, 40.0)
        self.assertEqual(job.used_gpu_seconds, 0.0)
        self.assertEqual(job.num_restarts, 3)

    def test_solver_failure_is_explicit_and_falls_back_to_s1(self):
        model = allocator("s2", "cpu_sensitive")
        failure = SimpleNamespace(success=False, x=None, status=1,
                                  message="synthetic solver failure")
        with patch("synergy_synthetic.milp", return_value=failure):
            result = model.allocate(
                {"job-0": ("aws", [0])}, {"job-0": FakeJob()}, 60)
        self.assertEqual(result["jobs"]["job-0"]["choice"], "cpu=1,ram=1")
        self.assertIn("MILP status=1", result["fallback"])
        self.assertEqual(model.fallback_count, 1)


if __name__ == "__main__":
    unittest.main()
