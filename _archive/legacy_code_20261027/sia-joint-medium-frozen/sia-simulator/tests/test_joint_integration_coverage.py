"""Integration coverage through Cluster, JobInfo, policy, and Job state paths."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

import simulator_config as sim_config
from joint_prototype.joint_integration import JointCluster, JointSiaPolicy
from joint_prototype.joint_profile import JointProfileOracle
from joint_prototype.joint_scheduler import PolicyResult


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT.parent / "research_sources" / "Synergy" / "simulator" / "models"


class DropB4SolvePolicy(JointSiaPolicy):
    """Corrupt the B4 solver result so production run_b4 exercises fallback."""

    def _scheduler(self, nodes, specs):
        scheduler = super()._scheduler(nodes, specs)
        original = scheduler.solve

        def solve(policy, columns, dimensions):
            result = original(policy, columns, dimensions)
            if policy == "B4_SERIAL_FIXED_GPU_AUX_PACKING" and result.selected:
                result.selected.pop(sorted(result.selected)[0])
            return result

        scheduler.solve = solve
        return scheduler


class FailB3SolvePolicy(JointSiaPolicy):
    fail_solver = False

    def _scheduler(self, nodes, specs):
        scheduler = super()._scheduler(nodes, specs)
        original = scheduler.solve

        def solve(policy, columns, dimensions):
            if self.fail_solver and policy == "B3_SIA_GOODPUT_JOINT_MIP":
                return PolicyResult(
                    policy, "FORCED_SOLVER_FAILURE", {}, sorted(j.job_id for j in specs),
                    0.0, 0.0, 0.0, audit={"valid": False, "violations": ["forced test failure"]}
                )
            return original(policy, columns, dimensions)

        scheduler.solve = solve
        return scheduler


class JointIntegrationCoverageTests(unittest.TestCase):
    def setUp(self):
        sim_config.num_parallel_jobs = 1
        sim_config.sia_solver_verbose = False
        sim_config.preserve_ckpt_allocs = True

    def _cluster(
        self,
        rows,
        mode="b3",
        nnodes=None,
        ngpus=None,
        capacities=None,
        project=True,
        policy_class=JointSiaPolicy,
        neutral=True,
    ):
        nnodes = nnodes or {"aws": 1, "dgx-ext": 1, "rtx": 1}
        ngpus = ngpus or {"aws": 4, "dgx-ext": 8, "rtx": 8}
        capacities = capacities or {
            cluster: {"cpu_cores": 3.0 * count, "ram_gb": 62.5 * count}
            for cluster, count in ngpus.items()
        }
        oracle = JointProfileOracle(MODELS, "fixed_reference", neutral_response=neutral)
        template_map = {row["name"]: "res18_1.py" for row in rows}
        policy = policy_class(
            mode=mode,
            oracle=oracle,
            node_capacities=capacities,
            template_map=template_map,
            aux_cpu_points=[1, 2, 3, 4, 6, 8],
            aux_ram_points=[20.83, 62.5, 125, 250],
            p_fairness=-0.5,
            restart_penalty=30,
            lambda_a=0.01,
            lambda_n=1.1,
            project_throughputs=project,
            share_max_replicas=False,
            sched_interval=60,
            enable_bsz_tuning=True,
        )
        max_nodes = dict(nnodes)
        policy.populate_valid_configs(nnodes, ngpus, max_nodes)
        cluster = JointCluster(pd.DataFrame(rows), policy, nnodes, ngpus, max_nodes)
        return cluster

    @staticmethod
    def _seed(cluster, nodes=1, replicas=8):
        for job in cluster.jobs:
            job.seed_profiles(nodes, replicas)

    def test_projected_profile_reaches_integrated_decision(self):
        rows = [{"name": "cifar10-projected", "time": 0, "application": "cifar10", "num_replicas": 1, "batch_size": 128}]
        cluster = self._cluster(rows, nnodes={"aws": 1, "rtx": 1}, ngpus={"aws": 4, "rtx": 8})
        self._seed(cluster, replicas=4)
        job = cluster.jobs[0]
        job.perf_params["rtx"] = None
        job.profiles["rtx"] = {}
        job.migrate("aws", [1])
        cluster.allocations[job.name] = ("aws", (0,))
        cluster.step(121)
        sources = {
            source
            for summary in cluster.policy.last_decision["candidate_summary"].values()
            for source in summary["goodput_sources"]
        }
        self.assertIn("direct_profile", sources)
        self.assertIn("projected_profile", sources)
        self.assertGreater(job.progress, 0)

    def test_rigid_job_holds_exact_gpu_count_and_progresses(self):
        rows = [{"name": "cifar10-rigid", "time": 0, "application": "cifar10", "num_replicas": 4, "batch_size": 512, "category": "rigid"}]
        cluster = self._cluster(rows)
        self._seed(cluster, replicas=8)
        for _ in range(4):
            cluster.step(60)
        selected = cluster.policy.last_decision["result"]["selected"]["cifar10-rigid"]
        self.assertEqual(selected["config"]["gpus"], 4)
        self.assertTrue(all(r["config"]["gpus"] == 4 for r in [selected]))
        self.assertGreater(cluster.jobs[0].progress, 0)

    def test_pmp_scale_units_are_legal_on_actual_jobinfo_path(self):
        rows = [{"name": "gpt_pmp-scale", "time": 0, "application": "gpt_pmp", "num_replicas": 1, "batch_size": 384, "category": "adaptive"}]
        cluster = self._cluster(
            rows,
            nnodes={"dgx-ext": 2, "rtx": 2},
            ngpus={"dgx-ext": 8, "rtx": 8},
            capacities={
                "dgx-ext": {"cpu_cores": 24, "ram_gb": 500},
                "rtx": {"cpu_cores": 24, "ram_gb": 500},
            },
        )
        self._seed(cluster, nodes=2, replicas=16)
        infos = cluster.get_job_infos()
        configs, provenance = cluster.policy._raw_candidates(infos["gpt_pmp-scale"])
        self.assertTrue(cluster.joint_profile_bridges.get("gpt_pmp"))
        for config in configs:
            scale_unit = provenance[config.config_id]["scale_unit"]
            self.assertEqual(config.gpus % scale_unit, 0)
        cluster.optimize_policy()
        chosen = cluster.policy.last_decision["result"]["selected"]["gpt_pmp-scale"]
        provenance = cluster.policy.last_decision["resources"]["gpt_pmp-scale"]["goodput_provenance"]
        self.assertEqual(chosen["config"]["gpus"] % provenance["scale_unit"], 0)

    def test_multinode_rigid_uses_complete_nodes(self):
        rows = [{"name": "cifar10-multinode", "time": 0, "application": "cifar10", "num_replicas": 16, "batch_size": 2048, "category": "rigid"}]
        cluster = self._cluster(
            rows,
            nnodes={"rtx": 2},
            ngpus={"rtx": 8},
            capacities={"rtx": {"cpu_cores": 24, "ram_gb": 500}},
        )
        self._seed(cluster, nodes=2, replicas=16)
        cluster.step(60)
        selected = cluster.policy.last_decision["result"]["selected"]["cifar10-multinode"]
        self.assertEqual(selected["config"]["num_nodes"], 2)
        self.assertEqual(sorted(selected["placement"].values()), [8, 8])

    def test_b4_repairs_placement_on_integrated_path(self):
        rows = [{"name": "cifar10-repair", "time": 0, "application": "cifar10", "num_replicas": 1, "batch_size": 128}]
        capacities = {
            "rtx": {
                "cpu_cores": 8,
                "ram_gb": 200,
                "nodes": {"1": {"cpu_cores": 24, "ram_gb": 500}},
            }
        }
        cluster = self._cluster(rows, mode="b4", nnodes={"rtx": 2}, ngpus={"rtx": 8}, capacities=capacities, neutral=False)
        self._seed(cluster)
        cluster.step(60)
        audit = cluster.policy.last_decision["result"]["audit"]
        self.assertEqual(audit["repair_rule"]["cifar10-repair"], "REPACK_FIXED_GPU_CONFIG")
        self.assertEqual(cluster.policy.last_decision["sia_fixed_gpu_configs"]["cifar10-repair"], cluster.policy.last_decision["resources"]["cifar10-repair"]["config_id"])

    def test_b4_forced_drop_uses_full_atomic_fallback(self):
        rows = [
            {"name": "cifar10-fallback", "time": 0, "application": "cifar10", "num_replicas": 1, "batch_size": 128},
            {"name": "bert-fallback", "time": 0, "application": "bert", "num_replicas": 1, "batch_size": 384},
        ]
        cluster = self._cluster(rows, mode="b4", policy_class=DropB4SolvePolicy)
        self._seed(cluster)
        cluster.step(60)
        result = cluster.policy.last_decision["result"]
        self.assertEqual(result["status"], "FALLBACK_B1_PROPORTIONAL")
        self.assertEqual(set(result["selected"]), set(cluster.policy.last_decision["sia_fixed_gpu_configs"]))
        self.assertTrue(result["audit"]["valid"])
        self.assertTrue(result["audit"]["fallback_reason"])

    def test_b3_solver_failure_preserves_valid_previous_state(self):
        rows = [{"name": "cifar10-solverfail", "time": 0, "application": "cifar10", "num_replicas": 1, "batch_size": 128}]
        cluster = self._cluster(rows, policy_class=FailB3SolvePolicy)
        self._seed(cluster)
        cluster.step(60)
        allocations_before = dict(cluster.allocations)
        resources_before = dict(cluster.policy.current_resources)
        cluster.policy.fail_solver = True
        cluster.optimize_policy()
        self.assertEqual(cluster.allocations, allocations_before)
        self.assertEqual(cluster.policy.current_resources, resources_before)
        result = cluster.policy.last_decision["result"]
        self.assertEqual(
            result["status"], "FALLBACK_PRESERVE_PREVIOUS_VALID_ALLOCATION"
        )
        self.assertTrue(result["audit"]["valid"])

    def test_checkpoint_lock_preserves_config_and_nodes_end_to_end(self):
        rows = [{"name": "bert-lock", "time": 0, "application": "bert", "num_replicas": 1, "batch_size": 384}]
        cluster = self._cluster(rows)
        self._seed(cluster)
        cluster.step(60)
        before = cluster.policy.current_resources["bert-lock"]
        self.assertGreater(cluster.jobs[0].rescale_time, 0)
        cluster.step(60)
        after = cluster.policy.current_resources["bert-lock"]
        self.assertEqual(after["config_id"], before["config_id"])
        self.assertEqual(after["placement"], before["placement"])
        self.assertEqual(cluster.joint_round_logs[-1]["execution"][0]["progress_delta"], 0)


if __name__ == "__main__":
    unittest.main()
