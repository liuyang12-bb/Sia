"""Controlled synthetic CPU/RAM allocation layered after Sia GPU placement.

This module deliberately does not implement Synergy-OPT or TUNE.  S1 accounts
for GPU-proportional resources.  S2 solves a separate multiple-choice MILP with
the GPU type, GPU count, and node placement fixed by Sia.
"""

from collections import Counter, defaultdict
import hashlib
import json
import re
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


SCENARIOS = ("neutral", "cpu_sensitive", "memory_sensitive", "mixed")
MODES = ("s1", "s2")


class SyntheticResourceModelError(ValueError):
    pass


def _job_cohort(job_name):
    """Assign a synthetic profile without using an application label or future state."""
    match = re.search(r"(\d+)$", job_name)
    if match:
        return int(match.group(1)) % 3
    digest = hashlib.sha256(job_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 3


class SyntheticAuxAllocator:
    def __init__(self, config_path, mode, scenario, cluster_num_nodes,
                 cluster_ngpus_per_node):
        if mode not in MODES:
            raise SyntheticResourceModelError(f"unknown mode: {mode}")
        if scenario not in SCENARIOS:
            raise SyntheticResourceModelError(f"unknown scenario: {scenario}")
        with open(config_path, encoding="utf-8") as stream:
            self.config = json.load(stream)
        if self.config.get("resource_model_type") != "synthetic":
            raise SyntheticResourceModelError(
                "resource_model_type must be exactly 'synthetic'")
        if not self.config.get("source", {}).get("not_real_hardware"):
            raise SyntheticResourceModelError("config must mark capacities as non-real")

        self.mode = mode
        self.scenario = scenario
        self.config_path = config_path
        self.capacities = self.config["clusters"]
        self.baseline = self.config["gpu_proportional_baseline"]
        self.ratios = tuple(float(v) for v in self.config["candidate_resource_ratios"])
        if self.ratios[0] != 1.0 or sorted(self.ratios) != list(self.ratios):
            raise SyntheticResourceModelError("candidate ratios must start at 1 and be sorted")
        if float(self.baseline["relative_progress_multiplier"]) != 1.0:
            raise SyntheticResourceModelError("baseline multiplier must equal 1")
        self._validate_cluster(cluster_num_nodes, cluster_ngpus_per_node)

        self.previous_choices = {}
        self.previous_maps = {}
        self.total_aux_reallocations = 0
        self.total_map_reallocations = 0
        self.fallback_count = 0
        self.solve_times = []
        self.last_result = self._empty_result()

    def _empty_result(self):
        return {
            "mode": self.mode,
            "scenario": self.scenario,
            "jobs": {},
            "node_usage": self._zero_node_usage(),
            "capacity_violations": [],
            "solve_seconds": 0.0,
            "fallback": None,
            "auxiliary_reallocations": 0,
            "map_reallocations": 0,
        }

    def _zero_node_usage(self):
        return {
            cluster: {
                str(node): {"gpus": 0, "cpu": 0.0, "ram_gb": 0.0}
                for node in range(int(capacity["nodes"]))
            }
            for cluster, capacity in self.capacities.items()
        }

    def _validate_cluster(self, cluster_num_nodes, cluster_ngpus_per_node):
        expected = set(cluster_num_nodes)
        if expected != set(self.capacities):
            raise SyntheticResourceModelError(
                f"cluster names differ: simulator={sorted(expected)}, config={sorted(self.capacities)}")
        for cluster in expected:
            capacity = self.capacities[cluster]
            if int(capacity["nodes"]) != int(cluster_num_nodes[cluster]):
                raise SyntheticResourceModelError(f"node count mismatch for {cluster}")
            if int(capacity["gpus_per_node"]) != int(cluster_ngpus_per_node[cluster]):
                raise SyntheticResourceModelError(f"GPU capacity mismatch for {cluster}")
            expected_cpu = float(self.baseline["cpu_per_gpu"]) * int(capacity["gpus_per_node"])
            expected_ram = float(self.baseline["ram_gb_per_gpu"]) * int(capacity["gpus_per_node"])
            if not np.isclose(expected_cpu, float(capacity["cpu_per_node"])):
                raise SyntheticResourceModelError(f"CPU proportional baseline mismatch for {cluster}")
            if not np.isclose(expected_ram, float(capacity["ram_gb_per_node"])):
                raise SyntheticResourceModelError(f"RAM proportional baseline mismatch for {cluster}")

    def profile_for(self, job_name):
        if self.scenario == "neutral":
            return "insensitive"
        cohort = _job_cohort(job_name)
        if self.scenario == "cpu_sensitive":
            return "cpu_sensitive" if cohort == 0 else "insensitive"
        if self.scenario == "memory_sensitive":
            return "memory_sensitive" if cohort == 1 else "insensitive"
        return ("cpu_sensitive", "memory_sensitive", "insensitive")[cohort]

    def response_multiplier(self, profile, cpu_ratio, ram_ratio):
        if cpu_ratio < 1.0 - 1e-12 or ram_ratio < 1.0 - 1e-12:
            raise SyntheticResourceModelError("candidate is below proportional baseline")
        if profile == "cpu_sensitive":
            fraction = np.clip((cpu_ratio - 1.0) / 0.5, 0.0, 1.0)
            return float(1.0 + 0.30 * fraction)
        if profile == "memory_sensitive":
            fraction = np.clip((ram_ratio - 1.0) / 0.5, 0.0, 1.0)
            return float(1.0 + 0.25 * fraction)
        return 1.0

    def candidates_for(self, job_name):
        profile = self.profile_for(job_name)
        candidates = []
        if self.mode == "s1" or profile == "insensitive":
            ratios = ((1.0, 1.0),)
        elif profile == "cpu_sensitive":
            ratios = tuple((ratio, 1.0) for ratio in self.ratios)
        else:
            ratios = tuple((1.0, ratio) for ratio in self.ratios)
        for cpu_ratio, ram_ratio in ratios:
            multiplier = self.response_multiplier(profile, cpu_ratio, ram_ratio)
            if multiplier < 1.0 - 1e-12:
                continue
            candidates.append({
                "id": f"cpu={cpu_ratio:g},ram={ram_ratio:g}",
                "cpu_ratio": cpu_ratio,
                "ram_ratio": ram_ratio,
                "multiplier": multiplier,
                "profile": profile,
            })
        return candidates

    def _normalize_allocations(self, allocations):
        normalized = []
        for job_name in sorted(allocations):
            cluster, node_ids = allocations[job_name]
            if cluster is None or not node_ids:
                continue
            if cluster not in self.capacities:
                raise SyntheticResourceModelError(f"unknown cluster in allocation: {cluster}")
            counts = Counter(int(node) for node in node_ids)
            for node, gpus in counts.items():
                capacity = self.capacities[cluster]
                if node < 0 or node >= int(capacity["nodes"]):
                    raise SyntheticResourceModelError(f"invalid node {cluster}/{node}")
                if gpus > int(capacity["gpus_per_node"]):
                    raise SyntheticResourceModelError(f"job exceeds GPU capacity on {cluster}/{node}")
            normalized.append((job_name, cluster, counts))
        return normalized

    def _baseline_choices(self, active):
        return {job_name: self.candidates_for(job_name)[0] for job_name, _, _ in active}

    def _solve_s2(self, active, jobs):
        variable_meta = []
        job_variables = defaultdict(list)
        for job_name, cluster, node_counts in active:
            base_goodput = float(jobs[job_name].estimate_current_goodput())
            if not np.isfinite(base_goodput) or base_goodput <= 0:
                raise SyntheticResourceModelError(f"invalid Sia goodput for {job_name}: {base_goodput}")
            for candidate in self.candidates_for(job_name):
                index = len(variable_meta)
                variable_meta.append((job_name, cluster, node_counts, base_goodput, candidate))
                job_variables[job_name].append(index)

        if not variable_meta:
            return {}, 0.0, None
        objective = np.asarray([
            -base_goodput * candidate["multiplier"]
            for _, _, _, base_goodput, candidate in variable_meta
        ], dtype=float)
        rows, lower, upper = [], [], []
        for job_name in sorted(job_variables):
            row = np.zeros(len(variable_meta), dtype=float)
            row[job_variables[job_name]] = 1.0
            rows.append(row)
            lower.append(1.0)
            upper.append(1.0)

        cpu_per_gpu = float(self.baseline["cpu_per_gpu"])
        ram_per_gpu = float(self.baseline["ram_gb_per_gpu"])
        for cluster, capacity in self.capacities.items():
            for node in range(int(capacity["nodes"])):
                cpu_row = np.zeros(len(variable_meta), dtype=float)
                ram_row = np.zeros(len(variable_meta), dtype=float)
                for index, (_, var_cluster, counts, _, candidate) in enumerate(variable_meta):
                    if var_cluster != cluster or node not in counts:
                        continue
                    gpu_count = counts[node]
                    cpu_row[index] = gpu_count * cpu_per_gpu * candidate["cpu_ratio"]
                    ram_row[index] = gpu_count * ram_per_gpu * candidate["ram_ratio"]
                rows.extend((cpu_row, ram_row))
                lower.extend((-np.inf, -np.inf))
                upper.extend((float(capacity["cpu_per_node"]),
                              float(capacity["ram_gb_per_node"])))

        start = time.perf_counter()
        result = milp(
            c=objective,
            integrality=np.ones(len(variable_meta), dtype=int),
            bounds=Bounds(np.zeros(len(variable_meta)), np.ones(len(variable_meta))),
            constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
            options={"time_limit": 30.0, "mip_rel_gap": 0.0},
        )
        solve_seconds = time.perf_counter() - start
        if not result.success or result.x is None:
            return self._baseline_choices(active), solve_seconds, f"MILP status={result.status}: {result.message}"
        choices = {}
        for job_name, indexes in job_variables.items():
            selected = max(indexes, key=lambda index: result.x[index])
            if result.x[selected] < 0.5:
                return self._baseline_choices(active), solve_seconds, f"no integral choice for {job_name}"
            choices[job_name] = variable_meta[selected][4]
        return choices, solve_seconds, None

    def _materialize(self, active, choices, jobs):
        result = self._empty_result()
        cpu_per_gpu = float(self.baseline["cpu_per_gpu"])
        ram_per_gpu = float(self.baseline["ram_gb_per_gpu"])
        for job_name, cluster, node_counts in active:
            candidate = choices[job_name]
            base_goodput = float(jobs[job_name].estimate_current_goodput())
            node_map = {}
            for node, gpu_count in sorted(node_counts.items()):
                cpu = gpu_count * cpu_per_gpu * candidate["cpu_ratio"]
                ram = gpu_count * ram_per_gpu * candidate["ram_ratio"]
                node_map[str(node)] = {"gpus": gpu_count, "cpu": cpu, "ram_gb": ram}
                usage = result["node_usage"][cluster][str(node)]
                usage["gpus"] += gpu_count
                usage["cpu"] += cpu
                usage["ram_gb"] += ram
            result["jobs"][job_name] = {
                "cluster": cluster,
                "profile": candidate["profile"],
                "choice": candidate["id"],
                "cpu_ratio": candidate["cpu_ratio"],
                "ram_ratio": candidate["ram_ratio"],
                "multiplier": candidate["multiplier"],
                "sia_goodput": base_goodput,
                "effective_goodput": base_goodput * candidate["multiplier"],
                "nodes": node_map,
            }
        result["capacity_violations"] = self.audit_capacity(result["node_usage"])
        if result["capacity_violations"]:
            raise SyntheticResourceModelError(
                f"capacity violation after allocation: {result['capacity_violations']}")
        return result

    def audit_capacity(self, node_usage):
        violations = []
        for cluster, nodes in node_usage.items():
            capacity = self.capacities[cluster]
            for node, usage in nodes.items():
                if usage["gpus"] > int(capacity["gpus_per_node"]):
                    violations.append(f"{cluster}/{node}: GPU {usage['gpus']}")
                if usage["cpu"] > float(capacity["cpu_per_node"]) + 1e-8:
                    violations.append(f"{cluster}/{node}: CPU {usage['cpu']}")
                if usage["ram_gb"] > float(capacity["ram_gb_per_node"]) + 1e-8:
                    violations.append(f"{cluster}/{node}: RAM {usage['ram_gb']}")
        return violations

    def allocate(self, allocations, jobs, current_time):
        active = self._normalize_allocations(allocations)
        if self.mode == "s1":
            choices, solve_seconds, fallback = self._baseline_choices(active), 0.0, None
        else:
            choices, solve_seconds, fallback = self._solve_s2(active, jobs)
            self.solve_times.append(solve_seconds)
            if fallback:
                self.fallback_count += 1
        result = self._materialize(active, choices, jobs)
        result["timestamp"] = current_time
        result["solve_seconds"] = solve_seconds
        result["fallback"] = fallback

        current_choices = {name: choice["id"] for name, choice in choices.items()}
        reallocations = sum(
            1 for name, choice in current_choices.items()
            if name in self.previous_choices and self.previous_choices[name] != choice
        )
        current_maps = {
            name: (job_result["cluster"], job_result["choice"],
                   tuple((node, values["gpus"], values["cpu"], values["ram_gb"])
                         for node, values in sorted(job_result["nodes"].items())))
            for name, job_result in result["jobs"].items()
        }
        map_reallocations = sum(
            1 for name, resource_map in current_maps.items()
            if name in self.previous_maps and self.previous_maps[name] != resource_map
        )
        self.previous_choices = current_choices
        self.previous_maps = current_maps
        self.total_aux_reallocations += reallocations
        self.total_map_reallocations += map_reallocations
        result["auxiliary_reallocations"] = reallocations
        result["map_reallocations"] = map_reallocations
        self.last_result = result
        return result

    def multiplier_for(self, job_name):
        return float(self.last_result.get("jobs", {}).get(job_name, {}).get("multiplier", 1.0))

    def metadata(self):
        solve_times = np.asarray(self.solve_times, dtype=float)
        return {
            "claim_boundary": self.config["claim_boundary"],
            "resource_model_type": self.config["resource_model_type"],
            "mode": self.mode,
            "scenario": self.scenario,
            "config_path": self.config_path,
            "auxiliary_reallocation_overhead_modelled": bool(
                self.config["auxiliary_reallocation_overhead_modelled"]),
            "auxiliary_reallocation_overhead_seconds": float(
                self.config["auxiliary_reallocation_overhead_seconds"]),
            "total_auxiliary_reallocations": self.total_aux_reallocations,
            "total_resource_map_reallocations": self.total_map_reallocations,
            "fallback_count": self.fallback_count,
            "solver_calls": int(solve_times.size),
            "solver_total_seconds": float(solve_times.sum()) if solve_times.size else 0.0,
            "solver_p99_seconds": float(np.percentile(solve_times, 99)) if solve_times.size else 0.0,
        }
