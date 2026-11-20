"""Offline, deterministic dataset design. Does not import Cluster/Job or run training.

Run once with a new output directory. R0 T0 is an offline diagnostic reference,
not the online Sia Goodput model and never a simulated completion prediction.
"""
import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(r"D:\科研启航\Sia")
TREE = ROOT / "sia-joint-service-horizon"
SIM = TREE / "sia-simulator"
sys.path.insert(0, str(SIM))
from joint_prototype.joint_profile import GpuConfig, JobSpec, JointProfileOracle

CONFIG = ROOT / "research_runs/sia_synergy/sia_r0_full160_comparison/inputs/config.json"
REGISTRY = CONFIG.with_name("registry.json")
TRACE = ROOT / "official-artifacts/sia-simulator/workloads/saturn/workload-1.csv"
MODELS = ROOT / "research_sources/Synergy/simulator/models"
WARNING = "SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main(out):
    if out.exists():
        raise FileExistsError("Refusing to reuse an existing output directory")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    reg = {r["key"]: r for r in registry}
    response = config["response"]
    mapping = response["application_template_rule"]
    cpu_grid = response["aux_cpu_points"]
    ram_grid = response["aux_ram_points"]
    oracle = JointProfileOracle(MODELS, "fixed_reference", response_exponent=0.5)

    def query(app, gpu, cpu, ram, count=1, exponent=0.5):
        entry = reg[f"{app}/{gpu}:1:{count}"]
        cfg = GpuConfig(entry["config_id"], gpu, count, 1, entry["T0"], entry["phi_c"], entry["phi_m"])
        job = JobSpec(app, mapping[app], (cfg,))
        engine = oracle if exponent == 0.5 else JointProfileOracle(MODELS, "fixed_reference", response_exponent=exponent)
        return asdict(engine.query(job, cfg, cpu, ram, (3.0, 62.5)))

    # All decisions below are explicit structural rules, not searches over JCT.
    scenarios = {
        "D1_auxiliary_exchange": {
            "kind": "static_exact_witness_only", "jobs": ["bert-28", "cifar10-0"],
            "nodes": {"aws": {"gpus": 2, "cpu": 8.0, "ram": 125.0}},
            "fixed_gpus_each": 1,
            "rule": "same served set/GPU/placement; proportional 4+4 CPU vs donor2+receiver6; RAM62.5 each",
        },
        "D2_gpu_auxiliary_coupling": {
            "kind": "static_exact_witness_only", "jobs": ["cifar10-0", "deepspeech2-8"],
            "nodes": {"aws": {"gpus": 1, "cpu": 6.0, "ram": 125.0}, "dgx-ext": {"gpus": 1, "cpu": 3.0, "ram": 125.0}},
            "fixed_gpus_each": 1,
            "rule": "two heterogeneous nodes; slow CIFAR GPU has twice CPU; exhaust both one-GPU assignments and existing auxiliary grid",
        },
        "D3_queued_training_proposal": {
            "kind": "future_training_not_executed", "jobs": ["cifar10-0", "cifar10-1", "cifar10-2", "deepspeech2-8", "deepspeech2-9", "deepspeech2-12"],
            "nodes": {"aws": {"gpus": 2, "cpu": 12.0, "ram": 250.0}, "dgx-ext": {"gpus": 2, "cpu": 6.0, "ram": 250.0}},
            "rule": "first three official rows of each of two preregistered applications; all arrive0; no workload amount/batch edits; all legal 1/2GPU candidates retained",
            "controls": {
                "neutral": "identical nodes/jobs; alpha0",
                "cpu_symmetric": "identical jobs/GPU/RAM; awsCPU12 and dgxCPU12; alpha0.5; removes CPU/GPU ratio mismatch, not all CPU contention",
            },
            "future_execution_contract": {
                "interval_seconds": 60, "max_rounds": 600, "initial_profiles": [1, 8],
                "policies": ["SIA-P", "SIA-AUX", "G112"],
                "no_tuning_no_extension": True, "online_gate_prefix_rounds": 60,
                "gate_is_not_policy_win": True,
            },
        },
    }
    # D1: fully in-domain witness, same GPU and RAM for both jobs.
    d1 = {}
    for label, cpus in [("proportional", [4.0, 4.0]), ("exchange", [2.0, 6.0])]:
        d1[label] = [query(a, "aws", c, 62.5) for a, c in zip(["bert", "cifar10"], cpus)]
        assert sum(p["cpu_per_gpu"] for p in d1[label]) == 8
        assert sum(p["ram_gb_per_gpu"] for p in d1[label]) == 125
        assert not any(p["source_coordinate_clipped"] or p["baseline_coordinate_clipped"] for p in d1[label])
    d1["receiver_gain_fraction"] = d1["exchange"][1]["rho"] / d1["proportional"][1]["rho"] - 1
    d1["donor_loss_fraction"] = 1 - d1["exchange"][0]["rho"] / d1["proportional"][0]["rho"]
    d1["simple_receiver_payback_seconds_if_extra_delay250"] = 250 * (1 + d1["receiver_gain_fraction"]) / d1["receiver_gain_fraction"]

    # D2 is an independent finite enumeration, NOT a replay of Sia MIP.
    apps = ["cifar10", "deepspeech2"]
    nodes = scenarios["D2_gpu_auxiliary_coupling"]["nodes"]
    norm = {a: min(reg[f"{a}/{g}:1:1"]["T0"] for g in nodes) for a in apps}

    def score(points, raw=False):
        vals = [p["raw_sia_goodput" if raw else "effective_goodput"] / norm[a] for a, p in zip(apps, points)]
        return {"normalized_rate_sum": sum(vals), "negative_power_utility_p_minus_half": -sum(v ** -0.5 for v in vals), "normalized_rates": vals}

    assignments = []
    for types in itertools.permutations(nodes):
        prop = [query(a, g, nodes[g]["cpu"], nodes[g]["ram"]) for a, g in zip(apps, types)]
        choices = []
        for a, g in zip(apps, types):
            n = nodes[g]
            pts = [query(a, g, c, m) for c in cpu_grid for m in ram_grid if c <= n["cpu"] and m <= n["ram"]]
            choices.append(pts)
        combinations = list(itertools.product(*choices))
        best = max(combinations, key=lambda ps: score(ps)["negative_power_utility_p_minus_half"])
        assignments.append({"gpu_assignment": dict(zip(apps, types)), "proportional": prop, "raw_score": score(prop, True), "proportional_score": score(prop), "aux_optimum": best, "aux_score": score(best), "enumerated_count": len(combinations)})
    a = max(assignments, key=lambda v: v["raw_score"]["negative_power_utility_p_minus_half"])
    c = max(assignments, key=lambda v: v["aux_score"]["negative_power_utility_p_minus_half"])
    d2 = {"scope": "offline R0 T0 proxy only; same two served jobs, all one-GPU assignments, no locks/delays; NOT official Sia's actual decision or online Goodput", "normalizers": norm, "all_assignments": assignments,
          "A_raw_GPU_then_proportional": a["proportional"], "B_same_GPU_aux_optimum": a["aux_optimum"], "C_joint_aux_optimum": c["aux_optimum"],
          "A_score": a["proportional_score"], "B_score": a["aux_score"], "C_score": c["aux_score"]}
    d2["C_vs_B_normalized_rate_gain"] = c["aux_score"]["normalized_rate_sum"] / a["aux_score"]["normalized_rate_sum"] - 1
    d2["per_job_C_vs_B"] = {app: qc["effective_goodput"] / qb["effective_goodput"] - 1 for app, qc, qb in zip(apps, c["aux_optimum"], a["aux_optimum"])}
    assert not any(p["source_coordinate_clipped"] or p["baseline_coordinate_clipped"] for p in [*a["aux_optimum"], *c["aux_optimum"]])
    assert a["gpu_assignment"] != c["gpu_assignment"]
    # Negative control: without auxiliary response, both exhaustive optima agree.
    neutral_assignments = [[query(app, g, nodes[g]["cpu"], 125, exponent=0) for app, g in zip(apps, ts)] for ts in itertools.permutations(nodes)]
    assert max(score(ps)["negative_power_utility_p_minus_half"] for ps in neutral_assignments) == a["raw_score"]["negative_power_utility_p_minus_half"]

    # Full registry denominator audit, plus full D3 CPU/RAM point coverage.
    denominator = []
    for entry in registry:
        app = entry["key"].split("/")[0]
        gpu, _, count = entry["config_id"].split(":")
        p = query(app, gpu, 3, 62.5, int(count))
        denominator.append({"key": entry["key"], "cpu_requested": p["requested_baseline_source_cpu_coordinate"], "ram_requested": p["requested_baseline_source_ram_coordinate"], "clipped": p["baseline_coordinate_clipped"]})
    domain = []
    for app in apps:
        for gpu, node in scenarios["D3_queued_training_proposal"]["nodes"].items():
            for count in [1, 2]:
                pts = [query(app, gpu, cp, mem, count) for cp in cpu_grid for mem in ram_grid if count * cp <= node["cpu"] and count * mem <= node["ram"]]
                prop = query(app, gpu, node["cpu"] / node["gpus"], node["ram"] / node["gpus"], count)
                domain.append({"app": app, "config": f"{gpu}:1:{count}", "all_node_feasible_points": pts, "point_count": len(pts), "numerator_clipped": sum(p["source_coordinate_clipped"] for p in pts), "denominator_clipped": sum(p["baseline_coordinate_clipped"] for p in pts), "proportional": prop})

    # Data extract: no fabricated runtime, no batch/workload edits.
    with TRACE.open(encoding="utf-8", newline="") as f:
        source = list(csv.DictReader(f))
    indexed = {r["name"]: (i, r) for i, r in enumerate(source)}
    out.mkdir(parents=True)
    for name, spec in scenarios.items():
        rows = []
        for job in spec["jobs"]:
            i, original = indexed[job]
            row = dict(original, time="0.0", original_arrival=original["time"], source_row_index=i)
            assert all(row[k] == original[k] for k in ["name", "application", "num_replicas", "batch_size"])
            rows.append(row)
        rows.sort(key=lambda r: r["source_row_index"])
        path = out / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        spec["workload_sha256"] = sha(path)
        spec["source_rows_zero_based"] = [r["source_row_index"] for r in rows]
    output = {"warning": WARNING, "training_executed": False, "response": response, "scenarios": scenarios, "D1": d1, "D2": d2, "registry_denominator_audit": denominator, "D3_full_domain": domain,
              "memory_response_example": {"same_config_cpu6_ram62_5": query("cifar10", "aws", 6, 62.5), "same_config_cpu6_ram125": query("cifar10", "aws", 6, 125)},
              "tests": {"source_fields_unchanged": True, "D1_capacity_and_no_clipping": True, "D2_exhaustive_complete": True, "D2_selected_no_clipping": True, "neutral_control_exact": True}}
    (out / "offline_evidence.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    paths = [CONFIG, REGISTRY, TRACE, Path(__file__).resolve(), SIM / "joint_prototype/joint_profile.py"] + [MODELS / f for f in mapping.values()]
    manifest = {"status": "OFFLINE_DESIGN_ONLY_NOT_EXECUTED", "source_paths_and_sha256": {str(p): sha(p) for p in paths}, "output_hashes": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                "tree": str(TREE), "head": subprocess.check_output(["git", "-C", str(TREE), "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "-C", str(TREE), "status", "--short"], text=True),
                "warning_head_not_full_source": "The working source includes uncommitted/untracked changes. Hashes and the prior run4 full source_snapshot.zip identify actual code; no simulation was executed here.",
                "argv": sys.argv, "warning": WARNING}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"D1_receiver_gain": d1["receiver_gain_fraction"], "D2_normalized_rate_gain": d2["C_vs_B_normalized_rate_gain"], "D2_individual": d2["per_job_C_vs_B"], "D3_coverage": [{k:v for k,v in d.items() if k not in {"all_node_feasible_points", "proportional"}} for d in domain], "training_executed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output.resolve())
