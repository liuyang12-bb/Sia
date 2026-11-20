"""Read-only response-data audit: no Job/Cluster import, no simulation/fit.

Historical selected columns are NOT treated as complete candidate profiles.
Missing phi is reported, never imputed. Outputs go to a fresh supplied folder.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from joint_prototype.joint_profile import CPU_COORDS, RAM_COORDS, load_synergy_template, JointProfileOracle, JobSpec, GpuConfig


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def stream_rounds(path):
    """Bound memory by a single round, including the historical 303 MB file."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as f:
        buf = ""
        marker = '"joint_round_logs"'
        while marker not in buf:
            chunk = f.read(65536)
            if not chunk:
                raise ValueError(f"missing round array in {path}")
            buf = buf[-len(marker):] + chunk
        buf = buf[buf.index(marker)+len(marker):]
        while "[" not in buf:
            buf += f.read(65536)
        buf = buf[buf.index("[")+1:]
        while True:
            buf = buf.lstrip(" \r\n\t,")
            if buf.startswith("]"):
                return
            try:
                obj, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = f.read(65536)
                if not chunk:
                    raise
                buf += chunk
                continue
            yield obj
            buf = buf[end:]


def flags(cpu, ram, phi_c, phi_m):
    points = {"numerator_cpu": cpu/phi_c, "numerator_ram": ram/phi_m,
              "denominator_cpu": 3/phi_c, "denominator_ram": 62.5/phi_m}
    return {name: bool(value < (CPU_COORDS[0] if name.endswith("cpu") else RAM_COORDS[0])
                   or value > (CPU_COORDS[-1] if name.endswith("cpu") else RAM_COORDS[-1]))
            for name, value in points.items()}


def scan_policy(path, config, oracle):
    counts = collections.Counter()
    config_flags = collections.defaultdict(set)
    known = collections.defaultdict(set)
    enumerated = set()
    states = {}
    fixed_point_rhos = collections.defaultdict(set)
    details = []
    for record in stream_rounds(path):
        counts["rounds"] += 1
        d = record["decision_for_next_interval"]
        for job, item in d.get("candidate_summary", {}).items():
            enumerated.update((job, cid) for cid in item["config_ids"])
        for job, r in d.get("resources", {}).items():
            provenance = r["goodput_provenance"]
            pc, pm = provenance["cpu_demand_scale"], provenance["memory_demand_scale"]
            cid = r["config_id"]
            key = (job, cid)
            known[key].add((pc, pm))
            counts["selected_column_observations"] += 1
            clipped = flags(r["cpu_per_gpu"], r["ram_gb_per_gpu"], pc, pm)
            for name, clipped_here in clipped.items():
                counts[name] += clipped_here
                if clipped_here:
                    config_flags[name].add(key)
            counts["numerator_any"] += clipped["numerator_cpu"] or clipped["numerator_ram"]
            counts["denominator_any"] += clipped["denominator_cpu"] or clipped["denominator_ram"]
            counts["either_any"] += any(clipped.values())
            fixed_point_rhos[(job, cid, r["cpu_per_gpu"], r["ram_gb_per_gpu"])].add(round(r["rho"], 12))
            gpu_type = r["gpu_type"]
            cap = config["cluster"]["capacity_scenarios"]["cpu_constrained"][gpu_type]
            ng = config["cluster"]["ngpus_per_node"][gpu_type]
            # Counterfactual proportional query on an OBSERVED phi state only.
            prop_cpu = max(cap["cpu_cores"]/ng, config["response"]["minimum_legal_profile_point"]["cpu_per_gpu"])
            prop_ram = max(cap["ram_gb"]/ng, config["response"]["minimum_legal_profile_point"]["ram_gb_per_gpu"])
            strict_flags = flags(prop_cpu, prop_ram, pc, pm)
            state_key = (job, cid, pc, pm)
            states[state_key] = {"job": job, "config": cid, "phi_c": pc, "phi_m": pm,
                                 "proportional_cpu": prop_cpu, "proportional_ram": prop_ram,
                                 "proportional_clipping": strict_flags,
                                 "strict_proportional_survives": not any(strict_flags.values())}
            details.append({"round": d.get("round"), "time": record["interval_end"], "job": job, "config": cid,
                            "phi_c": pc, "phi_m": pm, "cpu": r["cpu_per_gpu"], "ram": r["ram_gb_per_gpu"],
                            "rho": r["rho"], **clipped})
    by_job = collections.defaultdict(lambda: {"observed_states": 0, "strict_survivors": 0, "lost_config_ids": set(), "surviving_config_ids": set()})
    for state in states.values():
        x = by_job[state["job"]]
        x["observed_states"] += 1
        x["strict_survivors"] += state["strict_proportional_survives"]
        x["surviving_config_ids" if state["strict_proportional_survives"] else "lost_config_ids"].add(state["config"])
    for val in by_job.values():
        for key in ("lost_config_ids", "surviving_config_ids"):
            val[key] = sorted(val[key])
    grid_counts = collections.Counter()
    grid_flagged_configs = collections.defaultdict(set)
    for state in states.values():
        gpu_type, nn, ng = state["config"].split(":")
        cap = config["cluster"]["capacity_scenarios"]["cpu_constrained"][gpu_type]
        # Unit Sia goodput is used ONLY to cancel the common factor in the
        # existing proportional floor predicate, not as a profile or simulation.
        cfg = GpuConfig(state["config"], gpu_type, int(ng), int(nn), 1.0, state["phi_c"], state["phi_m"])
        job = JobSpec(state["job"], config["response"]["template_map"][state["job"]], (cfg,))
        prop = (state["proportional_cpu"], state["proportional_ram"])
        pq = oracle.query(job, cfg, *prop, prop)
        for cpu in sorted(set(config["response"]["aux_cpu_points"]+[prop[0]])):
            for ram in sorted(set(config["response"]["aux_ram_points"]+[prop[1]])):
                grid_counts["raw_state_quota_points"] += 1
                if cpu*int(ng) > cap["cpu_cores"] or ram*int(ng) > cap["ram_gb"]:
                    grid_counts["capacity_rejected"] += 1
                    continue
                grid_counts["capacity_feasible"] += 1
                q = oracle.query(job, cfg, cpu, ram, prop)
                if q.rho + 1e-9 < pq.rho:
                    grid_counts["proportional_progress_floor_rejected"] += 1
                    continue
                grid_counts["eligible_pre_pareto_candidates"] += 1
                fs = flags(cpu, ram, cfg.cpu_demand_scale, cfg.memory_demand_scale)
                grid_counts["strict_clipping_rejected"] += any(fs.values())
                for key, bad in fs.items():
                    grid_counts[key] += bad
                    if bad:
                        grid_flagged_configs[key].add((state["job"], state["config"]))
    # Full candidate-level conclusions cannot be reconstructed from selected-only provenance.
    return {"input": str(path), "sha256": sha(path), "counts_selected_observations": dict(counts),
            "config_level": {"distinct_job_config_with_phi": len(known), "enumerated_job_configs": len(enumerated),
                             "flagged_job_config_counts": {name: len(keys) for name, keys in config_flags.items()},
                             "enumerated_configs_without_any_logged_phi": [list(k) for k in sorted(enumerated-set(known))]},
            "path_dependence": {"job_configs_with_multiple_phi": sum(len(v)>1 for v in known.values()),
                                "identical_job_config_quota_with_multiple_rho": sum(len(v)>1 for v in fixed_point_rhos.values())},
            "strict_proportional_on_observed_phi_states": {"total": len(states), "lost": sum(not s["strict_proportional_survives"] for s in states.values()),
                                                         "by_job": dict(by_job),
                                                         "jobs_without_survivors_IN_OBSERVED_SUBSET_ONLY": sorted(k for k,v in by_job.items() if v["strict_survivors"]==0)},
            "all_candidate_phi_coverage": "UNAVAILABLE: candidate_summary records IDs, not every candidate's raw goodput/phi; no imputation",
            "aux_candidate_grid_ON_OBSERVED_PHI_STATES_ONLY": {
                "counts": dict(grid_counts), "flagged_job_config_counts": {k: len(v) for k,v in grid_flagged_configs.items()},
                "counting_unit": "distinct (job, GPU-config, observed phi_c/phi_m, CPU, RAM), capacity + proportional-floor eligible, before Pareto pruning"},
            "states": list(states.values()), "selected_observations": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refuse to overwrite response preflight")
    args.output.mkdir(parents=True)
    root = args.root.resolve()
    run = root / "research_runs/sia_synergy/b3_w10_late_ablation_2x2/run2"
    config_path = run / "frozen_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sim = root / "sia-joint-medium-frozen/sia-simulator"
    templates = {}
    for file in sorted((root / "research_sources/Synergy/simulator/models").glob("*_1.py")):
        try:
            t = load_synergy_template(file)
            templates[file.name] = {"sha256": sha(file), "shape": list(t.matrix.shape), "min": float(t.matrix.min()),
                                    "max": float(t.matrix.max()), "cpu_demand": t.cpu_demand, "ram_demand": t.memory_demand_gb}
        except ValueError as e:
            templates[file.name] = {"sha256": sha(file), "not_supported": str(e)}
    coverage = []
    apps = sorted(config["response"]["application_template_rule"])
    for app in apps:
        for gpu_type, ng in config["cluster"]["ngpus_per_node"].items():
            files = []
            observed_pairs = set()
            for kind in ("placements", "scalability"):
                path = sim / "traces" / app / f"{kind}-{gpu_type}.csv"
                if not path.exists():
                    files.append({"path": str(path), "missing": True})
                    continue
                with path.open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                pairs = set()
                seeded = 0
                bszs = []
                for row in rows:
                    if kind == "placements":
                        nn, g = len(row["placement"]), sum(int(x) for x in row["placement"])
                    else:
                        nn, g = int(row["num_nodes"]), int(row["num_replicas"])
                    pairs.add((nn, g))
                    seeded += nn <= 1 and g <= 8
                    bszs.append(float(row["local_bsz"]))
                observed_pairs |= pairs
                files.append({"path": str(path), "sha256": sha(path), "rows": len(rows), "seed_filter_rows": seeded,
                              "node_gpu_pairs": sorted(pairs), "batch_range": [min(bszs), max(bszs)] if bszs else None})
            gpu_configs = [1, 2, 4] if ng == 4 else [1, 2, 4, 8]
            coverage.append({"application": app, "gpu_type": gpu_type, "files": files,
                             "topology_config_count": len(gpu_configs),
                             "measured_geometry_present": [g for g in gpu_configs if (1,g) in observed_pairs],
                             "measured_geometry_missing": [g for g in gpu_configs if (1,g) not in observed_pairs],
                             "frozen_phi_available": False})
    policies = {}
    oracle = JointProfileOracle(root / "research_sources/Synergy/simulator/models", "fixed_reference", response_exponent=0.5)
    for policy in ("b1", "b4", "b3_prop", "b3"):
        data = scan_policy(run / "policies" / policy / "result.json", config, oracle)
        (args.output / f"{policy}_coverage.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        policies[policy] = {k:v for k,v in data.items() if k not in ("states", "selected_observations")}
        print(policy, policies[policy]["counts_selected_observations"], flush=True)
    manifest = root / "research_runs/sia_synergy/b3_multiworkload_medium/response_model_manifest_isolated.json"
    result = {"scope": "READ_ONLY_EXISTING_DATA_NO_TRAINING_SIMULATION", "config_hash": sha(config_path), "response_manifest_hash": sha(manifest),
              "source_head": "be03f95c7bc641cbef0e2a4452bc4c7a3b5404e4+archived_dirty_sources",
              "cpu_axis": CPU_COORDS.tolist(), "ram_axis": RAM_COORDS.tolist(), "templates": templates,
              "sia_source_coverage": coverage, "historical_selected_columns": policies,
              "symbolic_strict_proportional_boundary": "All frozen nodes yield CPU=1 and RAM=125 per GPU. For actual online phi_c in [1,4], numerator CPU is in-grid iff phi_c=1. Denominator CPU is in-grid iff phi_c<=3. phi_m in [1,2] gives no proportional RAM clipping. This is a bound, not imputation of phi=1.",
              "remaining_missing": ["static policy-independent phi table", "pre-registered batch/gradient/reference state for converting throughput to Goodput", "all-candidate raw goodput and phi per historical round", "real cross-hardware CPU/RAM calibration"],
              "no_global_job_loss_claim": "Cannot identify jobs losing ALL feasible configs without all-candidate phi. Per-job loss in observed subset is not proof of global infeasibility."}
    (args.output / "response_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("response preflight complete; no scheduler or Job.step was invoked")


if __name__ == "__main__":
    main()
