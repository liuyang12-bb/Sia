"""Static data gate: no Job/Cluster construction or step, no solver, no training.

Structural states are fixed before evaluation. Every queried candidate (including
out-of-support and node-infeasible candidates) is retained in the evidence file.
Historical completed traces are read, never modified or replayed.
"""
import csv
import gzip
import hashlib
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / "sia-joint-completion-time/sia-simulator"
PRIOR = OUT.parent / "joint_exchange_realization_20261026_run1"
RUNS = OUT.parent / "joint_opportunity_experiments_20261026_run1"
sys.path.insert(0, str(SOURCE))
from joint_prototype.joint_profile import GpuConfig, JobSpec, JointProfileOracle

WARNING = "SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE"
INPUTS = OUT / "inputs"
REG = {r["key"]: r for r in json.loads((INPUTS / "registry.json").read_text())}
CONFIG = json.loads((INPUTS / "D3_queued_training_proposal_config.json").read_text())
CPU_GRID = sorted(set(CONFIG["response"]["aux_cpu_points"]) | {8.0})
RAM_GRID = CONFIG["response"]["aux_ram_points"]
ORACLE = JointProfileOracle(INPUTS / "models", "fixed_reference", response_exponent=.5)
APPS = ("cifar10", "deepspeech2")
SCENARIOS = {
    "D2_existing": {"aws": dict(gpus=1, cpu=6., ram=125.),
                    "dgx-ext": dict(gpus=1, cpu=3., ram=125.)},
    "D3_existing_cpu_symmetric": {"aws": dict(gpus=2, cpu=12., ram=250.),
                    "dgx-ext": dict(gpus=2, cpu=12., ram=250.)},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def save(name, data):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def query(app, config_id, cpu, ram, raw=1., job=None):
    row = REG[f"{app}/{config_id}"]
    gpu, nn, ng = config_id.split(":")
    conf = GpuConfig(config_id, gpu, int(ng), int(nn), raw, row["phi_c"], row["phi_m"])
    spec = JobSpec(job or app, row["template"], (conf,))
    q = asdict(ORACLE.query(spec, conf, cpu, ram, (3., 62.5)))
    q["boundary"] = {
        "numerator_cpu": not 1. <= q["requested_source_cpu_coordinate"] <= 24.,
        "numerator_ram": not 20.83 <= q["requested_source_ram_coordinate"] <= 250.,
        "denominator_cpu": not 1. <= q["requested_baseline_source_cpu_coordinate"] <= 24.,
        "denominator_ram": not 20.83 <= q["requested_baseline_source_ram_coordinate"] <= 250.,
    }
    q["in_support"] = not any(q["boundary"].values())
    return q


def full_coverage():
    output = []
    for scenario, nodes in SCENARIOS.items():
        for app, gpu in itertools.product(APPS, nodes):
            node = nodes[gpu]
            for g in range(1, node["gpus"] + 1):
                config = f"{gpu}:1:{g}"
                points = []
                for cpu, ram in itertools.product(CPU_GRID, RAM_GRID):
                    q = query(app, config, cpu, ram)
                    q["node_feasible"] = g * cpu <= node["cpu"] and g * ram <= node["ram"]
                    q["node_capacity_reasons"] = (["CPU"] if g*cpu > node["cpu"] else []) + (["RAM"] if g*ram > node["ram"] else [])
                    points.append(q)
                feasible = [p for p in points if p["node_feasible"]]
                prop = query(app, config, node["cpu"]/node["gpus"], node["ram"]/node["gpus"])
                output.append(dict(scenario=scenario, app=app, config=config,
                    registry=REG[f"{app}/{config}"], node=node, proportional=prop,
                    all_points=points, counts=dict(total=len(points), node_feasible=len(feasible),
                        node_feasible_clean=sum(p["in_support"] for p in feasible),
                        node_infeasible=len(points)-len(feasible),
                        **{k: sum(p["boundary"][k] for p in feasible) for k in prop["boundary"]})))
    return output


def first_decision(path, condition):
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            row = json.loads(line)
            dec = row["decision_for_next_interval"]
            if dec["round"] in seen:
                continue
            seen.add(dec["round"])
            if condition(dec):
                return dict(path=str(path), gzip_line=index, input_sha256=sha(path),
                            interval_start=row["interval_start"], interval_end=row["interval_end"],
                            decision=dec, execution=row["execution"], job_end_state=row["job_end_state"])
    raise RuntimeError(f"STRUCTURAL_STATE_MISSING: {path}")


def coupling_enum(state):
    """Enumerate the same two served jobs; no admission, cold-profile, or JCT claims."""
    dec = state["decision"]
    specs = dec["probe"]["specs"]
    nodes = SCENARIOS["D2_existing"]
    assignments = []
    for types in itertools.permutations(nodes):
        opts = []
        for spec, gpu in zip(specs, types):
            app = spec["job_id"].rsplit("-", 1)[0]
            config = next(c for c in spec["configs"] if c["config_id"] == f"{gpu}:1:1")
            opts.append([query(app, config["config_id"], cpu, ram, config["sia_goodput"], spec["job_id"])
                         for cpu, ram in itertools.product(CPU_GRID, RAM_GRID)
                         if cpu <= nodes[gpu]["cpu"] and ram <= nodes[gpu]["ram"]])
        pairs = list(itertools.product(*opts))
        clean = [p for p in pairs if all(q["in_support"] for q in p)]
        def score(ps):
            return sum(q["effective_goodput"] for q in ps)
        def fair(ps):
            return -sum(q["effective_goodput"]**-.5 for q in ps)
        # Both choices reported; the comparison is an offline diagnostic, not a new scheduler.
        best = max(clean, key=score) if clean else None
        best_fair = max(clean, key=fair) if clean else None
        assignments.append(dict(gpu_assignment=dict(zip([s["job_id"] for s in specs], types)),
            all_combinations=len(pairs), clean_combinations=len(clean),
            all_candidate_points=opts, best_clean_rate=best,
            best_clean_rate_sum=score(best) if best else None,
            best_clean_fair=best_fair,
            best_clean_fair_score=fair(best_fair) if best_fair else None))
    official_types = {j: c["config"]["gpu_type"] for j,c in dec["result"]["selected"].items()}
    fixed = next(a for a in assignments if a["gpu_assignment"] == official_types)
    joint = max(assignments, key=lambda a:a["best_clean_rate_sum"])
    fairness_joint = max(assignments, key=lambda a:a["best_clean_fair_score"])
    return dict(state=state, scope="same served set, logged online candidate rates; no resumed simulation; rates normalized within each job, not physical cross-application throughput",
        all_assignments=assignments, official_gpu_intent=official_types,
        joint_rate_assignment=joint["gpu_assignment"], fair_joint_assignment=fairness_joint["gpu_assignment"],
        fixed_gpu_clean_score=fixed["best_clean_rate_sum"], joint_clean_score=joint["best_clean_rate_sum"],
        relative_normalized_rate_gain=joint["best_clean_rate_sum"]/fixed["best_clean_rate_sum"]-1,
        coupling_opportunity=joint["gpu_assignment"] != official_types and joint["best_clean_rate_sum"] > fixed["best_clean_rate_sum"]+1e-9,
        cost_scope="Zero switching cost upper bound. A failed opportunity under zero cost cannot be restored by positive switching cost.")


def common_warm_exchange():
    path = PRIOR / "training_probe_result.json"
    prior = json.loads(path.read_text())
    proof = []
    for job, cpu in [("cifar10-1",8.), ("deepspeech2-9",4.)]:
        app = job.rsplit("-",1)[0]
        baseline_seg = prior["results"]["proportional"][job]["steps"][0]["segments"][0]
        exchanged_seg = prior["results"]["exchange"][job]["steps"][0]["segments"][0]
        raw = baseline_seg["raw_rate_before_rho"]
        assert math.isclose(raw, exchanged_seg["raw_rate_before_rho"], rel_tol=1e-12)
        a = query(app,"aws:1:1",6.,125.,raw,job)
        b = query(app,"aws:1:1",cpu,125.,raw,job)
        assert a["in_support"] and b["in_support"]
        assert math.isclose(a["effective_goodput"],baseline_seg["effective_rate"],rel_tol=1e-12)
        assert math.isclose(b["effective_goodput"],exchanged_seg["effective_rate"],rel_tol=1e-12)
        proof.append(dict(job=job, baseline=a, exchange=b,
            predicted_relative_rate_change=b["effective_goodput"]/a["effective_goodput"]-1,
            unchanged_raw_rate=raw, shared_warm_state=prior["common_warm_state"][job],
            observed_600s_progress_change=prior["observed_progress_gain"][job],
            historical_training_seconds=prior["results"]["exchange"][job]["training_seconds"]))
    return dict(prior_file=str(path), prior_sha256=sha(path), proof=proof,
        equal_job_relative_rate_change=sum(p["predicted_relative_rate_change"] for p in proof)/2,
        evidence="Independent replay of response arithmetic from logged direct raw rates; prior training only, no new training performed",
        scope="Fixed AWS1 GPU and target batch each; RAM125 each; same node2GPU12CPU250GB; CPU6+6 versus8+4. Relative changes averaged only as diagnostic index; NOT sum of unlike progress units and NOT JCT gain.",
        no_new_restart=True, no_clipping=True,
        persistence="Historical 600-second productive-window witness already exists; epoch-dependent rates mean the first-segment ratio is not the whole-window ratio.")


def main():
    selection = dict(training_executed=False, static_only=True,
        scenario_rules="Existing D2 and existing D3 CPU-symmetric reused without capacity/application/template search; all input workload fields unchanged",
        states=["D2 SIA-P initial round", "D2 SIA-P first round where every candidate is not unprofiled fallback and both jobs remain active", "Previously preregistered common-warm-state AWS6+6 versus8+4"],
        future_small_cases=dict(D2=SCENARIOS["D2_existing"], D3=SCENARIOS["D3_existing_cpu_symmetric"]),
        note="This file freezes data selection only; training observation windows/strategy preregistration are a separate gate owned by the main task.")
    save("offline_selection_rules.json",selection)
    coverage = full_coverage()
    trace = RUNS / "cells/D2_gpu_auxiliary_coupling_medium_sia_p/rounds.jsonl.gz"
    initial = first_decision(trace, lambda d:d["round"]==0)
    warm = first_decision(trace, lambda d:len(d["probe"]["specs"])==2 and all(
        p["goodput_source"]!="unprofiled_sia_fallback" for ps in d["probe"]["provenance"].values() for p in ps.values()))
    coupling = {"cold":coupling_enum(initial),"first_warm":coupling_enum(warm)}
    common = common_warm_exchange()
    proxy = json.loads((INPUTS / "offline_evidence.json").read_text())["D2"]
    paths = [Path(__file__), INPUTS/"registry.json", INPUTS/"D2_gpu_auxiliary_coupling.csv", INPUTS/"D3_queued_training_proposal.csv", INPUTS/"D3_queued_training_proposal_config.json", PRIOR/"same_state_diagnostic.json", PRIOR/"training_probe_result.json", SOURCE/"joint_prototype/joint_profile.py", SOURCE/"joint_prototype/r0_liveness.py"] + list((INPUTS/"models").glob("*.py"))
    gaps = [dict(scenario=r["scenario"],app=r["app"],config=r["config"],counts=r["counts"], proportional_clean=r["proportional"]["in_support"]) for r in coverage if r["counts"]["node_feasible_clean"] < r["counts"]["node_feasible"]]
    summary=dict(warning=WARNING,training_executed=False,selection=selection,
        response=dict(formula="rho=[M(cpu/phi_c,ram/phi_m)/M(3/phi_c,62.5/phi_m)]^0.5",interpolation="Existing bilinear E-legacy with clip; boundary counts do not filter runtime candidates",cpu_grid=CPU_GRID,ram_grid=RAM_GRID,source_scope="Official Synergy matrix cells; application transfer, R0 reference, phi and multiplication are synthetic assumptions"),
        full_coverage=coverage,coverage_gaps=gaps,common_warm_exchange=common,coupling=coupling,
        r0_proxy_coupling=dict(scope=proxy["scope"],rate_gain=proxy["C_vs_B_normalized_rate_gain"],per_job=proxy["per_job_C_vs_B"],not_online_proof=True),
        gates=dict(auxiliary_exchange_clean_supported=common["equal_job_relative_rate_change"]>0,
            actual_online_warm_gpu_coupling_supported=coupling["first_warm"]["coupling_opportunity"],
            clean_full_configuration_domain=not gaps,
            no_jct_guarantee=True,
            recommendation="Auxiliary exchange and objective validation may proceed as bounded diagnostic tests. Existing D2 warm online state does not establish the stronger joint GPU-coupling opportunity; do not claim a new performance evaluation passes all data gates."),
        provenance_sha256={str(p):sha(p) for p in paths})
    save("offline_gate.json",summary)
    with (OUT/"coverage_table.csv").open("x",encoding="utf-8",newline="") as stream:
        fields=["scenario","app","config","node_feasible","node_feasible_clean","numerator_cpu","numerator_ram","denominator_cpu","denominator_ram","proportional_clean"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in coverage:writer.writerow({"scenario":row["scenario"],"app":row["app"],"config":row["config"],**{k:row["counts"][k] for k in fields[3:-1]},"proportional_clean":row["proportional"]["in_support"]})
    print(json.dumps({"gates":summary["gates"],"coverage":[{k:r[k] for k in ["scenario","app","config","counts"]} for r in coverage],"common_relative_rate_gain":common["equal_job_relative_rate_change"],"warm_coupling":{k:v for k,v in coupling["first_warm"].items() if k not in ["state","all_assignments"]}},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
