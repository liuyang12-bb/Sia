"""Bounded closed-loop regression of the experimental search implementations."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
QUANTUM = 60
MAX_JOBS = 12
MAX_ROUNDS = 600


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=ROOT / "inputs/core80_workload.csv")
    parser.add_argument("--jobs", type=positive_int, default=6, help="number of source CSV rows to retain")
    parser.add_argument("--rounds", type=positive_int, default=10, help="hard limit even with --until-complete")
    parser.add_argument("--until-complete", action="store_true", help="fail if any selected method hits the round limit")
    parser.add_argument("--include-local", action="store_true", help="also exercise the approximate LOCAL strategy")
    parser.add_argument("--include-full", action="store_true", help="include the original exhaustive FULL strategy")
    parser.add_argument("--include-old-lazy", action="store_true", help="also compare lazy FO before the recovery precheck")
    parser.add_argument("--response", choices=("neutral", "medium"), default="medium")
    parser.add_argument("--scenario", choices=("dual", "cpu_only", "ram_only"), default="dual")
    parser.add_argument("--topology", default="aws:2,dgx-ext:2", metavar="TYPE:COUNT,...")
    parser.add_argument("--unseeded", action="store_true", help="exercise original cold-start behavior")
    parser.add_argument("--search-budget-seconds", type=float, default=5.0, help="per-decision local search budget")
    parser.add_argument("--output", type=Path, help="also write JSON to a new file")
    args = parser.parse_args(argv)
    if args.jobs > MAX_JOBS or args.rounds > MAX_ROUNDS:
        parser.error(f"validation is bounded to {MAX_JOBS} jobs and {MAX_ROUNDS} rounds")
    if not math.isfinite(args.search_budget_seconds) or args.search_budget_seconds <= 0:
        parser.error("search budget must be finite and positive")
    limits = {"aws": 4, "dgx-ext": 8, "rtx": 8}
    topology = {}
    try:
        for item in args.topology.split(","):
            name, count = item.split(":")
            gpus = int(count)
            if name in topology or gpus not in (1, 2, 4, 8) or gpus > limits[name]:
                raise ValueError(item)
            topology[name] = gpus
    except (ValueError, KeyError):
        parser.error("topology needs distinct supported types and legal powers-of-two GPU counts")
    args.topology = topology
    return args


def prepare_workload(source, path, limit):
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or not {"name", "time", "application", "num_replicas", "batch_size"} <= set(fields):
            raise ValueError("workload must use the Core80 CSV schema")
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) == limit:
                break
    names = [row["name"] for row in rows]
    if not names or len(names) != len(set(names)):
        raise ValueError("validation needs a nonempty workload with unique job identities")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return names


def encode(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported diagnostic value {type(value).__name__}")


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=encode, separators=(",", ":"))


def choice(config_id, placement, cpu, ram, rho):
    return (config_id, tuple(sorted(placement.items() if isinstance(placement, dict) else placement)), cpu, ram, rho)


def logged_choice(column):
    return choice(column["config"]["config_id"], column["placement"],
                  column["cpu_per_gpu"], column["ram_gb_per_gpu"], column["rho"])


def ledger_choices(services):
    output = {}
    for name, record in services.items():
        column = record["column"]
        output[name] = {
            **{key: value for key, value in record.items() if key != "column"},
            "column": choice(column.config.config_id, column.placement.node_gpus,
                             column.query.cpu_per_gpu, column.query.ram_gb_per_gpu, column.query.rho),
        }
    return output


def execution_view(round_record):
    events = {}
    for event in round_record["execution"]:
        name = event["job_id"]
        if not event["independent_execution_audit"]["valid"]:
            raise RuntimeError(f"execution audit failed for {name}")
        events[name] = {key: event[key] for key in (
            "progress_before", "progress_after", "training_seconds", "waiting_seconds", "checkpoint_delay_consumed",
            "checkpoint_remaining_before_decision", "epoch_before", "epoch_after", "rho", "training_segments",
            "completion_time",
        )}
    return events


def snapshot(cluster, round_record, fresh):
    live = cluster.policy.liveness
    state = {
        "time": cluster.current_time,
        "new_decision": fresh,
        "ready": tuple(live.ready),
        "services": ledger_choices(live.services),
        "history": ledger_choices(getattr(live, "_lifetime_history", {})),
        "rescale_events": round_record["rescale_events"],
        "migrate_events": round_record["migrate_events"],
        "job_checkpoint_remaining": {job.name: job.rescale_time for job in cluster.jobs},
        "execution": execution_view(round_record),
    }
    if not fresh:
        # Cluster skips optimization when all arrived jobs have finished, and the
        # log still points at its most recent decision in that case.
        return state
    decision = round_record["decision_for_next_interval"]
    result = decision["result"]
    audit = result["audit"]
    columns = result["selected"]
    selected = {name: logged_choice(column) for name, column in columns.items()}
    execution = {event["job_id"]: event for event in round_record["execution"]}
    required = set(audit["required_jobs"])
    commitments = set(audit["service_commitments"])
    locks = set(audit["checkpoint_locks"])
    if not audit["valid"] or commitments != set(selected) or not locks <= required <= commitments:
        raise RuntimeError("required, checkpoint, selected, and service-ledger sets are inconsistent")
    if set(live.services) != commitments or set(live.ready) & commitments:
        raise RuntimeError("committed services overlap the ready queue or differ from the decision audit")
    if audit["drain"] and set(selected) != required:
        raise RuntimeError("admitted optional jobs during head-of-line drain")
    usage = {}
    for name, column in columns.items():
        for node, gpus in column["placement"].items():
            total = usage.setdefault(node, [0, 0.0, 0.0])
            total[0] += gpus
            total[1] += gpus * column["cpu_per_gpu"]
            total[2] += gpus * column["ram_gb_per_gpu"]
        if name in locks:
            locked = execution[name]["resource_allocation"]
            if locked is None or choice(locked["config_id"], locked["placement"], locked["cpu_per_gpu"],
                                        locked["ram_gb_per_gpu"], locked["rho"]) != selected[name]:
                raise RuntimeError(f"checkpoint geometry or quota changed for {name}")
    for node, total in usage.items():
        gpu_type = node.rsplit(":", 1)[0]
        cap = cluster.policy.node_capacities[gpu_type]
        bounds = (cluster.ngpus_per_node[gpu_type], cap["cpu_cores"], cap["ram_gb"])
        if any(value > maximum + 1e-7 for value, maximum in zip(total, bounds)):
            raise RuntimeError(f"capacity exceeded on {node}: {total} > {bounds}")
    scored_objective = float(result["scheduling_objective"])
    if not math.isfinite(scored_objective):
        raise RuntimeError("nonfinite scheduling objective")
    offers = audit["recovery_offers"]
    committed = audit["tail_recoveries"]
    grants = audit["tail_grants"]
    if len(offers) != audit["recovery_offer_count"]:
        raise RuntimeError("recovery offer count differs from the pre-commit permit audit")
    for recovery in committed:
        name = recovery["job"]
        permit = offers.get(name)
        if (permit is None or recovery["column_id"] not in permit.get("column_ids", [permit["column_id"]])
                or not grants[name]["used"]):
            raise RuntimeError(f"recovery committed without a matching permit for {name}")
    return {**state,
        "selected": selected,
        "objective": scored_objective,
        "required": sorted(required),
        "optional_selected": sorted(audit["optional_selected"]),
        "drain": audit["drain"],
        "startup_budget": audit["service_budget"],
        "locks": sorted(locks),
        "recovery_offers": offers,
        "recovery_commits": sorted(committed, key=lambda entry: (entry["job"], entry["column_id"])),
        "recovery_grants": grants,
        "phase1_proposals": [
            {column["job"]: choice(column["config"], column["placement"], column["cpu"], column["ram"], None)
             for column in entry["proposal"]}
            for entry in audit.get("serial_proposals", ())
        ],
    }


def quantile(values, probability):
    position = (len(values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def sample_stats(values):
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    return {"samples": len(clean), "total": sum(clean), "median": statistics.median(clean),
            "p95": quantile(clean, 0.95), "p99": quantile(clean, 0.99), "max": clean[-1]}


def compare_exact(reference, candidate):
    if reference.keys() != candidate.keys():
        return {"field": "state_fields", "fo": sorted(reference), "lazy_fo": sorted(candidate)}
    for field, expected in reference.items():
        observed = candidate[field]
        if field == "objective":
            matched = math.isclose(expected, observed, abs_tol=1e-9, rel_tol=0.0)
        else:
            matched = expected == observed
        if not matched:
            result = {"field": field}
            if isinstance(expected, dict) and isinstance(observed, dict):
                for key in sorted(expected.keys() | observed.keys()):
                    if key not in expected or key not in observed or expected[key] != observed[key]:
                        result.update(key=key, fo=expected.get(key), lazy_fo=observed.get(key))
                        break
            else:
                result.update(fo=expected, lazy_fo=observed)
            return result
    return None


MEASURES = ("step_seconds", "select_seconds", "input_columns", "materialized_columns", "initial_columns",
            "solve_calls", "phase1_columns", "aux_columns", "initial_generation_seconds",
            "aux_generation_seconds", "decision_seconds", "recovery_precheck_seconds")


def new_report():
    return dict(
        {field: [] for field in MEASURES}, last_decision_round=None,
        drain_rounds=0, lock_job_rounds=0, non_proportional_job_rounds=0,
        status_counts=Counter(), generation_counts=Counter(), budget_exhausted_rounds=0,
        local_status_counts=Counter(), trajectory=hashlib.sha256(),
        recovery_guard_rounds=0, recovery_eager_rounds=0, recovery_certificate_skip_rounds=0,
        recovery_offered_rounds=0, recovery_committed_rounds=0,
        recovery_offer_count=0, recovery_commit_count=0,
        recovery_guard_by_scope=Counter(), recovery_eager_by_scope=Counter(),
        recovery_certified_by_scope=Counter(), recovery_status_counts=Counter(),
        recovery_offers_by_scope=Counter(), recovery_commits_by_scope=Counter(),
    )


def record_decision(method, cluster, decision, state, report):
    result = decision["result"]
    audit = result["audit"]
    offers = state["recovery_offers"]
    commits = state["recovery_commits"]
    report["recovery_offered_rounds"] += bool(offers)
    report["recovery_committed_rounds"] += bool(commits)
    report["recovery_offer_count"] += len(offers)
    report["recovery_commit_count"] += len(commits)
    report["recovery_offers_by_scope"].update(
        offer.get("recovery_scope", "UNSPECIFIED") for offer in offers.values())
    report["recovery_commits_by_scope"].update(
        commit.get("recovery_scope", "UNSPECIFIED") for commit in commits)
    search = audit.get("search")
    decision_seconds = decision.get("engineering_decision_seconds")
    if decision_seconds is None and search is not None:
        decision_seconds = search.get("decision_seconds")
    report["decision_seconds"].append(decision_seconds)
    if method in {"LAZY_FO", "LAZY_FO_LEGACY", "LOCAL"} and search is None:
        raise RuntimeError(f"{method} omitted its search diagnostics")
    if search:
        generation = search["generation"]
        certificate = search["recovery_certificate"]
        potential = search["recovery_guard_potential"]
        scope = search["recovery_scope"]
        eager = generation == "RECOVERY_EAGER"
        certified = certificate == "proved_no_offer"
        if (potential != (scope in {"solo", "multi"}) or (eager or certified) and not potential
                or certified and (eager or offers)):
            raise RuntimeError("inconsistent no-offer certificate, enumeration, or permit audit")
        report["generation_counts"][generation] += 1
        report["recovery_status_counts"][certificate] += 1
        report["recovery_guard_rounds"] += potential
        report["recovery_eager_rounds"] += eager
        report["recovery_certificate_skip_rounds"] += certified
        if potential:
            report["recovery_guard_by_scope"][scope] += 1
        if eager:
            report["recovery_eager_by_scope"][scope] += 1
        if certified:
            report["recovery_certified_by_scope"][scope] += 1
        report["budget_exhausted_rounds"] += bool(search["budget_exhausted"])
        local = search.get("local")
        status = (local or {}).get("local_status")
        if status is not None:
            report["local_status_counts"][status] += 1
        if method == "LOCAL" and status != "accepted" and audit["serial_proposals"]:
            base = audit["serial_proposals"][-1]["selected"]
            baseline = {
                (c["job"], c["config"], tuple(tuple(pair) for pair in c["placement"]), c["cpu"], c["ram"])
                for c in base
            }
            committed = {
                (c.job_id, c.config.config_id, c.placement.node_gpus,
                 c.query.cpu_per_gpu, c.query.ram_gb_per_gpu)
                for c in cluster.policy._last_committed_columns.values()
            }
            if baseline != committed:
                raise RuntimeError("LOCAL fallback changed the same-round FO bundle")
        for field in ("initial_columns", "phase1_columns", "aux_columns", "initial_generation_seconds",
                      "aux_generation_seconds", "recovery_precheck_seconds"):
            report[field].append(search.get(field))
        report["materialized_columns"].append(search["initial_columns"] + search["aux_columns"])
    else:
        report["materialized_columns"].append(decision.get("engineering_repair", {}).get("materialized_columns"))
    report["select_seconds"].append(result["solve_seconds"])
    report["input_columns"].append(audit.get("scale80_candidate_columns"))
    report["solve_calls"].append(len(audit["solver_trace"]))
    report["drain_rounds"] += state["drain"]
    report["lock_job_rounds"] += len(state["locks"])
    report["non_proportional_job_rounds"] += sum(
        any(not math.isclose(used, reference, rel_tol=0.0, abs_tol=1e-9) for used, reference in zip(
            (column.query.cpu_per_gpu, column.query.ram_gb_per_gpu),
            (column.proportional_cpu_per_gpu, column.proportional_ram_per_gpu),
        )) for column in cluster.policy._last_committed_columns.values()
    )
    report["status_counts"][result["status"]] += 1


def record_step(method, cluster, report):
    started = time.perf_counter()
    cluster.step(QUANTUM)
    report["step_seconds"].append(time.perf_counter() - started)
    current = cluster.joint_round_logs[-1]
    decision = current["decision_for_next_interval"]
    decision_round = decision.get("round")
    fresh = decision_round is not None and decision_round != report["last_decision_round"]
    state = snapshot(cluster, current, fresh)
    if fresh:
        report["last_decision_round"] = decision_round
        record_decision(method, cluster, decision, state, report)
    # The objective is compared with its own numeric tolerance above.
    physical = {field: value for field, value in state.items() if field != "objective"}
    report["trajectory"].update((canonical(physical) + "\n").encode("utf-8"))
    cluster.joint_round_logs.clear()
    cluster.logs.clear()
    cluster.rescale_events.clear()
    cluster.migrate_events.clear()
    for name in ("exchange_columns", "exchange_scores", "continuity_scores", "completion_scores",
                 "actual_completion_columns"):
        if hasattr(cluster.policy, name):
            setattr(cluster.policy, name, [])
    return state


def finalize_report(cluster, report):
    jcts = {job.name: job.completion_time - job.submission_time for job in cluster.jobs
            if job.completion_time is not None}
    measures = {field: sample_stats(report.pop(field)) for field in MEASURES}
    final_jcts = sorted(jcts.values()) if cluster.all_complete() else []
    return {**report, **measures, "trajectory": report["trajectory"].hexdigest(),
            "completed": len(jcts), "total_jobs": len(cluster.jobs), "simulated_seconds": cluster.current_time,
            "mean_jct_if_all_completed": statistics.mean(final_jcts) if final_jcts else None,
            "p95_jct_if_all_completed": quantile(final_jcts, 0.95) if final_jcts else None,
            "p99_jct_if_all_completed": quantile(final_jcts, 0.99) if final_jcts else None,
            "completed_jct_seconds": jcts}


def run(args):
    from core80 import build

    methods = ["FO", "LAZY_FO"]
    if args.include_old_lazy:
        methods.append("LAZY_FO_LEGACY")
    if args.include_local:
        methods.append("LOCAL")
    if args.include_full:
        methods.append("FULL")
    with tempfile.TemporaryDirectory(prefix="verify-cc-search-") as directory:
        path = Path(directory) / "workload.csv"
        jobs = prepare_workload(args.workload, path, args.jobs)
        workload_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        clusters = {
            method: build(
                "LAZY_FO" if method == "LAZY_FO_LEGACY" else method,
                response=args.response, scenario=args.scenario, workload_path=path,
                topology=args.topology, seed=not args.unseeded,
                search_budget_seconds=args.search_budget_seconds, recovery_precheck=method == "LAZY_FO",
            )
            for method in methods
        }
        reports = {method: new_report() for method in methods}
        exact_methods = [method for method in methods if method in {"FO", "LAZY_FO", "LAZY_FO_LEGACY"}]
        matched_rounds = 0
        matched_decisions = 0
        failure = None
        for step in range(1, args.rounds + 1):
            states = {}
            for method, cluster in clusters.items():
                if cluster.all_complete():
                    continue
                try:
                    states[method] = record_step(method, cluster, reports[method])
                except Exception as error:
                    failure = {"step": step, "method": method, "exception": repr(error)}
                    break
            if failure:
                break
            active_exact = [method for method in exact_methods if method in states]
            if active_exact and active_exact != exact_methods:
                failure = {"step": step, "field": "completion_state", "stepped_methods": list(states)}
                break
            if "FO" in states:
                for method in exact_methods[1:]:
                    mismatch = compare_exact(states["FO"], states[method])
                    if mismatch:
                        failure = {"step": step, "method": method, **mismatch}
                        break
                if failure:
                    break
                matched_rounds += 1
                matched_decisions += states["FO"]["new_decision"]
            if all(cluster.all_complete() for cluster in clusters.values()):
                break
        if not failure and args.until_complete:
            incomplete = [method for method, cluster in clusters.items() if not cluster.all_complete()]
            if incomplete:
                failure = {"field": "round_limit", "incomplete_methods": incomplete}
        if not failure and matched_decisions == 0:
            failure = {"field": "coverage", "reason": "no active decision in the bounded window"}
        return {
            "valid": failure is None,
            "scope": "bounded regression; LOCAL and FULL (if requested) run independently of the exact comparison",
            "materialized_columns_definition": "FO uses the factor certificate; search uses initial plus expanded auxiliary columns",
            "workload": str(args.workload.resolve()),
            "workload_slice_sha256": workload_digest,
            "job_ids": jobs,
            "scenario": args.scenario,
            "response": args.response,
            "topology": args.topology,
            "search_budget_seconds": args.search_budget_seconds,
            "round_limit": args.rounds,
            "require_completion": args.until_complete,
            "seeded": not args.unseeded,
            "exact_methods": exact_methods,
            "fo_lazy_fo_matched_rounds": matched_rounds,
            "fo_lazy_fo_matched_decisions": matched_decisions,
            "failure": failure,
            "methods": {method: finalize_report(clusters[method], reports[method]) for method in methods},
        }


def main(argv=None):
    args = get_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            report = run(args)
    except Exception as error:
        report = {"valid": False, "failure": {"exception": repr(error)}}
    payload = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False, default=encode)
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload + "\n")
        except OSError as error:
            print(json.dumps({"valid": False, "failure": {"output": str(args.output), "exception": repr(error)}}))
            return 1
    print(payload)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
