"""Paired, process-isolated timings for the equivalent FO variants."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile

from verify_search import (
    MAX_JOBS, MAX_ROUNDS, QUANTUM, ROOT, canonical, encode, finalize_report, new_report,
    positive_int, prepare_workload, record_step, sample_stats,
)


METHODS = ("FO", "LAZY_OLD", "LAZY_FO")
TOPOLOGY = {"aws": 2, "dgx-ext": 2}
THREAD_ENV = {
    name: "1" for name in (
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
}
TIMED_FIELDS = ("step_seconds", "decision_seconds", "select_seconds")


def get_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=ROOT / "inputs/core80_workload.csv")
    parser.add_argument("--jobs", type=positive_int, default=6)
    parser.add_argument("--rounds", type=positive_int, default=48)
    parser.add_argument("--until-complete", action="store_true")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmup-decisions", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=22)
    parser.add_argument("--response", choices=("neutral", "medium"), default="medium")
    parser.add_argument("--scenario", choices=("dual", "cpu_only", "ram_only"), default="dual")
    parser.add_argument("--unseeded", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.jobs > MAX_JOBS or args.rounds > MAX_ROUNDS or args.repeats > 6:
        parser.error(f"limit the benchmark to {MAX_JOBS} jobs, {MAX_ROUNDS} rounds, and 6 repeats")
    if not 0 <= args.warmup_decisions < args.rounds:
        parser.error("warmup-decisions must be nonnegative and less than rounds")
    return args


def run_orders(repeats, seed):
    rng = random.Random(seed)
    for offset in range(0, repeats, len(METHODS)):
        order = rng.sample(METHODS, len(METHODS))
        for rotation in range(min(len(order), repeats - offset)):
            yield order[rotation:] + order[:rotation]


def worker(config):
    from core80 import build

    method = config["method"]
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        cluster = build(
            "FO" if method == "FO" else "LAZY_FO",
            recovery_precheck=method == "LAZY_FO",
            response=config["response"], scenario=config["scenario"],
            workload_path=Path(config["workload"]), topology=TOPOLOGY, seed=config["seeded"],
        )
        report = new_report()
        decisions = 0
        rounds = []
        for number in range(1, config["round_limit"] + 1):
            if cluster.all_complete():
                break
            state = record_step("FO" if method == "FO" else "LAZY_FO", cluster, report)
            active = state["new_decision"]
            decisions += active
            row = {
                "round": number,
                "measure": active and decisions > config["warmup_decisions"],
                "state_sha256": hashlib.sha256(canonical({
                    key: value for key, value in state.items() if key != "objective"
                }).encode("utf-8")).hexdigest(),
                "objective": state.get("objective"),
                "step_seconds": report["step_seconds"][-1],
            }
            if active:
                row.update({field: report[field][-1] for field in (
                    "decision_seconds", "select_seconds", "materialized_columns",
                )})
                search = cluster.policy.last_decision["result"]["audit"].get("search", {})
                row.update(
                    recovery_guard=search.get("recovery_guard_potential"),
                    recovery_certificate=search.get("recovery_certificate"),
                    recovery_scope=search.get("recovery_scope"),
                    recovery_precheck_seconds=search.get("recovery_precheck_seconds"),
                    generation=search.get("generation"),
                    initial_generation_seconds=search.get("initial_generation_seconds"),
                    aux_generation_seconds=search.get("aux_generation_seconds"),
                    recovery_offers=len(state["recovery_offers"]),
                    recovery_commits=len(state["recovery_commits"]),
                )
            rounds.append(row)
        if config["require_completion"] and not cluster.all_complete():
            raise RuntimeError(f"{method} did not finish all jobs by the round limit")
        totals = finalize_report(cluster, report)
    timed = [row for row in rounds if row["measure"]]
    if not timed:
        raise RuntimeError(f"{method} has no post-warmup decision samples")
    if any(row["decision_seconds"] is None for row in timed):
        raise RuntimeError(f"{method} is missing comparable decision timing")
    return {
        "totals": totals,
        "post_warmup": {field: sample_stats(row[field] for row in timed) for field in TIMED_FIELDS},
        "rounds": rounds,
    }


def verify_match(reference, candidate, label):
    if len(reference["rounds"]) != len(candidate["rounds"]):
        raise RuntimeError(f"{label} submitted a different number of rounds")
    for expected, observed in zip(reference["rounds"], candidate["rounds"]):
        if expected["state_sha256"] != observed["state_sha256"] or expected["measure"] != observed["measure"]:
            raise RuntimeError(f"{label} changed the physical state at round {expected['round']}")
        left, right = expected["objective"], observed["objective"]
        if ((left is None) != (right is None)
                or left is not None and not math.isclose(left, right, abs_tol=1e-9, rel_tol=0.0)):
            raise RuntimeError(f"{label} changed the objective at round {expected['round']}")
    if (reference["totals"]["completed_jct_seconds"] != candidate["totals"]["completed_jct_seconds"]
            or reference["totals"]["trajectory"] != candidate["totals"]["trajectory"]):
        raise RuntimeError(f"{label} changed the end-to-end simulation")


def run_isolated(config, environment):
    command = [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), "--worker"]
    result = subprocess.run(command, input=json.dumps(config), capture_output=True, text=True,
                            encoding="utf-8", env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{config['method']} failed:\n{result.stderr[-4000:]}\n{result.stdout[-1000:]}")
    return json.loads(result.stdout)


def compare_timings(repeats):
    comparisons = {}
    for baseline in ("FO", "LAZY_OLD"):
        for field in TIMED_FIELDS:
            by_run = []
            for repeat in repeats:
                timing = repeat["methods"]
                before = timing[baseline]["post_warmup"][field]["p95"]
                after = timing["LAZY_FO"]["post_warmup"][field]["p95"]
                by_run.append({"baseline_seconds": before, "optimized_seconds": after,
                               "reduction_fraction": (before - after) / before if before > 0 else None})
            changes = [pair["reduction_fraction"] for pair in by_run if pair["reduction_fraction"] is not None]
            comparisons[f"LAZY_FO_vs_{baseline}_{field}_p95"] = {
                "paired": by_run,
                "median_reduction_fraction": statistics.median(changes) if changes else None,
                "improved_repeats": sum(change > 0 for change in changes),
            }
    return comparisons


def run(args):
    environment = {**os.environ, **THREAD_ENV, "PYTHONHASHSEED": "0"}
    with tempfile.TemporaryDirectory(prefix="bench-cc-search-") as folder:
        workload = Path(folder) / "workload.csv"
        names = prepare_workload(args.workload, workload, args.jobs)
        workload_sha256 = hashlib.sha256(workload.read_bytes()).hexdigest()
        common = {
            "workload": str(workload), "response": args.response, "scenario": args.scenario,
            "seeded": not args.unseeded, "round_limit": args.rounds,
            "require_completion": args.until_complete, "warmup_decisions": args.warmup_decisions,
        }
        repeats = []
        reference = None
        for index, order in enumerate(run_orders(args.repeats, args.order_seed), 1):
            methods = {}
            for method in order:
                print(f"Repeat {index}/{args.repeats} · {method}", file=sys.stderr, flush=True)
                result = run_isolated({**common, "method": method}, environment)
                if reference is None:
                    reference = result
                else:
                    verify_match(reference, result, f"repeat {index} {method}")
                methods[method] = result
            repeats.append({"repeat": index, "order": order, "methods": methods})
    return {
        **common, "valid": True, "workload": str(args.workload.resolve()),
        "workload_slice_sha256": workload_sha256, "job_ids": names, "topology": TOPOLOGY,
        "quantum_seconds": QUANTUM, "order_seed": args.order_seed, "thread_environment": THREAD_ENV,
        "python": sys.version, "platform": platform.platform(),
        "timing_scope": {
            "step_seconds": "whole simulator step",
            "decision_seconds": "common optimize_policy wall timing",
            "select_seconds": "reported selection wall timing",
            "post_warmup": "active decisions after the configured number of fresh decisions",
            "isolated": "fresh worker per method and repeat, exactly one worker at a time",
        },
        "repeats": repeats, "p95_comparisons": compare_timings(repeats),
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False, default=encode))
        return 0
    args = get_args(argv)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            try:
                result = run(args)
            except Exception as error:
                result = {"valid": False, "failure": str(error)}
            json.dump(result, stream, indent=2, allow_nan=False, default=encode)
            stream.write("\n")
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 1
    if not result["valid"]:
        print(result["failure"], file=sys.stderr)
        return 1
    print(f"Validated {args.repeats} repeats with equivalent trajectories. Results in {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
