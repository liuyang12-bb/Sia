"""Run the pre-registered eight-configuration 20-job synthetic suite serially."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


SCENARIOS = ("neutral", "cpu_sensitive", "memory_sensitive", "mixed")


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--minimum-available-gib", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true",
                        help="keep successful manifest entries and run only missing/skipped configurations")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    simulator = Path(__file__).resolve().parent
    root = workspace / "research_runs" / "sia_synergy" / "synthetic"
    workload = root / "inputs" / "saturn_workload_20.csv"
    resource_config = simulator / "synthetic_resource_model.json"
    run_dir = root / "runs"
    log_dir = root / "run_logs"
    summary_dir = root / "summaries"
    for directory in (run_dir, log_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "run_manifest.json"
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("resume_events", []).append({"started_at": now_iso()})
        manifest.pop("finished_at", None)
        manifest.pop("status", None)
    else:
        manifest = {
            "claim_boundary": "SYNTHETIC RESOURCE MODEL — NOT REAL SATURN PERFORMANCE",
            "started_at": now_iso(),
            "python": sys.executable,
            "workload": str(workload),
            "resource_config": str(resource_config),
            "minimum_available_gib": args.minimum_available_gib,
            "runs": [],
        }
    environment = os.environ.copy()
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"

    for scenario in SCENARIOS:
        for mode in ("s1", "s2"):
            name = f"{scenario}_{mode}"
            if any(run.get("name") == name and run.get("status") == "SUCCESS"
                   for run in manifest["runs"]):
                continue
            replaced = [run for run in manifest["runs"]
                        if run.get("name") == name]
            if replaced:
                manifest.setdefault("gate_events", []).extend(replaced)
            manifest["runs"] = [run for run in manifest["runs"]
                                if run.get("name") != name]
            available_gib = psutil.virtual_memory().available / 2**30
            record = {
                "name": name,
                "scenario": scenario,
                "mode": mode,
                "available_gib_before": available_gib,
                "started_at": now_iso(),
            }
            if available_gib < args.minimum_available_gib:
                record.update({
                    "status": "SKIPPED_LOW_MEMORY",
                    "exit_code": None,
                    "reason": (f"available memory {available_gib:.3f} GiB is below "
                               f"{args.minimum_available_gib:.3f} GiB"),
                })
                manifest["runs"].append(record)
                manifest["finished_at"] = now_iso()
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(record["reason"], file=sys.stderr)
                return 2

            output = run_dir / f"{name}.jsonl"
            summary = summary_dir / f"{name}.json"
            console = log_dir / f"{name}.stdout.log"
            command = [
                sys.executable, str(simulator / "multi_simulator.py"), str(workload),
                "--policy=sia", "--policy_p_val=-0.5",
                "--mip_lambda_a=0.01", "--mip_lambda_n=1.1",
                "--interval=60", "--project_throughputs", "--share_max_replicas",
                f"--synthetic_mode={mode}", f"--synthetic_scenario={scenario}",
                f"--synthetic_resource_config={resource_config}",
                f"--output={output}", f"--synthetic_summary={summary}",
            ]
            record.update({
                "command": command,
                "output": str(output),
                "summary": str(summary),
                "console_log": str(console),
            })
            start = time.perf_counter()
            with console.open("w", encoding="utf-8", errors="replace") as stream:
                process = subprocess.run(
                    command, cwd=simulator, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, check=False)
            record.update({
                "exit_code": process.returncode,
                "duration_seconds": time.perf_counter() - start,
                "finished_at": now_iso(),
                "available_gib_after": psutil.virtual_memory().available / 2**30,
                "status": "SUCCESS" if process.returncode == 0 else "FAILED",
            })
            manifest["runs"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"{name}: exit={process.returncode}, duration={record['duration_seconds']:.3f}s")
            if process.returncode != 0:
                manifest["finished_at"] = now_iso()
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                return process.returncode

    manifest["finished_at"] = now_iso()
    manifest["status"] = "SUCCESS"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
