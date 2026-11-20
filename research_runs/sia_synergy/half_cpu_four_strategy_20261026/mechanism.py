"""Offline selected-column diagnostics; does not construct a simulator."""
import collections
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = json.loads((ROOT / "comparison_summary.json").read_text(encoding="utf-8"))
OUT = {}
for dataset, group in ANALYSIS["dataset_results"].items():
    OUT[dataset] = {}
    for method, info in group["metrics"].items():
        counts = collections.Counter()
        apps = collections.defaultdict(collections.Counter)
        last_round = None
        with gzip.open(Path(info["location"]) / "rounds.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                decision = record["decision_for_next_interval"]
                if last_round == decision["round"]:
                    continue
                last_round = decision["round"]
                selected = decision["result"]["selected"]
                counts["decisions"] += 1
                counts["nonempty_decisions"] += bool(selected)
                for job, column in selected.items():
                    app = job.rsplit("-", 1)[0]
                    rho = float(column["rho"])
                    cpu = float(column["cpu_per_gpu"])
                    ram = float(column["ram_gb_per_gpu"])
                    counts["selected"] += 1
                    apps[app]["selected"] += 1
                    for key, condition in (
                        ("nonproportional_cpu", abs(cpu - 1.5) > 1e-8),
                        ("nonproportional_ram", abs(ram - 62.5) > 1e-8),
                        ("rho_gt_1", rho > 1 + 1e-8),
                        ("rho_lt_1", rho < 1 - 1e-8),
                        ("rho_eq_1", abs(rho - 1) <= 1e-8),
                    ):
                        counts[key] += condition
                        apps[app][key] += condition
        OUT[dataset][method] = {"counts": dict(counts),
                                "by_application": {k: dict(v) for k, v in apps.items()}}

output = ROOT / "mechanism_summary.json"
assert not output.exists()
with output.open("x", encoding="utf-8") as stream:
    json.dump(OUT, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
for ds, group in OUT.items():
    for method, stats in group.items():
        print(ds, method, stats["counts"])
