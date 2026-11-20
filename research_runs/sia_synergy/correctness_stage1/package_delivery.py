"""Package immutable inputs and final hash index without touching old artifacts."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
WORK = ROOT / "sia-joint-correctness-stage1"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest().upper()

def main():
    if (OUT / "delivery_manifest.json").exists():
        raise RuntimeError("refuse to overwrite delivery")
    template_paths = sorted((ROOT / "research_sources/Synergy/simulator/models").glob("*_1.py"))
    with zipfile.ZipFile(OUT / "external_response_templates.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in template_paths:
            z.write(path, str(path.relative_to(ROOT)))
    history = ROOT / "research_runs/sia_synergy/b3_w10_late_ablation_2x2/run2"
    inputs = [history / name for name in ("frozen_config.json", "frozen_manifest.json", "audit_manifest.json", "workload.csv", "preregistration.json")]
    inputs.append(ROOT / "research_runs/sia_synergy/b3_multiworkload_medium/response_model_manifest_isolated.json")
    with zipfile.ZipFile(OUT / "historical_inputs_readonly_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in inputs:
            z.write(path, str(path.relative_to(ROOT)))
    original = json.loads((OUT / "snapshot_manifest.json").read_text(encoding="utf-8"))
    repair = json.loads((OUT / "repair_manifest.json").read_text(encoding="utf-8"))
    archive_checks = {}
    for archive, expected in [("source_snapshot.zip", original["source_hashes"]), ("repair_source_snapshot.zip", repair["source_hashes"])]:
        with zipfile.ZipFile(OUT / archive) as z:
            found = {name: hashlib.sha256(z.read(name)).hexdigest().upper() for name in z.namelist()}
        archive_checks[archive] = found == expected
        assert archive_checks[archive]
    assert all(sha(WORK/path) == h for path,h in repair["source_hashes"].items())
    checks = {"archive_byte_hashes": archive_checks,
              "protected_source_unchanged": all(sha(ROOT/"sia-joint-medium-frozen"/p)==h for p,h in original["source_hashes"].items()),
              "protected_old_artifacts_unchanged": all(sha(ROOT/p)==h for p,h in original["protected_hashes"].items())}
    assert checks["protected_source_unchanged"] and checks["protected_old_artifacts_unchanged"]
    for bundle in ("base.bundle", "repair.bundle"):
        r = subprocess.run(["git", "-C", str(WORK), "bundle", "verify", str(OUT/bundle)], capture_output=True)
        (OUT/(bundle+".verify.log")).write_bytes(r.stdout+r.stderr)
        assert r.returncode == 0
    reports = [ROOT/"research_notes/synergy"/name for name in ("B3_CORRECTNESS_STAGE1_REPORT.md", "B3_RESPONSE_DATA_PREFLIGHT.md")]
    test_log = (OUT/"stage1_final_tests.log").read_text(encoding="utf-8")
    count = re.search(r"Ran (\d+) tests in ([0-9.]+)s", test_log)
    payload = {"status": "ENGINEERING_STAGE1_COMPLETE_RESEARCH_SEMANTICS_PENDING", "repair_commit": repair["repair_commit"],
               "baseline_commit": repair["baseline_commit"], "checks": checks,
               "tests": {"count": int(count[1]), "unittest_seconds": float(count[2]), "passed": "\nOK\n" in test_log, "existing": 47, "new": 14},
               "historical_inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
               "reports": {str(p.relative_to(ROOT)): sha(p) for p in reports},
               "artifact_hashes": {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file()},
               "not_run": ["w10_late 20 jobs", "6000 rounds", "9 cohorts", "formal experiments"],
               "not_fixed": ["B3-PROP empty-optimum liveness", "2x2 objective/admission confounding", "online phi dependence and clipping", "unseeded Sia-intent/candidate boundary observed in interim fixture"],
               "next_research_decisions_only": ["liveness mechanism", "GPU comparison definition", "response boundary model"]}
    (OUT/"delivery_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k:payload[k] for k in ("status", "repair_commit", "checks", "tests")}))

if __name__ == "__main__":
    main()
