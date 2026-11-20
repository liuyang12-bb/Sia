"""Seal repair sources and verify protected bytes; no experiment execution."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SRC = ROOT / "sia-joint-medium-frozen"
WORK = ROOT / "sia-joint-correctness-stage1"

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest().upper()

def git(*args):
    return subprocess.check_output(["git", "-C", str(WORK), *args])

def function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return ast.dump(next(x for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name), include_attributes=False)

def main():
    target = OUT / "repair_manifest.json"
    if target.exists():
        raise RuntimeError("refuse to overwrite sealing record")
    original = json.loads((OUT / "snapshot_manifest.json").read_text(encoding="utf-8"))
    # Byte-for-byte proof for protected historical data and actual frozen code.
    altered = [p for p, h in original["protected_hashes"].items() if not (ROOT/p).is_file() or sha(ROOT/p) != h]
    altered_source = [p for p,h in original["source_hashes"].items() if sha(SRC/p) != h]
    if altered or altered_source:
        raise RuntimeError(f"protected bytes changed: {altered}, {altered_source}")
    stable = {}
    for relative, names in {
        "joint_prototype/joint_integration.py": ("_raw_candidates", "_b3_result", "_restart_factor"),
        "joint_prototype/joint_profile.py": ("query", "_bilinear"),
    }.items():
        for name in names:
            stable[f"{relative}::{name}"] = function(SRC/"sia-simulator"/relative, name) == function(WORK/"sia-simulator"/relative, name)
    assert all(stable.values())
    paths = [
        "job.py", "joint_prototype/joint_integration.py", "joint_prototype/joint_profile.py",
        "joint_prototype/joint_scheduler.py", "joint_prototype/synergy_adapter.py", "joint_prototype/execution_audit.py",
        "run_cpu_bottleneck_smoke.py", "tests/test_b3_failure_handling.py", "tests/test_b3_prop_ablation.py",
        "tests/test_joint_integration_coverage.py", "tests/test_joint_prototype.py", "tests/test_stage1_correctness.py",
        "preflight_stage1_response.py", "check_stage1_legacy_regression.py",
    ]
    paths = ["sia-simulator/" + p for p in paths]
    subprocess.run(["git", "-C", str(WORK), "add", "--", *paths], check=True)
    (OUT / "repair.patch").write_bytes(git("diff", "--cached", "--binary"))
    (OUT / "repair_diff_stat.txt").write_bytes(git("diff", "--cached", "--stat"))
    subprocess.run(["git", "-C", str(WORK), "commit", "-m", "fix: stage1 checkpoint integrity, fresh fallback, FIFO and execution accounting"], check=True)
    head = git("rev-parse", "HEAD").decode().strip()
    subprocess.run(["git", "-C", str(WORK), "bundle", "create", str(OUT / "repair.bundle"), "HEAD"], check=True)
    tracked = git("ls-files", "-z").decode().split("\0")[:-1]
    hashes = {p: sha(WORK/p) for p in tracked}
    with zipfile.ZipFile(OUT / "repair_source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in hashes:
            z.write(WORK/p, p)
    dependencies = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for package in ("numpy", "scipy", "cvxpy", "cylp", "pandas", "autograd"):
        dependencies[package] = importlib.metadata.version(package)
    dependencies["cbc_and_highs_version"] = "Not separately queried; linked SciPy/CyLP distributions pinned. No guessed engine version."
    dependencies["solver_options_unchanged"] = {"scipy_milp": {"time_limit": 30.0, "mip_rel_gap": 0.0}, "sia_solver_code": sha(WORK/"sia-simulator/sia.py")}
    (OUT / "dependencies.json").write_text(json.dumps(dependencies, indent=2), encoding="utf-8")
    record = {"status": "STAGE1_ENGINEERING_FIXES_ONLY_NOT_FORMAL_EXPERIMENT_READY", "baseline_commit": original["baseline_commit"],
              "repair_commit": head, "branch": git("branch", "--show-current").decode().strip(), "source_hashes": hashes,
              "protected_file_count": len(original["protected_hashes"]), "protected_original_source_count": len(original["source_hashes"]),
              "protected_byte_changes": altered, "frozen_source_byte_changes": altered_source, "unchanged_function_ast": stable,
              "changed_files": paths, "artifacts": {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != target.name},
              "limits": ["No 20-job/6000-round/cohort experiment", "No liveness/admission redesign", "No phi/clipping/capacity/workload modification", "No remote push"]}
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("repair_commit", "branch", "protected_file_count", "protected_original_source_count", "unchanged_function_ast")}))

if __name__ == "__main__":
    main()
