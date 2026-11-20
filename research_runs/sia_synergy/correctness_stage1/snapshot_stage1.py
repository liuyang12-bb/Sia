"""One-shot archival restore into a NEW worktree; never writes protected sources."""
import hashlib
import json
import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "sia-joint-medium-frozen"
DST = ROOT / "sia-joint-correctness-stage1"
BRANCH = "exp/b3-correctness-stage1"

def git(*args):
    return subprocess.check_output(["git", "-C", str(SRC), *args])

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()

def main():
    if DST.exists() or (OUT / "source_snapshot.zip").exists():
        raise RuntimeError("refuse to reuse snapshot or destination")
    tracked = git("ls-files", "-z").decode().split("\0")[:-1]
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")[:-1]
    sources = [p for p in untracked if "__pycache__" not in p and not p.endswith(".pyc")]
    names = sorted(set(tracked + sources))
    hashes = {p: digest(SRC / p) for p in names if (SRC / p).is_file()}
    manifest = json.loads((ROOT / "research_runs/sia_synergy/b3_w10_late_ablation_2x2/run2/frozen_manifest.json").read_text(encoding="utf-8"))
    checks = {}
    for p, expected in manifest["required_source_hashes"].items():
        key = p if p.startswith("sia-simulator/") else "sia-simulator/" + p
        checks[p] = {"expected": expected, "actual": hashes.get(key), "match": hashes.get(key) == expected}
    # Execution code must match. Post-run analysis scripts may have separate audit hashes.
    core = ("job.py", "run_cpu_bottleneck_smoke.py", "joint_prototype/joint_integration.py", "joint_prototype/joint_scheduler.py", "joint_prototype/joint_profile.py")
    for p in core:
        match = next((v["match"] for k, v in checks.items() if k.endswith(p)), None)
        if match is not True:
            raise RuntimeError(f"frozen execution source mismatch: {p}")
    (OUT / "tracked.patch").write_bytes(git("diff", "--binary", "HEAD"))
    (OUT / "index.patch").write_bytes(git("diff", "--cached", "--binary"))
    (OUT / "git_status_before.txt").write_bytes(git("status", "--short", "--branch"))
    subprocess.run(["git", "-C", str(SRC), "bundle", "create", str(OUT / "base.bundle"), "HEAD"], check=True)
    with zipfile.ZipFile(OUT / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in hashes:
            z.write(SRC / p, p)
    protected = {}
    for folder in [ROOT / "research_notes", ROOT / "research_runs/sia_synergy/b3_w10_late_ablation_2x2", ROOT / "official-artifacts", ROOT / "research_sources/Synergy"]:
        for path in folder.rglob("*"):
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                protected[str(path.relative_to(ROOT))] = digest(path)
    subprocess.run(["git", "-C", str(SRC), "worktree", "add", "-b", BRANCH, str(DST), "HEAD"], check=True)
    with zipfile.ZipFile(OUT / "source_snapshot.zip") as z:
        # All names originate in git ls-files, validated again before restoration.
        for name in z.namelist():
            if not (DST / name).resolve().is_relative_to(DST.resolve()):
                raise RuntimeError(name)
        z.extractall(DST)
    restored = {p: digest(DST / p) for p in hashes}
    assert restored == hashes
    changed = git("diff", "--name-only", "HEAD").decode().splitlines()
    subprocess.run(["git", "-C", str(DST), "add", "--", *changed, *sources], check=True)
    subprocess.run(["git", "-C", str(DST), "commit", "-m", "baseline: archive actual frozen 2x2 execution and analysis sources"], check=True)
    baseline = subprocess.check_output(["git", "-C", str(DST), "rev-parse", "HEAD"]).decode().strip()
    (OUT / "requirements.lock.txt").write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"]))
    record = {"source": str(SRC), "destination": str(DST), "branch": BRANCH, "original_head": git("rev-parse", "HEAD").decode().strip(), "baseline_commit": baseline, "source_hashes": hashes, "necessary_untracked": sources, "excluded_generated_files": sorted(set(untracked)-set(sources)), "frozen_manifest_checks": checks, "protected_hashes": protected, "python": sys.version, "restored_byte_identical": True}
    (OUT / "snapshot_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_commit": baseline, "files": len(hashes), "untracked_source_files": sources, "restored_byte_identical": True, "protected_files": len(protected)}))

if __name__ == "__main__":
    main()
