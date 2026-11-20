"""Capture bounded tests only. Not a simulation experiment runner."""
import json
import os
import pathlib
import subprocess
import sys
import time

OUT = pathlib.Path(__file__).resolve().parent
ROOT = OUT.parents[2]
WORK = ROOT / "sia-joint-correctness-stage1/sia-simulator"
label = sys.argv[1]
args = sys.argv[2:] or ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = OUT / (label + ".log")
if log.exists():
    raise RuntimeError("refusing to overwrite test evidence")
start = time.perf_counter()
with log.open("wb") as f:
    result = subprocess.run([sys.executable, *args], cwd=WORK, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                                 "STAGE1_EVIDENCE_DIR": str(OUT / (label + "_evidence"))})
summary = {"command": [sys.executable, *args], "cwd": str(WORK), "exit_code": result.returncode, "wall_seconds": time.perf_counter()-start, "log": str(log)}
(OUT / (label + ".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary))
print(log.read_text(encoding="utf-8", errors="replace")[-2000:])
sys.exit(result.returncode)
