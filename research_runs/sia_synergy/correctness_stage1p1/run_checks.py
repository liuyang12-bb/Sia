"""Capture bounded tests; refuse to overwrite any evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2] / 'sia-joint-correctness-stage1p1/sia-simulator'
label = sys.argv[1]
args = sys.argv[2:] or ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v']
log = OUT / (label+'.log')
if log.exists():
    raise RuntimeError('refuse to overwrite test evidence')
start = time.perf_counter()
with log.open('wb') as f:
    result = subprocess.run([sys.executable, *args], cwd=WORK, stdout=f, stderr=subprocess.STDOUT,
        env={**os.environ, 'PYTHONIOENCODING':'utf-8', 'PYTHONDONTWRITEBYTECODE':'1',
             'STAGE1_EVIDENCE_DIR':str(OUT/(label+'_evidence'))})
summary = dict(command=[sys.executable,*args], cwd=str(WORK), exit_code=result.returncode,
               wall_seconds=time.perf_counter()-start, log=str(log))
(OUT/(label+'.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print(json.dumps(summary))
print(log.read_text(encoding='utf-8',errors='replace')[-3500:])
sys.exit(result.returncode)
