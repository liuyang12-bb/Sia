"""Read-only original-Sia, Stage1 and Stage1.1 three-round regression."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent / 'legacy_regression'
ROOT = OUT.parents[3]
HELPER = ROOT/'sia-joint-correctness-stage1p1/sia-simulator/check_stage1_legacy_regression.py'
OUT.mkdir()  # Must not reuse previous output.
results = {}
commands = {}
for name in ('official-artifacts','sia-joint-medium-frozen','sia-joint-correctness-stage1','sia-joint-correctness-stage1p1'):
    cmd = [sys.executable,'-B',str(HELPER),'--probe',str(ROOT/name/'sia-simulator')]
    start = time.perf_counter()
    r = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
    (OUT/(name+'.log')).write_bytes(r.stdout)
    commands[name] = dict(command=cmd,exit_code=r.returncode,wall_seconds=time.perf_counter()-start)
    (OUT/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
    if r.returncode:
        raise RuntimeError(f'bounded probe failed: {name}')
    results[name] = json.loads(next(x.removeprefix('PROBE_JSON=') for x in r.stdout.decode().splitlines() if x.startswith('PROBE_JSON=')))
exact = all(v==results['official-artifacts'] for v in results.values())
previous = json.loads((ROOT/'research_runs/sia_synergy/correctness_stage1/legacy_regression/comparison.json').read_text())
prior_match = results['sia-joint-correctness-stage1']==previous['states']['sia-joint-correctness-stage1']
(OUT/'comparison.json').write_text(json.dumps(dict(exact_state_match=exact,prior_stage1_match=prior_match,
    jobs=1,rounds=3,states=results),indent=2),encoding='utf-8')
assert exact and prior_match
print('PASS: official Sia, frozen, Stage1, Stage1.1 and preserved prior state: 3/3 rounds identical')
