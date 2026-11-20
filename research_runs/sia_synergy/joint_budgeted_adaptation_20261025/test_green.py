import subprocess,sys,json,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;WORK=OUT.parents[2]/'sia-joint-budgeted-adaptation/sia-simulator'
cmd=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'];tick=time.perf_counter()
with (OUT/'green_tests.log').open('x',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=WORK,stdout=f,stderr=subprocess.STDOUT)
with (OUT/'green_result.json').open('x') as f:json.dump(dict(command=cmd,exit_code=r.returncode,wall_seconds=time.perf_counter()-tick),f,indent=2)
print('GREEN_EXIT',r.returncode)
