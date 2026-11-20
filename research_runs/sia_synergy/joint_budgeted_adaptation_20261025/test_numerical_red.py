import subprocess,sys,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;WORK=OUT.parents[2]/'sia-joint-budgeted-adaptation/sia-simulator'
cmd=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_bundle_solver_numerics.py','-v']
with (OUT/'numerical_red_tests.log').open('x',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=WORK,stdout=f,stderr=subprocess.STDOUT)
with (OUT/'numerical_red_result.json').open('x') as f:json.dump(dict(exit_code=r.returncode,command=cmd),f)
print('NUMERICAL_RED_EXIT',r.returncode)
