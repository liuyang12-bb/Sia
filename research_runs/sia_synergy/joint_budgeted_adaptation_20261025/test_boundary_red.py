import subprocess,sys,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;WORK=OUT.parents[2]/'sia-joint-budgeted-adaptation/sia-simulator'
cmd=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_checkpoint_rounding_audit.py','-v']
with (OUT/'boundary_red_tests.log').open('x',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=WORK,stdout=f,stderr=subprocess.STDOUT)
with (OUT/'boundary_red_result.json').open('x') as f:json.dump(dict(exit_code=r.returncode,command=cmd),f)
print('BOUNDARY_RED_EXIT',r.returncode)
