import subprocess,sys,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;WORK=OUT.parents[2]/'sia-joint-budgeted-adaptation/sia-simulator'
cmd=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_bundle_adaptation.py','-v']
with (OUT/'red_tests.log').open('x',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=WORK,stdout=f,stderr=subprocess.STDOUT)
with (OUT/'red_result.json').open('x') as f:json.dump(dict(command=cmd,exit_code=r.returncode,uses_legacy_fallback=True),f,indent=2)
print('RED_EXIT',r.returncode)
