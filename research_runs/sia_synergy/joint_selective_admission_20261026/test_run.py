import json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;WORK=OUT.parents[2]/'sia-joint-selective-admission/sia-simulator'
label=sys.argv[1];pattern=sys.argv[2];cmd=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p',pattern,'-v']
start=time.perf_counter()
with (OUT/(label+'.log')).open('x',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=WORK,stdout=f,stderr=subprocess.STDOUT)
with (OUT/(label+'.json')).open('x') as f:json.dump(dict(command=cmd,exit_code=r.returncode,seconds=time.perf_counter()-start),f,indent=2)
print(label,r.returncode,flush=True)
