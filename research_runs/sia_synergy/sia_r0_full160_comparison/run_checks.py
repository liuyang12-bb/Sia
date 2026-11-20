import json,os,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent; WORK=OUT.parents[2]/'sia-joint-full160/sia-simulator'
label=sys.argv[1]; args=sys.argv[2:] or ['-m','unittest','discover','-s','tests','-p','test_*.py','-v']
mem=psutil.virtual_memory().available/1024**3
if mem<2: raise RuntimeError(f'MEMORY_GATE {mem}')
t=time.perf_counter()
with (OUT/(label+'.log')).open('xb') as f:
    p=subprocess.run([sys.executable,'-B',*args],cwd=WORK,stdout=f,stderr=subprocess.STDOUT,
        env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1',
            'FULL160_EVIDENCE':str(OUT/(label+'_evidence'))})
with (OUT/(label+'.json')).open('x') as f: json.dump(dict(command=args,exit_code=p.returncode,memory_gib=mem,wall_seconds=time.perf_counter()-t),f,indent=2)
print((OUT/(label+'.log')).read_text(encoding='utf-8')[-6000:]);sys.exit(p.returncode)
