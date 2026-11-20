import json,os,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];WORK=ROOT/'sia-joint-full160/sia-simulator'
dest=OUT/sys.argv[1];dest.mkdir(exist_ok=False);runs={}
for label,sim in [('official',ROOT/'official-artifacts/sia-simulator'),('adapted',WORK)]:
    mem=psutil.virtual_memory().available/1024**3
    if mem<2:raise RuntimeError(f'MEMORY_GATE {mem}')
    cmd=[sys.executable,'-B',str(WORK/'check_full160_official.py'),'--sim',str(sim),'--kind',label,
         '--config',str(OUT/'inputs/config.json'),'--out',str(dest/(label+'.json'))]
    t=time.perf_counter()
    with (dest/(label+'.log')).open('xb') as f:
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
    runs[label]=dict(command=cmd,exit_code=r.returncode,memory_gib=mem,wall_seconds=time.perf_counter()-t)
    if r.returncode:raise RuntimeError(runs[label])
a=json.loads((dest/'official.json').read_text());b=json.loads((dest/'adapted.json').read_text())
diff=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
result=dict(exact_match=a==b,rounds_official=len(a),rounds_adapted=len(b),first_difference=diff[:1],runs=runs,
    completions={k:sum(j['completion'] is not None for j in rows[-1]['jobs'].values()) for k,rows in [('official',a),('adapted',b)]})
with (dest/'comparison.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result));assert result['exact_match']
