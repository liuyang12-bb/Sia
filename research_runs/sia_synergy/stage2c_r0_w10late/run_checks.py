import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
WORK=OUT.parents[2]/'sia-joint-stage2c-r0/sia-simulator'
label=sys.argv[1]; args=sys.argv[2:] or ['-m','unittest','discover','-s','tests','-p','test_*.py','-v']
mem=psutil.virtual_memory().available/1024**3
if mem<2: raise RuntimeError(f'MEMORY_GATE {mem}')
started=time.perf_counter()
with (OUT/(label+'.log')).open('xb') as f:
    p=subprocess.run([sys.executable,'-B',*args],cwd=WORK,stdout=f,stderr=subprocess.STDOUT,
        env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1',
            'STAGE1_EVIDENCE_DIR':str(OUT/(label+'_stage1')),'STAGE2B_EVIDENCE_DIR':str(OUT/(label+'_stage2b')),
            'STAGE2C_EVIDENCE_DIR':str(OUT/(label+'_stage2c'))})
data=dict(exit_code=p.returncode,command=[sys.executable,'-B',*args],memory_gib=mem,wall_seconds=time.perf_counter()-started,
    source_hashes={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest().upper()
        for folder in ('joint_prototype','tests') for p in (WORK/folder).glob('*.py')})
with (OUT/(label+'.json')).open('x',encoding='utf-8') as f: json.dump(data,f,indent=2)
print(json.dumps({k:v for k,v in data.items() if k!='source_hashes'}))
print((OUT/(label+'.log')).read_text(encoding='utf-8')[-5000:]); sys.exit(p.returncode)
