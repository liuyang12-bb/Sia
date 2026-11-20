"""Run bounded tests with immutable log/evidence labels, not formal experiments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
import hashlib
OUT=Path(__file__).resolve().parent
WORK=OUT.parents[2]/'sia-joint-stage2b-neutral/sia-simulator'
label=sys.argv[1]; args=sys.argv[2:] or ['-m','unittest','discover','-s','tests','-p','test_*.py','-v']
memory=psutil.virtual_memory().available/1024**3
if memory<2.: raise RuntimeError(f'2 GiB memory gate: available {memory:.3f}')
start=time.perf_counter()
source_hashes={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest().upper()
    for folder in (WORK/'joint_prototype',WORK/'tests') for p in folder.glob('*.py')}
commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD']).decode().strip()
with (OUT/(label+'.log')).open('xb') as log:
    r=subprocess.run([sys.executable,'-B',*args],cwd=WORK,stdout=log,stderr=subprocess.STDOUT,
        env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1',
             'STAGE1_EVIDENCE_DIR':str(OUT/(label+'_old_evidence')),
             'STAGE2B_EVIDENCE_DIR':str(OUT/(label+'_evidence'))})
summary=dict(command=[sys.executable,'-B',*args],cwd=str(WORK),exit_code=r.returncode,wall_seconds=time.perf_counter()-start,available_memory_gib=memory,commit=commit,source_hashes=source_hashes)
with (OUT/(label+'.json')).open('x',encoding='utf-8') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary)); print((OUT/(label+'.log')).read_text(encoding='utf-8')[-4000:])
sys.exit(r.returncode)
