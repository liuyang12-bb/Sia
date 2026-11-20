"""Only frozen six-cell stages. Never resume a partial output or full160 batch."""
import json,sys,subprocess,time,hashlib
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
def main():
    n=int(sys.argv[1]);assert n in [8,20]
    p=json.loads((OUT/'preregistration_v2.json').read_text());work=Path(p['tree'])/'sia-simulator'
    if n==20:
        gate=json.loads((OUT/'validation_8.json').read_text())
        assert gate['valid'] and gate['all_complete'],'EIGHT_JOB_GATE_NOT_PASSED'
    for response in p['responses']:
        for method in p['methods']:
            cid=f'n{n}_{response}_{method}';dest=OUT/cid
            if dest.exists():raise RuntimeError('REFUSE_EXISTING:'+cid)
            if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('MEMORY_GATE_2GiB')
            if psutil.disk_usage(str(OUT)).free<5*1024**3:raise RuntimeError('DISK_GATE_5GiB')
            cmd=[sys.executable,'-B',str(work/'run_scale_cell.py'),'--manifest',str(OUT/'preregistration_v2.json'),
                '--size',str(n),'--method',method,'--response',response,'--output',str(dest)]
            started=time.time();print('START',cid,flush=True)
            with (OUT/(cid+'.log')).open('x',encoding='utf-8') as log:rc=subprocess.run(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT).returncode
            event=dict(cell=cid,command=cmd,exit_code=rc,wall_seconds=time.time()-started,
                outputs={str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir() if f.is_file()} if dest.exists() else {})
            with (OUT/'execution_manifest.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            print('END',cid,rc,round(event['wall_seconds'],2),flush=True)
            if rc:raise RuntimeError('CELL_FAILED_STOP:'+cid)
if __name__=='__main__':main()
