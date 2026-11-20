import json,sys,subprocess,time,hashlib
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
def main():
    p=json.loads((OUT/'preregistration.json').read_text());work=Path(p['tree'])/'sia-simulator'
    for n in [8,20]:
        if n==20:
            gate=json.loads((OUT/'validation_8.json').read_text());assert gate['valid'] and gate['all_complete']
        for response in ['neutral','medium']:
            cid=f'n{n}_{response}_joint_g';dest=OUT/cid
            if dest.exists():raise RuntimeError('REFUSE_EXISTING:'+cid)
            if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('MEMORY_GATE_2GiB')
            if psutil.disk_usage(str(OUT)).free<5*1024**3:raise RuntimeError('DISK_GATE_5GiB')
            cmd=[sys.executable,'-B',str(work/'run_bundle_cell.py'),'--manifest',str(OUT/'preregistration.json'),
                '--size',str(n),'--response',response,'--output',str(dest)]
            tick=time.time();print('START',cid,flush=True)
            with (OUT/(cid+'.log')).open('x',encoding='utf-8') as f:rc=subprocess.run(cmd,cwd=work,stdout=f,stderr=subprocess.STDOUT).returncode
            ev=dict(cell=cid,command=cmd,exit_code=rc,wall_seconds=time.time()-tick,
                outputs={str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir() if f.is_file()} if dest.exists() else {})
            with (OUT/'execution_manifest.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(ev)+'\n')
            print('END',cid,rc,round(ev['wall_seconds'],2),flush=True)
            if rc:raise RuntimeError('CELL_FAILED_STOP:'+cid)
        rc=subprocess.run([sys.executable,'-B',str(OUT/'audit.py'),str(n)]).returncode
        if rc:raise RuntimeError('INDEPENDENT_AUDIT_FAILED')
if __name__=='__main__':main()
