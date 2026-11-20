import json,sys,subprocess,time,hashlib
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;p=json.loads((OUT/'preregistration.json').read_text());WORK=Path(p['tree'])/'sia-simulator'
def main():
    for cell in p['runs']:
        cid=cell['id'];dest=OUT/cid
        if dest.exists():raise RuntimeError('REFUSE_EXISTING:'+cid)
        if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('MEMORY_GATE_2GiB')
        if psutil.disk_usage(str(OUT)).free<5*1024**3:raise RuntimeError('DISK_GATE_5GiB')
        if cell['dataset']=='original20':
            for response in ['neutral','medium']:
                v=json.loads((OUT/f'original8_{response}_joint_s/validation.json').read_text());assert v['valid'] and v['all_complete']
        cmd=[sys.executable,'-B',str(WORK/'run_selective_cell.py'),'--manifest',str(OUT/'preregistration.json'),'--cell',cid,'--output',str(dest)]
        print('START',cid,flush=True);start=time.perf_counter()
        with (OUT/(cid+'.log')).open('x',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=WORK,stdout=f,stderr=subprocess.STDOUT)
        ev=dict(cell=cid,command=cmd,exit_code=r.returncode,seconds=time.perf_counter()-start,
            output_hashes={str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest().upper() for f in dest.iterdir() if f.is_file()} if dest.exists() else {})
        with (OUT/'execution_manifest.jsonl').open('a') as f:f.write(json.dumps(ev)+'\n')
        print('END',cid,r.returncode,round(ev['seconds'],2),flush=True)
        if r.returncode:raise RuntimeError('CELL_FAILED_STOP:'+cid)
        r=subprocess.run([sys.executable,'-B',str(OUT/'audit.py'),cid])
        if r.returncode:raise RuntimeError('AUDIT_FAILED_STOP:'+cid)
    print('COMPLETE_FROZEN_TEN_CELLS',flush=True)
if __name__=='__main__':main()
