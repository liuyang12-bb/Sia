"""One safety retry after no cluster was constructed; never overwrite failed log."""
import json,sys,subprocess,time,hashlib
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
def main():
    p=json.loads((OUT/'preregistration_v2.json').read_text());work=Path(p['tree'])/'sia-simulator'
    gate=json.loads((OUT/'validation_8.json').read_text());assert gate['valid'] and gate['all_complete']
    assert not list(OUT.glob('n20_*/*')),'PARTIAL_OUTPUT_MUST_NOT_BE_REUSED'
    assert 'MEMORY_GATE_2GiB' in (OUT/'n20_neutral_sia_p.log').read_text()
    # No running experiment processes may overlap this retry.
    others=[v.info['cmdline'] for v in psutil.process_iter(['name','cmdline'])
        if (v.info['name'] or '').lower()=='python.exe' and
        any('run_scale_cell.py' in arg for arg in (v.info['cmdline'] or []))]
    assert not others,others
    for path,h in p['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()==h
    plan=dict(reason='First child exited at memory gate before output directory/cluster creation; loaded-environment read-only probe later measured 2.1888885498046875 GiB.',
        retries=1,inputs_and_code='unchanged preregistration_v2',gate_8_sha256=hashlib.sha256((OUT/'validation_8.json').read_bytes()).hexdigest(),
        rule='Stop on any further gate/error; no automatic third attempt; original failed launch log preserved')
    with (OUT/'retry20_plan.json').open('x') as f:json.dump(plan,f,indent=2)
    for response in p['responses']:
        for method in p['methods']:
            cid=f'n20_{response}_{method}';dest=OUT/cid
            if dest.exists():raise RuntimeError('REFUSE_EXISTING')
            if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('MEMORY_GATE_2GiB')
            if psutil.disk_usage(str(OUT)).free<5*1024**3:raise RuntimeError('DISK_GATE_5GiB')
            cmd=[sys.executable,'-B',str(work/'run_scale_cell.py'),'--manifest',str(OUT/'preregistration_v2.json'),
                 '--size','20','--method',method,'--response',response,'--output',str(dest)]
            tick=time.time();print('START RETRY',cid,flush=True)
            with (OUT/('retry_'+cid+'.log')).open('x',encoding='utf-8') as f:rc=subprocess.run(cmd,cwd=work,stdout=f,stderr=subprocess.STDOUT).returncode
            event=dict(cell=cid,attempt=2,command=cmd,exit_code=rc,wall_seconds=time.time()-tick,
                outputs={str(v.relative_to(OUT)):hashlib.sha256(v.read_bytes()).hexdigest() for v in dest.iterdir() if v.is_file()} if dest.exists() else {})
            with (OUT/'execution_manifest.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            print('END RETRY',cid,rc,round(event['wall_seconds'],2),flush=True)
            if rc:raise RuntimeError('RETRY_STOP:'+cid)
if __name__=='__main__':main()
