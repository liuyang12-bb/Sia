"""Serial, bounded failure reproduction after the registered batch only."""
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'

def main():
    manifest=json.loads((OUT/'preregistration.json').read_text())
    ledger=OUT/'execution_recovery1/execution_manifest.jsonl'
    assert any(json.loads(x)['event']=='BATCH_END' for x in ledger.read_text().splitlines()),'BATCH_STILL_RUNNING'
    assert time.time()+900<manifest['formal_stop_unix'],'INSUFFICIENT_DIAGNOSTIC_BUDGET'
    assert psutil.virtual_memory().available>=2*2**30
    assert psutil.disk_usage(str(OUT)).free>=5*2**30
    cmd=[sys.executable,'-X','utf8','-B',str(ROOT/'maintenance/diagnose_core80_full_failure.py')]
    start=time.time();timeout=False
    log=OUT/'diagnostic_guard_console.log'
    with log.open('x',encoding='utf8') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while proc.poll() is None:
            if time.time()-start>=720:
                timeout=True
                try:
                    parent=psutil.Process(proc.pid);children=parent.children(recursive=True)
                    for q in reversed(children):
                        try:q.terminate()
                        except psutil.NoSuchProcess:pass
                    try:parent.terminate()
                    except psutil.NoSuchProcess:pass
                    psutil.wait_procs(children+[parent],timeout=10)
                except psutil.NoSuchProcess:pass
                break
            time.sleep(5)
        proc.wait()
    result=dict(start_unix=start,end_unix=time.time(),wall_seconds=time.time()-start,exit_code=proc.returncode,timeout=timeout,
                limit_seconds=720,scope='At most five60sec boundaries, unchanged source/parameters; no overall JCT evaluation; no extension.')
    with (OUT/'diagnostic_guard_result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
