"""Resume only the frozen, never-started suffix after the memory safety stop."""
import json, subprocess, sys, time
from experiment import OUT,LIVE,RUNS,gate,load,save,sha

def checkpoint():
    load();entries=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
    complete=[e for e in entries if e.get('exit_code')==0]
    assert [(e['response'],e['method']) for e in complete]==RUNS[:6]
    assert entries[-1]['status']=='SAFETY_STOP' and (entries[-1]['response'],entries[-1]['method'])==RUNS[6]
    hashes={}
    for response,method in RUNS[:6]:
      directory=OUT/'cells'/response/method
      assert json.loads((directory/'summary.json').read_text())['completed']==12
      assert not (directory/'failure.json').exists()
      for f in directory.iterdir():
        if f.is_file():hashes[f.relative_to(OUT).as_posix()]=sha(f)
      log=OUT/(response+'_'+method+'.log');hashes[log.name]=sha(log)
    for response,method in RUNS[6:]:
      assert not (OUT/'cells'/response/method).exists()
      assert not (OUT/(response+'_'+method+'.log')).exists()
    save(OUT/'safety_stop_snapshot.json',dict(completed_prefix=RUNS[:6],pending_suffix=RUNS[6:],
      prefix_hashes=hashes,preregistration_sha256=sha(OUT/'preregistration.json'),
      original_execution_manifest_sha256=sha(OUT/'execution_manifest.jsonl'),
      original_session=90386,original_session_exit=0,reason=entries[-1]))
    print('SAFETY_STOP_CHECKPOINT_VERIFIED_6_COMPLETE_2_NEVER_STARTED')

def resume():
    load();s=json.loads((OUT/'safety_stop_snapshot.json').read_text())
    assert sha(OUT/'preregistration.json')==s['preregistration_sha256']
    assert sha(OUT/'execution_manifest.jsonl')==s['original_execution_manifest_sha256']
    for n,h in s['prefix_hashes'].items():assert sha(OUT/n)==h,n
    for response,method in RUNS[6:]:assert not (OUT/'cells'/response/method).exists()
    resources=gate() # Check before taking the exclusive suffix-run marker.
    save(OUT/'resume_started.json',dict(resources=resources,time_unix=time.time(),
      completed_prefix_verified=True,original_session_exited=True,pending=RUNS[6:]))
    with (OUT/'resume_execution_manifest.jsonl').open('x',encoding='utf-8') as f:
      for response,method in RUNS[6:]:
        try:resources=gate()
        except RuntimeError as exc:
          f.write(json.dumps(dict(response=response,method=method,status='SAFETY_STOP',error=str(exc)))+'\n');f.flush();return
        cmd=[sys.executable,'-X','utf8','-B',str(OUT/'experiment.py'),'cell','--method',method,'--response',response]
        t=time.time()
        with (OUT/(response+'_'+method+'.log')).open('x',encoding='utf-8') as log:r=subprocess.run(cmd,cwd=LIVE,stdout=log,stderr=subprocess.STDOUT)
        e=dict(response=response,method=method,exit_code=r.returncode,wall_seconds=time.time()-t,resources=resources,command=cmd)
        f.write(json.dumps(e)+'\n');f.flush();print(e,flush=True)
        if r.returncode:print('FAILURE_STOP_PRESERVE_CELL',flush=True);return
    print('FROZEN_SUFFIX_COMPLETE',flush=True)

if __name__=='__main__':
    {'checkpoint':checkpoint,'resume':resume}[sys.argv[1]]()
