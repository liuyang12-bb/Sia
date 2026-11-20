"""Only the never-started eighth cell; preserve both earlier gate stops."""
import json,subprocess,sys,time
from experiment import OUT,LIVE,RUNS,load,save,sha,gate

load()
s=json.loads((OUT/'safety_stop_snapshot.json').read_text())
assert sha(OUT/'execution_manifest.jsonl')==s['original_execution_manifest_sha256']
for n,h in s['prefix_hashes'].items():assert sha(OUT/n)==h,n
entries=[json.loads(l) for l in (OUT/'resume_execution_manifest.jsonl').read_text().splitlines()]
assert len(entries)==2 and entries[0]['exit_code']==0 and entries[1]['status']=='SAFETY_STOP'
assert (entries[0]['response'],entries[0]['method'])==RUNS[6]
assert (entries[1]['response'],entries[1]['method'])==RUNS[7]
response,method=RUNS[7]
assert not (OUT/'cells'/response/method).exists()
assert not (OUT/(response+'_'+method+'.log')).exists()
hashes={}
for r,m in RUNS[:7]:
    d=OUT/'cells'/r/m;assert json.loads((d/'summary.json').read_text())['completed']==12
    assert not (d/'failure.json').exists()
    hashes.update({f.relative_to(OUT).as_posix():sha(f) for f in d.iterdir() if f.is_file()})
resources=gate()
save(OUT/'last_cell_started.json',dict(resources=resources,prefix_hashes=hashes,original_sessions_exited=[90386,96296],
    resume_manifest_sha256=sha(OUT/'resume_execution_manifest.jsonl'),time_unix=time.time(),pending=[response,method]))
cmd=[sys.executable,'-X','utf8','-B',str(OUT/'experiment.py'),'cell','--method',method,'--response',response]
t=time.time()
with (OUT/(response+'_'+method+'.log')).open('x',encoding='utf-8') as log:
    result=subprocess.run(cmd,cwd=LIVE,stdout=log,stderr=subprocess.STDOUT)
entry=dict(response=response,method=method,exit_code=result.returncode,wall_seconds=time.time()-t,resources=resources,command=cmd)
with (OUT/'last_execution_manifest.jsonl').open('x',encoding='utf-8') as f:f.write(json.dumps(entry)+'\n')
print('LAST_CELL_RESULT',entry,flush=True)
