"""Freeze before outcomes. No strategy tuning and no full160 resume."""
import hashlib, importlib.metadata, json, platform, subprocess, sys, zipfile, time
from pathlib import Path
import pandas as pd
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
TREE=ROOT/'sia-joint-scale-check';WORK=TREE/'sia-simulator'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def main():
    start=time.perf_counter()
    if psutil.virtual_memory().available<2*1024**3:raise RuntimeError('MEMORY_GATE_2GiB')
    old=json.loads((OUT.parent/'adaptive_service_20261025/preregistration.json').read_text())
    # Verify every historical prerequisite unchanged, then copy its hash contract.
    for p,h in old['hashes'].items():
        if sha(p)!=h:raise RuntimeError('HISTORICAL_FREEZE_CHANGED:'+p)
    prior=ROOT/'sia-joint-adaptive-service/sia-simulator'
    files=subprocess.check_output(['git','-C',str(prior),'ls-files','--others','--exclude-standard']).decode().splitlines()
    for f in files:
        rel=Path(f)
        if sha(prior/rel)!=sha(WORK/rel):raise RuntimeError('RESTORE_MISMATCH:'+f)
    testcmd=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v']
    if (OUT/'test_result.json').exists():
        test_result=json.loads((OUT/'test_result.json').read_text())
        assert test_result['command']==testcmd and test_result['exit_code']==0
    else:
        with (OUT/'tests.log').open('x',encoding='utf-8') as f:
            test=subprocess.run(testcmd,cwd=WORK,stdout=f,stderr=subprocess.STDOUT)
        save(OUT/'test_result.json',dict(command=testcmd,exit_code=test.returncode,wall_seconds=time.perf_counter()-start))
        if test.returncode:raise RuntimeError('TEST_GATE')
    source=ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv'
    df=pd.read_csv(source);apps=['bert','cifar10','deepspeech2']
    per={a:list(df.index[df.application==a]) for a in apps};cursor={a:0 for a in apps};indices=[]
    i=0
    while len(indices)<20:
        a=apps[i%3];i+=1
        if cursor[a]>=len(per[a]):continue
        indices.append(per[a][cursor[a]]);cursor[a]+=1
    workloads={};records={}
    for n in [8,20]:
        cohort=df.loc[indices[:n]].copy();cohort['original_arrival']=cohort.time
        cohort['source_row_index']=indices[:n];cohort['time']=0.
        path=OUT/f'workload_{n}.csv';cohort.to_csv(path,index=False)
        workloads[str(n)]=str(path);records[str(n)]=cohort.to_dict('records')
    hashes=dict(old['hashes'])
    for p in WORK.rglob('*.py'):hashes[str(p)]=sha(p)
    for p in workloads.values():hashes[p]=sha(p)
    src={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob('*.py')}
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in WORK.rglob('*.py'):z.write(p,p.relative_to(TREE))
    save(OUT/'source_hashes.json',src)
    packages={x:importlib.metadata.version(x) for x in ['numpy','scipy','pandas','cvxpy','cylp','psutil']}
    p=dict(base_commit=subprocess.check_output(['git','-C',str(TREE),'rev-parse','HEAD']).decode().strip(),
        branch='exp/joint-r-scale-check',tree=str(TREE),workloads=workloads,records=records,
        config=old['config'],registry=old['registry'],models=old['models'],hashes=hashes,
        interval=60,max_rounds=600,selection_rule='round-robin bert/cifar10/deepspeech2, each original order, skip exhausted class, no duplication; coarrival0; nested prefix',
        gate='all 8-job six cells complete and independent audits pass before 20-job six cells',
        methods=['sia_p','sia_aux','joint_r'],responses=['neutral','medium'],packages=packages,python=platform.python_version(),
        memory_gate_gib=2,disk_gate_gib=5,censor_rule='No imputed JCT; no cap extension; no complete-mean wins for censored pairs',
        interpretation='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
    save(OUT/'preregistration.json',p)
    print('FROZEN',packages,records,flush=True)
if __name__=='__main__':main()
