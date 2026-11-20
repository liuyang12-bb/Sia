"""104 cells; fixed order, serial isolated processes, no selective best-run choice."""
import argparse,gzip,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];WORK=ROOT/'sia-joint-full160/sia-simulator';TREE=WORK.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def git(*a):return subprocess.check_output(['git','-C',str(TREE),*a])
def freeze(label):
    dest=OUT/label;dest.mkdir(exist_ok=False)
    assert json.loads((OUT/'full3.json').read_text())['exit_code']==0
    assert json.loads((OUT/'official_regression3/comparison.json').read_text())['exact_match']
    assert all(x['status']=='PASS' for x in json.loads((OUT/'mixed2_evidence/summary.json').read_text()))
    assert not git('diff','--name-only','HEAD')
    assert not git('ls-files','--others','--exclude-standard')
    pre=json.loads((OUT/'data_preflight.json').read_text());rows=[]
    for path in [OUT/'preflight.py',OUT/'freeze_run.py',OUT/'audit_results.py',OUT/'full3.json',OUT/'official_regression3/comparison.json',OUT/'mixed2_evidence/summary.json']:
        pre['input_hashes'][str(path)]=sha(path)
    for trace in range(1,11):
        for response in ('neutral','medium'):
            for method in ('sia_p','sia_tune','joint_p','joint_o'):
                rows.append(dict(id=f'main_w{trace}_{response}_{method}',trace=trace,capacity='main',response=response,method=method))
    for trace in (1,5,10):
        for cap in ('half','double'):
            for method in ('sia_p','sia_tune','joint_p','joint_o'):
                rows.append(dict(id=f'{cap}_w{trace}_medium_{method}',trace=trace,capacity=cap,response='medium',method=method))
    assert len(rows)==104
    names=[p for p in git('ls-files','-z').decode().split('\0') if p and (TREE/p).is_file()]
    source={str((TREE/p).relative_to(WORK)):sha(TREE/p) for p in names if (TREE/p).is_relative_to(WORK)}
    packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','cvxpy','cylp','pandas','autograd','psutil']}
    manifest=dict(commit=git('rev-parse','HEAD').decode().strip(),base='141138d95fc51755e2f71c15f01964b4c20757da',
        runs=rows,traces=pre['traces'],config=str(OUT/'inputs/config.json'),registry=str(OUT/'inputs/registry.json'),models=str(ROOT/'research_sources/Synergy/simulator/models'),
        input_hashes=pre['input_hashes'],source_hashes=source,packages=packages,python=sys.version,platform=platform.platform(),
        solver=dict(CBC=dict(numberThreads=8,allowablePercentageGap=.01,engine_version='not independently available'),
            HiGHS=dict(time_limit=30.,mip_rel_gap=0.,engine_version='bundled scipy; not independently available')),
        interval_seconds=60,max_rounds=6000,memory_gate_gib=2,disk_gate_gib=5,initial_free_disk_gib=shutil.disk_usage(OUT).free/1024**3,
        random_seed='no added stochastic policy; existing solver defaults unchanged',seed_profiles=[1,8],Q=300,D_bar=250,
        responses=dict(neutral=0.,medium=.5),rule='fixed R0 full55 + E-legacy; no online-model R0 seeding',
        failure_rules=dict(censored='Record all 160 outcomes at 6000; continue matrix. No imputation or selective window extension.',
            implementation_error='Preserve attempt, stop batch, fix and rerun all affected paired cells with new version; never mix versions.',
            solver_undecided='Explicit failure with snapshot; do not label infeasible or silently empty.',
            sia_tune='Whole GPU intent required. TUNE failure -> audited proportional unlocked jobs + exact checkpoint locks; if impossible, explicit scientific-semantics stop.',
            memory_disk='Do not launch below 2/5 GiB; at most ten minutes safe wait then stop for external relief.'),
        statistics=dict(unit='trace',mean='Only all160 complete paired cells',percentiles='numpy linear',
            rmst='Per trace common tau=360000-last_arrival, mean min(JCT,tau); no infinity imputation.',
            contrasts=['joint_o vs sia_p','joint_o vs sia_tune','joint_o vs joint_p'],
            caveat='JOINT-P vs Sia confounds placement and admission; not separate identification'),
        warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
    save(dest/'preregistration.json',manifest)
    with zipfile.ZipFile(dest/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(TREE/n,n)
    with zipfile.ZipFile(dest/'external_tools_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in OUT.glob('*.py'):z.write(path,path.name)
    subprocess.run(['git','-C',str(TREE),'bundle','create',str(dest/'source.bundle'),'HEAD'],check=True)
    (dest/'code.patch').write_bytes(git('diff',manifest['base'],'HEAD','--'))
    (dest/'requirements.lock.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze','--all']))
    save(dest/'freeze_hashes.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    print('FROZEN',dest,flush=True)

def run(label):
    dest=OUT/label;manifest=dest/'preregistration.json';p=json.loads(manifest.read_text())
    commands={v['id']:[sys.executable,'-B','-u',str(WORK/'run_full160_cell.py'),'--manifest',str(manifest),'--cell',v['id'],'--output',str(dest/v['id'])] for v in p['runs']}
    if not (dest/'commands.json').exists():save(dest/'commands.json',commands)
    eventsfile=dest/'execution_manifest.jsonl';previous=[]
    if eventsfile.exists():previous=[json.loads(l) for l in eventsfile.read_text().splitlines()]
    done={e['cell'] for e in previous if e.get('event')=='END' and e['exit_code']==0}
    # Resume only exact proven completed prefix; no overwrites or best-of reruns.
    for cell in p['runs']:
        cid=cell['id'];folder=dest/cid
        if cid in done:
            h=json.loads((folder/'output_hashes.json').read_text());assert all(sha(folder/n)==v for n,v in h.items());continue
        if folder.exists():raise RuntimeError(f'PARTIAL_ATTEMPT_REQUIRES_AUDIT {cid}')
        for trial in range(21):
            mem=psutil.virtual_memory().available/1024**3;disk=shutil.disk_usage(OUT).free/1024**3
            if mem>=2 and disk>=5:break
            print('RESOURCE_WAIT',cid,mem,disk,flush=True)
            if trial==20:raise RuntimeError('RESOURCE_GATE_REQUIRES_EXTERNAL_RELIEF')
            time.sleep(30)
        command=commands[cid];t=time.perf_counter();utc=time.time()
        with eventsfile.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='START',cell=cid,command=command,memory_gib=mem,disk_gib=disk,unix_time=utc))+'\n')
        print('START',cid,flush=True)
        with gzip.open(dest/(cid+'.stdout.log.gz'),'xb') as log:
            process=subprocess.Popen(command,cwd=WORK,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
            for line in process.stdout:
                log.write(line)
                if line.startswith(b'PROGRESS ') or line.startswith(b'RESULT '):print(cid,line.decode('utf-8').strip(),flush=True)
            code=process.wait()
        event=dict(event='END',cell=cid,exit_code=code,wall_seconds=time.perf_counter()-t,unix_time=time.time())
        with eventsfile.open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        if code:raise RuntimeError('CELL_FAILED_PRESERVE_AND_AUDIT '+cid)
    print('MATRIX_COMPLETE_104',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);a.add_argument('label');args=a.parse_args();globals()[args.action](args.label)
