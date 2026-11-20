"""Freeze once then execute exactly eight registered separate processes."""
import argparse,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time,zipfile
from pathlib import Path
import pandas as pd
import psutil

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]; WORK=ROOT/'sia-joint-stage2c-r0/sia-simulator'
OLD=ROOT/'research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):
    with p.open('x',encoding='utf-8') as f: json.dump(v,f,indent=2)
def git(*args): return subprocess.check_output(['git','-C',str(WORK),*args])

def freeze(label):
    dest=OUT/label; dest.mkdir(exist_ok=False)
    config=OLD/'frozen_config_6000.json'; workload=OLD/'workload.csv'
    registry=ROOT/'research_runs/sia_synergy/stage2b_neutral/g0_verified/registry.json'
    cfg=json.loads(config.read_text(encoding='utf-8')); data=pd.read_csv(workload)
    trace=WORK/'workloads/saturn/workload-10.csv'; source=pd.read_csv(trace)
    sorted_source=source.sort_values('time',kind='mergesort'); expected=sorted_source.iloc[140:160].copy()
    indices=expected.index.tolist(); origin=int(expected.time.iloc[0]); expected['time']-=origin
    assert data.equals(expected.reset_index(drop=True))
    assert data.name.tolist()==cfg['workload_rows']
    for p in (config,workload,registry): (dest/p.name).write_bytes(p.read_bytes())
    models=ROOT/'research_sources/Synergy/simulator/models'
    inputs=[dest/config.name,dest/workload.name,dest/registry.name,trace]
    inputs += [models/f for f in cfg['response']['application_template_rule'].values()]
    inputs += sorted((WORK/'traces').rglob('*.csv'))
    tree=WORK.parent
    tracked=[tree/p for p in git('ls-files','--full-name','-z').decode().split('\0') if p]
    tracked=[p for p in tracked if p.is_file()]
    source_hashes={str(p.relative_to(WORK)):sha(p) for p in tracked if p.is_relative_to(WORK)}
    commit=git('rev-parse','HEAD').decode().strip()
    env=dict(python=sys.version,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','cvxpy','cylp','pandas','autograd','psutil']},
        CBC_engine_version='not independently available',HiGHS_engine_version='bundled SciPy engine; package version above',
        solver_options={'time_limit':30.,'mip_rel_gap':0.},random_seed='no stochastic policy/selection; no new seed override')
    freeze=dict(commit=commit,base='bd6105a37c6662336236a135eb7ad610ca6e16de',
        config=str(dest/config.name),workload=str(dest/workload.name),registry=str(dest/registry.name),models=str(models),
        runs=[[response,cell] for response in ('neutral','medium') for cell in ('u_fp','u_fo','u_jp','u_jo')],
        max_rounds=6000,interval=60,memory_gate_gib=2,Q=300,D_bar=250,
        rmst_tau_seconds=360000-float(data.time.max()),
        censor_rule='At 6000 rounds preserve incomplete jobs. Full mean/median/tails/makespan only if all 20 complete. Empirical mean min(JCT,tau), tau=360000-latest arrival, is identifiable for all jobs. No imputation.',
        percentile_rule='numpy linear interpolation; only 20 jobs, not precise population tails',
        source_trace_rows_zero_based=indices,original_first_arrival=origin,
        seed_rule='Preserve frozen official Sia seed_profiles(1,8); R0 never seeds online performance. Real unseeded paths are separately tested.',
        response_rule='fixed full-domain R0, E-legacy endpoint clipping, (3,62.5) reference, exponent0 or0.5',
        input_hashes={str(p):sha(p) for p in inputs},source_hashes=source_hashes,environment=env,
        warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
    save(dest/'preregistration.json',freeze)
    with zipfile.ZipFile(dest/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in tracked: z.write(p,str(p.relative_to(tree)))
    subprocess.run(['git','-C',str(tree),'bundle','create',str(dest/'source.bundle'),'HEAD'],check=True)
    (dest/'code.patch').write_bytes(git('diff','bd6105a37c6662336236a135eb7ad610ca6e16de','HEAD','--'))
    (dest/'requirements.lock.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    save(dest/'freeze_hashes.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    print('FROZEN '+str(dest),flush=True)

def regression(label):
    d=OUT/label; d.mkdir(exist_ok=False); states={}
    for name in ('official-artifacts','sia-joint-stage2c-r0'):
        cmd=[sys.executable,'-B',str(WORK/'check_stage1_legacy_regression.py'),'--probe',str(ROOT/name/'sia-simulator')]
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
        (d/(name+'.log')).write_bytes(r.stdout); assert r.returncode==0
        states[name]=json.loads(next(line[11:] for line in r.stdout.decode('utf-8').splitlines() if line.startswith('PROBE_JSON=')))
    same=states['official-artifacts']==states['sia-joint-stage2c-r0']; save(d/'comparison.json',dict(exact=same,states=states)); assert same
    print('ORIGINAL SIA EXACT',flush=True)

def run(label):
    dest=OUT/label; manifest=dest/'preregistration.json'; p=json.loads(manifest.read_text(encoding='utf-8'))
    commands=[]
    for response,cell in p['runs']:
        folder=dest/f'{response}_{cell}'
        commands.append([sys.executable,'-B',str(WORK/'run_stage2c_cell.py'),'--manifest',str(manifest),'--cell',cell,'--response',response,'--output',str(folder)])
    save(dest/'commands.json',commands)
    with (dest/'execution_manifest.jsonl').open('x',encoding='utf-8') as manifestlog:
        for command in commands:
            name=Path(command[-1]).name
            # Up to 10 minutes automatic safe wait, never lower memory threshold.
            for attempt in range(21):
                mem=psutil.virtual_memory().available/1024**3
                if mem>=2: break
                print(f'MEMORY_PAUSE {name} {mem:.3f} GiB attempt {attempt}',flush=True)
                if attempt==20: raise RuntimeError('MEMORY_GATE_REQUIRES_EXTERNAL_RELIEF')
                time.sleep(30)
            started=time.perf_counter(); print('START '+name,flush=True)
            with (dest/(name+'.log')).open('xb') as log:
                r=subprocess.run(command,cwd=WORK,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
            event=dict(cell=name,exit_code=r.returncode,memory_gib=mem,wall_seconds=time.perf_counter()-started)
            manifestlog.write(json.dumps(event)+'\n'); manifestlog.flush(); print(json.dumps(event),flush=True)
            if r.returncode: raise RuntimeError(f'CELL_FAILED {name}; preserve evidence and repair before continuing')

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('action',choices=['freeze','run','regression']); a.add_argument('label'); args=a.parse_args()
    globals()[args.action](args.label)
