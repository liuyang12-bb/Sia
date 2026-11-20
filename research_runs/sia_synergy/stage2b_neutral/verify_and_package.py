"""Bounded legacy regression and immutable Stage2B delivery; no cohort runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from collections import Counter

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
WORK=ROOT/'sia-joint-stage2b-neutral'
BASE='216573cb26f3136520ce9d4f609623b4c2c75df1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def dump(p,v):
    with p.open('x',encoding='utf-8') as f: json.dump(v,f,indent=2,allow_nan=False)
def git(*args): return subprocess.check_output(['git','-C',str(WORK),*args])

def legacy():
    out=OUT/'legacy_regression'; out.mkdir()
    helper=WORK/'sia-simulator/check_stage1_legacy_regression.py'
    states={}; commands={}
    for name in ('official-artifacts','sia-joint-correctness-stage1p1','sia-joint-stage2b-neutral'):
        cmd=[sys.executable,'-B',str(helper),'--probe',str(ROOT/name/'sia-simulator')]
        start=time.perf_counter()
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'})
        (out/(name+'.log')).write_bytes(r.stdout)
        commands[name]=dict(command=cmd,exit_code=r.returncode,wall_seconds=time.perf_counter()-start)
        if r.returncode: raise RuntimeError('legacy probe failed')
        states[name]=json.loads(next(s.removeprefix('PROBE_JSON=') for s in r.stdout.decode().splitlines() if s.startswith('PROBE_JSON=')))
    old=json.loads((ROOT/'research_runs/sia_synergy/correctness_stage1p1/legacy_regression/comparison.json').read_text())
    match=all(v==states['official-artifacts'] for v in states.values()) and states['official-artifacts']==old['states']['official-artifacts']
    dump(out/'comparison.json',dict(exact_state_match=match,jobs=1,rounds=3,states=states,commands=commands))
    assert match
    print('PASS original Sia: 3 versions plus archived original state, 3 rounds identical')

def analyze():
    evidence=OUT/'final_tests_evidence'
    summaries={}
    for p in sorted(evidence.glob('real_*.json')):
        data=json.loads(p.read_text(encoding='utf-8'))
        rounds=data if isinstance(data,list) else data['rounds']
        by_job={}; errors=[]; solver=Counter(); drain=0
        for r in rounds:
            decision=r['decision_for_next_interval']; audit=decision['result']['audit']
            if not audit['valid']: errors.append('allocation audit')
            solver.update(t['status'] for t in audit['solver_trace'])
            drain+=int(audit['drain'])
            for e in r['execution']:
                name=e['job_id']
                acc=by_job.setdefault(name,dict(waiting=0.,checkpoint=0.,productive=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,progress=0.,first_allocation=None,first_productive=None,completion=None,restart_events=0,wasted_gpu_seconds=0.))
                for key,src in [('waiting','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('productive','training_seconds')]: acc[key]+=e[src]
                acc['progress']=e['progress_after']; acc['completion']=e['completion_time']
                for key,src in [('first_allocation','first_allocation_time'),('first_productive','first_productive_training_time')]:
                    if e[src] is not None: acc[key]=e[src]
                resource=e['resource_allocation']
                if resource:
                    usage=resource['node_usage']; dt=e['allocated_seconds']
                    for metric,key in [('gpu_hours','gpus'),('cpu_core_hours','cpu'),('ram_gb_hours','ram_gb')]:
                        acc[metric]+=sum(u[key] for u in usage)*dt/3600
                    acc['wasted_gpu_seconds']+=sum(u['gpus'] for u in usage)*e['checkpoint_delay_consumed']
                if not e['independent_execution_audit']['valid'] or e['rho']!=1: errors.append('execution audit')
            # Migration/reallocation events are log records, not all are restarts
            # (suspensions and initial allocation differ). Use final Job count.
        final={} if isinstance(data,list) else {j['job']:j for j in data['final']}
        for name,acc in by_job.items():
            acc['restarts']=final.get(name,{}).get('restarts',0)
            acc.pop('restart_events')
        summaries[p.stem]=dict(jobs=by_job,rounds=len(rounds),last_time=rounds[-1]['interval_end'],
            drain_rounds=drain,solver_status_counts=dict(solver),violations=errors,
            completed=sum(j['completion'] is not None for j in by_job.values()),
            evidence_sha256=sha(p),service_issues=rounds[-1]['decision_for_next_interval']['liveness_events'])
    proof=json.loads((evidence/'state_boundary_enumeration.json').read_text())
    summary=dict(real_scenarios=summaries,state_boundary_cases=proof,
        bounds=dict(D_bar=250,Q=300,delta=60,L=610,W_alloc_N3=2440,W_first_N3=2810),
        all_runtime_audits_valid=all(not s['violations'] for s in summaries.values()),
        all_neutral=True,formal_experiments_run=False)
    dump(OUT/'mechanism_summary.json',summary)
    print(json.dumps({k:{a:v[a] for a in ('rounds','completed','drain_rounds','violations')} for k,v in summaries.items()}))

def package():
    base=json.loads((OUT/'snapshot_manifest.json').read_text())
    changed=[n for n,h in base['protected_hashes'].items() if not (ROOT/n).exists() or sha(ROOT/n)!=h]
    unchanged=[n for n,h in base['source_hashes'].items() if sha(WORK/n)!=h]
    assert not changed and not unchanged, (changed,unchanged)
    assert not git('diff','--name-only','HEAD')
    names=git('ls-files','-z').decode().split('\0')[:-1]
    hashes={n:sha(WORK/n) for n in names if (WORK/n).is_file()}
    with zipfile.ZipFile(OUT/'stage2b_source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in hashes: z.write(WORK/n,n)
    with zipfile.ZipFile(OUT/'stage2b_source_snapshot.zip') as z:
        assert all(hashlib.sha256(z.read(n)).hexdigest().upper()==h for n,h in hashes.items())
    subprocess.run(['git','-C',str(WORK),'bundle','create',str(OUT/'stage2b.bundle'),'HEAD'],check=True)
    r=subprocess.run(['git','-C',str(WORK),'bundle','verify',str(OUT/'stage2b.bundle')],capture_output=True)
    (OUT/'bundle_verify.log').write_bytes(r.stdout+r.stderr); assert r.returncode==0
    (OUT/'stage2b.patch').write_bytes(git('diff','--binary',BASE,'HEAD'))
    (OUT/'diff_stat.txt').write_bytes(git('diff','--stat',BASE,'HEAD'))
    models=ROOT/'research_sources/Synergy/simulator/models'
    with zipfile.ZipFile(OUT/'synergy_matrix_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(models.glob('*_1.py')): z.write(p,p.name)
    dump(OUT/'repair_manifest.json',dict(base_commit=BASE,commit=git('rev-parse','HEAD').decode().strip(),
        source_hashes=hashes,source_files=len(hashes),protected_files_checked=len(base['protected_hashes']),
        protected_changed=changed,historical_source_changed=unchanged,
        changed_files=git('diff','--name-only',BASE,'HEAD').decode().splitlines(),
        snapshot_sha256=sha(OUT/'stage2b_source_snapshot.zip'),bundle_sha256=sha(OUT/'stage2b.bundle'),
        patch_sha256=sha(OUT/'stage2b.patch')))
    print(json.dumps(dict(files=len(hashes),protected_changed=changed,historical_source_changed=unchanged)))

def commit():
    base=json.loads((OUT/'snapshot_manifest.json').read_text())
    assert git('rev-parse','HEAD').decode().strip()==BASE
    assert all(sha(WORK/n)==h for n,h in base['source_hashes'].items())
    assert not git('diff','--name-only','HEAD')
    # Refresh CRLF stat-only entries by an explicit, byte-verified allowlist.
    status=git('status','--porcelain','-z').decode().split('\0')
    refresh=[v[3:] for v in status if v.startswith(' M ')]
    assert all(n in base['source_hashes'] for n in refresh)
    if refresh: subprocess.run(['git','-C',str(WORK),'add','--',*refresh],check=True)
    assert not git('diff','--cached','--name-only')
    added=['sia-simulator/joint_prototype/neutral_liveness.py',
           'sia-simulator/joint_prototype/stage2b_neutral_config.json',
           'sia-simulator/stage2b_preflight.py','sia-simulator/tests/test_stage2b_neutral.py']
    subprocess.run(['git','-C',str(WORK),'add','--',*added],check=True)
    assert sorted(git('diff','--cached','--name-only').decode().splitlines())==sorted(added)
    subprocess.run(['git','-C',str(WORK),'commit','-m','feat: isolate Neutral L2 mechanism and offline R0 coverage audit'],check=True)
    print(git('rev-parse','HEAD').decode().strip())

if __name__=='__main__':
    {'legacy':legacy,'analyze':analyze,'package':package,'commit':commit}[sys.argv[1]]()
