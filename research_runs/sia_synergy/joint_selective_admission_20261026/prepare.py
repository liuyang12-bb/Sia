import hashlib,importlib.metadata,json,platform,subprocess,sys,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];TREE=ROOT/'sia-joint-selective-admission';WORK=TREE/'sia-simulator'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def save(f,v):
    with f.open('x') as out:json.dump(v,out,indent=2)
def main():
    assert json.loads((OUT/'green_final.json').read_text())['exit_code']==0
    parent=ROOT/'research_runs/sia_synergy/joint_budgeted_adaptation_20261025/attempt3'
    p=json.loads((parent/'preregistration.json').read_text())
    for f,h in p['hashes'].items():assert sha(f)==h,f
    oldsrc=json.loads((OUT/'restored_source_hashes.json').read_text())
    changes=[Path(rel).as_posix() for rel,h in oldsrc.items() if sha(TREE/rel)!=h]
    assert changes==['sia-simulator/joint_prototype/adaptive_service.py'],changes
    hashes=dict(p['hashes'])
    for f in WORK.rglob('*.py'):hashes[str(f)]=sha(f)
    hold=json.loads((OUT/'holdout_preregistration.json').read_text());assert sha(hold['csv'])==hold['sha256']
    datasets={'original8':p['workloads']['8'],'original20':p['workloads']['20'],'holdout20':hold['csv']}
    for f in datasets.values():hashes[f]=sha(f)
    runs=[]
    for ds in ['original8','original20','holdout20']:
        for response in ['neutral','medium']:
            for method in (['joint_s'] if ds!='holdout20' else ['sia_p','sia_aux','joint_s']):
                runs.append(dict(id=f'{ds}_{response}_{method}',dataset=ds,response=response,method=method))
    baseline={str(f):sha(f) for folder in [parent,ROOT/'research_runs/sia_synergy/joint_r_scale_check_20261025'] for f in folder.rglob('*') if f.is_file()}
    with zipfile.ZipFile(OUT/'execution_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in WORK.rglob('*.py'):z.write(f,f.relative_to(TREE))
    save(OUT/'preregistration.json',dict(base_commit=subprocess.check_output(['git','-C',str(TREE),'rev-parse','HEAD']).decode().strip(),
        branch=subprocess.check_output(['git','-C',str(TREE),'branch','--show-current']).decode().strip(),tree=str(TREE),new_code_committed=False,
        config=p['config'],registry=p['registry'],models=p['models'],hashes=hashes,baseline_hashes=baseline,datasets=datasets,
        runs=runs,interval=60,max_rounds=600,seed_profiles=[1,8],objective='unchanged p=-.5 lambda_n=1.1 bonus=0; exact primary then actual startup then geometry',
        admission='held+FIFO head mandatory, uncommitted optional; drain if head cannot coexist; no service eviction',
        max_active_jobs=20,quantum_productive_seconds=300,total_actual_delay_budget_seconds=250,
        memory_gate_gib=2,disk_gate_gib=5,censor_rule='no complete-mean comparison for incomplete pairs; no imputation/extension',
        gate='original8 N/M complete and raw audits before original20; holdout uses same frozen version; any implementation/audit failure stops',
        packages={x:importlib.metadata.version(x) for x in ['numpy','scipy','pandas','cvxpy','cylp','psutil']},python=platform.python_version(),
        conditional_numerical_retry='inherited G: validate optimal vectors; only invalid status0 retry identical model once with presolve=False',
        interpretation='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
    print('FROZEN',len(runs),'cells',len(hashes),'hashes; parent changes',changes,flush=True)
if __name__=='__main__':main()
