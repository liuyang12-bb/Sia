import json,hashlib,zipfile,subprocess,sys
from pathlib import Path
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
OLD=ROOT/'sia-joint-budgeted-adaptation';TREE=ROOT/'sia-joint-selective-admission'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def save(f,v):
    with f.open('x',encoding='utf-8') as out:json.dump(v,out,indent=2)
def main():
    files=list((OLD/'sia-simulator').rglob('*.py'))
    verified={str(f.relative_to(OLD)):sha(f) for f in files}
    for rel,h in verified.items():assert sha(TREE/rel)==h,rel
    with zipfile.ZipFile(OUT/'parent_complete_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.relative_to(OLD))
    save(OUT/'restored_source_hashes.json',verified)
    with (OUT/'parent_tracked.patch').open('xb') as f:
        f.write(subprocess.check_output(['git','-C',str(OLD),'diff','--binary']))
    # Before any policy result: same deterministic class-mix selection, independent trace.
    src=ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-5.csv'
    df=pd.read_csv(src);assert len(df)==160
    apps=['bert','cifar10','deepspeech2'];per={a:list(df.index[df.application==a]) for a in apps}
    cursor={a:0 for a in apps};indices=[];i=0
    while len(indices)<20:
        a=apps[i%3];i+=1
        if cursor[a]>=len(per[a]):continue
        indices.append(per[a][cursor[a]]);cursor[a]+=1
    cohort=df.loc[indices].copy();cohort['original_arrival']=cohort.time
    cohort['source_row_index']=indices;cohort['time']=0.
    dst=OUT/'holdout_workload_20.csv';cohort.to_csv(dst,index=False)
    save(OUT/'holdout_preregistration.json',dict(source=str(src),source_sha256=sha(src),
        selection='same round-robin bert,cifar10,deepspeech2; original within-class order; skip exhausted; coarrival0; no outcome selection',
        indices=indices,records=cohort.to_dict('records'),csv=str(dst),sha256=sha(dst),
        parameters='same R0, E-legacy, alpha0/.5, topology, Q300/D250, seed_profiles(1,8), 600 rounds',
        role='independent of the workload-1 diagnostic states; not claimed never used by any historical research'))
    print('RESTORED',len(verified),'files; holdout frozen',len(cohort),sha(dst),flush=True)
if __name__=='__main__':main()
