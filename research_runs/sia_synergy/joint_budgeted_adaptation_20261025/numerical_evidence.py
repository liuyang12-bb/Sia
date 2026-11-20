"""Independent proof of captured bad solver vectors, no new solve or simulation."""
import gzip,json,math,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;A=OUT/'attempt3'
def main():
    events=[]
    for n in [8,20]:
        for response in ['neutral','medium']:
            cid=f'n{n}_{response}_joint_g';last=None
            with gzip.open(A/cid/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);d=r['decision_for_next_interval']
                    if last==d['round']:continue
                    last=d['round']
                    for ev in d['result']['audit'].get('numerical_solver_retries',[]):
                        model=ev['model']
                        def violation(x):
                            x=np.asarray(x);v=max(float(np.max(-x)),float(np.max(x-1)),float(np.max(np.abs(x-np.round(x)))))
                            for c in model['constraints']:
                                a=np.asarray(c['A'])@x;v=max(v,float(np.max(np.asarray(c['lb'])-a)),float(np.max(a-np.asarray(c['ub']))))
                            return v
                        old=violation(ev['original_x']);new=violation(ev['retry_x']);rounded=violation(np.round(ev['retry_x']))
                        assert old>1e-7 and max(new,rounded)<=1e-7
                        events.append(dict(cell=cid,time=r['interval_end'],model_sha256=ev['model_fingerprint'],
                            original_status=ev['original_status'],original_message=ev['original_message'],original_max_violation=old,
                            broken_rows=ev['original_validation']['violations'],retry_status=ev['retry_status'],retry_message=ev['retry_message'],
                            retry_max_violation=new,retry_rounded_max_violation=rounded))
    with (OUT/'numerical_evidence.json').open('x',encoding='utf-8') as f:json.dump(dict(valid=True,count=len(events),events=events),f,indent=2)
    for ev in events:print(ev['cell'],ev['time'],ev['original_max_violation'],ev['broken_rows'][:2],flush=True)
if __name__=='__main__':main()
