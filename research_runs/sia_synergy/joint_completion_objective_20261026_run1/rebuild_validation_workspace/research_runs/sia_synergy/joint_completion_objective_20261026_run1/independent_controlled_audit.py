"""Recompute the controlled validation from direct training segments only."""
import hashlib,json,math
from pathlib import Path
OUT=Path(__file__).resolve().parent
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6),(a,b)
def main():
    p=json.loads((OUT/'controlled_validation.json').read_text())
    sums={};plans={};records=0
    for label, jobs in p['results'].items():
        sums[label]=0.;plans[label]={}
        assert sum(v['cpu'] for v in jobs.values())<=12
        assert sum(v['ram'] for v in jobs.values())<=250
        for jid,v in jobs.items():
            seconds=0.;progress=0.
            for step in v['steps']:
                for seg in step['segments']:
                    records+=1
                    close(seg['effective_rate'],seg['raw_rate_before_rho']*v['rho'])
                    assert abs(seg['progress_delta']-seg['effective_rate']*seg['seconds'])<=.5*seg['effective_rate']+1e-7
                    assert seg['seconds']>=0 and seg['progress_delta']>=0
                    seconds+=seg['seconds'];progress+=seg['progress_delta']
            close(seconds,600.);close(progress,v['progress_delta'])
            assert v['extra_restart']==0
            # Diagnostic freezes the observed FIRST rate. This is NOT an
            # observed JCT and is not used to modify the trained scheduler.
            direct=v['steps'][0]['segments'][0]['raw_rate_before_rho']
            eta=p['remaining_declared_work'][jid]/(direct*v['rho'])
            sums[label]+=eta
            plans[label][jid]=dict(progress=progress,training=seconds,
                wait=0.,checkpoint=0.,extra_restart=0,completion=v['completion'],
                jct=None,first_direct_raw=direct,observed_rate_frozen_etc=eta,
                model_raw_relative_error=v['predicted_raw']/direct-1,
                gpu_hours=seconds/3600,cpu_unit_hours=v['cpu']*seconds/3600,
                ram_gb_hours=v['ram']*seconds/3600)
    perjob={j:plans['completion_objective'][j]['observed_rate_frozen_etc']-
             plans['proportional'][j]['observed_rate_frozen_etc'] for j in plans['proportional']}
    payload=dict(valid=True,segment_checks=records,capacity_violations=0,
        completion_events=sum(v['completion'] is not None for js in plans.values() for v in js.values()),
        observed_rate_frozen_etc_sums=sums,receiver_donor_etc_delta=perjob,
        estimated_sign_robust=False,warning='Frozen-rate ETC is NOT actual completion time. No JCT imputation.',
        actual_segment_results=plans,
        input_sha256=hashlib.sha256((OUT/'controlled_validation.json').read_bytes()).hexdigest().upper())
    with (OUT/'controlled_independent_audit.json').open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    print(json.dumps({k:v for k,v in payload.items() if k!='actual_segment_results'},indent=2))
if __name__=='__main__':main()
