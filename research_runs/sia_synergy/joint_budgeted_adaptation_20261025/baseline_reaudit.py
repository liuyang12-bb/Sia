"""Replay immutable old logs with stronger audit; compare v1/v2 Neutral trajectory."""
import json,gzip,hashlib,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;A=OUT/'attempt2'
p=json.loads((A/'preregistration.json').read_text());OLD=Path(p['baseline_directory'])
module=importlib.util.spec_from_file_location('new_execution_audit_standalone',ROOT/'sia-joint-budgeted-adaptation/sia-simulator/joint_prototype/execution_audit.py')
m=importlib.util.module_from_spec(module);module.loader.exec_module(m)
def main():
    counts={}
    for n in [8,20]:
        for response in ['neutral','medium']:
            for method in ['sia_p','sia_aux','joint_r']:
                cid=f'n{n}_{response}_{method}';count=0
                with gzip.open(OLD/cid/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
                    for line in f:
                        r=json.loads(line)
                        for e in r['execution']:assert m.audit_execution(e)['valid'];count+=1
                counts[cid]=count
    def signature(folder):
        rows=[]
        with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                rows.append(dict(time=r['interval_end'],
                    execution=[{k:e[k] for k in ['job_id','progress_before','progress_after','rho','checkpoint_delay_consumed','training_seconds','completion_time','resource_allocation']} for e in r['execution']],
                    allocation={j:{k:v[k] for k in ['config','placement','cpu_per_gpu','ram_gb_per_gpu','rho']} for j,v in r['decision_for_next_interval']['result']['selected'].items()},
                    jobs=r['job_end_state']))
        return rows
    before=signature(OUT/'n8_neutral_joint_g');after=signature(A/'n8_neutral_joint_g')
    assert before==after,'AUDIT_CORRECTION_CHANGED_NEUTRAL_TRAJECTORY'
    for f,h in p['baseline_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()==h
    with (OUT/'baseline_reaudit.json').open('x',encoding='utf-8') as f:json.dump(dict(valid=True,old_execution_counts=counts,
        neutral_v1_v2_full_trajectory_identical=True,neutral_rounds=len(before),baseline_files_unchanged=len(p['baseline_hashes'])),f,indent=2)
    print('OLD_LOGS_REAUDITED',sum(counts.values()),'events; Neutral v1/v2 identical')
if __name__=='__main__':main()
