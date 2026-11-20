"""At fixed reference quotas, alpha changes no execution physics for Sia-P."""
import gzip,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
def chosen(d):
    return {j:{k:c[k] for k in ['config','placement','cpu_per_gpu','ram_gb_per_gpu','rho','raw_goodput','effective_goodput'] if k in c} for j,c in d['result']['selected'].items()}
def main():
    paths=[OUT/f'holdout20_{r}_sia_p' for r in ['neutral','medium']];count=0;decisions=0
    with gzip.open(paths[0]/'rounds.jsonl.gz','rt',encoding='utf-8') as a,gzip.open(paths[1]/'rounds.jsonl.gz','rt',encoding='utf-8') as b:
        import itertools
        for left,right in itertools.zip_longest(a,b):
            assert left is not None and right is not None
            x=json.loads(left);y=json.loads(right);count+=1
            assert chosen(x['decision_for_next_interval'])==chosen(y['decision_for_next_interval'])
            assert x['job_end_state']==y['job_end_state'];assert len(x['execution'])==len(y['execution'])
            for u,v in zip(x['execution'],y['execution']):
                for k in ['job_id','progress_before','progress_after','rho','training_seconds','waiting_seconds','checkpoint_delay_consumed','allocated_seconds','completion_time']:
                    assert u[k]==v[k],(count,k)
            decisions+=1
    s=[json.loads((p/'summary.json').read_text()) for p in paths]
    assert s[0]['jobs']==s[1]['jobs']
    with (OUT/'proportional_response_identity.json').open('x') as f:json.dump(dict(valid=True,rounds=count,job_completion_restart_and_timing_identical=True,decision_geometry_and_rates_identical=True,
        meaning='SIA-P at per-GPU3CPU/62.5GB reference has rho1 in both response settings; full observed trajectory identical'),f,indent=2)
    print('PROPORTIONAL_IDENTITY',count,'rounds',flush=True)
if __name__=='__main__':main()
