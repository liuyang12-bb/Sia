"""Read only completed service controls; no solver or training calls."""
import gzip,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'
CANON=json.loads((OUT/'canonical_result_paths.json').read_text())

def frames(cid):
    with gzip.open(Path(CANON['paths'][cid])/'rounds.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            selected=r['decision_for_next_interval']['result']['selected']
            allocation={j:{k:c[k] for k in ('config','placement','cpu_per_gpu','ram_gb_per_gpu','rho')} for j,c in selected.items()}
            # Model predictions can change without changing physical decisions;
            # compare GPU geometry, quota and executed training independently.
            for v in allocation.values():
                v['config']={k:v['config'][k] for k in ('config_id','gpu_type','gpus','num_nodes')}
            execution={e['job_id']:{k:e[k] for k in ('interval_start','interval_end','waiting_seconds','checkpoint_delay_consumed','training_seconds','progress_delta','completion_time','rho')} for e in r['execution']}
            yield allocation,execution

def main():
    a=list(frames('dual_neutral_A00'));b=list(frames('dual_neutral_A01'))
    mismatch=[i+1 for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    ans=dict(a00_a01_physical_allocation_and_execution_identical=len(a)==len(b) and not mismatch,rounds=[len(a),len(b)],mismatching_rounds=mismatch[:50],
       meaning='No effect on realized physical trajectory in this workload is not a proof that continuity never helps. Requeue changes ledger but objective/tie/eligibility may select the same incumbents.',
       relevant_source=['clean_ablation.py:291 service_expired','clean_ablation.py:170 FIFO head constraint','clean_ablation.py:185 common optimization','clean_ablation.py:300 shared_eligible'])
    with (OUT/'service_trajectory_diagnostic.json').open('x',encoding='utf8') as f:json.dump(ans,f,indent=2,ensure_ascii=False)
    print(json.dumps(ans,ensure_ascii=False))
if __name__=='__main__':main()
