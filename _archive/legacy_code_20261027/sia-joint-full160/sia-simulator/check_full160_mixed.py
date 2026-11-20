"""Fixed 3-job / 30-round gate, all capacities and methods; no tuning."""
import copy,json,os,sys,traceback
from pathlib import Path
import pandas as pd
from joint_prototype.full160 import make_cluster

def main():
    root=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    cfg=json.loads((out.parent/'inputs/config.json').read_text())
    # Stress arrivals while long checkpoint is in flight. No work/epoch edits.
    rows=pd.DataFrame([dict(name='imagenet-lock',time=0,application='imagenet',num_replicas=4,batch_size=1600),
        dict(name='bert-arrival',time=61,application='bert',num_replicas=4,batch_size=384),
        dict(name='cifar10-arrival',time=121,application='cifar10',num_replicas=4,batch_size=2048)])
    results=[]
    for capacity in ('main','half','double'):
        for method in ('sia_p','sia_tune','joint_p','joint_o'):
            c=make_cluster(cfg,rows,root/'research_runs/sia_synergy/stage2b_neutral/g0_verified/registry.json',
                root/'research_sources/Synergy/simulator/models',method,'medium',capacity,True)
            records=[]
            try:
                for i in range(30):
                    if c.all_complete():break
                    c.step(60);records.append(copy.deepcopy(c.joint_round_logs[-1]))
                result=dict(status='PASS',capacity=capacity,method=method,rounds=len(records),
                    progress={j.name:float(j.progress) for j in c.jobs})
            except Exception as e:
                result=dict(status='FAIL',capacity=capacity,method=method,rounds=len(records),
                    time=c.current_time,error=repr(e),traceback=traceback.format_exc(),failure=c.policy.last_failure,
                    allocations=c.allocations,resources=c.policy.current_resources)
            with (out/f'{capacity}_{method}.json').open('x') as f:json.dump(dict(result=result,records=records),f,default=lambda v:v.item() if hasattr(v,'item') else str(v))
            results.append(result)
            if result['status']=='FAIL':break
        if results[-1]['status']=='FAIL':break
    with (out/'summary.json').open('x') as f:json.dump(results,f,indent=2)
    print(json.dumps(results));assert all(r['status']=='PASS' for r in results)
if __name__=='__main__':main()
