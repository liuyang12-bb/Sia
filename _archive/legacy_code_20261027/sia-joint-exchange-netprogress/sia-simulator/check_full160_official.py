"""Separate-process exact official / adapted comparison; fixed 100-round cap."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def probe(a):
    sys.path.insert(0,str(a.sim));os.chdir(a.sim)
    import pandas as pd
    import simulator_config as sc
    sc.num_parallel_jobs=1;sc.num_parallel_traces=1;sc.sia_solver_verbose=False;sc.preserve_ckpt_allocs=True
    rows=pd.DataFrame([dict(name='cifar10-test',time=17,application='cifar10',num_replicas=1,batch_size=128),
                       dict(name='bert-test',time=130,application='bert',num_replicas=1,batch_size=384)])
    cfg=json.loads(a.config.read_text());root=a.sim.parents[1]
    if a.kind=='official':
        from sia import SiaPolicy
        from cluster import Cluster
        c0=cfg['cluster'];s=cfg['simulation']
        p=SiaPolicy(p_fairness=s['p_fairness'],restart_penalty=30,lambda_a=s['lambda_a'],lambda_n=s['lambda_n'],
            project_throughputs=s['project_throughputs'],share_max_replicas=s['share_max_replicas'],sched_interval=60,enable_bsz_tuning=True)
        p.populate_valid_configs(c0['nnodes'],c0['ngpus_per_node'],c0['max_physical_nodes'])
        c=Cluster(rows,p,c0['nnodes'],c0['ngpus_per_node'],c0['max_physical_nodes'])
        for j in c.jobs:j.seed_profiles(1,8)
    else:
        from joint_prototype.full160 import make_cluster
        c=make_cluster(cfg,rows,root/'research_runs/sia_synergy/stage2b_neutral/g0_verified/registry.json',
            root/'research_sources/Synergy/simulator/models','sia_p','neutral','main',True)
    records=[]
    for _ in range(100):
        if c.all_complete():break
        c.step(60)
        records.append(dict(time=c.current_time,allocations=c.allocations,jobs={j.name:dict(progress=j.progress,
            epoch=j.epoch,training=j.execution_time,delay=j.rescale_time,completion=j.completion_time,restarts=j.num_restarts,
            used=j.used_gpu_seconds,wasted=j.wasted_gpu_seconds,placement=j.placement,cluster=j.current_cluster) for j in c.jobs}))
    with a.out.open('x') as f:json.dump(records,f,default=lambda v:v.item())

def main():
    p=argparse.ArgumentParser();p.add_argument('--sim',type=Path);p.add_argument('--kind');p.add_argument('--config',type=Path);p.add_argument('--out',type=Path);a=p.parse_args();probe(a)
if __name__=='__main__':main()
