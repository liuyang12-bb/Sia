"""Six structural states fixed before bridge outcomes, offline MILP only."""
import copy,gzip,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import LinearConstraint
OUT=Path(__file__).resolve().parent;p=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(Path(p['tree'])/'sia-simulator'))
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec,ProfileQuery
from joint_prototype.joint_scheduler import JointColumn,Placement
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.neutral_liveness import FactorScheduler,geometry
from joint_prototype.bundle_adaptation import BundleAdmission
def job(v):
    v=copy.deepcopy(v);v['configs']=tuple(GpuConfig(**c) for c in v['configs']);v['current_nodes']=tuple(v['current_nodes']);v['current_placement']=tuple(map(tuple,v['current_placement']))
    return JobSpec(**v)
def column(v):
    v=copy.deepcopy(v);v['config']=GpuConfig(**v['config']);v['placement']=Placement(tuple(map(tuple,v['placement']['node_gpus'])))
    v['query']=ProfileQuery(**v['query']);v['node_usage']=tuple(map(tuple,v['node_usage']));return JointColumn(**v)
def main():
    results=[];config=json.loads(Path(p['config']).read_text())
    for dataset in ['original20','holdout20']:
        folder=(Path(next(c['folder'] for c in p['reference_cells'] if c['dataset']==dataset and c['response']=='medium' and c['method']=='joint_g'))
                if dataset=='original20' else OUT/'holdout20_medium_joint_g')
        criteria=['FIRST_DECISION','FIRST_SERVICE_EXPIRY','FIRST_BERT_SMALL_GPU_PRODUCTIVE_UNLOCKED'];found={}
        with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);d=r['decision_for_next_interval'];pb=d['probe'];b=pb['l2_before']
                expired=any(b['productive'].get(j,0)-v['productive_base']>=300 for j,v in b['services'].items() if j in b['productive'])
                small=any(s['job_id'].startswith('bert-') and s['current_config_id'] in ('rtx:1:1','aws:1:1','dgx-ext:1:1') and not s['checkpoint_locked'] and b['productive'].get(s['job_id'],0)>0 for s in pb['specs'])
                for crit,match in zip(criteria,[True,expired,small]):
                    if match and crit not in found:found[crit]=r
                if len(found)==3:break
        for crit in criteria:
            if crit not in found:results.append(dict(dataset=dataset,criterion=crit,status='NO_MATCH'));continue
            r=found[crit];d=r['decision_for_next_interval'];pb=d['probe'];specs=[job(v) for v in pb['specs']]
            oracle=R0Oracle(p['registry'],p['models'],{s.job_id:s.job_id.split('-')[0] for s in specs},'medium')
            s=FactorScheduler([NodeSpec(t+':0',t,g,g*3,g*62.5) for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]],specs,oracle,
                config['response']['aux_cpu_points'],config['response']['aux_ram_points'],allocation_bonus=0,minimum_cpu_per_gpu=1,minimum_ram_gb_per_gpu=20.83,verify_exact=False,candidate_policy='b3')
            cs=[column(v) for v in pb['joint_columns']];matrix,upper=s._constraint_matrix(cs,('gpu','cpu','ram'))
            locks={j.job_id for j in specs if s.lock_active(j)};selected=set(d['result']['selected'])
            guard=BundleAdmission(300,250,60);guard._diagnostic_job_order=[j.job_id for j in specs];guard.numerical_solver_retries=[]
            def solve(required,exact=False):
                lower=np.full(len(upper),-np.inf);ub=upper.copy()
                for i,j in enumerate(specs):
                    lower[i]=int(j.job_id in required)
                    if exact:ub[i]=int(j.job_id in required)
                res=guard._milp(-np.array([c.scheduling_utility for c in cs]),cs,[LinearConstraint(matrix,lower,ub)])
                assert res.status==0 and res.success
                chosen=[c for c,x in zip(cs,res.x) if x>.5];assert s.audit(chosen)['valid']
                return dict(objective=sum(c.scheduling_utility for c in chosen),count=len(chosen),
                    selected={c.job_id:dict(config=c.config.config_id,cpu=c.query.cpu_per_gpu,ram=c.query.ram_gb_per_gpu,rho=c.query.rho,utility=c.scheduling_utility) for c in chosen})
            free=solve(locks);fixed=solve(selected,True);gobjective=d['result']['scheduling_objective']
            assert free['objective']+1e-7>=fixed['objective']>=gobjective-1e-7
            results.append(dict(dataset=dataset,criterion=crit,time=r['interval_end'],status='SOLVED',locks=sorted(locks),
                g=dict(objective=gobjective,count=len(selected),selected=d['result']['selected']),free=free,fixed_g_job_set_without_service_filters=fixed,
                admission_objective_gap=free['objective']-fixed['objective'],candidate_filter_objective_gap=fixed['objective']-gobjective,
                note='Same logged online state, objective, complete raw grid and physical locks. Fixed-set relaxation isolates held-config filters; free relaxation also changes admission. Predicted utility, NOT JCT. Pareto-duplicate tie choice may differ from free runtime.'))
    with (OUT/'same_state_comparison.json').open('x') as f:json.dump(results,f,indent=2)
    print('OFFLINE_STRUCTURAL_STATES',len(results))
if __name__=='__main__':main()
