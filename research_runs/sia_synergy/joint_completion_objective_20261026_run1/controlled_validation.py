"""Fixed-state 2-job execution validation. No complete scheduling experiment."""
import copy,hashlib,itertools,json,math,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SRC=ROOT/'sia-joint-completion-time/sia-simulator'
sys.path.insert(0,str(SRC))
from applications import APPLICATIONS
from job import Job
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.joint_profile import JobSpec,GpuConfig
from joint_prototype.completion_time import predict_job,completion_cost,declare_work

def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)

def main():
    import psutil
    if psutil.virtual_memory().available<2*2**30:raise RuntimeError('MEMORY_GATE')
    if psutil.disk_usage(str(OUT)).free<5*2**30:raise RuntimeError('DISK_GATE')
    # Before constructing or stepping any Job, freeze the entire observation.
    save('controlled_preregistration.json',dict(
        jobs=['cifar10-1','deepspeech2-9'],source='official workload1 retained batch4096/320, demand12/8, full work',
        shared_warmup_seconds=240,observation_seconds=600,interval=60,
        fixed_gpu='AWS1 per job on 2GPU/12CPU/250GB',ram_per_job=125,
        fixed_batch_for_all=True,seed_profiles=[1,8],response='R0/E-legacy Medium alpha=.5',
        comparisons=['proportional6+6','previous_grid_witness8+4','minimum_current_ETA_clean_feasible_grid'],
        opportunity_definition='sum estimated remaining completion seconds, both donors and receivers counted',
        no_new_profile_points=True,no_GPU_changes=True,no_window_extension=True,
        full_performance_gate='NOT_PASSED: online GPU coupling not established; full runs not authorized by this gate',
        source_sha256=hashlib.sha256((SRC/'joint_prototype/completion_time.py').read_bytes()).hexdigest()))
    start=time.perf_counter()
    mapping={'cifar10-1':'cifar10','deepspeech2-9':'deepspeech2'}
    oracle=R0Oracle(OUT/'inputs/registry.json',OUT/'inputs/models',mapping,'medium')
    registry=oracle.registry
    warm={};specs={};cfgs={};queries={};pred={};work={}
    for jid,app in mapping.items():
        j=Job(jid,{t:APPLICATIONS[t][app] for t in APPLICATIONS},0,
              target_num_replicas=12 if app=='cifar10' else 8,
              target_batch_size=4096 if app=='cifar10' else 320)
        j.enable_bsz_tuning=False;j.enable_joint_accounting=True
        j.seed_profiles(1,8);j.migrate('aws',(1,))
        ref=registry[app+'/aws:1:1']
        cfgs[jid]=g=GpuConfig('aws:1:1','aws',1,1,1.,ref['phi_c'],ref['phi_m'])
        specs[jid]=s=JobSpec(jid,ref['template'],(g,))
        queries[jid]={cpu:oracle.query(s,g,cpu,125.,(6.,125.)) for cpu in [1.,2.,3.,4.,6.,8.,9.,12.,24.]}
        j.goodput_multiplier=queries[jid][6.].rho
        for _ in range(4):j.step(60)
        assert j.rescale_time==0 and j.completion_time is None
        warm[jid]=j;pred[jid]=predict_job(j,g)
        work[jid]=declare_work(j)['total_progress']-j.progress
    ids=list(mapping);feasible=[]
    for cpus in itertools.product(queries[ids[0]],queries[ids[1]]):
        if sum(cpus)>12+1e-9:continue
        q=[queries[j][cpu] for j,cpu in zip(ids,cpus)]
        if any(v.source_coordinate_clipped or v.baseline_coordinate_clipped for v in q):continue
        costs=[completion_cost(work[j],pred[j]['raw_rate'],v.rho,0.) for j,v in zip(ids,q)]
        feasible.append(dict(cpu=list(cpus),predicted_remaining_times=costs,sum_etc=sum(costs)))
    assert feasible
    best=min(feasible,key=lambda v:(v['sum_etc'],sum(v['cpu']),v['cpu']))
    plans={'proportional':[6.,6.],'prior_exchange':[8.,4.],'completion_objective':best['cpu']}
    result={}
    for label,cpus in plans.items():
        result[label]={}
        for jid,cpu in zip(ids,cpus):
            j=copy.deepcopy(warm[jid]);initial_progress=j.progress
            initial_restarts=j.num_restarts;q=queries[jid][cpu]
            assert not q.source_coordinate_clipped and not q.baseline_coordinate_clipped
            j.goodput_multiplier=q.rho;steps=[]
            for i in range(10):
                before=j.progress;j.step(60)
                assert j.progress>=before
                seg=copy.deepcopy(j.joint_step_segments)
                for v in seg:
                    assert math.isclose(v['raw_rate_before_rho']*q.rho,v['effective_rate'],rel_tol=1e-12)
                    assert abs(v['progress_delta']-v['effective_rate']*v['seconds'])<=.5*v['effective_rate']+1e-7
                steps.append(dict(step=i+1,progress=j.progress,epoch=j.epoch,segments=seg))
            assert j.num_restarts==initial_restarts
            result[label][jid]=dict(cpu=cpu,ram=125.,rho=q.rho,
                predicted_raw=pred[jid]['raw_rate'],predicted_remaining_time=completion_cost(work[jid],pred[jid]['raw_rate'],q.rho,0.),
                observed_first_raw=steps[0]['segments'][0]['raw_rate_before_rho'],
                progress_delta=j.progress-initial_progress,
                completion=j.completion_time,extra_restart=j.num_restarts-initial_restarts,steps=steps)
    payload=dict(plans=plans,all_clean_feasible_pairs=feasible,
        minimum_estimated_completion=best,current_model=pred,remaining_declared_work=work,
        results=result,capacity_violations=0,response_violations=0,
        note='600-second common-state training, no JCT comparison or projected completion substituted for an event',
        wall_seconds=time.perf_counter()-start)
    save('controlled_validation.json',payload)
    print(json.dumps(dict(best=best,plans=plans,
        totals={k:sum(v['predicted_remaining_time'] for v in values.values()) for k,values in result.items()},
        model_raw_errors={j:pred[j]['raw_rate']/result['proportional'][j]['observed_first_raw']-1 for j in ids}),indent=2))
if __name__=='__main__':main()
