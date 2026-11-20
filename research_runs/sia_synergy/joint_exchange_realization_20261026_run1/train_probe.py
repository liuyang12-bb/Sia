"""Controlled two real Job.step paths; common warm state and fixed GPU/batch."""
import copy,gzip,hashlib,json,math,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;PARENT=OUT.parent/'joint_opportunity_experiments_20261026_run1';SOURCE=PARENT/'source'
sys.path.insert(0,str(SOURCE))
from joint_prototype.joint_profile import JobSpec,GpuConfig
from joint_prototype.r0_liveness import R0Oracle
from job import Job
from applications import APPLICATIONS
from joint_prototype.exchange_netprogress import fair_utility
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def main():
 import psutil
 if psutil.virtual_memory().available<2*2**30:raise RuntimeError('MEMORY_GATE')
 if psutil.disk_usage(str(OUT)).free<5*2**30:raise RuntimeError('DISK_GATE')
 save(OUT/'training_preregistration.json',dict(jobs=['cifar10-1','deepspeech2-9'],source='official workload-1 original batch4096 and320',gpu='AWS1 each, same node2GPU12CPU250GB',warmup_wall_seconds=240,observation_wall_seconds=600,step_seconds=60,allocations={'proportional':[6,6],'exchange':[8,4]},ram_per_job=125,response='R0/E-legacy Medium alpha0.5',batch_semantics='original target batch fixed for BOTH clones, disable adaptive batch tuning only in this controlled diagnostic',common_state='deepcopy after shared warmup; no GPU movement or selection; CPU-only changes have zero extra delay in current execution model',forbidden='no JCT ranking, no window extension, no parameter changes based on results'))
 reg={v['key']:v for v in json.loads((PARENT/'inputs/registry.json').read_text())};apps={'cifar10-1':'cifar10','deepspeech2-9':'deepspeech2'}
 oracle=R0Oracle(PARENT/'inputs/registry.json',PARENT/'inputs/models',apps,'medium');warm={};queries={}
 for j,app in apps.items():
  job=Job(j,{t:APPLICATIONS[t][app] for t in APPLICATIONS},0,target_num_replicas=12 if app=='cifar10' else 8,target_batch_size=4096 if app=='cifar10' else 320)
  job.enable_bsz_tuning=False;job.enable_joint_accounting=True;job.first_productive_training_time=None
  job.seed_profiles(1,8);job.migrate('aws',(1,))
  v=reg[app+'/aws:1:1'];g=GpuConfig('aws:1:1','aws',1,1,1,v['phi_c'],v['phi_m']);sp=JobSpec(j,v['template'],(g,))
  queries[j]={cpu:oracle.query(sp,g,cpu,125,(6,125)) for cpu in [4,6,8]}
  job.goodput_multiplier=queries[j][6].rho
  for i in range(4):job.step(60)
  assert job.rescale_time==0 and job.completion_time is None
  warm[j]=job
 result={};states={}
 for j,job in warm.items():
  fn=job.get_speedup_fn('aws');raw=float(fn(1,1))
  states[j]=dict(time=job.current_time,progress=float(job.progress),epoch=job.epoch,atomic_bsz=job.atomic_bsz,accum_steps=job.accum_steps,profile_points=len(job.profiles['aws']),cold=False,online_speedup=raw)
 for label,cpus in [('proportional',[6,6]),('exchange',[8,4])]:
  result[label]={}
  for (j,original),cpu in zip(warm.items(),cpus):
   job=copy.deepcopy(original);startp=job.progress;starte=job.execution_time;startrestart=job.num_restarts
   job.goodput_multiplier=queries[j][cpu].rho;steps=[]
   for i in range(10):
    before=job.progress;job.step(60);segs=copy.deepcopy(job.joint_step_segments)
    for seg in segs:
     assert math.isclose(seg['effective_rate'],seg['raw_rate_before_rho']*seg['rho'],rel_tol=1e-12)
     residual=seg['progress_delta']-seg['effective_rate']*seg['seconds']
     assert abs(residual)<=.5*seg['effective_rate']+1e-7
    assert job.progress>=before
    steps.append(dict(step=i+1,time=job.current_time,progress=float(job.progress),epoch=job.epoch,atomic_bsz=job.atomic_bsz,accum_steps=job.accum_steps,segments=segs))
   assert job.num_restarts==startrestart and job.placement==(1,) and job.current_cluster=='aws'
   result[label][j]=dict(cpu=cpu,ram=125,rho=job.goodput_multiplier,clipped=queries[j][cpu].source_coordinate_clipped or queries[j][cpu].baseline_coordinate_clipped,progress_delta=float(job.progress-startp),training_seconds=job.execution_time-starte,completion=job.completion_time,additional_restarts=job.num_restarts-startrestart,steps=steps)
 gains={j:result['exchange'][j]['progress_delta']/result['proportional'][j]['progress_delta']-1 for j in warm}
 modelgains={j:queries[j][cpu].rho/queries[j][6].rho-1 for j,cpu in zip(warm,[8,4])}
 scored={label:sum(fair_utility(states[j]['online_speedup']*queries[j][cpu].rho) for j,cpu in zip(warm,cpus)) for label,cpus in [('proportional',[6,6]),('exchange',[8,4])]}
 payload=dict(common_warm_state=states,results=result,observed_progress_gain=gains,predicted_rho_gain=modelgains,equal_job_normalized_progress_gain=sum(gains.values())/2,fair_utility_after_warmup=scored,training_work_units='job-specific Sia progress units; not summed across different apps; equal-job index averages relative progress changes',no_clipping=not any(v['clipped'] for vs in result.values() for v in vs.values()),no_extra_restart=True)
 save(OUT/'training_probe_result.json',payload)
 print(json.dumps({k:v for k,v in payload.items() if k not in ['results','common_warm_state']},indent=2))
if __name__=='__main__':main()
