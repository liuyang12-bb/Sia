import collections,copy,gzip,importlib.util,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1'
sys.path[:0]=[str(OUT/'source'),str(OUT/'source/sia-simulator')]
from run_stage2c_cell import independent_rho
from breakpoint_study import build
from resource_study import CAPACITIES
def main():
 spec=importlib.util.spec_from_file_location('runner',OUT/'run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);runner.setup();runner.resources()
 CAPACITIES['RNEW']=(6.,125.)
 c=build('CC_FULL','RNEW','medium',OUT/'inputs/BASE_workload.csv')
 base=[next(j for j in c.jobs if j.name==n) for n in ['cifar10-0','bert-28']]
 witness=json.loads((OUT/'strict_witness.json').read_text());result={}
 for mode,pair in witness['pairs'].items():
  runner.resources();jobs=copy.deepcopy(base);totals=collections.defaultdict(float);per={}
  with gzip.open(OUT/f'fixed_{mode}.jsonl.gz','xt',encoding='utf8') as f:
   for j in jobs:
    q=pair[j.app_name];assert q['supported_including_floating_endpoint_tolerance']
    j.current_time=0.;j.enable_joint_accounting=True;j.migrate('aws',(1,));j.goodput_multiplier=independent_rho(c.policy.oracle,j.name,'aws:1:1',q['cpu'],q['ram']);assert math.isclose(j.goodput_multiplier,q['rho'],rel_tol=1e-12)
    j.target_batch_size=next(x.target_batch_size for x in base if x.name==j.name)
   assert pair['cpu_sum']<=6+1e-9 and pair['ram_sum']<=125+1e-9
   for i in range(6000):
    if all(j.completion_time is not None for j in jobs):break
    for j in jobs:
     if j.completion_time is not None:continue
     start=j.current_time;delay=min(j.rescale_time,60.);before=j.progress;j.step(60)
     segments=j.joint_step_segments
     for z in segments:
      assert math.isclose(z['effective_rate'],z['raw_rate_before_rho']*j.goodput_multiplier,rel_tol=1e-12)
      assert math.isclose(z['progress_delta'],z['effective_rate']*z['seconds']+z['rounding_progress_residual'],abs_tol=1e-7)
      assert z['start_time']>=start+delay-1e-8
     event=dict(round=i,job=j.name,progress_before=before,progress_after=j.progress,delay=delay,segments=segments,completion=j.completion_time)
     f.write(json.dumps(event)+'\n')
     dt=(j.completion_time if j.completion_time is not None else j.current_time)-start
     totals[j.name+':training']+=sum(s['seconds'] for s in segments);totals[j.name+':delay']+=delay
     totals['gpu_hours']+=dt/3600;totals['cpu_hours']+=dt*pair[j.app_name]['cpu']/3600;totals['ram_hours']+=dt*pair[j.app_name]['ram']/3600
   for j in jobs:
    per[j.name]=dict(completed=j.completion_time is not None,jct=j.completion_time,training=totals[j.name+':training'],checkpoint=totals[j.name+':delay'],rho=j.goodput_multiplier,restarts=j.num_restarts)
   result[mode]=dict(jobs=per,mean_jct=sum(j['jct'] for j in per.values())/2 if all(j['completed'] for j in per.values()) else None,resource_hours=dict(totals),rounds=i)
 runner.save(OUT/'fixed_mechanism.json',result)
 assert all(x['mean_jct'] is not None for x in result.values())
 assert all(result[m]['jobs']['bert-28']['jct']==result['P']['jobs']['bert-28']['jct'] for m in result)
 assert result['FULL']['mean_jct']<result['CPU']['mean_jct']<result['P']['mean_jct']
 assert result['FULL']['mean_jct']<result['MEM']['mean_jct']<result['P']['mean_jct']
 print('FIXED_MECHANISM_PASSED',json.dumps(result),flush=True)
if __name__=='__main__':main()
