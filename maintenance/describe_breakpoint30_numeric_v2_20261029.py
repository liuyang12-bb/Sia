"""Descriptive execution-response accounting only, never a new experiment."""
import collections,functools,gzip,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt2'
sys.path.insert(0,str(OUT/'source/sia-simulator'))
from joint_prototype.joint_profile import load_synergy_template
def main():
 p=json.loads((OUT/'preregistration.json').read_text());reg={r['key']:r for r in json.loads((OUT/'source/inputs/registry.json').read_text())}
 matrices={t:load_synergy_template(OUT/'source/inputs/models'/t).matrix for t in ['alexnet_1.py','deepspeech_1.py']}
 coverage=json.loads((OUT/'offline_coverage.json').read_text());work={r['application']:r['profiles'][0]['work'] for r in coverage}
 @functools.lru_cache(maxsize=None)
 def rho(app,cid,c,m,alpha):
  r=reg[app+'/'+cid];matrix=matrices[r['template']]
  def f(x,y):return float(np.interp(x,[1,2,3,4,5,6,9,12,24],[np.interp(y,[20.83,62.5,125,187.5,250],v) for v in matrix]))
  return (f(c/r['phi_c'],m/r['phi_m'])/f(3/r['phi_c'],62.5/r['phi_m']))**alpha
 data={}
 for mix,cap,response,method in p['cells']:
  name='_'.join((mix,cap,response,method));cpu,ram=p['capacities'][cap];alpha=p['response_exponents'][response]
  out=collections.Counter();apps=collections.defaultdict(collections.Counter);clips=collections.Counter()
  with gzip.open(OUT/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    r=json.loads(line);by_node=collections.defaultdict(list)
    for e in r['execution']:
     app=e['job_id'].rsplit('-',1)[0];a=e['resource_allocation']
     if not a:continue
     cid=a['config_id'];c,m=a['cpu_per_gpu'],a['ram_gb_per_gpu'];r00=rho(app,cid,cpu/2,ram/2,alpha)
     r10=rho(app,cid,c,ram/2,alpha);r01=rho(app,cid,cpu/2,m,alpha);r11=rho(app,cid,c,m,alpha)
     for z in e['training_segments']:
      assert math.isclose(z['rho'],r11,rel_tol=1e-12)
      w=z['raw_rate_before_rho']*z['seconds']/work[app];delta=w*(r11-r00)
      out['fixed_geometry_progress_fraction_delta']+=delta;apps[app]['fixed_geometry_progress_fraction_delta']+=delta
      out['receiver_progress_fraction']+=max(0,delta);out['donor_loss_fraction']+=max(0,-delta)
      out['cpu_local_shapley_fraction']+=.5*w*((r10-r00)+(r11-r01));out['ram_local_shapley_fraction']+=.5*w*((r01-r00)+(r11-r10))
      out['productive_job_seconds']+=z['seconds'];out['faster_than_proportional_job_seconds']+=z['seconds']*int(r11>r00+1e-10);out['slower_than_proportional_job_seconds']+=z['seconds']*int(r11<r00-1e-10)
      apps[app]['productive_job_seconds']+=z['seconds']
      if app=='cifar10' and cid=='dgx-ext:1:1':out['CF_DGX_single_productive_seconds']+=z['seconds']
      for tag,key in [('numerator','source_coordinates'),('denominator','baseline_source_coordinates')]:
       q=a[key]
       for axis,field,lo,hi in [('cpu','requested_cpu',1.,24.),('ram','requested_ram_gb',20.83,250.)]:
        v=q[field]
        clips[tag+'_'+axis+'_exact_requested']+=z['seconds']*int(v<lo or v>hi)
        clips[tag+'_'+axis+'_beyond_1e12_tolerance']+=z['seconds']*int(v<lo-1e-12 or v>hi+1e-12)
      for node,g in a['placement'].items():
       if g==1:by_node[node].append((app,z['start_time'],z['start_time']+z['seconds']))
    for spans in by_node.values():
     for i,(a,s,t) in enumerate(spans):
      for b,u,v in spans[i+1:]:
       if a!=b:out['heterogeneous_application_coproductive_node_seconds']+=max(0,min(t,v)-max(s,u))
  assert math.isclose(out['cpu_local_shapley_fraction']+out['ram_local_shapley_fraction'],out['fixed_geometry_progress_fraction_delta'],abs_tol=1e-9)
  data[name]=dict(totals=dict(out),applications={a:dict(v) for a,v in apps.items()},
   actual_requested_coordinate_clipping_seconds=dict(clips),
   interpretation='Fixed executed GPU/batch/raw rate/time compare response versus proportional. This is response accounting, NOT a feasible counterfactual schedule or JCT causal decomposition. Exact endpoint clipping and floating roundoff exposure distinguished; no model/tolerance was changed in execution.')
  print('RESPONSE_ACCOUNTED',name,flush=True)
 with (OUT/'analysis/execution_response_accounting.json').open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
if __name__=='__main__':main()
