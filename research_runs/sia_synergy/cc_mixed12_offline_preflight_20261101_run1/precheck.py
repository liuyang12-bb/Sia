"""Static reference-model resource envelope only; never advances training."""
import collections,copy,csv,gzip,hashlib,itertools,json,math,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1'
sys.dont_write_bytecode=True;sys.path[:0]=[str(T),str(T/'sia-simulator')]
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from job import Job
from cluster import Cluster
from applications import APPLICATIONS
from joint_prototype.completion_time import predict_job,declare_work
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.neutral_liveness import FactorScheduler
from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
from joint_prototype.physical_domain import restrict_single_node_domain
from residual_capacity import resource_points
from sia import SiaPolicy
from run_core80 import gate

def forbidden(*a,**k):raise AssertionError('TRAINING_SIMULATION_FORBIDDEN_IN_OFFLINE_PREFLIGHT')
Job.step=forbidden;Cluster.step=forbidden
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))
def supported(c):return not(c.query.source_coordinate_clipped or c.query.baseline_coordinate_clipped)
def prop(c):return abs(c.query.cpu_per_gpu-c.proportional_cpu_per_gpu)<1e-8 and abs(c.query.ram_gb_per_gpu-c.proportional_ram_per_gpu)<1e-8

def main():
 started=time.time();p=json.loads((O/'preregistration.json').read_text(encoding='utf8'));cfg=json.loads((O/'inputs/config.json').read_text(encoding='utf8'))
 for f,h in p['source_hashes'].items():assert sha(T/f)==h,f
 for f,h in p['input_hashes'].items():assert sha(f)==h,f
 save('resource_gate.json',gate(O));rows=p['workload'];rc=cfg['response'];cc=cfg['cluster']
 nodes=[NodeSpec(t+':0',t,cc['ngpus_per_node'][t],v['cpu_cores'],v['ram_gb']) for t,v in cc['capacity_scenarios']['main'].items()]
 pol=SiaPolicy(p_fairness=-.5,restart_penalty=30,lambda_a=.01,lambda_n=1.1,project_throughputs=False,share_max_replicas=False,sched_interval=60,enable_bsz_tuning=True)
 pol.populate_valid_configs(cc['nnodes'],cc['ngpus_per_node'],cc['max_physical_nodes']);restrict_single_node_domain(pol)
 domain=sorted((str(t),int(n),int(g)) for t,n,g in pol.idx_to_config_map.values())
 assert set(domain)=={(t,1,g) for t in ('aws','rtx') for g in (1,2,4)}
 oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{r['name']:r['application'] for r in rows},'medium')
 prototypes={};reference={};specs=[]
 for r in rows:
  app=r['application']
  if app not in prototypes:
   j=Job('offline_'+app,{t:APPLICATIONS[t][app] for t in ('aws','rtx')},0,target_num_replicas=int(r['num_replicas']),target_batch_size=int(r['batch_size']))
   j.seed_profiles(1,8);assert j.grad_params is None
   prototypes[app]=j
  cs=[]
  for typ,n,g in domain:
   cid=f'{typ}:{n}:{g}';ref=oracle.reference(r['name'],cid)
   cs.append(GpuConfig(cid,typ,g,n,1.,ref['phi_c'],ref['phi_m']))
  specs.append(JobSpec(r['name'],APPLICATION_TEMPLATES[app],tuple(cs),float(r['time']),source_row_index=r['source_row_index']))
  for state in p['reference_states']:
   j=copy.deepcopy(prototypes[app]);j.name=r['name'];j.target_num_replicas=int(r['num_replicas']);j.target_batch_size=int(r['batch_size'])
   epoch=int(state['fraction']*j.reference_app.max_epochs);j.epoch=epoch;j.progress=float(j.reference_app.get_progress(epoch))
   j.grad_params=tuple(float(x) for x in j.reference_app.get_grad_stats(j.reference_app.init_batch_size,epoch))
   work=declare_work(j)['total_progress']
   preds={c.config_id:dict(**predict_job(j,c),remaining=work-j.progress,total_work=work,delay=float(j.applications[c.gpu_type].rescale_time or 30)) for c in cs}
   assert all(x['raw_rate'] and x['raw_rate']>0 for x in preds.values())
   reference[state['id'],r['name']]=dict(epoch=epoch,progress=j.progress,gradient=j.grad_params,
      gradient_batch=j.reference_app.init_batch_size,predictions=preds,application=app,
      source='OFFLINE_SOURCE_GRADIENT_AND_SEEDED_GPU_PROFILE_REFERENCE_NOT_ONLINE_COLD_START')
 save('reference_models.json',{s:{r['name']:reference[s,r['name']] for r in rows} for s in ('E0','E25')})
 # Both waves' resource subproblems have identical application/order/remaining/raw
 # under the reference model; verify, never infer from convenient solver outputs.
 equality=[]
 for a,b in zip(rows[:6],rows[6:]):
  equal=all(reference[s,a['name']]==reference[s,b['name']] for s in ('E0','E25'))
  equality.append(dict(first=a['name'],second=b['name'],reference_equal=equal))
 assert all(x['reference_equal'] for x in equality)
 save('wave_reference_equivalence.json',dict(rows=equality,scope='reference-only; real online histories and arrival times are NOT equivalent',second_wave_solves_reused=True))
 # Construct exact current trusted candidate domain once; no optimizer use of raw=1.
 s=FactorScheduler(nodes,specs[:6],oracle,tuple(sorted(set(rc['aux_cpu_points'])|{8.})),tuple(rc['aux_ram_points']),
     verify_exact=False,enforce_proportional_progress_floor=False,minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,candidate_policy='b3')
 s.resource_point_provider=resource_points
 cols=s.build_joint_columns();coverage=[]
 for j in specs[:6]:
  for c in j.configs:
   own=[v for v in cols if v.job_id==j.job_id and v.config.config_id==c.config_id];assert own
   pp=[v for v in own if prop(v)];assert len(pp)==1
   q=pp[0].query
   coverage.append(dict(job=j.job_id,application=oracle.applications[j.job_id] if hasattr(oracle,'applications') else next(r['application'] for r in rows if r['name']==j.job_id),config=c.config_id,columns=len(own),fully_supported=sum(supported(x) for x in own),
     numerator_clipped=sum(x.query.source_coordinate_clipped for x in own),denominator_clipped=sum(x.query.baseline_coordinate_clipped for x in own),
     proportional=asdict(q),max_rho=max(v.query.rho for v in own),
     cpu_only_max_rho=max(v.query.rho for v in own if abs(v.query.ram_gb_per_gpu-pp[0].query.ram_gb_per_gpu)<1e-8),
     memory_only_max_rho=max(v.query.rho for v in own if abs(v.query.cpu_per_gpu-pp[0].query.cpu_per_gpu)<1e-8)))
 # Transfer coverage to second wave only after exact phi/template/config identity.
 for ja,jb in zip(specs[:6],specs[6:]):
  assert ja.configs==jb.configs and ja.template_file==jb.template_file
  for x in list(coverage):
   if x['job']==ja.job_id:
    y=copy.deepcopy(x);y['job']=jb.job_id;y['proportional']['job_id']=jb.job_id;y['coverage_reuse_exact_template_config']=ja.job_id;coverage.append(y)
 save('coverage.json',dict(job_configurations=len(coverage),gpu_domain=domain,candidate_columns_firstwave=len(cols),rows=coverage,
      source_scope='all72job/config pairs, identical response-domain reuse verified; raw=1 is enumeration placeholder only and NOT scoring/training'))
 print('COVERAGE',len(coverage),len(cols),flush=True)
 names=[j.job_id for j in specs[:6]];node_ids=[n.node_id for n in nodes]
 def values(c,state):
  x=reference[state,c.job_id]['predictions'][c.config.config_id];v=x['raw_rate']*c.query.rho
  return x,v,x['delay']+x['remaining']/v
 def slim(c,state):
  x,v,eta=values(c,state)
  return dict(job=c.job_id,config=c.config.config_id,cpu_per_gpu=c.query.cpu_per_gpu,ram_per_gpu=c.query.ram_gb_per_gpu,
      placement=c.placement.node_gpus,raw=x['raw_rate'],rho=c.query.rho,rate=v,remaining=x['remaining'],total_work=x['total_work'],startup=x['delay'],eta=eta,
      utility=1/eta,source_num_clipped=c.query.source_coordinate_clipped,source_den_clipped=c.query.baseline_coordinate_clipped,
      coordinates=dict(numerator=[c.query.requested_source_cpu_coordinate,c.query.requested_source_ram_coordinate],denominator=[c.query.requested_baseline_source_cpu_coordinate,c.query.requested_baseline_source_ram_coordinate]),
      atomic_batch=x['atomic_bsz'],accumulation=x['accum_steps'])
 def solve(pool,state,label):
  # Exact existing primary resource objective, mandatory common service set.
  # This offline optimizer does not impersonate full-controller admission/ties.
  rr=[];jj=[];vv=[]
  for k,c in enumerate(pool):
   rr.append(names.index(c.job_id));jj.append(k);vv.append(1.)
   for nid,g,cpu,ram in c.node_usage:
    n=node_ids.index(nid)
    for dim,val in enumerate((g,cpu,ram)):rr.append(len(names)+3*n+dim);jj.append(k);vv.append(val)
  mat=coo_matrix((vv,(rr,jj)),shape=(len(names)+3*len(nodes),len(pool))).tocsc()
  upper=np.array([1.]*len(names)+[v for n in nodes for v in (n.gpus,n.cpu_cores,n.ram_gb)])
  lower=np.array([1.]*len(names)+[-np.inf]*(3*len(nodes)))
  weights=np.array([1/values(c,state)[2] for c in pool]);scale=max(weights)
  before=time.perf_counter();res=milp(-weights/scale,integrality=np.ones(len(pool)),bounds=Bounds(0,1),constraints=LinearConstraint(mat,lower,upper),options={'time_limit':30.,'mip_rel_gap':0.})
  receipt=dict(label=label,state=state,status=int(res.status),message=res.message,time=time.perf_counter()-before,columns=len(pool),gap=getattr(res,'mip_gap',None),nodes=getattr(res,'mip_node_count',None))
  if res.x is None or res.status!=0:
   save(state+'_'+label+'_solver.json',receipt);raise RuntimeError('STATIC_SOLVER_NOT_CERTIFIED '+label)
  assert np.max(abs(res.x-np.round(res.x)))<1e-6
  sel=[c for c,x in zip(pool,res.x) if x>.5];assert {x.job_id for x in sel}==set(names)
  audit=s.audit(sel);assert audit['valid'],audit
  out=dict(**receipt,objective=sum(1/values(c,state)[2] for c in sel),sum_eta=sum(values(c,state)[2] for c in sel),selected=[slim(c,state) for c in sel],audit=audit)
  save(state+'_'+label+'.json',out);print('STATIC_SOLVE',state,label,receipt['time'],out['objective'],flush=True)
  return sel,out
 results={}
 for state in ('E0','E25'):
  gate(O)
  proposal,ps=solve([c for c in cols if prop(c)],state,'PROPORTIONAL_GPU_PROPOSAL')
  fixed={c.job_id:c.config.config_id for c in proposal}
  fo,a=solve([c for c in cols if fixed[c.job_id]==c.config.config_id],state,'FO_AUX_OPTIMUM')
  full,b=solve(cols,state,'FULL_RESOURCE_OPTIMUM')
  ax={x['job']:x for x in a['selected']};bx={x['job']:x for x in b['selected']};windows={}
  for h in (360,600):
   row={j:(min(bx[j]['remaining'],max(0,h-bx[j]['startup'])*bx[j]['rate'])-min(ax[j]['remaining'],max(0,h-ax[j]['startup'])*ax[j]['rate']))/ax[j]['total_work'] for j in names}
   base=sum(min(x['remaining'],max(0,h-x['startup'])*x['rate'])/x['total_work'] for x in ax.values())
   windows[str(h)]=dict(per_job_net_fraction=row,net=sum(row.values()),relative_to_fo_pct=100*sum(row.values())/base,
        receivers=sum(max(0,v) for v in row.values()),donor_losses=-sum(min(0,v) for v in row.values()),
        fo_constant_rate_completions=[j for j,x in ax.items() if x['eta']<=h],full_constant_rate_completions=[j for j,x in bx.items() if x['eta']<=h])
  diff=[j for j in names if ax[j]['config']!=bx[j]['config']]
  results[state]=dict(objective_gain_pct=100*(b['objective']/a['objective']-1),static_sum_eta_gain_pct=100*(1-b['sum_eta']/a['sum_eta']),gpu_differences=diff,
      changed_supported=[j for j in names if not any(x[k] for x in (ax[j],bx[j]) for k in ('source_num_clipped','source_den_clipped'))],windows=windows,
      warning='Static reference rates and held allocation, capped progress not reassigned after completion; not actual persistence, future execution, JCT or exact controller tie outcome.')
 save('conditional_opportunities.json',results)
 save('validation.json',dict(source_files=len(p['source_hashes']),source_unchanged=all(sha(T/f)==h for f,h in p['source_hashes'].items()),
     inputs_unchanged=all(sha(f)==h for f,h in p['input_hashes'].items()),training_steps=0,cluster_steps=0,
     complete_coverage=len(coverage)==72,static_states=len(results),wall_seconds=time.time()-started,
     caution='ONLINE_PERSISTENCE_AND_STARTUP_REACHABILITY_UNVERIFIED'))
 print(json.dumps(results,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  save('precheck_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
