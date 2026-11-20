"""Full-domain offline census, unchanged regression tests and bounded opportunity gates."""
import collections,gzip,hashlib,itertools,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 from cc_factory import verify
 verify()
 from breakpoint_study import resource_points
 from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
 from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
 from joint_prototype.joint_scheduler import Placement
 from joint_prototype.neutral_liveness import FactorScheduler
 from resource_study import ORIGINAL_CPU,ORIGINAL_RAM
 apps={'cifar10-0':'cifar10','bert-28':'bert'};o=R0Oracle(SRC/'inputs/registry.json',SRC/'inputs/models',apps,'medium');rows=[]
 def supported(q):return all([1-1e-12<=q.requested_source_cpu_coordinate<=24+1e-12,20.83-1e-12<=q.requested_source_ram_coordinate<=250+1e-12,1-1e-12<=q.requested_baseline_source_cpu_coordinate<=24+1e-12,20.83-1e-12<=q.requested_baseline_source_ram_coordinate<=250+1e-12])
 for name,app in apps.items():
  configs=[]
  for typ,g in itertools.product(['aws','dgx-ext'],[1,2]):
   cid=f'{typ}:1:{g}';ref=o.reference(name,cid);configs.append(GpuConfig(cid,typ,g,1,1.,ref['phi_c'],ref['phi_m']))
  j=JobSpec(name,APPLICATION_TEMPLATES[app],tuple(configs),0.)
  nodes=[NodeSpec(f'{typ}:0',typ,2,6.,125.) for typ in ['aws','dgx-ext']]
  s=FactorScheduler(nodes,[j],o,ORIGINAL_CPU,ORIGINAL_RAM,verify_exact=False,minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,enforce_proportional_progress_floor=False)
  for cfg in configs:
   dims=resource_points(s,j,cfg,Placement(((cfg.gpu_type+':0',cfg.gpus),)))
   for method in ['P','CPU','MEM','FULL']:
    cp=(3.,) if method in ['P','MEM'] else tuple(set(dims[0])|{3.});mm=(62.5,) if method in ['P','CPU'] else tuple(set(dims[1])|{62.5})
    counts=collections.Counter()
    for c,m in itertools.product(cp,mm):
     if c*cfg.gpus>6+1e-9 or m*cfg.gpus>125+1e-9:continue
     q=o.query(j,cfg,c,m,(3.,62.5));counts['feasible']+=1;counts['supported']+=supported(q);counts['numerator_clipped']+=q.source_coordinate_clipped;counts['denominator_clipped']+=q.baseline_coordinate_clipped
    rows.append(dict(application=app,config=cfg.config_id,method=method,**counts))
 save('offline_boundary_coverage.json',dict(rows=rows,physical_configs=8,missing_configs=[],execution_boundary='unchanged E-legacy',strict_witness=json.loads((OUT/'strict_witness.json').read_text())))
 start=time.time()
 with (OUT/'tests.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'],cwd=SRC/'sia-simulator',stdout=f,stderr=subprocess.STDOUT)
 save('tests.json',dict(exit_code=r.returncode,wall_seconds=time.time()-start));assert r.returncode==0
 gates={}
 for method in ['SIA_P','CC_CPU','CC_FULL']:
  with (OUT/f'GATE_{method}.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'run_scenario.py'),'run','--mix','GATE','--scenario','RNEW','--response','medium','--method',method],stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0,method
  pair=0.;gain=0.;supported_pair=0.
  with gzip.open(OUT/f'cells/GATE_RNEW_medium_{method}/rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    rr=json.loads(line);es=rr['execution']
    for a,b in itertools.combinations(es,2):
     if {a['job_id'].rsplit('-',1)[0],b['job_id'].rsplit('-',1)[0]}!={'cifar10','bert'}:continue
     if not a['resource_allocation'] or not b['resource_allocation']:continue
     aa,bb=a['resource_allocation'],b['resource_allocation']
     if not(set(aa['placement'])&set(bb['placement'])):continue
     for za,zb in itertools.product(a['training_segments'],b['training_segments']):
      dt=max(0,min(za['start_time']+za['seconds'],zb['start_time']+zb['seconds'])-max(za['start_time'],zb['start_time']))
      pair+=dt
      cf=a if a['job_id'].startswith('cifar10') else b;bt=b if cf is a else a
      if cf['rho']>1+1e-9 and abs(bt['rho']-1)<1e-12:gain+=dt
      def inside(x):
       return all(1-1e-12<=x[k]['requested_cpu']<=24+1e-12 and 20.83-1e-12<=x[k]['requested_ram_gb']<=250+1e-12 for k in ['source_coordinates','baseline_source_coordinates'])
      if inside(aa) and inside(bb):supported_pair+=dt
  gates[method]=dict(productive_coplace_seconds=pair,receiver_accelerated_donor_unchanged_seconds=gain,fully_supported_pair_seconds=supported_pair)
  print('GATE',method,gates[method],flush=True)
 passed=gates['CC_FULL']['receiver_accelerated_donor_unchanged_seconds']>=60
 save('actual_opportunity_gate.json',dict(passed=passed,methods=gates,boundary_warning='Actual E-legacy clipped points reported separately from domain-supported static witness.'))
 if not passed:raise RuntimeError('NO_ACTUAL_OPPORTUNITY; do not run formal6 or retune')
 with (OUT/'formal_batch_console.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'run_scenario.py'),'batch'],stdout=f,stderr=subprocess.STDOUT)
 print('FORMAL_BATCH_EXIT',r.returncode,flush=True)
if __name__=='__main__':main()
