"""Confirm budget-feasible complementarity in actual fully observed intervals.

Full60s productive co-residency only. Compare8/4 with6/6 at the SAME raw rate,
batch, GPU and node. Fraction-of-declared-work gain includes donor loss but
is not called JCT savings. No simulation or scheduler is invoked.
"""
import collections,csv,gzip,json,sys,argparse
from experiment import OUT,SOURCE,load,save

args=argparse.ArgumentParser();args.add_argument('--available-only',action='store_true');args=args.parse_args()
load();sys.path.insert(0,str(SOURCE))
from applications import APPLICATIONS
from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
from joint_prototype.joint_profile import JobSpec,GpuConfig
from run_stage2c_cell import independent_rho
with (OUT/'inputs/workload.csv').open(encoding='utf-8') as f:apps={r['name']:r['application'] for r in csv.DictReader(f)}
oracle=R0Oracle(OUT/'inputs/registry.json',OUT/'inputs/models',apps,'medium')
counts=collections.Counter();examples=[]
for method in ['SIA_P','SIA_ECT_AUX','JOINT_CC_P','JOINT_CC']:
  if not (OUT/'cells/medium'/method/'summary.json').exists():
    if args.available_only:continue
    raise RuntimeError('Medium cell not complete: '+method)
  count=collections.Counter();first=None
  with gzip.open(OUT/'cells/medium'/method/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
      r=json.loads(line);nodes=collections.defaultdict(list)
      for e in r['execution']:
        a=e['resource_allocation']
        if not a or a['gpus']!=1 or e['training_seconds']!=60 or e['checkpoint_delay_consumed']!=0:continue
        for node in a['node_usage']:nodes[node['node_id']].append(e)
      for node,ee in nodes.items():
        if len(ee)!=2 or {apps[e['job_id']] for e in ee}!={'cifar10','deepspeech2'}:continue
        count['full_productive_mixed_node_intervals']+=1
        cf=next(e for e in ee if apps[e['job_id']]=='cifar10');ds=next(e for e in ee if apps[e['job_id']]=='deepspeech2')
        rows=[];supported=True;net=0.
        for e,cpu in [(cf,8.),(ds,4.)]:
          job=e['job_id'];a=e['resource_allocation'];cid=a['config_id'];ref=oracle.reference(job,cid)
          cfg=GpuConfig(cid,cid.split(':')[0],1,1,1.,ref['phi_c'],ref['phi_m'])
          spec=JobSpec(job,APPLICATION_TEMPLATES[apps[job]],(cfg,))
          q=oracle.query(spec,cfg,cpu,125.,(6.,125.));prop=independent_rho(oracle,job,cid,6.,125.)
          support=not q.source_coordinate_clipped and not q.baseline_coordinate_clipped
          support=support and 1<=6/ref['phi_c']<=24 and 20.83<=125/ref['phi_m']<=250
          supported &= support
          # Average directly captured pre-rho rate; epoch changes are integrated.
          raw=sum(s['raw_rate_before_rho']*s['seconds'] for s in e['training_segments'])/60.
          work=APPLICATIONS['rtx'][apps[job]].get_progress(APPLICATIONS['rtx'][apps[job]].max_epochs)
          change=60*raw*(q.rho-prop)/work;net+=change
          rows.append(dict(job=job,config=cid,actual_cpu=a['cpu_per_gpu'],counterfactual_cpu=cpu,
            proportional_rho=prop,exchange_rho=q.rho,speed_ratio=q.rho/prop,
            declared_work_fraction_change=change,source_supported=support))
        if supported:
          count['supported_exchange_intervals']+=1
          count['positive_net_declared_work_intervals']+=net>0
          count['nonpositive_net_declared_work_intervals']+=net<=0
          if first is None:first=dict(interval_start=r['interval_start'],node=node,rows=rows,net_declared_work_fraction=net,
            budget=dict(gpu=2,cpu=12,ram_gb=250))
  counts[method]=dict(count);examples.append(dict(method=method,first_supported=first))
save(OUT/('observed_opportunities_available.json' if args.available_only else 'observed_opportunities.json'),dict(training_run=False,counts=dict(counts),examples=examples,
  any_supported_actual_complementary_overlap=any(x.get('supported_exchange_intervals',0)>0 for x in counts.values()),
  definition='Exactly full productive60s on one shared node; fixed GPUs and raw execution rates. Net normalized declared-work change includes donor loss, not global JCT proof.'))
print('OBSERVED_OPPORTUNITIES_READY',dict(counts))
