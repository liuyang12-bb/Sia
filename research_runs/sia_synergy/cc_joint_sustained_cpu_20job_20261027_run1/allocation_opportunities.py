"""All logged full-productive node states, not just cross-application pairs.

Enumerate the already-frozen CPU grid, fixing GPU/batch/placement/RAM125 and
the complete co-resident set. Denominator and numerator must both be supported.
Normalized work is progress / original declared work, not comparable samples/s.
This is offline opportunity diagnosis, not another training run or parameter search.
"""
import collections,gzip,itertools,json,sys,csv
from experiment import OUT,SOURCE,load,save
load();sys.path.insert(0,str(SOURCE))
from applications import APPLICATIONS
with (OUT/'inputs/workload.csv').open(encoding='utf-8') as f:apps={r['name']:r['application'] for r in csv.DictReader(f)}
coverage=json.loads((OUT/'coverage_preflight.json').read_text())
grids={(r['job'],r['config']):[q for q in r['points'] if q['ram_gb_per_gpu']==125]
    for r in coverage['coverage']}
totals={}
for method in ['SIA_P','SIA_ECT_AUX','JOINT_CC_P','JOINT_CC']:
  counts=collections.Counter();first=None;first_mixed=None;best_gain=0.;by_composition=collections.Counter()
  with gzip.open(OUT/'cells/medium'/method/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
      r=json.loads(line);nodes=collections.defaultdict(list)
      for e in r['execution']:
        a=e['resource_allocation']
        if a and e['allocated_seconds']>0:
          for u in a['node_usage']:nodes[u['node_id']].append(e)
      for node,ee in nodes.items():
        counts['allocated_node_intervals']+=1
        if not all(e['training_seconds']==60 and e['checkpoint_delay_consumed']==0 for e in ee):
          counts['excluded_partial_or_checkpoint_intervals']+=1;continue
        choices=[];rates=[];baseline=[]
        for e in ee:
          j=e['job_id'];a=e['resource_allocation'];grid=grids[j,a['config_id']]
          prop=next(q for q in grid if q['cpu_per_gpu']==6)
          valid=lambda q:not q['source_coordinate_clipped'] and not q['baseline_coordinate_clipped']
          if not valid(prop):break
          choices.append([q for q in grid if valid(q)])
          app=APPLICATIONS['rtx'][apps[j]];work=app.get_progress(app.max_epochs)
          raw=sum(s['raw_rate_before_rho']*s['seconds'] for s in e['training_segments'])/60
          rates.append(raw/work);baseline.append(prop)
        if len(choices)!=len(ee):counts['excluded_source_boundary_intervals']+=1;continue
        counts['supported_fixed_set_node_intervals']+=1
        base=sum(v*q['rho'] for v,q in zip(rates,baseline));best=None;best_value=base
        for combination in itertools.product(*choices):
          if sum(e['resource_allocation']['gpus']*q['cpu_per_gpu'] for e,q in zip(ee,combination))>12+1e-9:continue
          score=sum(v*q['rho'] for v,q in zip(rates,combination))
          if score>best_value+1e-14:best=combination;best_value=score
        if best is None:continue
        counts['positive_net_normalized_work_node_intervals']+=1
        counts['single_job_idle_CPU_opportunity_intervals']+=len(ee)==1
        label='+'.join(sorted(apps[e['job_id']] for e in ee));by_composition[label]+=1
        actual=sum(v*e['rho'] for v,e in zip(rates,ee));gain=best_value/base-1
        counts['actual_above_same_state_proportional_intervals']+=actual>base+1e-14
        counts['actual_below_same_state_proportional_intervals']+=actual<base-1e-14
        best_gain=max(best_gain,gain)
        rows=[dict(job=e['job_id'],config=e['resource_allocation']['config_id'],gpus=e['resource_allocation']['gpus'],
            baseline_cpu=6,alternative_cpu=q['cpu_per_gpu'],ram_per_gpu=125,
            baseline_rho=p['rho'],alternative_rho=q['rho'],actual_rho=e['rho'],actual_cpu=e['resource_allocation']['cpu_per_gpu'])
            for e,p,q in zip(ee,baseline,best)]
        ex=dict(interval_start=r['interval_start'],node=node,rows=rows,net_normalized_work_gain=gain,
          normalized_work_per_second=dict(proportional=base,alternative=best_value,actual=actual))
        if first is None:first=ex
        if len({apps[e['job_id']] for e in ee})>1 and first_mixed is None:first_mixed=ex
  totals[method]=dict(counts=dict(counts),positive_opportunities_by_composition=dict(by_composition),
      first_supported_positive=first,first_mixed_positive=first_mixed,maximum_local_relative_gain=best_gain)
save(OUT/'allocation_opportunities.json',dict(training_run=False,results=totals,
    interpretation='All complete service sets on a node, no donor omitted. Fixed observed GPU/batch/raw rates. Normalized progress gain is not a JCT counterfactual. RAM was held fixed.'))
print('ALLOCATION_OPPORTUNITIES_READY',{m:v['counts'] for m,v in totals.items()})
