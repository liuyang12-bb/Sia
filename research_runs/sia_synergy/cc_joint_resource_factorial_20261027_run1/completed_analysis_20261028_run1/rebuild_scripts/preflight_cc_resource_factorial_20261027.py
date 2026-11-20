"""Offline full-domain coverage and predetermined same-state opportunity gate."""
import collections,gzip,hashlib,itertools,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1';TREE=ROOT/'cc-joint-resource-study'
sys.path.insert(0,str(TREE));sys.path.insert(0,str(TREE/'sia-simulator'))
from resource_study import CAPACITIES,ORIGINAL_CPU,ORIGINAL_RAM
from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
from joint_prototype.joint_profile import JobSpec,GpuConfig
def save(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def supported(q):return not q.source_coordinate_clipped and not q.baseline_coordinate_clipped
def main():
 import csv
 with (TREE/'inputs/workload.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
 apps={r['name']:r['application'] for r in rows};oracle=R0Oracle(TREE/'inputs/registry.json',TREE/'inputs/models',apps,'medium')
 specs={}
 for name,app in apps.items():
  configs=[]
  for t,g in itertools.product(['aws','dgx-ext'],[1,2]):
   cid=f'{t}:1:{g}';ref=oracle.reference(name,cid);configs.append(GpuConfig(cid,t,g,1,1.,ref['phi_c'],ref['phi_m']))
  specs[name]=JobSpec(name,APPLICATION_TEMPLATES[app],tuple(configs))
 coverage={};details=[]
 for scenario,(cpu,ram) in CAPACITIES.items():
  count=collections.Counter()
  for name,j in specs.items():
   for g in j.configs:
    count['job_configs']+=1
    for cp,mm in itertools.product(sorted(set(ORIGINAL_CPU+(cpu/2,))),sorted(set(ORIGINAL_RAM+(ram/2,)))):
     if cp*g.gpus>cpu+1e-9 or mm*g.gpus>ram+1e-9:continue
     q=oracle.query(j,g,cp,mm,(cpu/2,ram/2));assert math.isfinite(q.rho) and q.rho>0
     count['queries']+=1
     for label,c,m in [('numerator',q.requested_source_cpu_coordinate,q.requested_source_ram_coordinate),('denominator',q.requested_baseline_source_cpu_coordinate,q.requested_baseline_source_ram_coordinate)]:
      count[label+'_cpu_clip']+=not 1<=c<=24;count[label+'_ram_clip']+=not 20.83<=m<=250
     details.append(dict(scenario=scenario,**q.to_dict()))
    q=oracle.query(j,g,cpu/2,ram/2,(cpu/2,ram/2));assert cpu/2*g.gpus<=cpu and ram/2*g.gpus<=ram
  coverage[scenario]=dict(count)
 save('full_domain_coverage.json',dict(summary=coverage,queries=details,missing=[]))
 # First structural state per type, never chosen by JCT or gain magnitude.
 selected={};old=ROOT/'research_runs/sia_synergy/cc_joint_tail_recovery_20261027_run1/cells/ram250_JOINT_CC_TAIL/rounds.jsonl.gz'
 with gzip.open(old,'rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);d=r['decision_for_next_interval'];cols=d['result']['selected'];by_node=collections.defaultdict(list);sp={s['job_id']:s for s in d['probe']['specs']}
   scores={s['column_id']:s for s in d['continuity_scores']}
   for name,c in cols.items():
    if c['config']['gpus']==1 and not sp[name]['checkpoint_locked'] and scores[c['column_id']]['raw_rate'] is not None:
     for node in c['placement']:by_node[node].append((name,c))
   for node,pair in by_node.items():
    typ=node.rsplit(':',1)[0]
    if typ in selected or len(pair)!=2 or {apps[n] for n,c in pair}!={'cifar10','deepspeech2'}:continue
    selected[typ]=dict(time=r['interval_end'],node=node,jobs=[dict(name=n,config=c['config']['config_id'],raw_rate=scores[c['column_id']]['raw_rate'],remaining_work=scores[c['column_id']]['remaining_work'],total_work=d['declared_work'][n]['total_progress'],batch=scores[c['column_id']]['atomic_bsz'],accum=scores[c['column_id']]['accum_steps']) for n,c in pair])
   if len(selected)==2:break
 assert selected,'NO_STRUCTURAL_STATE'
 save('selected_states.json',dict(rule='First known nonlocked CF+DS oneGPU each per GPUtype',states=selected,source_sha256=hashlib.sha256(old.read_bytes()).hexdigest().upper()))
 results=[]
 for scenario,(cpu,ram) in CAPACITIES.items():
  for typ,state in selected.items():
   js=state['jobs'];gs=[next(g for g in specs[j['name']].configs if g.config_id==j['config']) for j in js]
   prop=[oracle.query(specs[j['name']],g,cpu/2,ram/2,(cpu/2,ram/2)) for j,g in zip(js,gs)]
   base=sum(j['raw_rate']*q.rho/j['total_work'] for j,q in zip(js,prop))
   base_eta=sum(j['remaining_work']/j['raw_rate']/q.rho for j,q in zip(js,prop))
   for factor in ['CPU','MEM','FULL']:
    options=[]
    for j,g in zip(js,gs):
     cps=[cpu/2] if factor=='MEM' else sorted(set(ORIGINAL_CPU+(cpu/2,)))
     rms=[ram/2] if factor=='CPU' else sorted(set(ORIGINAL_RAM+(ram/2,)))
     qs=[oracle.query(specs[j['name']],g,c,m,(cpu/2,ram/2)) for c,m in itertools.product(cps,rms) if c<=cpu and m<=ram]
     options.append(qs)
    best=None;positive=0;feasible=0
    for pair in itertools.product(*options):
     if sum(q.cpu_per_gpu for q in pair)>cpu+1e-9 or sum(q.ram_gb_per_gpu for q in pair)>ram+1e-9:continue
     feasible+=1
     if not all(supported(q) for q in pair+tuple(prop)):continue
     net=sum(j['raw_rate']*q.rho/j['total_work'] for j,q in zip(js,pair))-base
     positive+=net>1e-12
     if best is None or net>best['net_normalized_work_rate']:
      best=dict(net_normalized_work_rate=net,relative_work_rate_gain=net/base,
       eta_sum_delta=sum(j['remaining_work']/j['raw_rate']/q.rho for j,q in zip(js,pair))-base_eta,
       points=[dict(job=j['name'],cpu=q.cpu_per_gpu,ram=q.ram_gb_per_gpu,rho=q.rho,speed_ratio=q.rho/p.rho,source_cpu=q.requested_source_cpu_coordinate,source_ram=q.requested_source_ram_coordinate) for j,q,p in zip(js,pair,prop)])
    results.append(dict(scenario=scenario,gpu=typ,factor=factor,feasible_grid_pairs=feasible,supported_positive_pairs=positive,best=best))
 cpu_gate=any(x['scenario'] in ['R10','R11'] and x['factor']=='CPU' and x['supported_positive_pairs'] for x in results)
 mem_gate=any(x['scenario'] in ['R01','R11'] and x['factor']=='MEM' and x['supported_positive_pairs'] for x in results)
 # Preserve the known outside-grid witness; it is not inserted into execution.
 outside=json.loads((ROOT/'research_runs/sia_synergy/cc_joint_queue_ram_tail_20261027_run3/memory_candidate_gap.json').read_text())
 save('opportunity_preflight.json',dict(valid=True,cpu_gate=bool(cpu_gate),memory_gate=bool(mem_gate),run_gate=bool(cpu_gate and mem_gate),
  rule='At least one supported positive net declared-work-rate grid exchange for each intended resource pressure; donor losses included; no JCT guarantee',
  results=results,known_outside_grid_memory_witness=outside,grid_modified=False,
  status='READY_FOR_ADAPTER_TESTS' if cpu_gate and mem_gate else 'STOP_KEY_OPPORTUNITY_NOT_IN_CURRENT_GRID'))
 print('G0',cpu_gate,mem_gate)
 for x in results:print(x['scenario'],x['gpu'],x['factor'],x['supported_positive_pairs'],x['best']['relative_work_rate_gain'] if x['best'] else None)
if __name__=='__main__':main()
