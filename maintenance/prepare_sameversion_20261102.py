"""Freeze approved inputs and audit responses offline. Never starts a simulator.

No historical file is edited. A nonempty output directory is always refused.
"""
import collections,csv,hashlib,itertools,json,math,sys,time,zipfile
from pathlib import Path
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'cc-joint-core80-full-lex-repair-v8'
OUT=ROOT/'research_runs/sia_synergy'/('cc_paper_sameversion_20261102_run2' if len(sys.argv)<2 else sys.argv[1])
sys.path[:0]=[str(BASE),str(BASE/'sia-simulator')]
from joint_prototype.joint_profile import GpuConfig,JobSpec,NodeSpec
from joint_prototype.joint_scheduler import Placement
from joint_prototype.neutral_liveness import FactorScheduler
from joint_prototype.r0_liveness import R0Oracle
from breakpoint_study import resource_points

GPUS={'aws':4,'dgx-ext':8,'rtx':8}
LEVELS={'dual':(3.,62.5),'cpu_relative':(3.,125.),'ram_relative':(6.,62.5),'relaxed':(6.,125.)}
WARNING='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest().upper()
def save(p,x):
 with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def csvwrite(p,rows):
 with Path(p).open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def inputs():
 records=[]
 original=BASE/'inputs/core80_workload.csv';rows=read(original)
 assert len(rows)==80 and collections.Counter(r['application'] for r in rows)=={'cifar10':48,'deepspeech2':16,'bert':16}
 for name in ('waves','uniform'):
  selected=[dict(r) for r in rows]
  if name=='uniform':
   for i,r in enumerate(selected):r['time']=format(1200*i/79,'.15g')
  target=OUT/'inputs'/f'{name}.csv';csvwrite(target,selected)
  records.append(dict(id=name,path=str(target),sha256=sha(target),source=str(original),source_sha256=sha(original),
   rows=80,applications=dict(collections.Counter(r['application'] for r in selected)),
   rule='Historical frozen order and full training parameters; only uniform changes time to 1200*i/79',
   warning='Controlled task instances, including historical record reuse; not independent production jobs'))
 for family in ('saturn','philly'):
  for index in (1,2,3):
   source=ROOT/f'official-artifacts/sia-simulator/workloads/{family}/workload-{index}.csv'
   raw=read(source);assert len(raw)>=80
   order=sorted(range(len(raw)),key=lambda i:(float(raw[i]['time']),i))[:80]
   shift=float(raw[order[0]]['time']);selected=[]
   for stable,i in enumerate(order):
    r=dict(raw[i]);r.update(time=format(float(r['time'])-shift,'.15g'),original_arrival=raw[i]['time'],
      original_source_row_index=i,source_row_index=stable,source_trace=f'{family}/workload-{index}.csv')
    selected.append(r)
   target=OUT/'inputs'/f'{family}_{index}.csv';csvwrite(target,selected)
   records.append(dict(id=f'{family}_{index}',path=str(target),sha256=sha(target),source=str(source),source_sha256=sha(source),
    rows=80,source_csv_data_row_indices=order,job_ids=[r['name'] for r in selected],arrival_shift=shift,
    applications=dict(collections.Counter(r['application'] for r in selected)),
    rule='First80 by stable arrival order from artifact workload files1,2,3. No result-based selection.',
    independence='Production-trace-derived artifact samples; overlap in underlying original trace is not recoverable from these CSVs; no independent-replicate claim'))
 save(OUT/'input_manifest.json',records)
 return records

def matrix():
 cells=[]
 def add(group,work,cap,response,method,scale=1):
  cells.append(dict(id=f'{group}_{work}_{cap}_{response}_{method}_n{scale}',group=group,workload=work,
   capacity=cap,response=response,method=method,nodes_per_type=scale,max_rounds=6000,interval=60))
 for work in ('waves','uniform'):
  for method in ('Q00','Q10','Q01','Q11'):add('service',work,'dual','neutral',method)
 for cap in ('cpu_relative','ram_relative','dual'):
  for method in ('PROP','CPU','MEM','FULL'):add('resources','waves',cap,'medium',method)
 for method in ('SIA_P','SIA_AUX','FULL'):add('negative','waves','relaxed','medium',method)
 for family in ('saturn','philly'):
  for index in (1,2,3):
   for method in ('SIA_P','SIA_AUX','FULL'):add('trace',f'{family}_{index}','dual','medium',method)
 for family in ('saturn','philly'):
  for method in ('SIA_P','SIA_AUX','FULL'):add('scale',f'{family}_1','dual','medium',method,2)
 assert len(cells)==47 and len({c['id'] for c in cells})==47
 save(OUT/'preregistration.json',dict(status='INPUTS_AND_SCIENTIFIC_MATRIX_FROZEN_IMPLEMENTATION_ACCEPTANCE_PENDING',
  cells=cells,capacities_per_gpu=LEVELS,gpus_per_node=GPUS,reference_point=[3.,62.5],alpha={'neutral':0.,'medium':.5},
  service_contract='Qxy: groupB common completion objective, jointGPU+proportional aux, ordered=x, until-completion=y; off uses existing300 productive seconds and lifetime budget safety. No change to restart/checkpoint semantics.',
  history_baseline='v8 original SIA_P/SIA_AUX retained; SIA_AUX minimizes sumETA, FULL maximizes reciprocalETA; disclose objective/admission differences, no pure GPU-factor claim.',
  scale_contract='40GPU by duplicating each type; each task remains single-node; same workload/arrivals => capacity sensitivity, not fixed-utilization scaling.',
  gates={'memory_gib':2,'disk_gib':5,'full_suite':335,'original_sia_exact_rounds':3,'no_new_matrix_before_acceptance':True},
  order='Listed order, independent Python processes, serial',
  response='Unchanged fixedR0/E-legacy; no refitting; all55 registry entries audited; clipping separately reported',
  source_plan='Copy all actual v8 files, verify hashes, add isolated topology/config harness and tests; no historical source edits',
  censoring='6000 rounds; keep censored jobs; complete paired means only; no selective extension; engineering errors preserve logs and stop paired block',
  inference='Trace-sample paired descriptive analysis, not1600 independent trials; no winner-based parameter selection',warning=WARNING))
 return cells

def coverage():
 registry=json.loads((BASE/'inputs/registry.json').read_text());apps=sorted({r['application'] for r in registry})
 oracle=R0Oracle(BASE/'inputs/registry.json',BASE/'inputs/models',{a:a for a in apps},'medium')
 details=[];groups={};counts=collections.Counter()
 for level,(pc,pm) in LEVELS.items():
  nodes=[NodeSpec(t+':0',t,g,pc*g,pm*g) for t,g in GPUS.items()]
  for ref in registry:
   typ,nn,gg=ref['config_id'].split(':');g=int(gg);assert int(nn)==1 and g<=GPUS[typ]
   cfg=GpuConfig(ref['config_id'],typ,g,1,1.,ref['phi_c'],ref['phi_m'])
   job=JobSpec(ref['application'],ref['template'],(cfg,))
   s=FactorScheduler(nodes,[job],oracle,[1.,2.,3.,4.,6.,8.,9.,12.,24.],[20.83,62.5,125.,187.5,250.],verify_exact=False,
       minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,candidate_policy='b3')
   pl=Placement(((typ+':0',g),));cp,mp=resource_points(s,job,cfg,pl)
   points=[]
   for c,m in itertools.product(sorted(set(cp+(pc,))),sorted(set(mp+(pm,)))):
    if g*c>pc*GPUS[typ]+1e-9 or g*m>pm*GPUS[typ]+1e-9:continue
    q=oracle.query(job,cfg,c,m,(pc,pm));assert math.isfinite(q.rho) and q.rho>0
    row=dict(capacity=level,application=job.job_id,config=cfg.config_id,cpu=c,ram=m,rho=q.rho,
      numerator_cpu_clip=not 1<=c/ref['phi_c']<=24,numerator_ram_clip=not 20.83<=m/ref['phi_m']<=250,
      denominator_cpu_clip=not 1<=3/ref['phi_c']<=24,denominator_ram_clip=not 20.83<=62.5/ref['phi_m']<=250)
    row['supported']=not any(row[k] for k in row if k.endswith('_clip'))
    points.append(row)
    for k in ('numerator_cpu_clip','numerator_ram_clip','denominator_cpu_clip','denominator_ram_clip'):counts[k]+=row[k]
   base=oracle.query(job,cfg,pc,pm,(pc,pm))
   details.append(dict(capacity=level,application=job.job_id,config=cfg.config_id,template=job.template_file,
    phi_c=ref['phi_c'],phi_m=ref['phi_m'],proportional_cpu=pc,proportional_ram=pm,proportional_rho=base.rho,
    points=len(points),fully_supported=sum(r['supported'] for r in points),
    best_cpu_only=max(r['rho'] for r in points if r['ram']==pm),best_ram_only=max(r['rho'] for r in points if r['cpu']==pc),
    best_rho=max(r['rho'] for r in points),proportional_fits=g*pc<=GPUS[typ]*pc and g*pm<=GPUS[typ]*pm))
   if g==1:groups[level,job.job_id,typ]=points
 csvwrite(OUT/'coverage.csv',details)
 witnesses=[]
 for level,(pc,pm) in LEVELS.items():
  for typ in GPUS:
   for a,b in [('cifar10','deepspeech2'),('cifar10','bert')]:
    # Strictly within-support points, budget of two proportional1GPU jobs.
    left=[r for r in groups[level,a,typ] if r['supported']]
    right=[r for r in groups[level,b,typ] if r['supported']]
    best=None
    for x in left:
     feasible=[y for y in right if x['cpu']+y['cpu']<=2*pc+1e-8 and x['ram']+y['ram']<=2*pm+1e-8]
     if not feasible:continue
     y=max(feasible,key=lambda z:z['rho'])
     baseline=[d['proportional_rho'] for d in details if d['capacity']==level and d['config']==f'{typ}:1:1' and d['application'] in (a,b)]
     gain=x['rho']+y['rho']-sum(baseline)
     if best is None or gain>best['sum_relative_speed_gain']:best=dict(receiver=x,peer=y,baseline_rhos=baseline,sum_relative_speed_gain=gain)
    witnesses.append(dict(capacity=level,gpu_type=typ,applications=[a,b],best_supported_pair=best,
      interpretation='Offline source-response opportunity only; raw Sia rate set to dimensionless1 for response query, never used as execution input. Sum relative speeds is not global throughput or JCT proof.'))
 save(OUT/'opportunity_witnesses.json',witnesses)
 assert len(details)==220 and all(r['proportional_fits'] for r in details)
 save(OUT/'offline_validation.json',dict(status='PASS_RESPONSE_FINITE_AND_PROPORTIONAL_CAPACITY',registry_configs=55,
   capacity_config_rows=len(details),candidate_query_clipping_counts=dict(counts),
   supported_positive_pairs=sum(bool(r['best_supported_pair'] and r['best_supported_pair']['sum_relative_speed_gain']>1e-8) for r in witnesses),
   same_state_global_raw_rate_net_benefit='NOT_YET_VERIFIED',actual_sustained_competition='NOT_RUN',
   tests='Offline assertions only; existing335-suite and simulator acceptance have not been rerun',warning=WARNING))
 return details,witnesses

def main():
 if OUT.exists():raise RuntimeError('REFUSE_EXISTING_OUTPUT')
 OUT.mkdir();(OUT/'inputs').mkdir()
 start=time.time();resources=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
 (OUT/'CURRENT_TASK.md').write_text('''# 同版论文实验恢复入口\n用户已授权完成服务消融、CPU/RAM消融、两类trace和20/40GPU场景。\n本目录为新任务；正式实验首次预检磁盘不足5GiB，未启动。\n先读preregistration.json、resource_gate.json、REPORT.md和source_hashes.json。\n仅离线准备完成；禁止把计划矩阵当作运行结果。335项回归、同版开关与双节点适配验收尚未执行。\n恢复前核验无重复进程，内存>=2GiB、磁盘>=5GiB；完成新隔离执行源码、测试和快照冻结后才能启动。\n不改目标/响应/工作量，不按结果调参；保留历史源码和产物；不推送。\n''',encoding='utf8')
 save(OUT/'resource_gate.json',dict(status='BLOCKED_DISK_GATE' if resources['disk_gib']<5 else 'GATE_CURRENTLY_PASS',resources=resources,minimum_disk_gib=5,minimum_memory_gib=2,formal_workers_started=0))
 records=inputs();cells=matrix()
 source={p.relative_to(BASE).as_posix():sha(p) for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 expected=json.loads((ROOT/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5/source_hashes.json').read_text())
 assert source==expected,'V8_SOURCE_DIFFERS_FROM_FROZEN_EXECUTION'
 save(OUT/'source_hashes.json',source)
 with zipfile.ZipFile(OUT/'baseline_source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for rel in source:z.write(BASE/rel,rel)
 with zipfile.ZipFile(OUT/'baseline_source_snapshot.zip') as z:
  assert all(hashlib.sha256(z.read(rel)).hexdigest().upper()==h for rel,h in source.items())
 save(OUT/'snapshot_receipt.json',dict(files=len(source),matches_v8_frozen_execution=True,sha256=sha(OUT/'baseline_source_snapshot.zip')))
 details,witnesses=coverage()
 blocked=resources['disk_gib']<5 or resources['memory_gib']<2
 report=f'''# 同版实验输入冻结与离线预检\n\n状态：{'BLOCKED_RESOURCE_GATE' if blocked else 'IMPLEMENTATION_ACCEPTANCE_PENDING'}。正式仿真0/{len(cells)}格启动；没有新增JCT结果。\n\n首次资源：内存{resources['memory_gib']:.3f}GiB，磁盘{resources['disk_gib']:.3f}GiB。既有门槛2/5GiB。没有删除历史文件、降低门槛或启动工作进程。\n\n输入：受控80作业waves/uniform；Saturn/Philly各workload1/2/3前80个稳定到达记录。共8份CSV，逐项hash与行号见input_manifest.json。trace来源CSV不能恢复所有上游抽样身份，不能宣称六份样本相互独立。\n\n矩阵：服务8格，资源12格，宽松对照3格，trace18格，40GPU6格，总47格。40GPU相同到达序列用于容量敏感性，不宣称保持负载强度的可扩展性。\n\n实际v8源码{len(source)}文件与历史执行hash完全一致，并保存逐文件校验的完整ZIP。尚未生成/验收新的执行适配版本。\n\n覆盖：4容量×55配置={len(details)}项，比例配额均符合逐节点静态容量，响应有限且为正。完整配置与CPU/RAM分开变化结果见coverage.csv。支持域内两作业互补证据见opportunity_witnesses.json；相对速率和不等于不同应用绝对Goodput之和，不是全局净收益或JCT证明。\n\n剩余验收：完成评分下的服务四开关、双节点但单节点任务域、335项旧测试、新适配测试、原Sia精确回归、同状态实际raw-rate净收益，以及逐轮持续触发。SIA_AUX与FULL历史评分不同，主表只评价整体方法，不称纯GPU单因子效应。\n\n任何正式运行必须在资源门槛满足后独立进程串行启动，冻结执行源码与参数。历史结果没有复用成新实验完成格；没有仿真、没有报告性能改善。\n\n{WARNING}\n'''
 (OUT/'REPORT.md').write_text(report,encoding='utf8')
 save(OUT/'preparation_receipt.json',dict(status='BLOCKED_RESOURCE_GATE' if blocked else 'PREPARED_PENDING_ACCEPTANCE',wall_seconds=time.time()-start,
  formal_runs_started=0,planned_cells=len(cells),files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
 print(json.dumps(dict(output=str(OUT),resources=resources,planned_cells=len(cells),formal_runs_started=0,source_files=len(source)),ensure_ascii=False))
if __name__=='__main__':main()
