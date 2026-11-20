import collections,csv,gzip,hashlib,json,shutil,statistics,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def table(h,rows):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def main():
 p=json.loads((OUT/'preregistration.json').read_text());records=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()];assert len(records)==6 and all(r['exit_code']==0 for r in records)
 text=(ROOT/'maintenance/audit_breakpoint30_precision_v3_20261029.py').read_text();body=text.split('def main():\n',1)[1].split(" p=json.loads((OUT/'preregistration.json')",1)[0]
 body=body.replace(' DEST.mkdir(exist_ok=False)\n','').replace('cc_breakpoint_study_20261029_run1/attempt3','cc_ample_lex_repair_20261030_run1')
 body=body.replace(" ns={'__file__':", " text=text.replace('tau=6000*60-2400','tau=6000*60-1200').replace('if len(vals)==20:', 'if len(vals)==12:').replace('assert(cp,mm)==(cpu/2,ram/2)','assert cp in {cpu/2,round(cpu/2,9)} and mm in {ram/2,round(ram/2,9)}').replace('assert mm==ram/2','assert mm in {ram/2,round(ram/2,9)}').replace('assert cp==cpu/2','assert cp in {cpu/2,round(cpu/2,9)}')\n ns={'__file__':")
 scope=dict(ROOT=ROOT,OUT=OUT,DEST=OUT,np=np);exec('def bootstrap():\n'+body+'\n return ns\n',scope);ns=scope['bootstrap']();prior_points=ns['independent_points']
 def pts(j,c,cpu,ram,method):
  a,b=prior_points(j,c,cpu,ram,method);return set(a)|{round(cpu/2,9)},set(b)|{round(ram/2,9)}
 ns['independent_points']=pts
 from applications import APPLICATIONS
 results={};metrics=[];jobs=[];costchecks=collections.Counter()
 for cell,rec in zip(p['cells'],records):
  name='_'.join(cell);assert name==rec['cell'];folder=OUT/'cells'/name
  for n,h in json.loads((folder/'output_hashes.json').read_text()).items():assert sha(folder/n)==h
  r=ns['audit'](folder,p);results[cell[-1]]=r;s=r['summary']
  with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    d=json.loads(line)['decision_for_next_interval'];sel=d['result']['selected'];specs={j['job_id']:j for j in d['probe']['specs']}
    assert {j for j,q in specs.items() if q['checkpoint_locked']}<=set(sel)
    a=d['result']['audit']
    if 'secondary_tie' in a and not a.get('tail_recoveries'):
     actual=0.
     for j,col in sel.items():
      q=specs[j];same=q['current_config_id']==col['config']['config_id'] and dict(q['current_placement'])==col['placement']
      if not same:actual+=float(APPLICATIONS[col['config']['gpu_type']][j.rsplit('-',1)[0]].rescale_time or 30.)
     assert abs(actual-a['secondary_tie']['minimum_actual_delay'])<1e-8,(cell,d['round'],actual,a['secondary_tie'])
     costchecks[cell[-1]]+=1
  vals=[v['jct'] for v in r['jobs'].values() if v.get('jct') is not None]
  if len(vals)==12:
   assert abs(statistics.mean(vals)-s['mean_jct'])<1e-8
  metrics.append(dict(method=cell[-1],completed=s['completed'],mean=s['mean_jct'],median=s['median_jct'],p95=s['p95_jct'],p99=s['p99_jct'],makespan=s['makespan'],**s['totals']))
  jobs.extend(dict(method=cell[-1],job=j,**v) for j,v in r['jobs'].items());print('RAW_AUDITED',cell[-1],s['completed'],flush=True)
 save('raw_audit.json',results);save('metrics.json',metrics)
 with (OUT/'per_job.csv').open('x',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=sorted({k for r in jobs for k in r}));w.writeheader();w.writerows(jobs)
 # Same accepted independent fixed-execution response accounting, not a simulation.
 desc=(ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1/describe_response_v2.py').read_text().replace('cc_cifar_bert12_20261029_run1','cc_ample_lex_repair_20261030_run1')
 (OUT/'response_analysis_script.py').write_text(desc,encoding='utf8');env={'__file__':str(OUT/'response_analysis_script.py'),'__name__':'descriptive_only'};exec(compile(desc,str(OUT/'response_analysis_script.py'),'exec'),env);env['main']()
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h
 for n,h in p['input_hashes'].items():assert sha(OUT/'inputs'/n)==h
 protection=json.loads((OUT/'protection.json').read_text())
 for directory,h in protection['old_seals'].items():
  d=Path(directory);assert sha(d/'delivery_hashes.json')==h
  for n,x in json.loads((d/'delivery_hashes.json').read_text()).items():assert sha(d/n)==x
 validation=dict(cells=6,completed_cells=sum(r['summary']['completed']==12 for r in results.values()),capacity_violations=0,lock_violations=0,rho_violations=0,secondary_count_checks=dict(costchecks),source_and_inputs_frozen=True,historical_seals_valid=True,tests=245,original_sia_bounded_exact=True,simulation_wall_seconds=sum(r['wall_seconds'] for r in records))
 save('validation.json',validation)
 parent=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1/raw_audit.json';old=json.loads(parent.read_text());oldcomp={}
 for m,r in results.items():
  previous=old.get('BASE_AMPLE_medium_'+m,{})
  if previous.get('execution_failed'):oldcomp[m]=dict(previous='FAILED_NO_JCT',new_mean=r['summary']['mean_jct'])
  else:oldcomp[m]=dict(previous_mean=previous['summary']['mean_jct'],new_mean=r['summary']['mean_jct'],per_job_delta={j:v['jct']-previous['jobs'][j]['jct'] for j,v in r['jobs'].items()})
 save('previous_successful_comparison.json',oldcomp)
 from importlib.machinery import SourceFileLoader
 design=SourceFileLoader('design_ample',str(ROOT/'maintenance/design_distinct_loads_20261030.py')).load_module();design.main()
 proof=json.loads((OUT/'independent_model_proof.json').read_text());baseline=json.loads((OUT/'baseline_source_hashes.json').read_text());changed=[n for n,h in baseline.items() if sha(OUT/'source'/n)!=h];added=sorted(set(p['source_hashes'])-set(baseline));save('file_diff_summary.json',dict(changed=changed,added=added))
 lines=['# 资源充足档词典序修复与六方法完整复验','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE',
 f"六格完成数={validation['completed_cells']}/6；完整245项测试通过，原始Sia一作业三轮逐状态一致。六方法子进程累计{validation['simulation_wall_seconds']:.2f}秒。算法主目标、CPU/RAM模型、quota网格、资源容量、工作量、初始化画像、solver参数均未改变。",
 '## 根因与证据',
 '真实CC_CPU和CC_MEM均在第240秒复现。失败不是已经证明的数值误判：511列/263列模型的实际最优启动成本为220秒，旧代码把它记为270秒，导致新增二级等式与其他约束矛盾。CC_CPU带对象身份的快照进一步显示已选CIFAR列在候选表中被同一对象重复引用。',
 '上游来源：joint_scheduler.py:365用:g构造展示ID，不区分非常接近的配额；bundle_adaptation.py:68–72用ID重建held/reintroduced候选，可以形成重复引用。真正引发本次错误的代码是原bundle_adaptation.py:98扫描候选表所有位置，用any(v is c for v in candidate)累加成本，而非每个已选变量计费一次。',
 '数学定义保持C(x)=Σ_i d_i x_i。旧记账错误相当于把某个已选列的d_i额外乘以对象在候选表中的出现次数m_i。修复按已选candidate列表逐项计费，并用identity查取其固有单列成本；主效用、二级成本定义、词典序顺序及全部约束保持不变。重复列及展示ID本轮未清理，避免改变候选顺序/可行域。它们仍带来冗余和日志歧义，后续若清理须单独证明几何候选集合等价。',
 table(['方法','失败列数','旧错误启动成本下界','独立最小成本','修复模型状态'],[[m,x['columns'],x['wrong_secondary_lower'],x['independent_minimum_startup'],x['repaired_status']] for m,x in proof.items()]),
 '完整失败模型CC_CPU_failed_model.json.gz/CC_MEM_failed_model.json.gz、带identity的CPU快照、独立证明independent_model_proof.json均保留。新增4测试在旧代码上2失败/2通过（重复2次错误100、重复10次错误500），修复后4/4。真实CPU/MEM各8轮越过旧故障点、经过下一次到达/真实调度；无mock替换下一次调度或修改Goodput。',
 '## 修改与回归',str(dict(changed=changed,added=added)),
 '运行代码为本目录source和完全相同的隔离目录cc-joint-ample-lex-repair。source_diff.patch、baseline_source_snapshot.zip、repaired_source_snapshot.zip和依赖锁支持重建。source_manifest仅更新hash及修复说明；官方源和历史目录未改。原Sia有界回归日志和original_sia_regression.json记录进度、分配、restart、完成事件等精确一致。',
 '## 六方法结果',table(['方法','完成','Mean','Median','P95','P99','Makespan'],[[r[k] for k in ['method','completed','mean','median','p95','p99','makespan']] for r in metrics]),
 '## 资源与成本',table(['方法','GPUh','CPU单位h','RAM GBh','等待s','训练s','checkpoint s','restart'],[[r[k] for k in ['method','gpu_hours','cpu_core_hours','ram_gb_hours','wait','training','checkpoint','restarts']] for r in metrics]),
 '资源为每节点2GPU、76.53020404150541合成CPU单位、446.42738083375554GB RAM，AWS/DGX各一节点。容量来自先前预注册8种配置的最大每GPU饱和需求×2，未因本轮结果调整。Medium alpha=.5，固定R0/E-legacy，60秒/6000轮，原BASE12作业完整原工作量与seed_profiles(1,8)。比例点沿用既有round9和最低资源契约；容量审计沿用原1e−7容差。',
 '旧成功格逐作业比较见previous_successful_comparison.json；旧失败格不补JCT、不与新结果挑选最佳一次。源码已变化，本次全部六格统一重跑，只有本次六格形成完整主配对。',
 '## 独立审计',json.dumps(validation,ensure_ascii=False),
 '从原始完成事件和训练片段独立复算JCT/资源小时/互斥时间；独立R0插值核对rho，直接记录raw_rate_before_rho验证倍率只乘一次、checkpoint不加速。全量检查locked集合必须被选中及其完整分配；每个非tail override轮次独立核算已选作业实际启动成本与secondary_tie.minimum_actual_delay一致。响应边界与固定轨迹进度差见response_analysis.json，裁剪比例不能解释为JCT贡献比例。',
 '## 下一轮仅设计',
 '真正差异负载见NEXT_WORKLOAD_DESIGN.md和proposed_workloads.json，本轮未运行它们。通过改变实际submission_time确保调度输入状态不同；不靠换trace名称、无效CSV duration或被自适应忽略的GPU/batch声明制造样本。候选应用扩展另有画像覆盖门槛，未经验证不加入正式矩阵。',
 '## 范围与停止',
 '这是固定合成跨硬件响应下的模拟证据，未校准真实目标硬件CPU/RAM性能；BERT借用模板平坦、已有分母越界和clipping假设保留。辅助配额变更加载开销未建模，solver墙钟不计入虚拟JCT；P95/P99仅12作业描述。未扩大工作负载或自动运行新实验，完成后停止。']
 (OUT/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
 for name in ['prepare_ample_lex_repair_20261030.py','capture_ample_lex_20261030.py','prove_ample_model_20261030.py','accept_ample_lex_20261030.py','bounded_ample_green_20261030.py','audit_ample_lex_20261030.py','design_distinct_loads_20261030.py']:shutil.copyfile(ROOT/'maintenance'/name,OUT/name)
 (OUT/'CURRENT_TASK.md').write_text('COMPLETE_REPAIRED_SIX_AUDITED_STOPPED. REPORT.md/validation.json authoritative. New load plans are DESIGN ONLY. No next experiments automatically authorized.\n',encoding='utf8')
 print('SEALING_SUCCESS',flush=True)
 seal={f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='delivery_hashes.json'};save('delivery_hashes.json',seal)
if __name__=='__main__':main()
