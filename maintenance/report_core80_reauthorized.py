"""Report all fixed outcomes, including failures; never choose a best run."""
import collections,csv,hashlib,json,time,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research_runs/sia_synergy/cc_core80_repair_full_20261031_run1';F=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run3';OLD=R/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def fmt(x):return '—' if x is None else f'{x:.2f}' if isinstance(x,(float,int)) else str(x)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 ledger=[json.loads(x) for x in (O/'execution_manifest.jsonl').read_text().splitlines()];assert ledger[-1]['event']=='BATCH_END'
 p=read(O/'preregistration.json');m=read(O/'audit/raw_recomputed_metrics.json');pairs=read(O/'audit/paired_comparisons.json');factor=read(O/'audit/resource_factor_effects.json');service=read(O/'audit/service_factor_effects.json');ends={x['cell']:x for x in ledger if x['event']=='CELL_END'}
 coverage=[]
 for c in p['cells']:
  x=m.get(c['id'],{});e=ends.get(c['id'],{});status=x.get('status','NOT_STARTED')
  if (O/'cells'/c['id']/'COMPUTE_TIMEOUT.json').exists():status='COMPUTE_TIMEOUT'
  coverage.append(dict(cell=c['id'],method=c['method'],response=c['response'],status=status,completed=x.get('completed'),declared=80,mean_jct=x.get('mean_jct'),rounds=x.get('rounds'),wall_seconds=e.get('wall_seconds'),directory=str(O/'cells'/c['id'])))
 save(O/'coverage.json',dict(expected=11,counts=dict(collections.Counter(x['status'] for x in coverage)),cells=coverage))
 with (O/'coverage.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,list(coverage[0]));w.writeheader();w.writerows(coverage)
 jobs={}
 with (O/'audit/per_job.csv').open(encoding='utf8') as f:
  for x in csv.DictReader(f):jobs.setdefault(x['cell'],{})[x['job']]=x
 oldjobs={}
 with (OLD/'audit/per_job.csv').open(encoding='utf8') as f:
  for x in csv.DictReader(f):oldjobs.setdefault(x['cell'],{})[x['job']]=x
 regression=[]
 for cid in ('dual_medium_SIA_P','dual_medium_SIA_AUX','dual_neutral_A00','dual_neutral_A01','dual_neutral_A11'):
  if cid not in jobs:continue
  fields=('completion','jct','wait','checkpoint','training','restarts','migrations','gpu_changes','wasted_gpu_seconds');delta={j:{k:[oldjobs[cid][j][k],v[k]] for k in fields if oldjobs[cid][j][k]!=v[k]} for j,v in jobs[cid].items()};delta={j:v for j,v in delta.items() if v}
  regression.append(dict(cell=cid,new_status=m[cid]['status'],exact_job_event_metrics=not delta,differences=delta))
 save(O/'audit/old_complete_regressions.json',regression)
 tails=[]
 for q in pairs:
  if not q.get('complete_pair'):continue
  worst=sorted(q['per_job_saved_seconds'],key=q['per_job_saved_seconds'].get)[:8]
  tails.append(dict(candidate=q['candidate'],baseline=q['baseline'],worst_eight=[dict(job=j,saved_seconds=q['per_job_saved_seconds'][j],candidate=jobs[q['candidate']][j],baseline=jobs[q['baseline']][j]) for j in worst]))
 save(O/'audit/tail_diagnostics.json',tails)
 complete=sum(x['status']=='ALL_COMPLETE' for x in coverage);viol=sum(x['capacity_and_execution_violations'] for x in m.values());s=[];add=s.append
 add(f'# 冻结修复版：11格80作业补跑（{complete}/11格完整）\n\n**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**\n')
 add(f'本轮固定矩阵完成 **{complete}/11格**，独立原始事件审计违规 **{viol}项**。下表保留全部完成、删失、失败或计算超时结果；不为未完成作业插补JCT，也不以已完成子集均值代替总体均值。\n')
 add('### 直接结论\n')
 for candidate,baseline,meaning in [('SIA_AUX','SIA_P','相同官方Sia接口后的辅助资源优化'),('PROP','SIA_P','CC比例配额整套策略，含目标/准入/配置变化'),('CPU','PROP','同一CC策略下仅开放CPU差异化配额'),('MEM','PROP','同一CC策略下仅开放内存差异化配额')]:
  q=next((q for q in pairs if q['candidate']=='dual_medium_'+candidate and q['baseline']=='dual_medium_'+baseline and q.get('complete_pair')),None)
  if q:add(f'- {meaning}：{candidate}相对{baseline}平均JCT改善{fmt(q["improvement_percent"])}%，P99改善{fmt(q["p99_improvement_percent"])}%，Makespan改善{fmt(q["makespan_improvement_percent"])}%；负值为退化。')
 if m.get('dual_medium_FULL',{}).get('status')!='ALL_COMPLETE':add('- 完整双资源联合格FULL未完整完成，不能声称它相对Sia、串行FO或单资源优化更好，也不能计算完整CPU×内存交互效应。工程求解失败不等于训练性能退化。')
 add('- 仅将完整且原始审计有效的配对用于JCT比较；等待/训练/检查点分解属于会计归因，不能据此单独证明某机制的因果贡献。\n')
 add('## 1. 授权、冻结与公平性\n')
 add('研究者明确解除本轮11格80作业暂停后，使用全新目录重跑；旧阻止记录、失败尝试、源码及历史结果未覆盖。源树`cc-joint-core80-repair-v3`保持315文件冻结，源码zip SHA-256 `7894EA3C16587113301D4568FAA09C2116E7EB88A171864C5B3D6AFBAD584DCC`；38科学输入hash、依赖及302/302测试验收在启动前核验。原Sia有界回归一致，FULL12/A1020/PROP12故障状态回归通过。\n')
 add('没有修改目标、p、lambda、allocation bonus、CPU/RAM网格、R0/E-legacy、phi、alpha、容量、workload、训练工作量或服务语义。仅使用已验收的规范化整数解验证、精确变量消去/对偶证书/可行上界及独立CBC安全重试。HiGHS30秒/zero-gap、原无presolve30秒重试，CBC30秒/zero-gap不变；只有最优且原始行通过的解可提交。数值后端重试与调度fallback分开计数。\n')
 add('固定80声明为48CIFAR10/16DeepSpeech2/16BERT，真实Saturn记录的batch/GPU需求/训练工作量保留，40@0、20@600、20@1200秒受控到达。不是自然完整trace重放。AWS4GPU/12合成CPU/250GB，DGX-ext8/24/500、RTX8/24/500；所有策略相同。60秒、6000模拟轮、seed_profiles(1,8)。每格独立进程串行，启动门槛2GiB/5GiB；程序计算上限6h/格、24h/批次不等于训练时长或JCT。\n')
 add('## 2. 方法与可识别的因果因素\n')
 add('|组|方法|定义|\n|---|---|---|\n|资源 Medium|SIA_P|实际官方Sia GPU/placement决策＋比例CPU/RAM执行适配，无L2|\n|资源 Medium|SIA_AUX|实际Sia当轮意图固定后的自有辅助资源优化；不是官方Synergy-TUNE|\n|资源 Medium|PROP/CPU/MEM/FULL|共享完成优先目标、候选、FIFO/持续服务及切换成本；分别比例、仅CPU自由、仅内存自由、双资源自由|\n|资源 Medium|FO|同一CC目标/准入先以比例域选GPU及服务集合，再冻结GPU配置优化辅助资源；不是原Sia|\n|服务 Neutral|A00/A01/A10/A11|共同公平效用与历史/锁约束；第一位关闭/开启FIFO队首准入，第二位300秒有效训练量子/持续到完成|\n')
 add('A组和资源组目标不同，不能把两组百分比相加。JOINT-P/PROP相对Sia还同时改变目标、准入及配置选择，不能将其全部收益归因CPU感知。A00仍保留共同的服务状态、候选与历史约束，不等于官方Sia。A11 Neutral与SIA_P Medium的跨组比较仅作描述：两者比例点均需经独立执行审计确认rho=1，不能替代同目标同响应的因子消融。JCT的等待/训练/检查点分解是会计分解，真正资源因子效果应看同组受控消融。\n')
 add('## 3. 11格绝对结果\n')
 add('|方法|状态|完成|均值JCT(s)|Median|P95|P99|Makespan|轮数|计算墙钟(s)|\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
 for c in coverage:
  x=m.get(c['cell'],{});add('| '+' | '.join(map(str,[c['method'],c['status'],str(c['completed'])+'/80']+[fmt(x.get(k)) for k in ('mean_jct','median_jct','p95_jct','p99_jct','makespan')]+[c['rounds'],fmt(c['wall_seconds'])]))+' |')
 add('\nP95/P99为单一80作业场景的经验分位数，不能当高精度总体尾延迟估计；没有把80作业视作80次独立实验。\n')
 add('### 未完成/失败的解释\n')
 for c in coverage:
  if c['status']=='ALL_COMPLETE':continue
  x=m.get(c['cell'],{});add(f'- {c["method"]}：{c["status"]}，已记录{x.get("rounds")}个完整轮次，失败时间{x.get("failure_simtime")}，最后完整时间{x.get("observed_simulated_end")}；未记录的失败间隔{x.get("unlogged_failed_interval_seconds")}秒不补写、不插补。错误`{x.get("failure_error")}`。')
  diag=O/'cells'/c['cell']/'failure_solver_diagnostics.json.gz'
  if diag.exists():
   import gzip
   d=json.loads(gzip.decompress(diag.read_bytes()));model=d.get('model',{});proof=d.get('dual_proof') or {};names={v['job_id'] for v,w in zip(model.get('columns',[]),model.get('c',[])) if w!=0.}
   add(f'  求解器首次/重试status={d.get("original_status")}/{d.get("retry_status")}；CBC={d.get("independent_cbc")}；当前成本={proof.get("objective")}，下界={proof.get("lower_bound")}；本阶段非零成本任务={sorted(names)}。没有证明可用最优解不等于资源违规、训练死锁或JCT性能退化；主结果不计算失败格总体均值。')
 add('## 4. 完整配对、排队与辅助资源贡献\n')
 add('|候选 vs 对照|均值JCT改善%|平均节省秒|等待/检查点/训练节省秒|改善/退化/持平|P99改善%|Makespan改善%|\n|---|---:|---:|---|---|---:|---:|')
 for q in pairs:
  if not q.get('complete_pair'):continue
  d=q['mean_jct_decomposition_seconds'];label=lambda v:v.removeprefix('dual_medium_').removeprefix('dual_neutral_')
  add(f'| {label(q["candidate"])} vs {label(q["baseline"])} | {fmt(q["improvement_percent"])} | {fmt(q["mean_jct_saved"])} | {fmt(d["wait"])}/{fmt(d["checkpoint"])}/{fmt(d["training"])} | {q["improved_jobs"]}/{q["worse_jobs"]}/{q["ties"]} | {fmt(q["p99_improvement_percent"])} | {fmt(q["makespan_improvement_percent"])} |')
 add('\n正值为降低JCT/延迟，负值为退化；未完成配对不给出改善率，见paired_comparisons.json的complete_pair。\n')
 add('资源2×2（仅四格完整才计算）：\n```json\n'+json.dumps(factor,ensure_ascii=False,indent=2)+'\n```\n')
 add('有序准入×持续服务2×2：\n```json\n'+json.dumps(service,ensure_ascii=False,indent=2)+'\n```\n')
 add('主效应正数表示平均节省JCT秒；交互项Y11−Y10−Y01+Y00为负表示超加性改善。A00/A01无差异只说明本场景下量子重入队未改变最终服务轨迹，不证明持续服务普遍无效。\n')
 add('## 5. 资源、执行、求解与边界审计\n')
 add('|方法|GPU-h|CPU单位-h|RAM GB-h|等待(s)|检查点(s)|训练(s)|restart|wasted GPU-s|违规|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
 for c in coverage:
  x=m.get(c['cell'],{});t=x.get('totals',{});add('| '+' | '.join(map(str,[c['method']]+[fmt(t.get(k)) for k in ('gpu_hours','cpu_core_hours','ram_gb_hours','wait','checkpoint','training','restarts','wasted_gpu_seconds')]+[x.get('capacity_and_execution_violations')]))+' |')
 add('\n从原始完成事件、训练片段和allocated_seconds独立积分，作业周期中完成后不继续计资源。检查未选中锁作业、完整锁几何/配额、节点容量、单次rho、互斥时间、进度及原Sia/FO意图。完整输出summary只用于交叉核对，不作为唯一指标来源。\n')
 for c in coverage:
  x=m.get(c['cell'],{});add(f'- {c["method"]}：执行检查{x.get("execution_audits")}，锁检查{x.get("checkpoint_lock_checks")}，等待轮{x.get("waiting_rounds")}，CPU饱和/GPU空闲节点轮{x.get("cpu_saturated_gpu_idle_node_rounds")}；fallback={x.get("fallback_count")}；数值事件`{x.get("numerical_event_counts")}`；实际整数后端/LP/证明统计`{x.get("engineering_solver")}`；clipping`{x.get("clipping_counts")}`；裁剪率%`{x.get("clipping_rates_percent")}`；训练暴露`{x.get("training_response_exposure_seconds")}`。')
 add('\n整数后端总次数包含支撑集上界子问题、重试和CBC；外层solver_trace不是实际后端次数。后端计时含其调用封装，选择/优化计时还含候选构建和审计。仿真不把求解墙钟加入作业训练时钟，开销应单列，不应当作真实性能。\n')
 add('## 6. 长尾与局部训练收益\n')
 for q in pairs:
  if not q.get('complete_pair'):continue
  add(f'- {q["candidate"]} vs {q["baseline"]}：最大受益{q["top_beneficiary"]}（原{q["top_original_name"]}），节省{fmt(q["top_saved_seconds"])}秒；占净节省{fmt(None if q["top_fraction_of_net"] is None else 100*q["top_fraction_of_net"])}%；剔除此单作业的**诊断**改善{fmt(q["diagnostic_without_top_improvement_percent"])}%，不替代80作业主结果。应用净贡献`{json.dumps(q["by_application"],ensure_ascii=False)}`。')
 for name in ('PROP','CPU','MEM','FULL','FO','SIA_AUX'):
  x=m.get('dual_medium_'+name)
  if x:add(f'- {name}同状态/同raw/config/batch的比例参考等效训练秒`{x.get("local_proportional_reference_equivalent_training_seconds")}`，其中数据内/裁剪支持分层`{x.get("local_reference_gain_by_data_support")}`；这不是完整反事实重放或JCT因果贡献。')
 add('\n最差8作业的逐项JCT、首轮/后续等待、训练、检查点和重启见audit/tail_diagnostics.json；主分析不删长任务。\n')
 add('## 7. 历史回归及可复现性\n')
 for x in regression:add(f'- {x["cell"]}：新状态{x["new_status"]}；与旧完整结果逐作业完成/JCT/互斥时间/restart等精确一致={x["exact_job_event_metrics"]}。')
 add('\n若有差异，old_complete_regressions.json逐项保留，不因均值接近就认定相同，不挑选较好版本。源码及依赖快照见工程冻结目录；本目录保留新授权、预检、固定命令、逐轮压缩JSON、每格manifest/hash、逐作业CSV和独立审计。科学条件一致不等于预测模型已被真实硬件校准。\n')
 hours=(ledger[-1]['unix']-ledger[0]['unix'])/3600;add(f'本轮批次实测墙钟 **{hours:.3f}小时**。全部11格来自同一315文件修复版，没有与旧5个完成格混成新矩阵。\n')
 add('## 8. 结论边界\n')
 add('这是一组受控80作业、三类GPU、合成CPU/RAM迁移的机制对照，不能证明Sia普遍存在缺陷，也不是同平台真实Saturn测量。E-legacy分子/分母裁剪仍存在，报告必须与clipping暴露一同解释。更大规模/跨负载结论不由这一个场景保证。对于未完成或失败格，保持证据、不给完整JCT结论；不通过删任务、改响应、改预算内配额或改目标使方法获胜。本轮结束后停止，不自动追加可选4格、104/160作业或调参。\n')
 with (O/'REBUILD_CURRENT_RUN.md').open('x',encoding='utf8') as f:f.write('This is the actual reauthorized11-cell run; see authorization.json,preregistration.json and coverage.json. REBUILD.md copied from the engineering packet preserves old pause instructions as historical test material. Those were superseded ONLY for this batch by the new explicit human authorization. After this batch, stop again: no automatic rerun or new scenarios. Verify315source files and38inputs, use dependency lock, restore external test fixtures at their recorded layout. source_snapshot.zip is the actual immutable source, not a bare Git HEAD. Analysis commands are archived in analysis_scripts.zip; original raw logs never rewritten.\n')
 report='\n'.join(s)
 for path in (O/'REPORT.md',R/'research_notes/synergy/CC_CORE80_FROZEN_REPAIR_FULL_COMPARISON_20261031.md'):
  with path.open('x',encoding='utf8') as f:f.write(report)
 save(O/'result_summary.json',dict(complete=complete,expected=11,violations=viol,wall_hours=hours,metrics=m,paired_comparisons=pairs,resource_factor=factor,service_factor=service,source_sha256=p['source_snapshot_sha256']))
 print('FINAL_REPORT_READY',complete,'/11','violations',viol,'hours',hours)
if __name__=='__main__':main()
