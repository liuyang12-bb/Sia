"""Build an outcome-neutral repair report from independent raw-event audits."""
import collections,csv,hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run1';O=P/'attempt2';OLD=R/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def fmt(v):return '—' if v is None else f'{v:.2f}' if isinstance(v,(float,int)) else str(v)
def main():
 a=O/'audit';m=read(a/'raw_recomputed_metrics.json');pairs=read(a/'paired_comparisons.json');p=read(O/'preregistration.json');ledger=[json.loads(x) for x in (O/'execution_manifest.jsonl').read_text().splitlines()]
 assert ledger[-1]['event']=='BATCH_END','Do not finalize while the batch is running'
 end={x['cell']:x for x in ledger if x['event']=='CELL_END'};old=read(OLD/'audit/raw_recomputed_metrics.json');start=1793419473.
 rows=[]
 for cell in p['cells']:
  cid=cell['id'];x=m.get(cid,{});e=end.get(cid,{});status=x.get('status','NOT_STARTED')
  if status=='COMPUTE_TIMEOUT_OR_INTERRUPTED' and (O/'cells'/cid/'COMPUTE_TIMEOUT.json').exists():status='COMPUTE_TIMEOUT'
  rows.append(dict(cell=cid,method=cell['method'],status=status,completed=x.get('completed'),total=80,rounds=x.get('rounds'),mean_jct=x.get('mean_jct'),p95=x.get('p95_jct'),p99=x.get('p99_jct'),makespan=x.get('makespan'),wall_seconds=e.get('wall_seconds'),violations=x.get('capacity_and_execution_violations'),directory=str(O/'cells'/cid)))
 save(a/'repair_coverage.json',dict(cells=rows,complete=sum(x['status']=='ALL_COMPLETE' for x in rows),expected=11,counts=dict(collections.Counter(x['status'] for x in rows))))
 with (a/'repair_coverage.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 jobs={}
 with (a/'per_job.csv').open(encoding='utf8') as f:
  for x in csv.DictReader(f):jobs.setdefault(x['cell'],{})[x['job']]=x
 oldjobs={}
 with (OLD/'audit/per_job.csv').open(encoding='utf8') as f:
  for x in csv.DictReader(f):oldjobs.setdefault(x['cell'],{})[x['job']]=x
 regressions=[]
 for cid in ['dual_medium_SIA_P','dual_medium_SIA_AUX','dual_neutral_A00','dual_neutral_A01','dual_neutral_A11']:
  if cid not in m:continue
  fields=('jct','completion','wait','checkpoint','training','restarts','migrations','gpu_changes','wasted_gpu_seconds')
  different={j:{k:[oldjobs[cid][j][k],v[k]] for k in fields if oldjobs[cid][j][k]!=v[k]} for j,v in jobs[cid].items()}
  different={j:v for j,v in different.items() if v}
  regressions.append(dict(cell=cid,old_complete=old[cid]['status']=='ALL_COMPLETE',new_status=m[cid]['status'],old_mean=old[cid]['mean_jct'],new_mean=m[cid]['mean_jct'],all_job_event_metrics_exact=not different,differences=different))
 save(a/'historical_complete_regressions.json',regressions)
 tail=[]
 for q in pairs:
  if not q.get('complete_pair'):continue
  worst=sorted(q['per_job_saved_seconds'],key=q['per_job_saved_seconds'].get)[:8]
  tail.append(dict(candidate=q['candidate'],baseline=q['baseline'],worst_eight=[dict(job=j,saved_seconds=q['per_job_saved_seconds'][j],candidate=jobs[q['candidate']][j],baseline=jobs[q['baseline']][j]) for j in worst]))
 save(a/'tail_diagnostics.json',tail)
 s=[];add=s.append
 add('# 80作业：数值验收与等价计算修复报告\n')
 add('**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**\n')
 good=sum(x['status']=='ALL_COMPLETE' for x in rows);viol=sum(x.get('capacity_and_execution_violations',0) for x in m.values())
 add(f'本次11格修复重跑完成 **{good}/11格**；独立原始事件审计记录 **{viol}项违规**。未完成格不计算总体平均JCT，不插补作业。本轮仅修复工程数值验收与等价计算，未改变公平/完成目标、准入/持续服务、冷启动规则、候选资源网格、容量、工作量或响应模型。\n')
 add('## 1. 现场与修复证据\n')
 add('上轮FULL在第5次真实调度附近被原子保护拒绝，A10在第13次附近出现词典序目标不一致；PROP/CPU/MEM因程序计算墙钟上限未完成。旧树和封存产物全部保留。复现的FULL模型含68394列：HiGHS首次返回status0，原始整数误差6.048878e-7，原始行满足约束，舍入后全部原始行仍满足；旧Bundle先用1e-7整数门槛拒绝，导致上层已有1e-6规范化无机会执行。不能把求解器超时的可行向量当作最优解。\n')
 add('修复在`numeric_repair.py:canonical_bundle`先执行既有规范化并检查全部原始行；不放宽主目标±1e-9带或行容差。窄带数值问题才使用可逆二进制幂缩放和原有presolve=False重试。作用域内status1只允许一次固定30秒/zero-gap新求解，只有新status0且原始约束通过才能提交；两次失败保留模型并停止该格。\n')
 add('`factor_cache.py:fast_factor`用同一完整资源网格计算公共归一化最大值，再仅物化原有因子会保留的点；每轮重算当前训练率、剩余工作和delay。`exact_grid_presolve.py`只消去可证明不可能出现在最终词典序最优解中的变量：同任务/配置/放置的资源支配点，以及既有零效用、正启动成本定理覆盖的可选任务。FIFO队首/服务/checkpoint必须选中者始终保留。完整点集仍用于normalizer/oracle/提交审计。已证明达到逐任务松弛下界的当前可行向量才可跳过重复求解。\n')
 add('SIA_AUX仍调用原Sia：只把原先立即被丢弃的离意图配置提前过滤，固定GPU/placement域和目标不变。新修复通过显式context启用；其他历史类不受隐式猴子补丁影响。\n')
 add('## 2. 红绿测试与版本保护\n')
 add('原276项测试完整保留，新13项，总计 **289/289通过（244.460秒）**。捕获模型红测有1failure+1error，真实触发旧整数验收和窄带不可行问题；绿测覆盖超时不得冒充最优、非法容量/主目标拒绝、固定一次重试、穷举验证精确变量消去、FIFO/lock保留、全部资源因子同状态一致及真实未seed BERT链路。早期fixture字段/上下文错误已分别保留并说明，不计作算法缺陷。\n')
 add('真实80作业有界回归：FULL8次、A10前20次均穿越旧失败点；PROP前12次正常。FULL前4次、A10/PROP前12次与旧成功前缀的选择、分数、执行逐轮精确匹配，唯一忽略项是迁移后的template_source路径字符串，模板SHA仍比较。原始Sia3轮分配/进度/restart/完成事件逐项一致。\n')
 freeze=read(P/'freeze_receipt.json');delta=read(P/'source_delta.json')
 add(f'独立源码`cc-joint-core80-repair/`保存{freeze["source_files"]}文件；旧302文件中{delta["unchanged_count"]}个未改，3个已有文件修改、8个新增文件，0删除；38个输入内容hash相同。源码zip SHA-256：`{freeze["source_snapshot_sha256"]}`。完整patch见`source_changes.patch`。本轮以实际源码快照为版本依据，不把裸HEAD冒称执行版本。\n')
 add('初次启动遗漏`cells/`父目录，11个子进程均在创建输出目录、构造cluster之前退出；完整日志留在根目录。补齐父目录后的实际运行在`attempt2/`，源码与科学参数完全相同，没有重写初次失败记录。\n')
 add('## 3. 冻结输入与运行规则\n')
 add('同一80个Saturn真实训练声明：48 CIFAR10、16 DeepSpeech2、16 BERT；保留batch、原工作量和profile，受控到达40@0、20@600、20@1200秒。这是合成竞争到达，不是自然80作业trace重放。AWS4GPU/12CPU/250GB，DGX-ext8/24/500、RTX8/24/500，CPU为合成单位。R0、E-legacy、alpha=0或0.5、seed_profiles(1,8)、60秒、6000轮未变；分子/分母裁剪照实记录。\n')
 add('新修复任务已与上轮8小时任务分开。程序共同计算墙钟预算6小时/格、24小时/批次，原生MIP仍30秒、gap0；扩大程序预算不增加模拟训练工作或观察轮数。每格独立进程串行，启动门槛2GiB/5GiB。11格固定顺序FULL→A10→PROP→CPU→MEM→FO→SIA_P→SIA_AUX→A00→A01→A11；不运行旧4个可选格或160作业。\n')
 add('SIA_P=原Sia GPU/placement+比例执行；SIA_AUX=同Sia意图+自有辅助资源优化，不能称官方Synergy-TUNE。PROP/CPU/MEM/FULL共享完成优先、FIFO和持续服务，仅比例/仅CPU/仅RAM/双资源自由不同；FO为同控制器先GPU后aux，不能称原Sia。A00/01/10/11共享Sia公平效用，分别开关有序准入和持续服务；它们与资源组目标不同，收益不可直接相加。\n')
 add('## 4. 全部结果\n')
 add('| 方法 | 状态 | 完成 | 平均JCT秒 | P95 | P99 | Makespan | 轮数 | 墙钟秒 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|')
 for x in rows:add('| '+' | '.join(map(str,[x['method'],x['status'],str(x['completed'])+'/80',fmt(x['mean_jct']),fmt(x['p95']),fmt(x['p99']),fmt(x['makespan']),x['rounds'],fmt(x['wall_seconds'])]))+' |')
 add('\n每格仅80作业的经验P95/P99，不是高精度总体尾延迟估计。计算超时、求解失败与模拟6000轮删失分别记录。\n')
 add('## 5. 完整配对与两条主线\n')
 add('| 候选 vs 对照 | 平均JCT改善率% | 平均节省秒 | 等待/检查点/训练节省秒 | 改善/退化/持平作业 | P99改善% | Makespan改善% |\n|---|---:|---:|---|---|---:|---:|')
 for q in pairs:
  if q.get('complete_pair'):
   d=q['mean_jct_decomposition_seconds'];add(f'| {q["candidate"].removeprefix("dual_medium_").removeprefix("dual_neutral_")} vs {q["baseline"].removeprefix("dual_medium_").removeprefix("dual_neutral_")} | {fmt(q["improvement_percent"])} | {fmt(q["mean_jct_saved"])} | {fmt(d["wait"])}/{fmt(d["checkpoint"])}/{fmt(d["training"])} | {q["improved_jobs"]}/{q["worse_jobs"]}/{q["ties"]} | {fmt(q["p99_improvement_percent"])} | {fmt(q["makespan_improvement_percent"])} |')
 add('\n正值=改善；负值=退化。总JCT差严格分解为互斥等待+checkpoint+有效训练，不能把“更早服务”全部归于CPU感知。\n')
 add('服务2×2效应：\n```json\n'+json.dumps(read(a/'service_factor_effects.json'),ensure_ascii=False,indent=2)+'\n```\n')
 add('CPU×内存2×2效应：\n```json\n'+json.dumps(read(a/'resource_factor_effects.json'),ensure_ascii=False,indent=2)+'\n```\n')
 add('正主效应表示降低JCT；交互项Y11−Y10−Y01+Y00为负才表示超加性改善。空对象表示配对不足，不能推断无效。\n')
 add('## 6. 资源、执行与长尾审计\n')
 add('| 方法 | GPU小时 | CPU单位小时 | RAM-GB小时 | 总等待秒 | checkpoint秒 | restart | wasted GPU秒 | 执行审计数 | 违规数 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
 for x in rows:
  v=m.get(x['cell'],{});t=v.get('totals',{});add('| '+' | '.join(map(str,[x['method']]+[fmt(t.get(k)) for k in ('gpu_hours','cpu_core_hours','ram_gb_hours','wait','checkpoint','restarts','wasted_gpu_seconds')]+[v.get('execution_audits',0),v.get('capacity_and_execution_violations',0)]))+' |')
 add('\n资源小时由原始allocated_seconds积分，作业周期中完成后不继续计费；训练倍率用施加前raw_rate_before_rho×独立重算rho校验，不反推raw。锁定作业既检查完整几何/配额，又检查是否被漏选。原始完成事件逐作业复算并与summary交叉核对。\n')
 for q in pairs:
  if not q.get('complete_pair') or q['baseline'] not in ('dual_medium_SIA_P','dual_medium_PROP','dual_medium_FO'):continue
  add(f'- {q["candidate"]} vs {q["baseline"]}：最大获益作业{q["top_beneficiary"]}（原名{q["top_original_name"]}），节省{fmt(q["top_saved_seconds"])}秒，占净节省{fmt(None if q["top_fraction_of_net"] is None else 100*q["top_fraction_of_net"])}%；去掉该作业的诊断改善{fmt(q["diagnostic_without_top_improvement_percent"])}%。主结果保留全部80作业。应用贡献：`{json.dumps(q["by_application"],ensure_ascii=False)}`。')
 add('\n最差8作业逐项JCT/等待/训练/restart变化见`audit/tail_diagnostics.json`，不能删除长任务来改善主均值。\n')
 add('## 7. 数值状态、计算效率与历史桥接\n')
 for x in rows:
  v=m.get(x['cell'],{});add(f'- {x["method"]}：round/job/optimize总墙钟`{v.get("computation_wall_seconds")}`；实际后端/证明统计`{v.get("engineering_solver")}`；fallback={v.get("fallback_count")}；状态`{v.get("solver_status")}`；clipping`{v.get("clipping_counts")}`。')
 add('\n证书是数学下界/原始约束验证，不能把证书计成一次实际后端调用。raw timer中的optimize包含候选构建、审计和多MIP控制，不只是单个求解。仿真时钟不加求解墙钟，必须将工程开销单独披露。\n')
 for x in regressions:add(f'- 旧完整格{x["cell"]}：新状态{x["new_status"]}，逐作业完成/JCT/互斥时间/restart等精确相同={x["all_job_event_metrics_exact"]}；旧/新均值{x["old_mean"]}/{x["new_mean"]}。')
 add('\n详细差异即使数值微小也留在`historical_complete_regressions.json`；不得因接近就声称一致。\n')
 add('## 8. 结论适用范围与恢复材料\n')
 add('数据可证明受限资源下存在可落地的CPU/RAM互补，但训练效率的局部预测与JCT因果贡献不同；E-legacy跨硬件合成和坐标裁剪仍是外部有效性边界。冷启动零效用最小启动成本导致的早期1→2→3任务启动行为本轮未改，不能包装为已解决的活性/效率创新。本次只有一个固定80作业场景，没有新增随机重复或多负载统计独立性。\n')
 add('新源码zip、完整逐文件SHA、patch、38输入SHA、依赖锁、289项测试、红测失败、未采用技术版本均在父目录。实际原始日志/summary/manifest及独立复算在attempt2。原始旧run3与旧报告不覆盖。后续是否扩大验证须审阅完整配对、尾部和裁剪结果；本轮停止，不自动调参或增跑。\n')
 now=time.time();add(f'墙钟：实际批次{(ledger[-1]["unix"]-ledger[0]["unix"])/3600:.3f}小时；从本次修复任务起（04:04:33UTC）至报告生成{(now-start)/3600:.3f}小时。源码及输出最终seal另列，避免自引用。\n')
 report='\n'.join(s)
 for path in [P/'REPORT.md',R/'research_notes/synergy/CC_CORE80_REPAIR_REPORT_20261031.md']:
  with path.open('x',encoding='utf8') as f:f.write(report)
 print('REPORT_READY',good,'/11','violations',viol)
if __name__=='__main__':main()
