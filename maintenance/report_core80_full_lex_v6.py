"""Report every preregistered pair; never choose a best run."""
import collections,csv,json,time,ast
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run3';OLD=R/'research_runs/sia_synergy/cc_core80_repair_full_20261031_run1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):
 with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def fmt(x):return '—' if x is None else f'{x:.3f}' if isinstance(x,(int,float)) else str(x)
def main():
 ledger=[json.loads(x) for x in (O/'execution_manifest.jsonl').read_text().splitlines()];assert ledger[-1]['event']=='BATCH_END'
 p=read(O/'preregistration.json');m=read(O/'audit/raw_recomputed_metrics.json');pairs=read(O/'audit/paired_comparisons.json');factor=read(O/'audit/resource_factor_effects.json');ends={x['cell']:x for x in ledger if x['event']=='CELL_END'}
 coverage=[]
 for c in p['cells']:
  v=m.get(c['id'],{});coverage.append(dict(cell=c['id'],method=c['method'],state=v.get('status','NOT_RUN'),completed=v.get('completed'),mean_jct=v.get('mean_jct'),rounds=v.get('rounds'),wall_seconds=ends.get(c['id'],{}).get('wall_seconds')))
 save(O/'coverage.json',coverage)
 with (O/'coverage.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,list(coverage[0]));w.writeheader();w.writerows(coverage)
 def jobcsv(path):
  ans={}
  with path.open(encoding='utf8') as f:
   for x in csv.DictReader(f):ans.setdefault(x['cell'],{})[x['job']]=x
  return ans
 jobs=jobcsv(O/'audit/per_job.csv');oldjobs=jobcsv(OLD/'audit/per_job.csv');regressions=[]
 for method in ('SIA_P','SIA_AUX','PROP','CPU','MEM','FO'):
  cid='dual_medium_'+method
  if cid not in jobs:continue
  fields=('completion','jct','wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','migrations','gpu_changes','wasted_gpu_seconds')
  diff={j:{k:[oldjobs[cid][j][k],x[k]] for k in fields if x[k]!=oldjobs[cid][j][k]} for j,x in jobs[cid].items()};diff={j:x for j,x in diff.items() if x}
  regressions.append(dict(method=method,new_state=m[cid]['status'],exact_event_metrics=not diff,differences=diff))
 save(O/'audit/control_regressions.json',regressions)
 tail=[]
 for q in pairs:
  if not q.get('complete_pair'):continue
  worst=sorted(q['per_job_saved_seconds'],key=q['per_job_saved_seconds'].get)[:8]
  tail.append(dict(candidate=q['candidate'],baseline=q['baseline'],worst_jobs=[dict(job=j,saved=q['per_job_saved_seconds'][j],candidate=jobs[q['candidate']][j],baseline=jobs[q['baseline']][j]) for j in worst]))
 save(O/'audit/tail_diagnostics.json',tail)
 n=sum(x['state']=='ALL_COMPLETE' for x in coverage);viol=sum(v['capacity_and_execution_violations'] for v in m.values());hours=(ledger[-1]['unix']-ledger[0]['unix'])/3600
 s=[];add=s.append
 add(f'# FULL词典序修复及80作业完整资源对照\n\n**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**\n\n正式矩阵完整 **{n}/7格**，独立原始事件审计违规 **{viol}项**，补跑墙钟 **{hours:.3f}小时**。不插补删失JCT，不将不同源码版本混成配对。\n')
 add('## 1. 直接回答\n')
 for q in pairs:
  if q['candidate']=='dual_medium_FULL':
   add(f'- FULL vs {q["baseline"].removeprefix("dual_medium_")}：'+(f'平均JCT改善{fmt(q["improvement_percent"])}%，P95/P99/Makespan改善{fmt(q["p95_improvement_percent"])}/{fmt(q["p99_improvement_percent"])}/{fmt(q["makespan_improvement_percent"])}%，逐作业改善/退化/持平={q["improved_jobs"]}/{q["worse_jobs"]}/{q["ties"]}。' if q.get('complete_pair') else '不完整配对，禁止给出总体JCT改善率。'))
 add('正值为改善、负值为退化。单轮目标更优不保证平均JCT更低；FULL不必优于FO或每个单资源版本。\n')
 add('## 2. 根因、证据与最小修复\n')
 add('旧冻结修复版在第19次决策失败：131244列，前级主目标区间约[1.3341274568444956,1.3341274588444958]，启动代价275±1e-9，当前排名对象deepspeech2-10003；合法整数可行解排名成本5，旧LP排名下界0，两次HiGHS超时及CBC optimal_inaccurate不能证明成本5最优。失败不是容量不可行，也不是训练死锁。旧现场和封存hash未改。\n')
 add('旧版只用当前排名目标的下界和逐配置支配关系消元，遗漏已固定主目标对其他变量的约束；在窄浮点带、大量近似/重复列下搜索困难。单独把整数启动行等价改写成=275仍超时，probe_lattice.json保留此证据。\n')
 add('实现还将可行域消元前移到当前排名LP及其支持集证明之前；这些证明仅在已认证的等价域上执行，再把支持集、约化成本及整数格点索引还原到原变量编号。没有调整LP/MIP时间限或公平/完成目标。V4部分回归日志与源码见相邻run1/DEVELOPMENT_SUPERSEDED.json；中断日志未修补、没有启动正式实验。\n')
 add('第二个封存故障：V5在第37次决策的第二级启动目标超时，200462列，旧可行启动成本250，旧LP下界102.6124不足以证明最优，36完整轮/9完成。V6不再要求启动最优先固定才启用主目标域证明；仅主目标带已固定即可。离线模型排除171141列，保留29321列，仍以原约束和原30秒零gap求解，证明启动最优值250。没有更换数据、增大时间限或用未证明点继续。\n')
 add('最小工程修复：①整数行gcd格点向内取界，只有原数学域和原audit容差域都相同时才改写；该启动行化为精确275。②使用既有主目标带进行可证明安全的条件弱对偶消元。设c=−w，外包约束Ax≤b、单位盒；任意y≤0给出B=bᵀy+Σmin(0,cᵢ−Aᵢᵀy)。若xᵢ=1，目标下界至少B+max(0,cᵢ−Aᵢᵀy)。仅当它严格超过−L及原容差/浮点误差的保守上界时删除变量。这里L是原主目标下界，未增加新优化目标。\n')
 add('使用80位有向Decimal区间；包含原验证容差与保守点积误差。缓存绑定当前列对象及前三组原始约束的字节hash，后续加约束可复用，前缀改变则失效；LP失败不删变量；丢失现有合法incumbent则明确停止。全部返回解还须全局最优且逐行通过原始约束。候选网格、Goodput/评分、目标带、CPU/RAM容量、R0/phi/clipping和任何训练/锁语义均未修改。\n')
 old=read(O/'probe_old.json');green=read(O/'probe_repair.json')
 add(f'同一捕获模型红/绿：旧版{fmt(old["seconds"])}秒后再次拒绝；修复重放{fmt(green["seconds"])}秒取得最优排名{green.get("objective")}，主目标条件证书排除{green.get("primary_band_variables_removed")}列。只是求解实现改进，不是训练速度或JCT收益。\n')
 change=read(O/'source_change_inventory.json');add('改动文件：`'+ '`, `'.join(change['changed'])+'`；新增`'+'`, `'.join(change['new'])+'`。supervise80只新增默认关闭的工程失败停止选项；本次开启以在新工程失败时停止批次，非调度规则。完整diff在source_changes.patch。\n')
 for rel,name in [('sia-simulator/joint_prototype/clean_ablation.py','_decide'),('numeric_repair.py','canonical_bundle'),('integer_row_lattice.py','tighten_integer_rows'),('primary_band_presolve.py','primary_certificate'),('primary_band_presolve.py','certificate_on_domain')]:
  path=Path(p['source_root'])/rel;node=next(x for x in ast.walk(ast.parse(path.read_text(encoding='utf8'))) if isinstance(x,ast.FunctionDef) and x.name==name)
  add(f'- 代码依据：[{rel}::{name}]({path.as_posix()}:{node.lineno})。')
 add('## 3. 验收、冻结及公平条件\n')
 a=read(O/'software_acceptance.json');add('```json\n'+json.dumps(a,ensure_ascii=False,indent=2)+'\n```\n')
 add('红测red_backend_contract2.log捕获原后端未使用等价整数行；red_literal_replay.log复现真实131244列失败。red_backend_contract.log仅是首次测试cwd导入错误，不能算缺陷红测。probe_primary.log的离线工具稠密/稀疏兼容错误保留，修正后probe_primary2.log完成；未修改原矩阵。\n')
 add(f'来源：实际315文件旧源码完整复制，保持旧树及官方源；最终{len(p["source_hashes"])}文件，source_snapshot.zip SHA256 `{p["source_snapshot_sha256"]}`，输入{len(p["input_hashes"])}项逐一核验。没有可假定代表实际源码的裸Git HEAD；完整快照、patch和测试fixture共同重建，不推送。\n')
 add('320项测试含原302项与新增18项；真实FULL45轮跨越原第19轮及第三到达波，A1020/PROP12短回归，原Sia3轮精确回归。V4开发版315/316项测试通过，但其证明顺序仍在大矩阵上执行冗余LP；正式运行前已停止并封存V4，不作性能结果。V5新增证明域顺序和非连续索引映射回归，V5第37次调度在最小启动目标再次停止，已封存run2且不填补JCT。V6把相同主目标证书扩展到启动阶段之前，新增阶段和200462列真实模型回归，最终320项再通过。这里是有界正确性验收，不是80作业完成证据；下表来自全程正式补跑。\n')
 add('80作业、48CIFAR10/16DeepSpeech2/16BERT；40@0、20@600、20@1200的既有受控到达。原始训练模型、work、batch不变。AWS4GPU/12合成CPU/250GB、DGX-ext8/24/500、RTX8/24/500。R0+E-legacy、Medium alpha=.5、seed_profiles(1,8)、60秒/6000轮，7格独立进程串行。门槛2GiB/5GiB，每格计算watchdog6h、批次24h，不调整求解器30秒/零gap或原重试设置。\n')
 add('FULL/CPU/MEM/PROP共享完成导向目标、GPU域、服务/准入、切换成本，分别开放双资源、仅CPU、仅RAM、比例配额。FO共享CC目标/准入，先在比例域确定服务集合与GPU，再冻结GPU配置做辅助优化。FULL vs FO检验联合与串行决策，不单独识别服务集合和GPU选择贡献。SIA_P实际调用官方Sia GPU/placement后比例执行，不加CC/L2；SIA_AUX固定该意图后用自有辅助优化，不是官方Synergy-TUNE。\n')
 add('## 4. 七格绝对数值\n\n|方法|状态|完成|Mean JCT|Median|P95|P99|Makespan|轮数|墙钟(s)|\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
 for c in coverage:
  x=m.get(c['cell'],{});add('| '+' | '.join(map(str,[c['method'],c['state'],str(c['completed'])+'/80']+[fmt(x.get(k)) for k in ('mean_jct','median_jct','p95_jct','p99_jct','makespan')]+[c['rounds'],fmt(c['wall_seconds'])]))+' |')
 add('\nP95/P99是单一80作业场景经验分位数，不把80个作业当80次独立实验。失败/删失格均值留空。\n')
 for c in coverage:
  if c['state']=='ALL_COMPLETE':continue
  x=m.get(c['cell'],{});add(f'- 未完成格{c["method"]}：{c["state"]}，已记录完整轮次{x.get("rounds")}，最后记录时间{x.get("observed_simulated_end")}，失败时间{x.get("failure_simtime")}；错误`{x.get("failure_error")}`。失败区间不补写；NOT_RUN不等于算法失败。')
 add('## 5. 配对收益、互斥时间和双资源交互\n\n|候选 vs 对照|平均JCT改善%|节省秒/作业|等待/检查点/训练节省秒|改善/退化/平|\n|---|---:|---:|---|---|')
 for q in pairs:
  if not q.get('complete_pair'):continue
  d=q['mean_jct_decomposition_seconds'];add(f'|{q["candidate"].removeprefix("dual_medium_")} vs {q["baseline"].removeprefix("dual_medium_")}|{fmt(q["improvement_percent"])}|{fmt(q["mean_jct_saved"])}|{fmt(d["wait"])}/{fmt(d["checkpoint"])}/{fmt(d["training"])}|{q["improved_jobs"]}/{q["worse_jobs"]}/{q["ties"]}|')
 add('\nCPU×RAM四格（只有PROP/CPU/MEM/FULL全部完成才计算）：\n```json\n'+json.dumps(factor,ensure_ascii=False,indent=2)+'\n```\n')
 add('主效应为开启该因子的平均JCT节省秒；交互YFULL−YCPU−YMEM+YPROP，负值代表超加性改善。局部实际训练加速可能通过释放资源间接降低排队，时间分解只是会计恒等式，不能将全部等待减少独占归因某模块。\n')
 add('## 6. 资源、重启与约束审计\n\n|方法|GPU-h|CPU单位-h|RAM GB-h|等待(s)|检查点(s)|训练(s)|restart|wasted GPU-s|违规|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
 for c in coverage:
  x=m.get(c['cell'],{});t=x.get('totals',{});add('| '+' | '.join(map(str,[c['method']]+[fmt(t.get(k)) for k in ('gpu_hours','cpu_core_hours','ram_gb_hours','wait','checkpoint','training','restarts','wasted_gpu_seconds')]+[x.get('capacity_and_execution_violations')]))+' |')
 for c in coverage:
  x=m.get(c['cell'],{});add(f'- {c["method"]}：执行审计{x.get("execution_audits")}、完整锁检查{x.get("checkpoint_lock_checks")}、fallback{x.get("fallback_count")}，求解状态`{x.get("solver_status")}`；工程统计`{x.get("engineering_solver")}`；选择总/P99秒`{x.get("selection_reported_seconds")}`。')
 add('资源小时从执行片段allocated_seconds积分，周期中完成后不继续占用计时。checkpoint与训练互斥、rho只乘原始执行率一次；节点容量和锁检查不只看已选中的任务。仿真未将求解墙钟加入训练时间，计算开销须与JCT分开解读。\n')
 add('## 7. 长尾、资源响应和局限\n')
 for q in pairs:
  if not q.get('complete_pair') or q['candidate']!='dual_medium_FULL':continue
  add(f'- FULL vs {q["baseline"]}：最大受益{q["top_beneficiary"]}（原{q["top_original_name"]}），节省{fmt(q["top_saved_seconds"])}秒，占正净收益{fmt(100*q["top_fraction_of_net"] if q["top_fraction_of_net"] is not None else None)}%；去除此单作业诊断改善{fmt(q["diagnostic_without_top_improvement_percent"])}%，不替代包含全部80作业的主结果。应用净贡献`{json.dumps(q["by_application"],ensure_ascii=False)}`。')
 for c in coverage:
  x=m.get(c['cell'],{});add(f'- {c["method"]}：分子/分母分CPU/RAM的裁剪率% `{x.get("clipping_rates_percent")}`；数据内/裁剪训练秒`{x.get("training_response_exposure_seconds")}`；同节点真实差异化配额且receiver加速的重叠训练对秒{x.get("co_resident_different_quota_pair_seconds")}，其中全坐标数据内{x.get("fully_supported_co_resident_pair_seconds")}。同raw/config/batch比例参考的局部净等效训练秒`{x.get("local_proportional_reference_equivalent_training_seconds")}`；这不是完整反事实JCT。')
 add('所有策略共享R0/E-legacy合成跨硬件执行函数；比例点rho=1是参考归一化，不是无CPU瓶颈证明。裁剪允许且未改，尤其不能把边界外推假设下的收益称为目标硬件实测。真实共驻片段、局部预测/执行收益和全程JCT分别报告，禁止互相替代。最差8作业详见audit/tail_diagnostics.json；主结果不删长尾。\n')
 add('## 8. 同版本回归与交付\n')
 for x in regressions:add(f'- {x["method"]}：{x["new_state"]}；逐作业完成事件/JCT/等待/训练/检查点/资源小时/重启精确复现旧完整格={x["exact_event_metrics"]}。')
 add('若有差异，control_regressions.json保留逐项证据，不因均值接近就认定相同，不选择较好一次。完整正式结果均来自新320文件版本；旧结果只用于回归核验，不填入新矩阵。\n')
 add('原始rounds.jsonl.gz、solver诊断、输入/源码hash、source_snapshot.zip、source_changes.patch、320项测试和短回归均保留。本轮停止，不追加容量/响应档、九cohort、104/160或参数搜索。\n')
 report='\n'.join(s)
 for path in (O/'REPORT.md',R/'research_notes/synergy/CC_CORE80_FULL_LEX_REPAIR_V6_COMPARISON_20261031.md'):
  with path.open('x',encoding='utf8') as f:f.write(report)
 save(O/'result_summary.json',dict(complete=n,expected=7,violations=viol,wall_hours=hours,coverage=coverage,metrics=m,paired_comparisons=pairs,resource_factor=factor,regressions=regressions,source_sha256=p['source_snapshot_sha256']))
 print('REPORT_READY',n,'/7','violations',viol,flush=True)
if __name__=='__main__':main()
