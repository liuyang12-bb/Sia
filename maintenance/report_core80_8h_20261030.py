"""Build a new report from independently recomputed events, not partial JCTs."""
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3';A=OUT/'audit'
def read(n):return json.loads((A/n).read_text(encoding='utf8'))
def number(v):return '—' if v is None else f'{v:.2f}'
def main():
 m=read('raw_recomputed_metrics.json');pairs=read('paired_comparisons.json');effects=read('service_factor_effects.json');resource_effects=read('resource_factor_effects.json');cov=read('coverage.json');p=json.loads((OUT/'preregistration.json').read_text());budget=json.loads((OUT/'budget.json').read_text());elapsed=time.time()-budget['start_unix']
 terminal=read('termination_analysis.json');classified={c['cell']:c for c in terminal['cells']}
 conclusions=['## 结论先行','']
 for candidate,baseline,label in [
    ('dual_medium_SIA_AUX','dual_medium_SIA_P','保留官方Sia决策接口后，串行辅助资源优化的闭环增量'),
    ('dual_neutral_A11','dual_medium_SIA_P','有序启动＋持续服务组合相对官方Sia决策适配'),
    ('dual_neutral_A11','dual_neutral_A01','持续服务开启时，有序启动的条件增量'),
    ('dual_neutral_A01','dual_neutral_A00','无有序头部约束时，持续服务的条件增量'),
    ('dual_medium_CPU','dual_medium_PROP','共同完成导向控制器下CPU优化增量'),
    ('dual_medium_MEM','dual_medium_PROP','共同完成导向控制器下内存优化增量'),
    ('dual_medium_FULL','dual_medium_PROP','共同完成导向控制器下CPU＋内存优化增量')]:
  q=next((q for q in pairs if q['candidate']==candidate and q['baseline']==baseline),None)
  if q and q['complete_pair']:
   conclusions.append(f"- {label}：平均JCT减少{q['mean_jct_saved']:.2f}s（{q['improvement_percent']:.2f}%），P99变化按改善率计为{q['p99_improvement_percent']:.2f}%，makespan改善率{q['makespan_improvement_percent']:.2f}%。正值改善，负值退化。")
   d=q['mean_jct_decomposition_seconds']
   conclusions.append(f"  每作业平均节省分解：等待{d['wait']:.2f}s（首次{d['first_wait']:.2f}s、后续{d['later_wait']:.2f}s）、checkpoint {d['checkpoint']:.2f}s、实际训练{d['training']:.2f}s。此分解是观察到的时间账户，不把训练时间变化全部归因于辅助资源。")
  else:conclusions.append(f'- {label}：缺少完整有效配对，**本轮不可给出完整平均JCT贡献**。')
 conclusions+=['',f"核心11格全部尝试：{terminal['counts'].get('ALL_COMPLETE',0)}格完整完成、{terminal['counts'].get('FAILED',0)}格失败、{terminal['counts'].get('COMPUTE_TIMEOUT',0)}格计算超时。可选4格未启动，完整主格与配对控制未通过扩展门槛，剩余保留时间也不足以完成两组可选配对。",
  'FULL、FO触发数值验收保护，A10触发词典序一致性保护，失败原始现场保留；没有更改目标或容差后择优重跑。未完成资源格不能由已完成作业均值、局部加速或静态机会替代整体JCT结论。四格主效应与交互只在全部有效完成时计算。',
  'SIA_P是实际官方GPU/placement决策的执行适配；A组共同控制器还含资格、生命周期启动预算等约束，不能把A11对Sia的全部差异归因于单一机制。资源组采用另一完成导向目标，A/B组效果不叠加。','']
 aux=m.get('dual_medium_SIA_AUX',{})
 if aux.get('status')=='ALL_COMPLETE':
  support=aux['local_reference_gain_by_data_support'];exposure=aux['training_response_exposure_seconds']
  conclusions+=['辅助资源局部增益并非只能通过裁剪点产生：SIA_AUX中完整受矩阵支持的训练段，相对同状态比例点净增益为'+number(support['fully_supported']['net'])+'个训练等价秒；裁剪段净值为'+number(support['clipped']['net'])+'秒。两者合计'+number(sum(x['net'] for x in support.values()))+'秒。这只是固定GPU/raw/batch训练段诊断，不能直接解释为整体JCT的因果分解。实际训练仍有'+number(100*exposure['clipped']/sum(exposure.values()))+'%秒暴露于边界假设，保留模型不确定性。','']
 rows=['# CC联合调度：80作业、8小时核心验证','',
 '**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**','',
 f"本轮核心完成{sum(m.get(c['id'],{}).get('status')=='ALL_COMPLETE' for c in p['cells'] if not c['optional'])}/11格；可选完成{sum(m.get(c['id'],{}).get('status')=='ALL_COMPLETE' for c in p['cells'] if c['optional'])}/4格。所有完成格均保留80个原始声明的完整训练工作，未按结果调整参数。准备、工程验收、执行与本次审计合计约{elapsed/3600:.3f}小时（最终封存另有精确时点）。",'',*conclusions,
 '## 1. 预注册输入与实验条件','',
 '80条官方Saturn作业声明：48 CIFAR10、16 DeepSpeech2、16 BERT；固定交错CF/CF/CF/DS/BERT，0秒到达40个、600/1200秒各20个。原始声明、batch、完整epoch工作及源trace/行号保留。这是受控到达场景，不是自然trace replay。',
 'AWS4GPU/12合成CPU/250GB，DGX-ext8GPU/24合成CPU/500GB，RTX8GPU/24合成CPU/500GB；每GPU比例3CPU/62.5GB。实际GPU训练模型、batch自适应来自Sia；辅助响应R0+E-legacy和alpha0/.5保持不变。所有节点规格与跨硬件映射为合成假设。',
 '60秒调度、6000轮上限，seed_profiles(1,8)。启动门槛2GiB可用内存/5GiB磁盘，每格独立进程串行。初始操作预算1800秒过紧，因此公开冻结一次仅操作性修订：统一最多3000秒计算、服务4格提前，科学参数/源码/模拟轮上限与全局8小时不变。原30分钟前缀全部保留。Sia完整结果与Full数值失败都在原限额前已结束，按哈希核验复用；其他9格只选择重跑目录，不选择最好结果。COMPUTE_TIMEOUT不等于完成，不插补总体JCT；模拟轮删失与计算中断分别报告。',
 'SIA_P主配置Medium比例点与固定参考相等，33种配置rho均1，真实有界Neutral/Medium轨迹严格相同，故可作Neutral外部参考；没有增加重复样本。', '',
 '## 2. 机会预检及适用范围','',
 '初始40作业、20GPU即产生服务竞争。完整33种应用/GPU配置有registry；49090个单列可行辅助点，30390个分子/分母都在数据范围内。27/33配置的比例点完全受数据支持；分母CPU越界10961候选点，分子CPU/RAM越界4095/7333点，分母RAM无越界。运行期裁剪比例另列，不能以算法运行成功验证跨硬件模型。',
 'AWS上1GPU CF的4.213316CPU/93.468723GB点rho1.534810，1GPU BERT的1.676408CPU/26.969888GB点rho1；合计不超过6CPU/125GB，分子与分母均不裁剪。同样20GPU预算可放两组该配对、其余16个CF/DS按比例，所有20个点有数据支持。这只证明可落地的相对速度机会，不能证明当前策略会选择它或整体JCT必然下降。',
 '源矩阵、完整配置覆盖、边界和可行配额见preflight/full_domain_points.csv.gz、coverage.json、pair_opportunities.json、global_embedding_witness.json。', '',
 '## 3. 保留算法与工程验收','',
 '权威基线291个文件逐字节保留。新入口扩展80队列与安装范围，使用等价稀疏矩阵、固定响应缓存和紧凑日志；未改p/lambda、完成优先目标、响应/phi/裁剪、候选网格、服务承诺、checkpoint或训练工作。完整源码302文件及依赖锁已保存。',
 'run1暴露20作业安装入口守卫，FULL在训练前被拒绝；run1/run2与中断证据均保留。修复进入新隔离源与新目录后整组重跑。为控制计算开销，在严格的已有主目标和启动代价次目标下，用当前可行向量证明正启动成本、零效用可选作业的最大presence必为0，省去冗余求解；不改变候选或最终词典序。此证书不引入新准入机制。',
 '新增/既有完整276项测试通过；真实80三轮的完整评分向量、最终分配和训练推进与未省略求解调用版本完全相同，已验证39/38/37个冗余调用。官方Sia有界回归逐状态一致。此前两个源码版本的Sia完整80作业309轮语义严格一致（仅忽略SHA已核对的迁移路径）。各原始失败、字面路径差异和后续语义核对均保留。', '',
 '## 4. 各格独立复算指标','',
 '|格|状态|完成|Mean JCT|Median|P95|P99|Makespan|','|---|---|---:|---:|---:|---:|---:|---:|']
 for cell in p['cells']:
  r=m.get(cell['id'],{});rows.append('|'+cell['id']+'|'+classified[cell['id']]['status']+'|'+str(r.get('completed','—'))+'/80|'+ '|'.join(number(r.get(k)) for k in ('mean_jct','median_jct','p95_jct','p99_jct','makespan'))+'|')
 rows+=['','所有JCT/尾部/完成事件由原始execution与job_end_state独立复算。没有用已完成子集的均值替代总体；P95/P99是本固定80作业样本的描述统计。部分格的完成数仅表示已持久化前缀，分母80包含未来才到达的已声明任务；各自观测结束时间、实际到达数和逐作业删失时长另见audit，不把不同模拟终点的完成数作性能排名。','',
 '|格|总等待s|首次等待s|后续等待s|checkpoint s|训练s|GPU h|CPU单位h|RAM GBh|restart|wasted GPU-s|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for cid,r in m.items():
  t=r['totals'];rows.append('|'+cid+'|'+ '|'.join(number(t.get(k)) for k in ('wait','first_wait','later_wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds'))+'|')
 rows+=['','首次等待按arrival至首次allocation，后续等待为其余waiting；checkpoint和productive training互斥。资源小时积分至实际完成，不按整轮60秒多计。部分格的积分仅表示日志前缀，不能与完整格做总量性能比较。', '',
 '## 5. 资源主线：条件贡献','',
 'SIA_P实际调用官方Sia GPU决策、无L2；SIA_AUX冻结同一官方Sia GPU/placement意图后用既有ECT辅助分配，不等于完整官方Synergy。PROP/CPU/MEM/FULL共同使用完成导向目标、有序准入和持续服务，仅辅助自由度不同；FO先在比例域选服务/GPU意图，再冻结配置优化辅助资源。FULL对FO是共同控制器下联合vs串行；FULL对SIA_AUX同时含目标、准入差异。', '',
 '|候选vs对照|完整配对|Mean节省s|Mean改善%|P95改善%|P99改善%|Makespan改善%|改善/退化作业数|','|---|---|---:|---:|---:|---:|---:|---|']
 for q in pairs:
  if q['candidate'].startswith('dual_neutral'):continue
  rows.append('|'+q['candidate']+' vs '+q['baseline']+'|'+str(q['complete_pair'])+'|'+ '|'.join(number(q.get(k)) for k in ('mean_jct_saved','improvement_percent','p95_improvement_percent','p99_improvement_percent','makespan_improvement_percent'))+'|'+(f"{q['improved_jobs']}/{q['worse_jobs']}" if q['complete_pair'] else '—')+'|')
 rows+=['','CPU对PROP、MEM对PROP分别给出单资源增量；FULL对CPU/MEM给出再开放另一资源的条件增量，不可把多个百分比直接相加。统一响应方程的局部接收/让出训练时间积分为解释性诊断，不是重放得到的因果JCT。','',json.dumps(resource_effects,ensure_ascii=False,indent=2),'',
 '## 6. 服务主线：有序启动与持续有效服务','',
 'A00/A01/A10/A11保持同一Sia公平效用、候选、比例执行、历史资格过滤与250秒生命周期启动预算，两个二元因素只改变有序头部准入和300有效服务秒/持续至完成。它们不是官方Sia，且与资源组的完成优先目标不同，不能把A/B百分比叠加。','',json.dumps(effects,ensure_ascii=False,indent=2),'',
 '四格全完成时：有序准入主效应＝[(Y00−Y10)+(Y01−Y11)]/2，持续服务主效应＝[(Y00−Y01)+(Y10−Y11)]/2（正值为平均JCT下降）；交互＝Y11−Y10−Y01+Y00（负值为额外下降）。缺失任一格则不估算完整因子主效应/交互，仍可报告已完成配对的条件效应。',
 'A00和A01虽然分别记录579次重新入队与零次重新入队，395轮物理分配和训练执行轨迹逐轮完全一致（service_trajectory_diagnostic.json）。因此本负载中，单独关闭服务到期没有新增实际收益；到期后既有目标和次目标又选择了相同活动作业。不能以“记录了到期”推断真正发生了服务中断，也不能把A11改善全部称为持续服务的独立贡献。','',
 '## 7. 逐作业与长尾诊断','']
 for q in pairs:
  if q['complete_pair']:
   rows.append(f"- {q['candidate']} vs {q['baseline']}：最大受益{q['top_beneficiary']}（原始{q['top_original_name']}，{q['top_source_trace']}），节省{q['top_saved_seconds']:.2f}s，占净节省比例{number(q['top_fraction_of_net'])}；诊断性去除该作业后的平均节省{q['diagnostic_without_top_saved_mean']:.2f}s。主结果仍保留全部80作业。")
 rows+=['','|完整配对|应用|作业数|总JCT节省s|改善/退化|','|---|---|---:|---:|---|']
 for q in pairs:
  if not q['complete_pair']:continue
  for app,v in q['by_application'].items():rows.append(f"|{q['candidate']} vs {q['baseline']}|{app}|{v['count']}|{v['sum_saved_seconds']:.2f}|{v['improved']}/{v['worse']}|")
 rows+=['','服务组合A11对Sia：平均训练时间实际增加912.60秒/作业，靠等待减少2394.75秒和checkpoint减少124.375秒抵消，得到平均JCT净减1606.525秒。DeepSpeech2组净退化46508秒，BERT组净改善113588秒；与P99/makespan小幅退化相容，不能把平均改善说成训练普遍变快。',
  'SIA_AUX对Sia：CIFAR10净节省98725秒、BERT节省53960秒，DeepSpeech2净损失3048秒；收益并非由单个DeepSpeech2长尾驱动。最大受益单作业占净节省8.73%，去掉它后的诊断改善仍为18.75%，主结果始终包含它和24个退化作业。']
 rows+=['','完整逐作业完成/等待/训练/资源指标：audit/per_job.csv；逐作业配对差值：audit/paired_comparisons.json。', '',
 '## 8. 正确性、机会兑现与裁剪','',
 '|格|审计违规|执行事件|锁核验|排队轮|峰值等待作业|CPU满但GPU空闲节点轮|实际互补共训pair-s|分子CPU裁剪|分子RAM裁剪|分母CPU裁剪|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for cid,r in m.items():
  clip=r['clipping_counts'];den=max(1,clip.get('selected',0));rows.append('|'+cid+'|'+ '|'.join(str(r.get(k,'—')) for k in ('capacity_and_execution_violations','execution_audits','checkpoint_lock_checks','waiting_rounds','max_waiting_jobs','cpu_saturated_gpu_idle_node_rounds','co_resident_different_quota_pair_seconds'))+'|'+ '|'.join(f'{100*clip.get(k,0)/den:.2f}%' for k in ('num_cpu','num_ram','den_cpu'))+'|')
 rows+=['','共训pair-s按同节点、不同配额、至少一方rho>1且另一方少于比例配额、真实训练片段交叠统计；它是任务对秒数，不等于作业等待秒，也不能当作新增独立样本。明细含接收/让出双方rho与配额。CPU满GPU空闲是节点级轮事件，与总等待秒不同。实际rho只乘直接记录的raw训练率一次；checkpoint/恢复倒计时和服务秒数均未乘rho。', '',
 ]
 rows+=['## 9. 计算开销与数据支持强度','',
 '|格|已记录模拟终点s|已到达数|评分目录峰值列数|建模/优化/提交墙钟s|报告的选择器P99 s|完整支持训练占比%|完整支持互补pair-s|',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
 for cid,r in m.items():
  e=r.get('training_response_exposure_seconds',{});total=sum(e.values());share=100*e.get('fully_supported',0)/total if total else None
  rows.append('|'+cid+'|'+ '|'.join(number(v) for v in (r.get('observed_simulated_end'),r.get('arrived_by_observed_end'),r.get('peak_logged_score_catalog_columns'),r.get('computation_wall_seconds',{}).get('optimize_and_commit'),r.get('selection_reported_seconds',{}).get('p99'),share,r.get('fully_supported_co_resident_pair_seconds')))+'|')
 rows+=['','优化提交墙钟包含在线建模、候选构建、多个MIP和提交；PolicyResult选择时间也不是单个求解器调用。实际调用状态与可证明省略的冗余调用分别记录，不把证书复用当作额外HiGHS成功。程序启动/序列化时间见dispatcher逐格墙钟。评分峰值只统计实际日志中保留计数的目录，不代替完整预检上界。','',
 '完整支持＝当前分子和分母都在官方矩阵坐标范围。中性alpha0即使坐标越界仍保持rho1；Medium下E-legacy边界假设会影响性能。各格审计分别记录受支持/裁剪训练段相对同状态比例点的局部收益与让出损失。它们是固定raw/config/batch的训练时间等价诊断，不是反事实重跑，不可将整体JCT收益按比例拆成实测与合成部分。','',
 '## 10. 故障与当前最小改进建议（本轮未实施）','',
 'FULL在模拟300秒、FO在1080秒发生INVALID_SOLVER_RESULT_AFTER_NUMERICAL_RETRY；A10在780秒发生LEXICOGRAPHIC_OBJECTIVE_INCONSISTENT。这些是整体执行失败，不能与正常的试探集合不可行status2混为一谈。A11的143次status2与DRAIN_FOR_HEAD相对应，是可审计的排空选择；未发生运行崩溃。统一fallback只按实际FALLBACK状态计数，不按存在fallback_reason字段计数。',
 '主批次停止后，仅串行重放FULL前5个60秒边界，外部720秒上限，未做完整JCT评估或参数修改。前4轮分配、训练和完整评分hash与正式失败前缀一致，仍在第5轮复现相同异常。捕获模型68394列、49行资源/作业约束与1行主目标带。首次HiGHS报告Optimal，最大近整数误差6.048878e-7；BundleAdmission先以1e-7拒绝，关闭presolve的原30秒重试超时；NumericalAdmission只有retry_status=0才进入已有1e-6近整数规范化。',
 '离线将两条捕获向量二进制化后，都满足所有原始行及已有严格主目标带，零行违规，启动次目标245。该证据定位到验收顺序/数值路径，不是证明物理资源不足；缺少原始dual bound，不能宣称额外证明全局精确最优。建议下一轮先统一“求解状态→已有近整数规范化→全部原始行/锁/候选审计→必要重试”的顺序，保留严格主目标带，不放宽容量或改效用。本轮没有实施该修复；FO同异常标签不证明内部向量相同，A10仍需持久化词典序模型与incumbent。代码：bundle_adaptation.py:24–64、completion_continuity.py:110–149、clean_ablation.py:137–165；完整捕获与独立复算见diagnostics_full_five_boundaries/。',
 '三个计算超时格也有清楚的开销证据：PROP已记录优化/提交2178.97秒，而选择器仅131.82秒；MEM为2158.00/211.30秒。大量时间在候选、全域归一化与模型准备，不应全归因于HiGHS难解。建议先做保持评分向量、候选集合与最终分配完全等价的缓存/提前固定GPU过滤，特别是PROP为共同归一化计算整个辅助域、SIA_AUX先建全域再过滤官方GPU意图；不改网格、目标或容量。',
 '此外，既有资源组冷启动合同UNKNOWN_ETA_ZERO_PROP_BOOTSTRAP与最小启动次目标，使首轮仅选1作业/1GPU，随后2、3作业；Sia与公平服务组首轮都是20作业/20GPU。FULL在180秒仅用4GPU却已用51合成CPU。这是实际同状态日志可确认的准入/估计限制，不能把资源组前缀慢全部解释成CPU优化无用。后续如需改冷启动必须单独设计与隔离对照，本轮未改；不向在线模型灌入R0真实未来速率。',
 '276项测试通过但不意味着80作业全矩阵稳定性通过。应先修数值验收与等价计算规模，再冻结新版本重跑受影响的完整配对；没有证据支持现在扩大负载、改响应或改变目标直到获胜。','',
 '## 11. 结论边界与交付','',
 '本轮只对固定80作业受控负载、一个主要合成容量及可选单资源压力档成立。CPU/RAM-only配额调整沿用零附加启动开销假设，没有新增数据加载/NUMA/IO模型；GPU重配置仍收取原Sia实际开销。调度器计算墙钟单独报告，模拟时钟未按优化器耗时额外推进，因此JCT不包含真实部署的控制器计算等待。不能推出所有trace、真实GPU集群或一般JCT最优。若完整联合格未完成，不能用局部预测、部分完成或仅CPU格替代完整联合JCT结论；时间/求解失败本身是规模稳定性证据。',
 '完整配置、源码/输入hash与固定执行顺序见preregistration.json；302文件源码zip、环境和依赖锁支持重建。执行状态和墙钟见execution_manifest.jsonl。独立原始复算见audit/，完整/缺失覆盖见audit/coverage.json；run1/run2保留失败与中断，不与run3混合。官方源、旧CC-TAIL和历史报告均保留，没有推送。',
 '本轮结束后停止；不自动开展新的负载、网格、响应或算法优化。','']
 report=ROOT/'research_notes/synergy/CC_CORE80_8H_REPORT_20261030.md'
 with report.open('x',encoding='utf8') as f:f.write('\n'.join(rows))
 with (OUT/'REPORT.md').open('x',encoding='utf8') as f:f.write('\n'.join(rows))
 print(report)
if __name__=='__main__':main()
