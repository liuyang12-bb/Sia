import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def pct(x):return f'{100*x:.2f}%'
def main():
 p=json.loads((OUT/'preregistration.json').read_text());m=json.loads((OUT/'metrics.json').read_text());v=json.loads((OUT/'validation.json').read_text());cs=json.loads((OUT/'comparisons.json').read_text());ef=json.loads((OUT/'factor_effects.json').read_text());raw=json.loads((OUT/'raw_audit.json').read_text())
 records=[json.loads(x) for x in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
 assert len(records)==15 and all(r['exit_code']==0 for r in records)
 for n,h in p['source_hashes'].items():assert sha(OUT/'frozen_source'/n)==h,n
 lines=['# 有序启动／持续服务与联合资源配置：干净消融A、B及响应预检C','',
 '**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**','',
 f"状态：15/15格有原始结果；全作业完成={v['all_completed']}；逐节点容量、完整checkpoint锁、rho重复施加违规均为0。正式子进程累计墙钟{sum(r['wall_seconds'] for r in records):.2f}秒；准备/测试/静态分析不计入此数。",'',
 '## 1. 本轮范围与控制变量','',
 '本轮只增加独立消融适配器。当前生产/历史CC-TAIL、官方Sia/Synergy、旧报告和结果均未修改。父版本为已验收词典序修复版的完整实际源码；289个既有文件逐文件不变，增加clean_ablation.py与test_clean_ablation.py。源码版本以完整snapshot+SHA256为准，本树不是Git仓库，不虚构commit。','',
 'A组只研究服务组织；共同使用Sia式归一化Goodput公平效用和原重配置proxy（p=-0.5、lambda_n=1.1），共同保留原Sia冷启动效用。因此A11不是原CC-TAIL或官方Sia，不将其全部收益包装为完成评分收益。B组共同使用当前倒数剩余ETA评分（含真实startup），不改变未知ETA零效用/比例bootstrap规则。B各单元先在共同完整候选域评分和计算数值归一化，再过滤因素，避免候选域改变带来的词典序容差单位混杂。','',
 '两组均共享真实训练/Goodput/batch、完整资源容量、checkpoint强制选择及完整分配、原子提交、当前轮oracle、生命周期250秒启动预算、相同一次TAIL例外及实际startup次级目标。候选资格由所有已启动任务的历史决定，不因服务到期退出held集合而自动放宽。6000轮、60秒、同画像初始化seed_profiles(1,8)。Neutral用于A；Medium alpha0.5用于B。','',
 '|组|因素单元|精确定义|','|---|---|---|',
 '|A|A00|目标驱动可选准入；300秒有效服务段|',
 '|A|A01|目标驱动可选准入；持续服务至完成|',
 '|A|A10|FIFO队首/排空；300秒有效服务段|',
 '|A|A11|FIFO队首/排空；持续服务至完成|',
 '|B|FP|共同控制器在比例列域先选GPU与服务集合；比例执行|',
 '|B|FO|同一比例域GPU提案；冻结GPU类型/数量/节点数及服务集合，再优化CPU/RAM和placement|',
 '|B|JP|在比例列域共同选择GPU与placement；比例执行|',
 '|B|JO|共同选择GPU、CPU/RAM与placement|',
 '|外部|SIA|实际官方Sia GPU+placement决策；比例辅助资源执行适配，不加新服务控制器|','',
 'B的固定GPU提案不是原Sia，而是为隔离决策耦合而定义的共同控制器串行对照；实际Sia另作外部基线。FP和JP在只有比例点时数学等价，逐轮精确轨迹是预注册负对照。FO允许固定GPU配置下调整placement，因此JO对FO不单独隔离节点放置贡献。','',
 'C0在剩余启动预算不足以安全恢复时不强行撤出任务，并单独计数。这是非破坏性安全底线，不是无限预算；A00/A10并非任意抢占。无FIFO单元不声称一般到达负载下的个体防饥饿保证。持续服务不是无限冻结配置。','',
 'A使用原20作业（18 CIFAR+2 DeepSpeech），AWS/DGX各2GPU、6合成CPU、187.5GB RAM；Neutral、比例。B使用原COMPACT/SPREAD12作业（6 CIFAR+6 BERT），AWS/DGX各2GPU、6CPU、125GB；Medium。原记录训练工作量及顺序不变。A、B负载/目标/响应不同，不能把它们的贡献百分比相加分解同一整体收益。','',
 '## 2. 完整指标','',
 '|组/负载|方法|完成|Mean JCT|Median|P95|P99|Makespan|Restart|','|---|---|---|---|---|---|---|---|---|']
 for name,s in m.items():
  vals=[name,s['method'],f"{s['completed']}/{s['total']}"]+[f"{s[k]:.2f}" if s[k] is not None else 'CENSORED' for k in ['mean_jct','median_jct','p95_jct','p99_jct','makespan']]+[s['totals']['restarts']]
  lines.append('|'+ '|'.join(map(str,vals))+'|')
 lines+=['','JCT从原始完成时间减到达时间独立复算；P95/P99仅12或20个作业的描述性样本插值，不是高精度总体尾延迟估计。若有删失，不比较完整均值，不插补未完成JCT。','',
 '## 3. A：服务组织贡献','',
 '|条件比较（后者作对照）|平均JCT差秒|改善率|首次等待差总秒|后续等待差总秒|checkpoint差总秒|训练差总秒|','|---|---|---|---|---|---|---|']
 for key in ['A20/A10_vs_A00','A20/A11_vs_A01','A20/A01_vs_A00','A20/A11_vs_A10','A20/A11_vs_SIA']:
  c=cs.get(key)
  if not c or c.get('incomplete'):continue
  lines.append(f"|{key}|{c['mean_delta']:.2f}|{pct(c['improvement'])}|{c['first_wait_delta']:.0f}|{c['later_wait_delta']:.0f}|{c['totals_delta']['checkpoint']:.0f}|{c['totals_delta']['training']:.0f}|")
 if 'A20' in ef:
  a=ef['A20'];lines+=['',f"等权因子主效应：有序启动{a['order_main']:.2f}秒/作业，持续服务{a['continuity_main']:.2f}秒/作业；交互{a['interaction']:.2f}秒/作业。定义均为新减旧，负数有利。交互=A11−A10−A01+A00，负数表示组合下降超过简单相加。主效应是另一个开关两状态上的等权平均，不是跨独立负载统计。"]
 lines+=['','必须结合首次等待、后续等待和checkpoint判断机制。持续服务可能让首次启动更晚而降低后续重排队；FIFO可能帮助启动也可能排空阻挡。A组不比较新完成评分，不能从这些差值宣称完成评分优于Sia。','',
 '## 4. B：辅助资源与联合选择贡献','',
 '|负载/比较|平均JCT差秒|改善率|改善/退化/持平作业|最大受益作业|去掉最大受益后的诊断均值差秒|','|---|---|---|---|---|---|']
 for w in ['COMPACT','SPREAD']:
  for pair in ['B_FO_vs_B_FP','B_JO_vs_B_JP','B_JP_vs_B_FP','B_JO_vs_B_FO','B_JO_vs_SIA']:
   key=w+'/'+pair;c=cs.get(key)
   if not c or c.get('incomplete'):continue
   lines.append(f"|{key}|{c['mean_delta']:.2f}|{pct(c['improvement'])}|{c['improved']}/{c['worsened']}/{c['equal']}|{c['largest_beneficiary']}|{c['leave_largest_beneficiary_out_mean_delta']:.2f}|")
  if w in ef:
   a=ef[w];lines+=['',f"{w}：辅助优化主效应{a['aux_main']:.2f}秒，联合选择主效应{a['joint_main']:.2f}秒；交互{a['interaction']:.2f}秒，定义(JO−JP)−(FO−FP)。FP=JP的精确物理轨迹审计见validation.json。"]
 lines+=['','JO−JP是相同联合控制器下开放辅助配额的闭环增量，含后续GPU和队列反馈；不是固定GPU纯加速份额。JO−FO是相同控制器下联合决策优于串行阶段的增量；FP=JP时，该增量也体现在交互项。CPU与RAM各自贡献仍引用既有六方法实验，本轮不新增CPU/MEM单因素格。','',
 '## 5. 时间与资源','', '|格|等待任务秒|有效训练任务秒|checkpoint任务秒|GPUh|CPU单位h|RAM GBh|wasted GPU-s|调度求解总秒|','|---|---|---|---|---|---|---|---|---|']
 for name,s in m.items():
  t=s['totals'];lines.append('|'+ '|'.join(map(str,[name]+[f"{t[k]:.2f}" for k in ['wait','training','checkpoint','gpu_hours','cpu_core_hours','ram_gb_hours','wasted_gpu_seconds']]+[f"{s['solve_total_seconds']:.3f}"]))+'|')
 lines+=['','资源小时为实际分配×完成边界截断后的占用时间，不是实测CPU忙碌率/RSS或能耗。辅助配额变化的额外加载成本沿用旧模型未单独计入；solver墙钟未加入虚拟JCT。','',
 '## 6. C：响应数据预检与执行曝光','',
 '完整预检见C_RESPONSE_PREFLIGHT.md、C_coverage.json、C_points.csv、C_pair_witnesses.json。96行/3286个物理可行查询点，registry与源哈希缺失0；不等于全部源矩阵域内。A相关8种比例配置7种域内，B为5/8。CIFAR DGX双GPU和BERT DGX参考缺口没有修复；不补phi1、不删配置、不改E-legacy。BERT借用模板完全平坦，是理想donor；非平坦DeepSpeech donor的静态净收益另列，不把静态响应和当作JCT。','',
 '|格|训练任务秒|分子CPU裁剪%|分子RAM裁剪%|分母CPU裁剪%|分母RAM裁剪%|','|---|---|---|---|---|---|']
 for name,r in raw['results'].items():
  train=r['summary']['totals']['training'];z=r['clipped_productive_job_seconds'];lines.append('|'+ '|'.join([name,f'{train:.0f}']+[f"{100*z.get(k,0)/train:.2f}" if train else '0' for k in ['numerator_cpu','numerator_ram','denominator_cpu','denominator_ram']])+'|')
 lines+=['','裁剪比例按有效训练任务秒加权；不同类别可重叠，不能相加，也不是收益归因比例。Neutral仍有坐标越界但rho=1。内部共享响应保证公平执行，不证明跨硬件迁移准确。','',
 '## 7. 版本、失败尝试与测试','',
 f"最终冻结配置SHA256：{sha(OUT/'preregistration.json')}。依赖、源文件、输入和脚本hash在preregistration.json；完整源码frozen_source/及frozen_source_snapshot.zip；源码新增diff在source_additions_final.patch。",'',
 '原245测试保留，257项完整套件通过；最终12新增测试又在元数据标签澄清后复跑。原始Sia一作业三轮进度/分配/restart/完成状态精确一致。先有缺模块红测；恢复旧held成员依赖的候选过滤与旧候选域归一化时，新测试分别按预期失败，证明不是只测试新增接口存在。真实unseeded BERT在八个因素路径走过checkpoint、首次有效训练及后续真实调度。','',
 'run1在A00期间因控制变量审查停止，部分gzip和原日志保持原样、不修补、不计算其总体JCT。问题是B比例/优化单元可能用不同候选域计算数值归一化，固定词典序容差因此不完全同单位；在任何B结果出现前修正。run2共同完整域评分后再过滤，新增红绿与257项回归，全部15格使用新统一快照。没有根据JCT调整目标、容量、响应、工作负载、窗口或网格。','',
 '## 8. 结论适用范围','',
 '本轮能分别检验服务组织和联合资源配置的条件贡献；A与B使用不同负载/目标，不能把两个改善率加总为整体贡献。每个条件是确定性受控场景，作业不是独立重复。没有恢复104格、运行160作业或自动调参。',
 '仍待解决：响应参考覆盖/裁剪与借用模板的外部可信度、更多有效应用和规模、持续服务/启动预算的长任务代价。是否推进下一轮应依据完整效应和反例，不要求两个因素处处获胜。']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
 events=['# 服务状态机事件与真实分配转换','',
  '到期重新入队不等于物理抢占；若同轮选回同一配置，可能没有停训。预算安全保持是轮次计数，不是独立作业数。','',
  '|格|服务发放|到期重新入队|队首排空轮|预算安全保持轮|活动到等待转换|活动中GPU几何改变|TAIL恢复|',
  '|---|---|---|---|---|---|---|---|']
 for name,s in m.items():
  e=s['liveness_events'];c=raw['results'][name]['checks']
  events.append('|'+ '|'.join(map(str,[name,e.get('ISSUE_SERVICE',0),e.get('REQUEUE_PRODUCTIVE_QUANTUM',0),e.get('DRAIN_FOR_HEAD',0),e.get('QUANTUM_HELD_NO_LIFETIME_RESUME_BUDGET',0),c.get('active_to_wait_transitions',0),c.get('gpu_geometry_changes_while_active',0),e.get('TAIL_RECOVERY',0)]))+'|')
 (OUT/'SERVICE_EVENTS.md').write_text('\n'.join(events),encoding='utf8')
 note=ROOT/'research_notes/synergy/CC_CLEAN_ABLATION_AB_C_20261030.md'
 with note.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
 print('REPORT_WRITTEN',note)
if __name__=='__main__':main()
