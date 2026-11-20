# 论文实验图说明

统一数据入口：`../evidence/paper_results.json`；绘图代码：`../scripts/plot_results.py`。所有图均有300 dpi PNG、可编辑SVG及矢量PDF。脚本只读取既有结果，不运行调度器或训练仿真。各数据组/版本与来源哈希见JSON的`provenance`。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。

图式采用系统论文常见的分组柱状图、归一化对照、时间分解、经验累积分布及逐作业差值图。此处仅借鉴比较和组织证据的方法；全部图形由本项目数据重新绘制，未复制Sia、Synergy或其他论文中的原图，也未把其数值用于本项目结果。具体相关论文出处由文献汇编统一给出。

## 图3 有序启动和持续有效服务的条件贡献

文件：`fig02_service_ablation.png/.svg/.pdf`。

左：A00、A01、A10、A11及SIA-P的平均JCT，纵轴小时；右：各方法每作业的等待、有效训练、启动/checkpoint时间分解。Axy中x表示是否启用FIFO有序准入，y表示是否启用持续服务至完成；y=0采用300秒有效服务段。各A单元共享公平评分和候选契约，因此该图不检验完成优先目标的独立贡献。SIA-P是实际官方Sia决策加统一辅助资源执行适配。20作业、Neutral、同一合成容量。图表显示平均JCT下降主要对应等待减少，而非所有任务训练加快；A01与A00相同，说明持续服务效果依赖准入条件。

来源：`paper_results.json`中`dataset=clean_ab, cohort=A20`；实际文件`research_runs/sia_synergy/cc_clean_ab_20261030_run3/metrics.json`与既有独立原始日志审计。无误差棒，因为这是固定场景的确定性执行，不是独立随机重复。

## 图4 差异化CPU/RAM配额及联合决策的增量

文件：`fig03_resource_ablation.png/.svg/.pdf`。

左：BURST、COMPACT、SPREAD三种受控到达下，比例、仅CPU、仅RAM和CPU+RAM优化的平均JCT，分别除以同场景比例控制的平均JCT。右：干净控制下串行FO与联合JO，分别归一化至同场景FO。小于1有利。左图`arrival18`与右图`clean_ab`属于不同冻结版本，不能直接混合统计；三种到达也不是三个独立自然trace。FO是共同完成控制器先在比例候选域确定GPU/服务集合，再优化辅助资源的串行对照，不是官方Sia GPU MIP的别名。JO对FO改善6.18%和8.53%；SPREAD增量存在单个BERT长尾贡献集中的局限，必须结合正文逐作业分析。

来源：`paper_results.json`中`dataset=arrival18`的CC_P/CC_CPU/CC_MEM/CC_FULL；右图`dataset=clean_ab`、COMPACT/SPREAD的B_FO/B_JO。原响应含固定E-legacy边界假设及平坦BERT模板，这些并不是目标硬件实测响应。

## 图5 80作业整体改善与尾部代价

文件：`fig04_core80.png/.svg/.pdf`。

上左：同版本七方法平均JCT，单位小时；上右：SIA-P、FO、FULL的80个逐作业JCT经验CDF；下：FULL相对SIA-P和FO的均值、P95、P99及makespan改善率。改善率定义为`100×(baseline−FULL)/baseline`，正数有利，负数为退化。FULL对SIA-P均值下降29.62%，但P99和makespan恶化18.29%和14.50%；相对FO均值只下降1.09%。不得仅用均值柱图推断长尾全面改善。80个作业是同一受控负载中的相关任务，不是80个独立重复。

来源：`paper_results.json`中`dataset=core80`。原始依据为`cc_core80_full_lex_repair_20261031_run5/result_summary.json`、`audit/`与`cells/*/summary.json`；完整七格均80/80。该组源码329文件版本，与图3/3和图6的后期12作业版本不同。

## 图6 逐作业得失及平均JCT分解

文件：`fig05_tradeoffs.png/.svg/.pdf`。

上：最新12作业FO/FULL完整对照的逐作业JCT节省量`FO−FULL`，蓝色正值表示改善，橙红色负值表示损害。作业顺序固定为原输入顺序，C/D/B分别为CIFAR10/DeepSpeech2/BERT，保留全部DeepSpeech2退化。下：三个不同版本/场景下，等待、训练、checkpoint节省的任务秒除以作业数，仅作机制分解示例；不合并估计、不把三组当作同一版本重复。最新12作业等待减少4200任务秒，但训练增加3799、checkpoint增加120，完整JCT净节省仅281任务秒（0.336%）。

来源：`paper_results.json`中`dataset=mixed12`及clean_ab A11/A_SIA、core80 FULL/SIA_P的`totals`。最新12作业结果来自`cc_mixed12_event_pair_20261101_run1/REPORT_v2.md`、逐作业JSON及原始审计，不能使用保留的初版失败门槛报告代替。

## 重建命令

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe -X utf8 -B D:\科研启航\Sia\research_notes\paper\cc_joint_draft_20261101_v1\scripts\plot_results.py
```

本轮已逐图查看PNG，轴、标签、正负值、单位、图例和限定文字均可读。无拟合曲线、无伪造显著性/置信区间、无删除长尾作业。数据SHA及图形SHA见`plot_manifest.json`。

排版更新：图3至图6均按约8英寸母版宽生成，主要字体11–12 pt，图5为上排均值/CDF、下排整宽改善率，图6为上下两panel；适配170 mm正文宽。长篇边界限定移入本图注，不在图内缩成小字。已重新逐图检查PNG。
