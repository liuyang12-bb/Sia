# 最终审查与交付入口

## 结论边界

已完成一个机制改进版本JOINT-G，没有按JCT追加参数寻优。最终只使用attempt3的8/20×Neutral/Medium四格，全部8/8或20/20，600轮内正常完成。完整141项测试通过；新四格的原始执行/分配/锁/预算/主次目标校验通过。旧12格15408个执行事件经新审计复查全部合法，84份旧产物hash未变。

Medium：8任务新G1385.50秒，对旧R改善5.16%、对AUX改善6.15%；20任务新G3157.15秒，对旧R改善3.66%、对SIA-P改善9.20%，仍比AUX慢1.43%。因此不建议现在用新G直接替代强串联基线或恢复104格。

8任务均值收益主要由deepspeech2-9贡献651/726=89.7%的净提前；诊断性去掉它仍有0.87%均值改善，但主结果保留它。20任务相对AUX10改善/10退化，净慢889秒。bert-54/99净慢810/720秒；其他任务部分收益抵消。数据有可用辅助资源机会，当前策略尚未充分兑现为普遍JCT收益。

20任务的median 2264.5秒差于AUX1917秒；makespan6127秒优于AUX6244秒，P95/P99略优。平均/中位数与尾部存在冲突，不能以尾部获胜代替平均JCT主指标。GPU-h31.710低于AUX33.208，RAM-GB-h2041.214高于AUX2027.449；更低资源小时不自动等于更低JCT。

## 工程与数值证据

- 第一次失败为checkpoint完成边界的审计假阳性。未改动官方Job真实复现相同取整结果，修复仅涉及审计，并新增拒绝锁内伪进度的测试。Job.step、训练工作量、JCT和checkpoint时间没有修改。
- 第二次失败为求解器“最优”向量遗漏必选deepspeech2-9。最终同一1860秒轮次捕获5个子问题的等式违反量1.0；同模型关闭presolve重求后，原始和取整向量都满足所有约束。5次不是5个等待轮，更不是5次已提交资源违规。细节numerical_evidence.json和NUMERICAL_GUARD.md。
- 不改算法数学定义或数据，完整配对在新目录重跑。attempt2已完成的3格与attempt3的逐轮配置、配额、rho、进度、restart、completion完全一致，见trajectory_regression.json；未混合或择优选取旧结果。
- 仍未解决的研究问题：FIFO服务集合先验过宽、即时公平效用与完成时间不完全一致、GPU预测误差及合成跨硬件响应/clipping。下一步建议保持锁与有界服务，隔离“队首必选＋其他未承诺任务可选”的准入机制，不是简单删除L2；本轮未实施下一种算法。

## 版本标识更正

实际树：`D:/科研启航/Sia/sia-joint-budgeted-adaptation`。
实际分支：`exp/joint-budgeted-adaptation`。
基线HEAD：`2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2`。
本轮没有提交或推送，新代码在快照中，不可把裸HEAD当成最终执行版本。

冻结preregistration中的`branch`字段误继承了旧分支`exp/joint-r-scale-check`；保持原manifest不重写。其`tree`、源码/输入hash、命令及实际执行文件均为新树，未用错源码。最终以source_version.json的actual_branch及delivery_manifest.json为权威；这是元数据标签更正，不是算法/配置更改。

## 复现与文件

1. 以基线Git版本建立全新目录/工作树，覆盖恢复attempt3/source_snapshot.zip内的源码；不要覆盖当前用户源码或旧结果。
2. 使用frozen_prerequisites.zip恢复其相对项目根的输入/源码前置文件；旧对照资料独立保存为baseline_snapshot.zip。保持目录布局，或只做绝对路径前缀重定位并重新校验文件内容hash，不改变科学参数。
3. 独立Python环境使用requirements-lock.txt/依赖JSON，运行对应测试；不得升级系统Python。环境/本机信息见source_version.json。
4. 读取attempt3/preregistration.json，在全新输出目录运行四格入口，8两格通过后20两格。旧run.py不允许覆盖已存在目录。所有原始命令和退出码在对应execution_manifest.jsonl中。
5. attempt3/audit.py用于原始事件复核；baseline_reaudit.py和trajectory_regression.py保存对照不可变及轨迹回归证据；report.py输出主报告。旧两版目录是故障证据，不补写或混入完整矩阵。

主报告JOINT_GLOBAL_EXCHANGE_REPORT.md；完整机器可读16格（12个旧对照＋4个新格）在all_comparisons.json。新4格、旧尝试、输入、源码、报告及代码diff最终hash见delivery_manifest.json。

原始Sia仓库干净。Synergy仓库原有未跟踪论文PDF保留；未修改官方代码。sia-104确认PAUSED。没有运行160-job、九cohort或恢复104格。完成后停止。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
