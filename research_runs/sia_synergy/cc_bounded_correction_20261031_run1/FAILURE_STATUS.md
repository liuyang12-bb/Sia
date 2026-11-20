# 本轮失败与停止状态

本轮未完成，不能形成新旧FO/FULL四格性能比较。流水线结束只代表执行及失败后审计脚本结束，不代表两格实验成功。

- 新FO：224个完整记录轮次、70/80完成；模拟时间13500秒的后续调度触发`RuntimeError: GLOBAL_BUNDLE_OBJECTIVE_REGRESSION`。
- 新FULL：依照预注册工程失败停止门槛，未启动。
- 正式运行墙钟6746.87秒（约112.45分钟）；包含验收和失败后审计的pipeline墙钟7484.62秒。
- 224个完整轮次独立审计违规0，已提交的7次多作业恢复记录完整保留。其存在不能证明整体JCT改善。
- 新FO未完成，全体平均JCT、P95/P99、makespan留空；不得用已完成70项均值推断全体。
- 原生进程核对：pipeline/supervise80/run_core80均已结束；331项冻结源码哈希无变化。

触发位置：`sia-simulator/joint_prototype/bundle_adaptation.py:107–123`。保护逻辑用旧held作业完整geometry组成物理可行baseline，在`chosen_objective + 1e-7 < baseline_objective`时抛错。该检查不只是资源审计。

尚未确定：本次是否存在真实的选中目标退化，还是旧几何基线未满足FO两阶段固定GPU意图、当前资格或恢复后几何冻结等约束，或数值差异。不能直接删除保护、增加容差、放宽锁或重新启动。下一步应在新诊断产物中重建失败轮两个分配的当前评分、完整候选资格及串行GPU意图，先查明原因。

现场：`cells/dual_medium_FO/failure.json`、`rounds.jsonl.gz`、`execution_manifest.jsonl`、`audit/`、`recovery_events.json`。原始报告和失败现场均保持不动。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE
