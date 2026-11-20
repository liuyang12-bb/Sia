# CC Joint 排队收益 内存场景与长尾诊断

状态：FIVE_COMPLETE_VALID。五格均20/20完成，独立原始日志审计通过；212既有测试、7新增离线验收、原始Sia独立3轮逐状态回归通过。训练子进程总墙钟759.986秒。
结果与分析见REPORT.md、validation_and_comparison.json。算法未修改，后续收尾恢复/候选修复只是建议。不得再次启动batch或重跑已完成格。

用户授权：分析排队优势与尾部退化，并扩展资源场景做实验。算法不改。
当前代码来源 cc-joint 全量快照在 source/；哈希冻结于 preregistration.json。
新场景保留原20作业、GPU/CPU、R0/E-legacy、alpha=.5、60秒、6000轮，只将每节点RAM250改为187.5GB。
五格：SIA_P、SIA_ECT_AUX、JOINT_CC_P、JOINT_CC_CPU、JOINT_CC。CPU消融RAM固定93.75GB/GPU。
维护运行器 maintenance/cc_joint_extension_20261027.py，批次会话53352。不得重复启动。
每格独立进程串行；2GiB内存、5GiB磁盘门槛；失败或门槛停止则保留现场，不自动调参或重跑。
run1导入失败与run2离线无损donor假设失败均保留，均未训练；run3明确计入donor损失，不保证JCT获益。
完整JCT均值只用于全部完成配对；任何删失不插补。禁止扩大160作业或恢复104格。
旧20任务原始日志诊断：queue_tail_raw_diagnostics.json、concurrency_diagnostics.json、tail_budget_witness.json。
212既有测试日志位于run2/existing_tests.log；6新增离线验收见offline_tests.log。
尾部修复仅形成建议；不解除checkpoint或启动预算、不把新目标混入本批。
SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
