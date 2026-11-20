# 有限纠偏恢复隔离验证
用户授权：FO/FULL共同使用有限恢复规则，保护checkpoint，计入真实切换成本；先同状态红绿测试及完整回归，通过后原80作业FO、FULL新进程串行补跑。
基线：cc-joint-core80-full-lex-repair-v8；旧产物cc_core80_full_lex_repair_20261031_run5封存不动。新树cc-joint-bounded-correction-v1。
不改变目标、候选网格、执行响应、负载或求解器参数。2GiB内存、5GiB磁盘门槛，6000轮、60秒。历史失败与新失败均保留。
先验证规则：当前ready为空、已有多个运行作业、仅利用固定其他当前分配后的余量、已有效训练、目标已画像、预计ETA含真实启动且严格改善。每任务最多一次额外恢复，与原尾部恢复共享一次许可；无未来到达信息。
恢复入口：本目录证据及execution_manifest.jsonl。正式运行中不修改冻结源码，不重复启动。完成同版本FO/FULL后独立事件审计并对比旧两格，停止。

FROZEN 1793509381.0128944;348tests accepted. FO→FULL serial new processes. Resume only after inventory and prefix hashes; no automatic rerun of failed directories.
