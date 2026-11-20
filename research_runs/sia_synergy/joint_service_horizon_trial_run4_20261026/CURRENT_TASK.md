# G112 服务时域评分配对实验：已完成

当前目录为唯一成功配对版本。冻结输入与源码见 `preregistration.json`、`source_snapshot.zip`；结果见 `cells/G112/`、`cells/G111/`，两格独立进程运行且各完成6/6。G112前60轮门槛见 `cells/G112/opportunity_gate.json`，四项均通过。`comparison_audit.json` 为valid；逐作业与聚合差异见 `paired_analysis.json`。完整测试见 `full_tests.log`，173次执行通过。

G112平均JCT5825.33秒；同版本G111为5900.67秒，改善1.28%。G112依旧比先前同输入SIA-P的5151.33秒慢。详细中文解释在 `research_notes/synergy/JOINT_SERVICE_HORIZON_PAIRED_REPORT_20261026.md`。

旧尝试目录 `joint_service_horizon_trial_20261026`、`joint_service_horizon_trial_run2_20261026`、`joint_service_horizon_trial_run3_20261026` 保留语法、内存、指标适配与到期状态失败证据。没有覆盖历史产物，也未恢复104格实验。
