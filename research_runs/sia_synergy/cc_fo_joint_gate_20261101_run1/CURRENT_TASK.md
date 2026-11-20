# 当前授权：FO验收修复＋同状态机会门槛＋至多一个12–20作业FO/FULL配对
基线cc-joint-bounded-correction-v1，331项实际源码逐项核验复制至cc-joint-fo-audit-repair-v2。原80作业失败和所有旧版本不改。不得重跑80作业，不修改目标、网格、响应、负载工作量或调参。
先只读提取失败轮数据，定位GLOBAL_BUNDLE_OBJECTIVE_REGRESSION验收基准；红测再修复验收逻辑与诊断持久化。完整348测试及短原始Sia回归。不得绕过容量/锁/合法性。若日志不能精确恢复失败轮，明确缺失，不把近似反事实称原轮精确重放。
离线门槛使用同一状态、资源、知识、服务集合；FO先比例GPU提案再优化CPU/RAM，FULL联合，比较计入donor损失和真实切换成本。优先已有12作业COMPACT数据，不按JCT搜索新负载。只在机会、覆盖、正确性及预注册都闭合后运行单个FO/FULL配对，各独立进程串行，2GiB内存/5GiB磁盘、60秒/6000轮。机会不成立则停止并说明，不为取胜继续换数据。
SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

结束：FO验收修复完成，353测试/原始Sia3轮通过。12作业离线门槛未过，训练配对0格，80作业未重跑。详见REPORT.md。无后台实验。
