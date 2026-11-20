# 最终核验与机制补充

本轮完成：8/20作业各6格，12个完整实验单元；有一次20首格启动在2GiB内存门槛前退出（没有创建输出目录、构造集群或推进任务），所以execution_manifest记录13次启动尝试，而不是13个科学实验格。确认无重复实验进程、重验完成前缀hash和只加载环境可用内存2.1889GiB后，唯一一次重试成功。没有降低门槛、改参数或挑选更好的结果；首次错误日志保留。

125项测试通过。两个规模原始审计valid/all_complete均true；额外15408个执行片段单调性及checkpoint先于训练检查通过。Neutral中SIA-P与SIA-AUX的逐作业JCT、完成、等待、checkpoint、训练、restart及wasted GPU-s完全一致。12格没有容量/进度/锁/rho审计违规；Medium增益来自同一冻结执行响应，不是重复计乘。

策略进程启动总墙钟1441.58秒（约24.03分钟，含首次4秒内存退出），测试墙钟约39.35秒；准备、分析和等待另计。封存前745个冻结源码/输入hash全部重新核验一致。

## 为什么20作业联合仍慢于串联：具体已观察证据

|Job|SIA-AUX有效训练分布（秒）|JOINT-R有效训练分布（秒）|JCT变化R-AUX|
|---|---|---|---:|
|bert-54|AWS1:60，DGX1:1557|RTX1:1200，DGX1:1324|+907|
|bert-99|DGX1:1591|RTX1:1200，DGX1:1324|+1053|

两任务的服务候选过滤各累计出现`NO_PRODUCTIVE_SERVICE_SINCE_GEOMETRY=283`、`NO_REMAINING_SERVICE_PAYBACK=5986`。这是被拒绝候选列的累计次数，绝不是等待轮次或等待秒数。它提示应审查短服务回本过滤与GPU类型选择，但不足单凭计数断定所有被拒绝列都全局可行、更能改善JCT。真实checkpoint锁不能因这项损失被放松。

数据中有辅助资源机会，当前联合也确实利用了部分机会；但“更高rho”“更少等待”和“更多任务同时运行”均不足保证更低JCT。本组20作业Medium R相比AUX的额外训练耗时7019秒，抵消了等待及checkpoint节省3730秒。不能再用“只有3个任务”作为唯一解释，也不能据单个受控负载家族否定联合方向。

下一步建议先做已有日志的同状态候选资格/全局可行性诊断，定位上述RTX长驻与payback过滤，再审定最小隔离实验；不是盲目继续加作业或调整参数。此轮不实施下一步。

## 产物权威性

- `SCALE_COMPETITION_REPORT.md`：完整主报告及12格指标、逐任务结果、资源小时、尾延迟、长尾诊断。
- `preregistration_v2.json`、`source_snapshot_v2.zip`：唯一正式执行冻结；v1保留启动前Windows路径适配证据。
- `validation_8.json`、`validation_20.json`、`extra_audit.json`：原始日志复核。
- `mechanism_by_config.json`：上述训练分布与过滤计数来源。
- `final_manifest.json`：主报告生成时的文件快照；此后只新增收尾资料并更新CURRENT_TASK完成标记。
- `delivery_manifest.json`：最终交付的权威全文件hash索引，包含上述快照及收尾资料。
- `frozen_prerequisites.zip`：全部冻结前置输入/源码独立副本，路径相对项目根；不只保存hash。

新树仅新增显式测试的20任务上限适配、运行器和规模测试；原AdaptiveAdmission/历史Sia/AUX算法逐字节不变。未提交/推送，没有修改官方源码。sia-104自动化确认PAUSED；没有恢复104格或运行160任务。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
