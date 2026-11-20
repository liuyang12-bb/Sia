# 六任务四策略完整训练对照报告

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

本轮按用户批准的六任务场景执行，保留相同工作量、Medium响应和既有算法版本。首次预注册之后未按结果调整参数。

六任务为CIFAR0/1/2与DeepSpeech8/9/12，同时到达；AWS与DGX各2 GPU、12合成CPU单位、250 GB RAM。比例点6CPU/125GB每GPU。

## 实际结果

|策略|状态|完成|平均JCT(s)|Median|P95|P99|Makespan|等待(s)|训练(s)|checkpoint(s)|restart|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|SIA_P|COMPLETE|6/6|5636.500|6765.000|10352.500|11170.500|11375.000|14400.000|18794.000|625.000|14.000|
|SIA_ECT_AUX|COMPLETE|6/6|5616.333|6734.000|10316.500|11124.900|11327.000|14280.000|18793.000|625.000|14.000|
|JOINT_ECT_P|COMPLETE|6/6|7308.833|7394.000|10716.750|10748.150|10756.000|9420.000|28983.000|5450.000|163.000|
|JOINT_ECT|COMPLETE|6/6|6832.500|6843.500|10470.000|10473.200|10474.000|8040.000|28005.000|4950.000|157.000|

## 配对与归因

- JOINT-ECT相对SIA_P平均JCT改善-21.219%；改善/退化/不变任务数2/4/0。净JCT节省-7176.000秒。最大受益任务cifar10-2贡献4178.000秒；去掉该任务的诊断改善-42.790%，不替代主结果。
  总等待/checkpoint/训练时间变化：{'wait': -6360.0, 'checkpoint': 4325.0, 'training': 9211.0}。
- JOINT-ECT相对SIA_ECT_AUX平均JCT改善-21.654%；改善/退化/不变任务数2/4/0。净JCT节省-7297.000秒。最大受益任务cifar10-2贡献4178.000秒；去掉该任务的诊断改善-43.445%，不替代主结果。
  总等待/checkpoint/训练时间变化：{'wait': -6240.0, 'checkpoint': 4325.0, 'training': 9212.0}。
- JOINT-ECT相对JOINT_ECT_P平均JCT改善6.517%；改善/退化/不变任务数6/0/0。净JCT节省2858.000秒。最大受益任务cifar10-2贡献892.000秒；去掉该任务的诊断改善4.933%，不替代主结果。
  总等待/checkpoint/训练时间变化：{'wait': -1380.0, 'checkpoint': -500.0, 'training': -978.0}。

C对D（JOINT-ECT对JOINT-ECT-P）共享目标、L2和GPU候选，识别辅助配额优化的增量。JOINT-ECT对Sia的差异还包含目标、准入和placement，不能全归因CPU。SIA-ECT-AUX冻结完整Sia GPU意图和placement，不加入L2。

## 逐任务JCT(s)

|任务|SIA_P|SIA_ECT_AUX|JOINT_ECT_P|JOINT_ECT|
|---|---:|---:|---:|---:|
|cifar10-0|786.000|799.000|3711.000|3269.000|
|cifar10-1|843.000|819.000|4199.000|3323.000|
|cifar10-2|7285.000|7285.000|3999.000|3107.000|
|deepspeech2-8|6808.000|6807.000|10599.000|10474.000|
|deepspeech2-9|11375.000|11327.000|10756.000|10458.000|
|deepspeech2-12|6722.000|6661.000|10589.000|10364.000|

## 资源、约束与模型边界

|策略|GPU-h|CPU-unit-h|RAM-GB-h|低于比例进度点的选中次数|选中列数|截断计数|
|---|---:|---:|---:|---:|---:|---|
|SIA_P|10.2025|61.2150|1275.3125|0|328|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 7, 'denominator_ram': 0}|
|SIA_ECT_AUX|10.1883|42.3911|1273.5417|2|327|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 7, 'denominator_ram': 0}|
|JOINT_ECT_P|11.6964|70.1783|1462.0486|0|576|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 1, 'denominator_ram': 0}|
|JOINT_ECT|11.4019|59.5450|1425.2431|6|552|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 0, 'denominator_ram': 0}|

CPU敏感性来自Synergy矩阵的合成跨应用/跨硬件迁移，CIFAR用AlexNet模板、DeepSpeech用DeepSpeech模板。两类都可能在低资源时减速，DeepSpeech不是绝对不敏感donor。

保留已知E-legacy分子及分母clip，CIFAR DGX2固定分母越界未修复或删除。计数以逐轮选中列为分母，CPU/RAM类别可重叠。任何收益均不能据此声称同平台测量或迁移模型已验证。

独立审计从原始完成事件复算JCT，从施加rho前raw及训练段复算倍率与进度，核对等待/训练/checkpoint互斥时间、完成后的资源释放、节点容量及完整锁。通过范围见independent_audit.json。

188项既有测试证据沿用上一轮已核对的同源码版本；本轮未修改算法，另有四策略构造验收。没有以构造验收替代实际逐轮审计。

本轮从预注册至审计墙钟312.815秒；四子进程各自实测时间见execution_manifest.jsonl。每组仅6作业，P95/P99为描述性指标，没有总体尾延迟精度含义。

所有源码、输入、脚本与依赖冻结见preregistration.json。逐轮完整证据在cells/<method>/rounds.jsonl.gz；简明轨迹在audited_decisions.jsonl.gz；逐任务结果见per_job.csv。

本轮结束后停止。不按结果修改参数、不扩展负载、不恢复104格或160-job。
