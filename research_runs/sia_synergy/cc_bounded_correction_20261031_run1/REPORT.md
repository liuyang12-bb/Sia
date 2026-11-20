# 有限纠偏恢复：80作业FO/FULL共同规则对照

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

旧结果保持封存；新FO/FULL使用相同有限恢复规则。当前ready为空、多个已分配作业、目标已画像且有实际训练证据，固定其他当前分配后目标配置仍可行，预计完成时间含真实启动且严格改善。新恢复只开放一个现有比例点；不修改普通辅助优化。与旧单作业尾部恢复共用一次许可，用后固定GPU几何。

余量见证证明无需驱逐peer即可落地；最终普通优化仍可调整其他peer，不声称其全程配置完全固定。不得读取未来到达。估计改善不是实际JCT保证。

|版本/方法|完成|平均JCT|P95|P99|Makespan|等待秒|训练秒|checkpoint秒|GPU-h|CPU-h|RAM-h|restart|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_FO | 80/80 | 6610.688 | 15320.500 | 17021.070 | 18171 | 282060.000 | 231055.000 | 15740.000 | 83.822 | 258.022 | 5265.187 | 266 |
| old_FULL | 80/80 | 6538.913 | 15735.100 | 20286.420 | 21204 | 260820.000 | 245903.000 | 16390.000 | 79.981 | 266.585 | 5331.912 | 280 |
| new_FO | 70/80 | None | None | None | None | 282060.000 | 201505.000 | 16030.000 | 72.189 | 210.884 | 4379.665 | 270 |

## 配对变化

- new_FO_vs_old_FO：未完成配对，不插补JCT。
- new_FULL_vs_old_FULL：未完成配对，不插补JCT。
- new_FULL_vs_new_FO：未完成配对，不插补JCT。
- old_FULL_vs_old_FO：均值改善1.086%；61改善/18退化/1持平。等待、checkpoint、训练总节省分别21240.0、-650.0、-14848.0秒。

恢复规则×联合选择交互：None秒/作业；负值为FULL对FO优势扩大。实际恢复事件7次，逐事件见recovery_events.json。

## 验收与局限

348项测试（原335+新13）、真实unseeded BERT12轮链路、原始Sia3轮精确回归、新旧FO/FULL各3轮前缀桥接记录见software_acceptance.json。红测两项失败，捕获旧规则无法多作业恢复。初次扩展测试另有锁定peer零进度超过watchdog的测试输入错误，修正到180秒后重跑；失败日志保留。

独立原始事件审计违规总数：0。所有资源、锁、rho与互斥时间检查详见audit/。比较表来自原始完成事件和执行片段独立复算。

单个80作业受控负载，响应依赖R0/E-legacy跨硬件合成及边界裁剪。不能推出所有负载获益；FO/FULL共享恢复的收益不能全部归为联合优化。求解墙钟不计入仿真JCT。预测ETA误差和少数长尾贡献仍须检查，主结果保留全部作业。

## 可复现材料

source_snapshot.zip、source_hashes.json、source_changes.patch、input_hashes.json、preregistration.json、execution_manifest.jsonl；历史源329项逐项核验不变。新树是完整源码副本，没有虚构Git commit。依赖版本固定在预注册environment。

预注册从历史基线继承的source_snapshot_path和matrix两个说明字段仍指向旧路径/旧七格描述；未修改运行中的预注册。实际执行以cells中的FO/FULL两格、source_root和逐项源码/输入hash为准。重建使用本目录source_snapshot.zip，其哈希与预注册一致；路径及矩阵澄清见snapshot_and_matrix_receipt.json。
