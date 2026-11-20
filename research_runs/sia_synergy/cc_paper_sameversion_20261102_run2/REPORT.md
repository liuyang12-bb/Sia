# 同版实验输入冻结与离线预检

状态：BLOCKED_RESOURCE_GATE。正式仿真0/47格启动；没有新增JCT结果。

首次资源：内存5.030GiB，磁盘4.257GiB。既有门槛2/5GiB。没有删除历史文件、降低门槛或启动工作进程。

输入：受控80作业waves/uniform；Saturn/Philly各workload1/2/3前80个稳定到达记录。共8份CSV，逐项hash与行号见input_manifest.json。trace来源CSV不能恢复所有上游抽样身份，不能宣称六份样本相互独立。

矩阵：服务8格，资源12格，宽松对照3格，trace18格，40GPU6格，总47格。40GPU相同到达序列用于容量敏感性，不宣称保持负载强度的可扩展性。

实际v8源码329文件与历史执行hash完全一致，并保存逐文件校验的完整ZIP。尚未生成/验收新的执行适配版本。

覆盖：4容量×55配置=220项，比例配额均符合逐节点静态容量，响应有限且为正。完整配置与CPU/RAM分开变化结果见coverage.csv。支持域内两作业互补证据见opportunity_witnesses.json；相对速率和不等于不同应用绝对Goodput之和，不是全局净收益或JCT证明。

剩余验收：完成评分下的服务四开关、双节点但单节点任务域、335项旧测试、新适配测试、原Sia精确回归、同状态实际raw-rate净收益，以及逐轮持续触发。SIA_AUX与FULL历史评分不同，主表只评价整体方法，不称纯GPU单因子效应。

任何正式运行必须在资源门槛满足后独立进程串行启动，冻结执行源码与参数。历史结果没有复用成新实验完成格；没有仿真、没有报告性能改善。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE
