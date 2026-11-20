# 离线验收与当前阻塞

5项独立离线检查全部通过：329文件源码及ZIP逐字节校验；6份trace提取的全参数保留；受控波次/均匀到达规则；47格矩阵及最低容量；55项R0原始GPU画像源文件与Synergy模板hash。

现有335项调度测试尚未复跑；20/40GPU执行适配尚未实施/验收。不能把5项离线检查称为完整系统验收。

完整4容量×55配置覆盖中，找到11个支持域内、两任务相对速率和为正的互补配对。并非11个已兑现JCT收益。

- 双资源：CIFAR/DeepSpeech的1GPU同配置支持域内正互补仅AWS；DGX/RTX该受限预算下最好为持平。
- CPU相对受限：AWS该配对增益很小，其余两类最好为持平，不能提前宣称强CPU优化机会。
- 内存相对受限：三类GPU都有CIFAR/DeepSpeech支持域内正互补。
- CIFAR/BERT的DGX/RTX1GPU支持域配对缺失，不能把旧E-legacy下的理想donor证据当成支持域内验证。

这些只是固定1GPU两任务响应机会，不是不同应用raw训练率之和，也没有证明完整80任务持续净收益。剩余GPU数量配置覆盖见coverage.csv；未按机会结果修改任何容量、任务或模型。

正式矩阵0/47启动。最后资源采样写在execution_manifest.jsonl。磁盘不足5GiB时停止，不删除历史数据，不降低门槛。恢复时先完成执行适配与完整验收，再冻结最终执行源码；当前baseline_source_snapshot.zip只冻结历史v8基线。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
