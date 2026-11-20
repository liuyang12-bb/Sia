# 可重建性收尾补充

首次包`rebuildable_source_inputs.zip`完整保存了本轮实际源码、输入和诊断证据，但最终依赖检查发现：既有测试还读取多个历史目录中的小型fixture。为避免“当前机器能测试、独立恢复后缺输入”，保留首次包和manifest，不覆盖，另外生成：

- `rebuildable_source_inputs_complete.zip`：首次包内容，加既有测试所需历史输入、官方Sia只读源码副本及Synergy矩阵副本；均按原相对路径保存并记录hash。
- `final_delivery_manifest.json`：最终入口，引用首次manifest、完整包、逐文件哈希，以及从空目录恢复后的有界测试结果。
- `rebuild_validation_workspace/`：在本轮新目录中解压验证的副本，不修改任何原树或旧输入。

恢复后唯一需要路径重定位的fixture是`small_mechanism_20261025/preregistration.json`的config/registry/models/workload绝对路径。在**新恢复副本**里仅替换原项目根前缀；原文件、原参数、hash溯源字段和所有算法源码保持不动。路径映射前后记录在最终manifest；不能把这个操作用于修改实验数据。

完整测试仍是固定上限正确性用例，不会启动160-job/104格实验。原报告中的算法、门槛和性能结论不变。最终交付应同时查看REPORT.md和本补充，以`final_delivery_manifest.json`及`rebuildable_source_inputs_complete.zip`为独立重建入口。
