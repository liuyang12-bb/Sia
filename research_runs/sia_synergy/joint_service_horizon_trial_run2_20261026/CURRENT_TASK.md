# 服务时域评分实验：当前状态

- 来源：`sia-joint-protected-admission` 的实际 116 个 Python 文件，复制到隔离工作树 `sia-joint-service-horizon`；原有文件的哈希与上一轮预注册逐项相等。基线 HEAD 为 `2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2`，不能把该 HEAD 单独当作完整执行源码。
- 新策略：`joint_prototype/service_horizon.py`，代号 G112。唯一算法变化：已承诺任务按剩余有效服务秒数（至少一个60秒周期，最多300秒）评分；新任务仍用300秒。资源交换和保护式准入继承 G111。
- 对照：同一新工作树中的 G111。完全沿用先前的六作业 workload、节点规格、R0/E-legacy Medium、模型、求解参数、Q=300、D=250、600轮上限。新输入已冻结在 `preregistration.json`，源码见 `source_snapshot.zip`。
- 首次冻结目录 `joint_service_horizon_trial_20261026` 在运行前因复制来的脚本出现 `IndentationError` 而停止，没有训练结果；该目录保留。当前 run2 的脚本已修复并通过语法检查。
- run2 执行 G112 前被原有2 GiB门槛停止，退出码1。2026-10-26现场读数：可用内存0.435 GiB、可用磁盘21.70 GiB；没有启动任何训练轮次，`cells/`未创建。不降低安全门槛。
- 单项集成测试和完整已有测试套件通过（完整套件退出码0）。没有声称JCT收益或损失。

内存达标后，在当前已冻结目录分别以独立进程顺序执行：

`D:/科研启航/Sia/.venv-sia-baseline/Scripts/python.exe -X utf8 -B cell.py --method G112`

仅当其 `opportunity_gate.json` 为 passed、且20/160作业实验仍保持暂停时，再执行 `--method G111`。如果G112被门槛拒绝，只报告机制没有触发，不评价JCT。每种方法的 `cells/<method>` 目录必须此前不存在。运行后执行 `audit.py` 并对比原始完成事件。不得覆盖上一轮有效结果或本轮任何失败目录。

注意：G112只检验“已承诺作业评分与服务期限一致”这一项，仍未显式计入被停跑任务未来恢复损失；不能称为完整多轮JCT预测或保证改进。
