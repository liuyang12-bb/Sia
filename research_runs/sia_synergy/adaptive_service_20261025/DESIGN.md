# JOINT-R：保留服务资格的有界配置适应

本轮只修正已观测的“在线模型已知更快GPU，但服务承诺仍固定旧几何/配额”的决策限制。不是宣称所有JCT退化均由此导致。

保持原目标 `sum(lambda_n - clip(round((G_sia*rho*f_restart)^p,2)))`、p=-.5、lambda_n=1.1、零bonus、候选域、比例进度floor、原末级tie-break。没有为了赢得BERT单独指定GPU或使用未来时长。

每个服务承诺记录issued_time、productive_base、累计已承诺延迟B、上次GPU几何变化时的productive计数。任务继续强制选择一列，真实checkpoint继续固定完整GPU/placement/CPU/RAM。

checkpoint外辅助配额可以改变，但新有效Goodput不得低于同一当前模型重新查询的现有配额。辅助重新加载开销沿现模型为0，不声称真实系统没有该成本。

GPU几何变化额外要求：已发生首次有效训练且自上次GPU变化以来有正训练；目标为direct/projected在线候选，不能用unprofiled占位；`B+d <= D_bar`；以及

`max(Q - productive_done - d, 0)*G_new > (Q-productive_done)*G_current`。

d来自目标Job.applications[gpu_type].rescale_time or30的实际模拟器启动延迟（CIFAR50，DeepSpeech25，BERT120）；不是Sia目标惩罚代理30/25/120。GPU变化记账d，quota-only记0。当前列始终刷新并保留。服务productive_base、issue时刻不因配置改变重置。

初次发服务也计入启动预算。例如BERT初始120+一次纠正120=240，仍不超过既有250；ImageNet初始250后不能在本服务段迁移，但可在下一服务段调整。预算耗尽后GPU几何固定，直到服务完成/到期；不能靠不断重启延长承诺。

活性论证：同一服务任务每轮被选中，所有启动延迟之和≤D_bar，有效训练计数不重置且达到Q退出，故从发服务到退出的墙钟≤Q+D_bar+2Δ（保守计入轮次量化）。正执行率、固定容量、有限时间正确求解、所有队首始终存在单独可行合法列是必要假设。排空期间不发无限新承诺；有限先行服务均有界退出，因此FIFO队首最终获服务；有限前序和有限负载下重复服务。最终完成还需有限总工作量和正执行进度下界。模型/profile变动使当前列非法或零执行率时显式报错，不伪造此保证。

本改进不保证JCT下降。回本门槛可能过于保守，逐任务不降速也可能拒绝全局更高效的资源让渡；线上GPU模型可能有误差。本轮只验证一个预注册版本，禁止看JCT后探索不同预算/Q或目标权重。
