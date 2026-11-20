# 交叉错配场景真实策略门槛核验

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

## 结论

**GPU收缩触发门槛未通过，按用户“通过后再运行12格”的条件停止。12格正式对照启动数为0。** 本轮没有新的12作业配对平均JCT，也不把局部速度改善当成JCT改善。

这否定了“此前静态反例已经足够支持当前策略必然进入该机会”的推断，没有否定FULL机制在其他可触达条件下可能有效。上一轮FULL相对FO平均JCT改善8.475%的完整结果未改变。

## 隔离及执行适配

由已验收 `cc-joint-residual-capacity-v1` 完整342文件逐哈希复制为 `cc-joint-mismatch-v1`，仅新增 `mismatch_adapter.py`。原342文件均不改。没有虚构Git commit；完整源码zip及逐文件hash可重建。

适配器显式在第一次真实调度前设置冻结的CPU/RAM容量；所有scheduler构造均逐节点断言CPU/RAM/GPU与配置一致。JointCluster的节点构造、分配及执行都读取同一policy.node_capacities。避免默认core80.build覆盖配置回3CPU/62.5GB每GPU的问题。目标、候选provider、响应、训练工作量、画像初始化、checkpoint和恢复规则不改。

## 门槛的预注册与实际调用

`gate_plan.json`在运行前写入：使用原COMPACT12前两条记录（cifar10-0、bert-28），不修改其batch/GPU demand/工作量；两者原到达均0。S3容量：AWS2GPU/9CPU/62.5GB，DGX2GPU/3CPU/187.5GB。Medium alpha=.5，seed_profiles(1,8)。固定最多30轮，每轮60秒，完成可提前停止；2GiB/5GiB启动门槛。

真实FULL训练轨迹在每次 `optimize_policy()` 前复制当前状态；两个副本分别用 `clean_gpu=fixed/joint` 调用真实 `optimize_policy()`，共同进入completion评分、service/恢复预算过滤、FO两阶段或FULL联合选择和完整原子校验。没有mock候选、没有额外seed、没有伪造Goodput、没有去掉checkpoint/服务保护。副本只执行一次真实调度，不向后训练。

完整暖启动轨迹2/2完成，实际27轮（1620秒），产生27对同状态决策。**这是一个FULL暖启动轨迹上的同状态门槛测试，不是两条独立FO/FULL完整训练对照。** 不报告其FO总体JCT。

触发标准是FO分配更多GPU、FULL收缩CIFAR GPU且有效预测速度提高，同时完整资源/锁合法。触发数0。不能据此推断全部12作业或所有可行状态都没有机会；本测试只能判断这条预注册短链路未触发。

## 实际证据

|时刻|FO|FULL|解释|
|---|---|---|---|
|60s|CIFAR AWS1，CPU4.5/RAM31.25|相同|冷启动对称比例候选|
|120s|CIFAR DGX2，BERT AWS1|相同|CIFAR首次具备当前模型，真实评分选择DGX2|
|300–600s|CIFAR DGX2，BERT AWS2|相同GPU配置|资源优化后CIFAR每GPU CPU1/RAM93.75，BERT每GPU CPU1/RAM20.83|
|720s|只剩BERT，AWS2|只剩BERT，DGX2|存在GPU类型选择差异，但不是所预注册的CIFAR收缩机会；不能用它替代通过标准|

真实120秒决策中的CIFAR候选最佳预计剩余完成时间（已含当前模型、batch、rho和切换延迟）：

|候选|raw rate|rho|delay|预计完成秒|
|---|---:|---:|---:|---:|
|AWS1 CPU9/RAM62.5|5.161261|1.611957|0|4691.198|
|AWS2 CPU4.5/RAM31.25|7.882730|0.618327|50|8057.520|
|DGX1 CPU3/RAM103.2809|15.286747|1.308235|50|2001.606|
|DGX2 CPU1.5/RAM93.75|18.356676|1.224716|50|1786.056|

AWS内部“1GPU优于2GPU”的机会真实存在，且两个比较点无需裁剪，但真实优化器可以选择更快的DGX2。先前静态反例把BERT固定在DGX、CIFAR留在AWS；真实策略没有采取这项GPU分工。仅证明AWS内部有排序反转，不等于证明FO实际选择了这个次优AWS配置。

先前预检还使用旧对称运行420秒的归一化Sia速度及“归一化速度之和”，本轮使用真实在线模型/batch/剩余工作和当前完成优先目标。两者不能互相替代。该差异源于状态、评分和服务集合选择，不是本轮暗改目标。

420秒时：CIFAR的DGX2预计完成310.084秒，AWS1为2825.234秒。即便暂看过滤前评分，AWS收缩也不是更优替换。实际共享资格过滤还拒绝400个CIFAR替代列（NO_SHARED_GEOMETRY_EVENT）；BERT334列被NO_PRODUCTIVE_SERVICE_SINCE_GEOMETRY拒绝。该轮没有checkpoint锁，不能把服务历史过滤与真实checkpoint锁混为一谈。这些边界均保留，未为触发机会而解除。

## 测试与审计

- 两项新增参数化测试通过：四容量×FO/FULL的首次真实scheduler容量/原子audit；FO和FULL各三轮对称容量与旧build+残余候选精确回归，包含GPU/配额/rho、progress、restart、rescale及完成事件。
- 原始Sia官方源码与本树3轮逐状态精确回归通过。
- 门槛内27×2次真实决策audit均valid；FO两阶段最终配置保持其比例提案，完整checkpoint列未丢失。
- 主暖启动训练每轮通过既有Metrics检查：独立rho公式、容量、checkpoint及执行倍率检查；本轮未新增独立完成事件性能复算，因为正式对照未启动。
- 旧393项测试的已验收证据保留；本轮没有改其342源文件，也没有把旧393通过冒称本轮又全套重跑。
- 本轮没有80作业训练，没有参数搜索，没有改变Medium/phi/clipping/响应或任务总工作量。

## 后续最小建议

当前不应仅凭AWS内部速度反例启动12格。下一次设计先补足一个条件：在真实完成评分下，FO确实把敏感任务放在辅助资源不匹配的GPU配置，同时FULL有一项满足当前服务/恢复约束且集合净收益为正的替换。必须同时检查其余GPU类型和同服务集合，不能人为固定BERT占据DGX而忽略另一种合法分工。

可优先只读检查已有12作业的多任务共驻状态：额外敏感任务是否让DGX不足，从而把部分任务挤到AWS；再用真实完成评分和当前资格域核验收缩。这里是机会预检建议，不是已证明的收益，不应通过修改到达/容量反复跑JCT来挑中有利结果。本轮未执行这项扩展。

门槛失败的价值是定位了具体缺口：**需要证明FO实际受困，而不是仅证明一个人为指定的GPU分工受困。** 这属于场景适用性证据不足，当前没有证据要求改目标或响应模型。

## 产物

`gate_plan.json`、`gate_result.json`、`gate_rounds.jsonl.gz`、`gate.log`保存预注册和全部同状态决策；`decision_evidence.json`保存紧凑评分/资格证据；`tests.log`及`original_sia_regression.json`保存测试；`source_hashes.json`、`source_snapshot.zip`和`integrity.json`用于重建与完整性验证。正式12格状态为BLOCKED_GATE_NOT_TRIGGERED，未生成虚构JCT表。
