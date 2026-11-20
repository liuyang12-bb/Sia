# 半 CPU 场景离线预检结论

**结论：有条件值得进入四策略对照，但只能作为冻结合成模型下的机制实验。** 半 CPU 配置确实使 CPU 成为绑定资源，并且现有任务组合含有对该资源敏感与不敏感的任务；静态资源分配也能找到满足容量限制的速度改善方案。关键限制是：四个固定 GPU 状态中的“让出资源且不降低任何作业速度”都需要至少一个落在 E-legacy 响应曲线裁剪边界上的资源释放动作。故这组数据能测试调度器是否利用当前模型定义的机会，不能证明目标 GPU 集群会有同等收益。

> SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

## 范围与复核

- 输入为版本桥接冻结的 `original20` 和 `holdout20`，各 20 个作业，统一到达时间为 0；保留原始 job ID、到达和 workload 字段。两组应用构成为 original20：BERT 6、CIFAR10 7、DeepSpeech2 7；holdout20：BERT 4、CIFAR10 8、DeepSpeech2 8。
- 固定 GPU/RAM 和模型映射，仅把每 GPU 合成 CPU 配额从 3 降到 1.5。节点总量从 AWS/DGX-ext/RTX 的 12/24/24 降为 6/12/12；GPU 为 4/8/8，RAM 为 250/500/500 GB。参考坐标仍固定为 3 CPU、62.5 GB，不按新容量重新归一化。
- 沿用固定 R0 registry、E-legacy、alpha=0.5 与现有模板映射：BERT→`bert_1.py`、CIFAR10→`alexnet_1.py`、DeepSpeech2→`deepspeech_1.py`。离线吞吐只用于建立 phi 与响应查询，不注入 Sia Goodput。
- 预检的两个状态是每组的首次调度状态（t=60）及首个三应用均有有效训练且无 checkpoint 锁的暖态（t=240）。每个状态固定当时 GPU 配置、placement 和任务集合；用精确 CPU-only DP 与 CPU/RAM 离线整数分配核验可行点。这是静态反事实，不是模拟器调度或 JCT 结果。
- 版本：`sia-joint-version-bridge`，分支 `exp/b3-version-bridge`，HEAD `2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2`。实际冻结源码以 `execution_source.zip` SHA-256 `F64B64EA06474A39AE1EC28B3E73ADCEC7ABAE439820F5F04EA5CBF798B82951` 标识；R0 registry SHA-256 `78F676E7695CBE09AA9A3DB71F539FE2A2CEF1DBC243E57CC064E4E294DCF1C9`。原始运行清单声明 `OFFLINE_ONLY_NO_SIMULATION`：未导入训练模拟器、未构造 Cluster/Job、训练 step 调用为 0；完成 22,038 次独立响应查询。

复核使用的机器可读证据：

- `research_runs/sia_synergy/half_cpu_offline_preflight_20261026/attempt2/run_manifest.json`
- `.../attempt2/preregistration.json`
- `.../attempt2/validation.json`
- `.../attempt2/summary.json`
- `.../attempt2/task_coverage.json`
- `.../attempt2/configuration_coverage.json`
- `.../attempt2/same_geometry_resource_witnesses.json`
- `.../attempt2/boundary_release_certificates.json`

预检验证结果为 `valid=true`：覆盖 440 个 job-config 记录、320 个状态分配、48 项节点容量检查、16 个资源分配见证；比例资源点全部可行。验证器确认 1,233 个历史桥接输入/源码哈希未变，并确认正式 104 格实验仍为暂停状态。

## 是否出现资源竞争

出现。半 CPU 场景的比例点为每 GPU 1.5 CPU；当三个节点 GPU 全部使用时，节点 CPU 容量也恰好耗尽：AWS `4×1.5=6`，DGX-ext 与 RTX 各 `8×1.5=12`。暖态两组的 20 个作业均有合法固定 GPU 分配，节点资源审计通过。因而这里不是“配置里写了 CPU 字段但没有约束”：若把同一节点上的 CPU 从敏感任务转给其他作业，就必须同时考虑前者速度下降以及后者速度恢复。

在完整的 55 个应用×GPU 配置响应域中：

| 应用 | 半 CPU 时变慢的配置数 | 半 CPU rho 范围 | 存在更高 CPU 恢复点的配置数 | 解释 |
|---|---:|---:|---:|---|
| BERT | 0/11 | 1.000–1.000 | 0/11 | 当前固定映射与 E-legacy 裁剪下对半 CPU 不敏感；不提供 CPU 速度恢复机会。 |
| CIFAR10 | 10/11 | 0.692–1.000 | 8/11 | 多数配置对减半 CPU 敏感，部分可由更多 CPU 恢复。 |
| DeepSpeech2 | 11/11 | 0.692–0.881 | 8/11 | 全部配置在半 CPU 下减速，多个配置存在更高 CPU 恢复点。 |

实际两组 workload 中，14/20（original20）和 16/20（holdout20）作业的历史合法候选至少有一项半 CPU 减速且可通过增加 CPU 恢复的配置。这里的“可恢复”是固定响应模型内候选点的比较，不代表在线时 Sia 一定选择该配置，也不代表净 JCT 改善。

## 离线互补见证及其边界

对固定 GPU/placement 与固定作业集合，离线模型在半 CPU 节点容量下找到了下列资源可行的分配。`mean relative speed` 是相对于该状态比例配额的任务速率比平均值，不能换算成 JCT。

| 负载/状态 | 辅助分配 | 平均相对速率 | 获得速率提升的作业 | 提升点在曲线内部 | 提升涉及裁剪 |
|---|---|---:|---:|---:|---:|
| original20 / 首次决策 | 仅优化 CPU | 1.0762 | 4/20 | 1 | 3 |
| original20 / 首次决策 | 优化 CPU+RAM | 1.1358 | 7/20 | 1 | 6 |
| original20 / 暖态 | 仅优化 CPU | 1.0329 | 3/20 | 1 | 2 |
| original20 / 暖态 | 优化 CPU+RAM | 1.0998 | 5/20 | 1 | 4 |
| holdout20 / 首次决策 | 仅优化 CPU | 1.0614 | 4/20 | 0 | 4 |
| holdout20 / 首次决策 | 优化 CPU+RAM | 1.0954 | 5/20 | 0 | 5 |
| holdout20 / 暖态 | 仅优化 CPU | 1.0247 | 2/20 | 0 | 2 |
| holdout20 / 暖态 | 优化 CPU+RAM | 1.0659 | 2/20 | 0 | 2 |

全部离线见证通过容量和比例进度下限检查，8 个固定状态×容量×辅助资源模型求解均为最优。它们表明在当前数学模型中，CPU/RAM 的重新分配能提高固定 GPU 集合的总预测速率。但 `boundary_release_certificates.json` 同时指出：四个固定 GPU 的半 CPU 状态都需要至少一次被裁剪的资源释放，才能在不降低任何任务速度的前提下把资源转给受益任务。holdout20 的四个优化结果中，受益点本身也全部落在裁剪边界；original20 虽各有一个收益点在内部，但完成无损资源转移仍依赖边界释放者。

这一区别很关键：模型给出的配置在容量内可执行，但部分互补收益来自把 E-legacy 的 clipping 当作平台区使用。裁剪在当前实现中是预注册公式的一部分，不是计算错误；然而裁剪区间之外的真实响应未知，边界平台是否成立没有目标硬件数据支持。因此当前可以说“模型内存在可利用机会”，还不能说“真实 CPU 敏感任务可无损让渡这些 CPU”。

## 对四策略对照的建议

**建议进入，但限定为机制验证，且把裁剪依赖作为预注册的主要解释边界。** 半 CPU 场景能触发执行速率变化、CPU 容量是紧约束，而且负载中同时有 BERT 不敏感作业与 CIFAR10/DeepSpeech2 敏感作业；这足以测试 Sia、Sia+辅助优化、自由联合策略和 JOINT-G 是否将 CPU 从低边际收益位置转到高边际收益位置，以及静态速率收益能否抵消等待、重启和训练推进损失。

对照必须使用同一冻结负载、GPU/RAM、rho、初始化画像、工作量、观察上限与执行规则；逐轮报告实际 CPU 配额、rho、Goodput、等待、checkpoint/restart 和所有边界裁剪。四策略的主结论应限定为：当前合成响应模型下的调度行为及模拟 JCT。若收益集中在裁剪释放点，应单独标示为边界敏感结果；不把它推广成真实集群结论。

如果研究问题要求在边界规则获得目标硬件依据前就证明“可验证的、非边界依赖的资源互补”，则当前预检**不满足该证据要求**，应先补充独立数据或明确承认合成假设。无需为了得到正收益更改现有负载或响应曲线。

## 本轮没有做的事情

未启动任何训练模拟或四策略运行，未修改调度器、负载、phi、响应、容量或桥接树源码，也未恢复正式 104 格实验。之前的 attempt1 验证器失败证据保留；本结论依据有效的 attempt2 输出。
