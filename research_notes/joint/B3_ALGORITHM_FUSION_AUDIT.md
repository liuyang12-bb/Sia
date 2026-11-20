# B3 联合调度算法融合审计

> 结论范围：代码审计与两作业机制验证。本文不构成真实 Saturn 或真实异构 GPU 集群性能结论。

## 1. 审计结论

B3 确实把 **Sia 的异构 GPU 候选、弹性配置、实际 Goodput 模型、公平性指数与重配置折损**放进了一个逐节点 GPU/CPU/RAM 有限列 MIP；它不是“先运行 Sia 得到唯一 GPU 配置，再分配 CPU/RAM”。因此，B3 在本原型定义下是真正的联合选择器。

但是，B3 **没有移植 Synergy 的完整调度算法**。它复用了 Synergy 官方 CPU×RAM 响应矩阵的形状，并采用“不劣于 proportional 执行点”的候选筛选思想；Synergy-TUNE 的 demand→fair-share→peer downgrade 流程、Synergy-OPT 的两阶段求解、固定 GPU demand 和官方 FIFO/SRTF 策略均不属于 B3。B3 应表述为“受 Sia 与 Synergy 启发的新联合有限列算法”，不能称为官方 Synergy 算法的直接组合。

本轮未根据 JCT 调整目标权重、响应指数或模板。发现并修复的是 SY 固定 GPU demand 在早期画像不足时随 Sia profile 状态变化的问题；修复后 demand 始终由预注册配置决定。

## 2. 版本与保护状态

- 实验仓库：`sia-joint-experiments/`
- 分支：`exp/synergy-baselines-cpu-bottleneck`
- HEAD：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 工作区在任务开始时已含任务 12–14 的未提交原型；本轮保留这些修改，没有 reset、clean 或覆盖历史产物。
- `official-artifacts/`：`main...origin/main`，干净，未修改。
- `research_sources/Synergy/`：只读审计，未修改。

可复核哈希见 `research_runs/sia_synergy/cpu_bottleneck_smoke/code_manifest.json`，运行时输入与源画像哈希见同目录 `run_manifest.json`。

## 3. B3 的真实调用链

1. `JointSiaPolicy.optimize()` 在 `joint_prototype/joint_integration.py:467-533` 接收 Sia 当前轮 `JobInfo` 与节点状态。
2. `_raw_candidates()`（同文件 `:111-190`）遍历 Sia 的 `configs`，检查 `scale_unit`、min/max replicas、rigid 约束，并在 `:139/142` 直接调用 `SiaPolicy._compute_goodputs_v2()`；后者位于 `sia.py:423`。direct profile 与 projected profile 在 `:136-146` 分开标记。
3. 每个合法 GPU 配置被扩展为 CPU、RAM、placement 列。`JointScheduler.build_joint_columns()` 位于 `joint_scheduler.py:233-293`，checkpoint 中的任务只保留当前配置和节点（`:242-246`）。
4. `JointProfileOracle.query()`（`joint_profile.py:234-295`）查询官方 Synergy 矩阵并计算 `rho` 和候选 effective Goodput。
5. `_b3_result()`（`joint_integration.py:357-372`）使用 Sia 的 `p`、`lambda_n` 与基于当前状态的 restart factor 生成效用，然后调用一个 MIP 同时选择 GPU 配置、CPU、RAM 和 placement。
6. `JointScheduler._constraint_matrix()/solve()`（`joint_scheduler.py:357-416`）加入每任务至多一列以及逐节点 GPU/CPU/RAM 上界，解为二元列选择。
7. 解在 `joint_integration.py:530-533` 通过容量审计后才一次性提交；无合法完整解时不能提交部分资源状态。
8. `JointCluster.optimize_policy()`（`:573-578`）把本轮 `rho` 写入 Job；`JointCluster.step()`（`:580-635`）沿用 Sia 的时间推进，并记录训练时间、checkpoint delay、raw/effective Goodput 和完成事件。
9. `job.py:419` 的原始 `Job.step()` 继续负责 checkpoint/restore、batch、epoch 和 progress；新增 multiplier 只作用于训练推进，不作用于 delay 倒计时。

## 4. 数学模型

对任务 `j` 的每个联合列 `k=(GPU 类型、GPU 数、节点集合、CPU/GPU、RAM/GPU)` 定义二元变量 `x_jk`。候选性能为：

`g_eff(j,k) = g_sia(j,gpu_config) × rho(j,gpu_config,cpu,ram)`。

固定参考模式下：

`source_cpu = 3 × (cpu_alloc / 3) / phi_c`

`source_ram = 62.5 × (ram_alloc / 62.5) / phi_m`

`raw_rho = F_template(source_cpu, source_ram) / F_template(3/phi_c, 62.5/phi_m)`

`rho = raw_rho^alpha`，本轮 neutral 的 `alpha=0`，medium 的 `alpha=0.5`。

B3 的列效用与当前 Sia 原型一致：

`u_jk = sign(p) × (round((g_eff × restart_factor)^p, 2) - lambda_n)`，并沿用 `sia_goodput_clip_val`。

约束为：

- `Σ_k x_jk ≤ 1`；
- 对每个节点 `n`：`Σ gpu_jkn x_jk ≤ G_n`；
- `Σ cpu_jkn x_jk ≤ C_n`；
- `Σ ram_jkn x_jk ≤ M_n`；
- checkpoint 锁定任务只能选择当前 GPU 配置和当前节点；
- 每列必须是 Sia 合法 GPU 配置和完整合法 placement；
- 启用 floor 时，候选性能不得低于该 placement 的合法 proportional 执行点。

## 5. 机制对照表

| 机制 | Sia 原始 | Synergy 原始 | B3 当前实现 | 归属判断 |
|---|---|---|---|---|
| 异构 GPU 类型选择 | 有 | 无；同构、固定 demand | 有，枚举 Sia cluster/config | Sia |
| 弹性 GPU 数与合法 scale unit | 有 | 固定 GPU demand | 有 | Sia |
| direct/projected Goodput | 有 | 无 | 直接调用 `_compute_goodputs_v2` | Sia |
| `p` 公平效用 | 有 | 无 | 有 | Sia |
| restart/reallocation 折损 | 有 | 无同等语义 | 有，当前状态计算 | Sia |
| checkpoint 配置锁 | 有 | 无同等语义 | 有 | Sia |
| CPU×RAM 响应矩阵 | 无 | 有 | 借用官方矩阵数据 | Synergy 数据 |
| GPU-proportional/fair-share 参考 | 无 | 有 | 作为执行点/收益下界 | Synergy 思想 |
| Synergy-OPT 第一阶段矩阵优化 | 无 | 有 | 未移植 | 未实现 |
| Synergy-OPT 第二阶段 placement | 无 | 有 | 未移植 | 未实现 |
| TUNE demand→fair→peer downgrade | 无 | 有 | B3 中没有；仅 SY/B5 使用 | 未实现于 B3 |
| 固定 GPU demand | 无 | 有 | B3 不固定 | 与 Synergy 不同 |
| 联合 GPU+CPU+RAM+placement MIP | 无 | 无 | 有 | 本项目新设计 |
| GPU 性能映射到 `phi_c/phi_m` | 无 | 无 | 有 | 本项目合成假设 |
| 跨 GPU/应用矩阵迁移 | 无 | 无 | 有 | 本项目合成假设 |

## 6. `rho=4.2981` 的可复算来源

历史 smoke 中的精确记录位于 `research_runs/sia_synergy/joint_response_audit/rho_4_2981_reproduction.json`：

- 任务：`cifar10-29`
- GPU 配置：`dgx-ext:1:1`
- 分配：8 CPU/GPU、125 GB/GPU
- 模板：`res18_1.py`
- 合成需求缩放：`phi_c=4`、`phi_m=2`
- 当前源坐标：`(2, 62.5)`，官方矩阵双线性查询值 `0.72`
- baseline 请求坐标：`(0.75, 31.25)`；CPU 被 clip 到官方网格最小值 1，查询值 `0.1675161987041037`
- 因此 `rho = 0.72 / 0.1675161987041037 = 4.298091799896853`

这不是同一轮重复乘倍率造成的；它来自低归一化分母、`phi` 跨硬件缩放、网格边界 clip 与固定参考点的共同作用。只有矩阵单元本身来自 Synergy 官方画像；固定参考、`phi`、跨 GPU/跨应用映射、插值和乘到 Sia Goodput 上均是合成假设。故 4.2981 不能解释为真实硬件加速比。

本轮预注册 medium 使用 `alpha=0.5` 后，20 个两作业配置中的最大 `rho` 为 3.4155136484（CPU 充足、B3、`alexnet_1.py`）。它仍然偏高，说明响应模型不确定性尚未闭合；没有事后截断该值。

## 7. 发现的限制与影响

1. **决策/执行 Goodput 口径并非数值恒等。** 调度候选使用 Sia 的归一化 `_compute_goodputs_v2`；实际推进仍由 `Job.step()` 根据实时 batch/profile 产生，再只乘一次 `rho`。这保留了 Sia 的执行语义，但候选预测与实际瞬时 Goodput 可能有差异。
2. **`phi_c/phi_m` 是合成迁移。** 当前以每 GPU Goodput 比率生成并 clip 到 `[0.5,4]` 与 `[0.5,2]`，没有真实跨 GPU CPU/RAM 测量校准。
3. **B3 不等于 Synergy。** 使用响应矩阵不等于采用 Synergy 的调度器。
4. **逐节点联合约束改变可行域。** 即使 neutral `rho=1`，B3 与原 Sia 也只保证目标构造相近，不保证逐轮同解；本轮 neutral 平均 JCT 实际差 0.5 秒。
5. **收益对响应模型敏感。** medium 下 B3 相对 B5 的平均 JCT 差为 -23 秒（CPU 充足）与 -6.5 秒（CPU 受限），neutral 下均为 +0.5 秒。这不足以证明真实优势。

## 8. 审计判定

`B3_FUSION_MECHANISM_CONFIRMED_WITH_SYNTHETIC_MODEL_LIMITATIONS`

B3 的联合决策结构是真实存在的，但 Synergy 贡献目前主要是画像与 fair/proportional 概念，而不是其完整调度算法。真实结论仍需校准跨硬件响应并扩大预注册 workload；当前不建议据此宣称 B3 优于 Sia 或 Synergy。
