# Synergy 独立基线与 B5 串行基线实现说明

> 本文中的 `SY` 是对官方 Synergy-TUNE 核心流程的异构模拟器适配，不是未经修改的官方 Synergy 系统；`B5` 是 Sia GPU 决策后再运行该适配器。所有跨硬件性能响应均为 synthetic cross-hardware transfer。

## 1. 官方源码依据

审计使用 `research_sources/Synergy/simulator/`：

- FIFO 调度入口：`schedulers/fifo_synergy_new.py:24-112`，按到达顺序处理可运行任务。
- placement：`resources/cluster.py:382-431`。先尝试 consolidation；否则按 GPU-normalized demand 做多服务器放置。
- TUNE 入口：`resources/cluster.py:575-586`。
- TUNE 核心：`resources/cluster.py:589-666`。先按任务 demand 分配；失败后把当前任务降为 fair share；仍失败时调整 peer，再最终重试。
- fair-share 转换：`resources/cluster.py:733-745`，超过 GPU-proportional fair 需求时降低 CPU/RAM/SSpeed，并把 `synergy_speedup` 重置为 1。
- Synergy-OPT 第一阶段：`opt_algo/synergy_opt.py:61-116`，选择 CPU×RAM 矩阵点、最大化总 normalized throughput，并要求每任务不低于 fair-share。
- Synergy-OPT 第二阶段：`opt_algo/synergy_opt.py:174-230`，在固定 demand 上做 server placement。
- 实际训练倍率与 demand 状态：`jobs/job.py:90-190`。

本轮选择移植 **TUNE** 而不是 OPT：TUNE 是官方模拟器运行时、placement-aware 的 demand 降级流程，且能形成与 B4（本项目有限列最优辅助资源分配）清晰不同的基线。没有把 TUNE 写成 OPT，也没有宣称移植两阶段 OPT。

## 2. 适配器实现

新增 `sia-joint-experiments/sia-simulator/joint_prototype/synergy_adapter.py`。

`SynergyTuneAdapter.run()`（`:132-219`）按 `(arrival_time, job_id)` 排序，执行：

1. 在固定 GPU 配置下尝试画像 demand（`:145-152`）；
2. 失败则降为合法 GPU-proportional fair-share（`:154-158`）；
3. 再失败则把已放置 peers 与当前任务一起原子重装为 fair-share（`:160-177`）；
4. 最终仍不可行则当前任务等待，不提交部分状态（`:178-189`）。

placement 使用 Sia 集成层已经枚举出的整数、同 GPU 类型配置，按节点数和节点 ID 做确定性 tie-break（`:80-87`）。所有提交仍经过统一逐节点 GPU/CPU/RAM 审计。

### 明确的适配差异

- GPU 类型和固定 demand 由调用者提供；TUNE 不参与 GPU 类型选择。
- Sia 的整数 placement 候选替换官方同构 GPU 对象。
- 官方最终失败会在 `cluster.py:608` 调用 `sys.exit(1)`；适配器改为任务等待，以便离散事件模拟继续运行。
- actual progress 统一由 Sia `Job.step()` 与共同 Oracle 控制。
- 当前适配器不实现 storage、network、server fragmentation 的全部官方细节。
- peer downgrade 以“构造完整新 map→全量审计→原子提交”实现，避免官方逐项 deallocate/reallocate 在共同模拟器中留下半提交状态。

## 3. SY：Synergy-adapted

SY 不调用 Sia MIP 决定 GPU。`JointSiaPolicy._sy_fixed_configs()` 位于 `joint_integration.py:306-342`：

- `bert-28` 固定映射到 `aws`，工作负载 demand=12，裁为不超过 demand 的最大合法 Sia 配置 `aws:1:4`；
- `cifar10-29` 固定映射到 `dgx-ext`，demand=6，裁为 `dgx-ext:1:4`；
- 映射与 demand 均来自预注册配置，且不读取 CPU/RAM 响应或 JCT。

`optimize()` 的 SY 分支位于 `joint_integration.py:518-524`，随后只调用 TUNE 适配器。

一个集成问题是：新任务尚未形成所需 Sia profile 时，原候选表可能只给 1 GPU fallback。若直接使用该表，SY 的“固定 demand”会随模拟时间变化。`_raw_candidates()` 的 `sy_fixed_demand_unprofiled_placeholder`（`:150-165`）只为已预注册 SY 配置保留候选；它不参与 GPU 选择，也不替代实际 `Job.step()` 训练速度。最终验证要求所有 SY 轮次始终保持上述两个固定配置。

## 4. B5：Sia GPU + Synergy-TUNE

B5 的路径位于 `joint_integration.py:471-517`：

1. 调用 `super().optimize()` 运行原 Sia MIP；
2. 将当轮 Sia GPU 类型、数量和配置冻结为 `fixed_configs`；
3. 调用 `SynergyTuneAdapter.run()` 决定 CPU/RAM 与 placement；
4. 提交前逐任务断言输出配置与 Sia 决策完全相同（`:515-517`）。

B5 不允许用 CPU/RAM 响应反向改变 GPU 配置。验证器检查全部 B5 日志中的 Sia intent 与提交配置一致。

## 5. B4 与 B5 不能混同

| 策略 | GPU 决策 | CPU/RAM 决策 | 官方 Synergy 核心 |
|---|---|---|---|
| B4 | 当轮原 Sia MIP，随后冻结 | 本项目有限列 effective-Goodput 优化，可修复 placement | 否 |
| B5 | 当轮原 Sia MIP，随后冻结 | TUNE：demand→fair→peer downgrade | 是，带明确适配 |
| SY | 预注册固定 GPU type/demand | 同上 TUNE | 是，带明确适配 |
| B3 | GPU/CPU/RAM/placement 同时选择 | 同一 MIP | 否；仅借用矩阵 |

## 6. 测试

新增 `tests/test_synergy_adapter.py`：

- `test_tune_atomically_downgrades_peer_to_fair_share`：需求冲突时 peer 降级且全量原子提交；
- `test_profile_grid_minimum_makes_oversized_gpu_config_wait`：6 CPU 节点不能容纳最小 1 CPU/GPU 的 8-GPU 任务，任务等待且不超配；
- `test_neutral_execution_has_exactly_one_multiplier`：neutral 下 `rho` 严格为 1。

全套结果：30 tests，0 failure，0 error，`Ran 30 tests in 8.488s`。日志：`research_runs/sia_synergy/cpu_bottleneck_smoke/all_tests.log`。另有 12 项运行产物验证全部通过，包括 SY 固定 demand、B5 GPU 冻结、checkpoint 无训练推进和逐节点容量审计。

## 7. 适配可信度判定

`SYNERGY_TUNE_CORE_ADAPTED_WITH_DOCUMENTED_BOUNDARIES`

已忠实保留 TUNE 的核心资源竞争处理顺序，但异构 GPU 映射、Sia placement 表、最终等待语义和统一执行模型是适配层。SY 的较差 JCT 主要反映固定 GPU demand/type 与本两作业 workload 的不匹配，不能据此宣称官方 Synergy 弱于 B3。
