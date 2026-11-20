# Sia + Synergy 完整模拟器接入 Smoke 报告

- 日期：2026-10-24
- 工作区：`D:\科研启航\Sia`
- 实验仓库：`sia-joint-experiments`
- 分支：`exp/sia-synergy-joint-integration`
- 基线 HEAD：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 判定：**END_TO_END_SMOKE_VALID — NOT YET APPROVED FOR 20-JOB PERFORMANCE CLAIMS**
- 数据性质：**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 1. 结论

现有联合调度原型已经接入 Sia 的完整时间推进路径。B1、B4、B3 均完成了两个实际 Sia workload 作业，而不是只求解单轮分配；JCT 来自 `Job.step()` 产生的完成事件。训练进度、checkpoint/restore、restart、等待、epoch 和完成判定继续使用原 Sia 实现。

本轮最重要的正确性结果是：

1. 原始 `multi_simulator.py --policy sia` 在官方仓库与实验仓库输出逐字节相同。
2. neutral 场景中 B1、B4 与原始 Sia 的 JCT 完全相同。
3. 所有联合运行的逐节点 GPU/CPU/RAM 容量审计均通过。
4. 每个训练区间满足 `effective_goodput = raw_goodput × rho`，没有重复乘倍率。
5. 完整 checkpoint/restore 区间没有训练进度。
6. B4 每轮均保持其当轮原 Sia GPU 配置，未发现偷偷改变 GPU 类型或数量。
7. B3 不调用原 Sia MIP 先固定唯一配置；它从 Sia 合法配置和 Goodput 接口生成多个候选后执行联合 MIP。

敏感响应场景中 B3 平均 JCT 比 B1 低 33.88%，但最大 `rho=4.2981`。这只是合成机制证据，不是可信的真实硬件收益估计。

## 2. 保护范围与 Git 状态

开始前状态：

- `official-artifacts/`：`main...origin/main`，干净。
- 原型分支：`exp/sia-synergy-joint-prototype`，只有任务 12 的未提交新增文件。
- 本轮新建分支：`exp/sia-synergy-joint-integration`。
- 未修改 `official-artifacts/`、`research_sources/Synergy/` 或历史实验仓库。

正式的 `sia.py`、`cluster.py` 和 `multi_simulator.py` 没有修改。唯一触及既有模拟器代码的改动是 `job.py` 的默认执行倍率；默认值严格为 1，因此原始路径保持不变。

## 3. 修改文件与函数

| 文件 | 修改/新增内容 |
|---|---|
| `sia-simulator/job.py:110,491` | 增加默认 `goodput_multiplier=1.0`；只在原始 Goodput 和 calibration 之后相乘。 |
| `joint_prototype/joint_integration.py:37` | `JointSiaPolicy`；实现 B1/B4/B3 接入。 |
| `joint_integration.py:97` | 从 Sia 配置表及 `_compute_goodputs_v2()` 生成 direct/projected/fallback 候选。 |
| `joint_integration.py:210` | 将原 Sia 分配转换为 B1 proportional 联合执行记录。 |
| `joint_integration.py:266` | B3 的 Sia 公平性、未分配惩罚及 restart factor 列目标。 |
| `joint_integration.py:358` | 三种策略的实际每轮入口。 |
| `joint_integration.py:387` | `JointCluster`；原子提交倍率并记录执行/决策日志。 |
| `joint_profile.py:200-252` | neutral response 开关；neutral 时严格 `rho=1`。 |
| `joint_scheduler.py:143-349` | 为端到端运行增加可关闭的穷举复核；原 2-job 单元场景仍默认执行穷举。 |
| `run_joint_integration_smoke.py:55` | 独立 smoke runner；原始 `multi_simulator.py` 不增加新策略。 |
| `validate_joint_integration.py:17` | 持久化日志的 15 项自动审计。 |
| `summarize_joint_integration.py:13` | JCT、资源小时、配置序列和活跃轮次摘要。 |
| `tests/test_joint_integration.py:17` | 5 项新的集成语义测试。 |

## 4. 实际 Sia Goodput 接入路径

### 4.1 调度候选

调用链为：

```text
Cluster.get_sia_multi_job_info(job)
  -> Job.get_speedup_fn(cluster)
  -> JointSiaPolicy._raw_candidates(job_info)
  -> SiaPolicy.configs[cluster]
  -> SiaPolicy._compute_goodputs_v2(job_info, cluster, nnodes, ngpus)
  -> per-job Sia normalization
  -> GpuConfig.sia_goodput
```

合法性仍使用原 Sia 的：

- `min_replicas`、`max_replicas`；
- PMP/MP `scale_unit`；
- `CONFIGS_4GPU` / `CONFIGS_8GPU`；
- 各 GPU 类型的节点数和每节点 GPU 数；
- checkpoint 中当前配置和节点锁。

Goodput 来源被逐候选记录为：

- `direct_profile`：目标 GPU 类型已有 `Job.get_speedup_fn()`；
- `projected_profile`：启用 projection 且从另一 GPU 类型投影；
- `unprofiled_sia_fallback`：尚无可用 profile，采用原 Sia 的最小 scale-unit 启动语义。

本次 smoke 实际观察到 117 次 direct 和 24 次 fallback 选择。projected 分支通过单元测试验证，但本次配置 `project_throughputs=false`，因此端到端日志中没有 projected 记录。

### 4.2 训练执行

调度器使用的是当前 Sia profile 给出的候选 Goodput。真正的训练推进仍在原 `Job.step()` 中完成：

```text
checkpoint/restore delay
  -> Application.get_grad_stats()
  -> Application.get_throughput()
  -> 原始 gain / total_time
  -> calibration_factor
  -> goodput_multiplier (= rho)
  -> progress / epoch / completion_time
```

`rho` 不参与等待时间或 checkpoint/restore 倒计时。每轮日志分别保存：

- 原始平均 Goodput；
- `rho`；
- effective Goodput；
- progress 前后值；
- 实际训练秒数；
- checkpoint 延迟秒数；
- 完成时间。

## 5. B1、B4、B3 的实际差异

### B1

每轮直接调用 `SiaPolicy.optimize()`。GPU 类型、GPU 数量和原 Sia placement 均来自原调度路径；CPU/RAM 按所在节点每 GPU 容量比例分配。B1 的 Sia MIP 不读取 CPU/RAM response，但任务执行时使用统一 Oracle 的实际 `rho`。

### B4

每轮先调用与 B1 相同的 `SiaPolicy.optimize()`，随后冻结每个任务的 `(GPU type, num_nodes, num_gpus)`。联合列只能在这个固定配置内选择 CPU/RAM 和 placement。

若固定配置 MIP 丢失任何原 Sia 已分配任务，整轮回退到经过三资源审计的 B1 proportional map；不能提交部分解。本次没有触发回退，27 个已分配 job-round 的配置一致性检查全部通过。

### B3

B3 不调用 `SiaPolicy.optimize()`。它直接枚举 Sia 合法候选，为每个候选查询 CPU/RAM response 和 placement，一次 MIP 选择 GPU 类型、GPU 数、CPU、RAM 和节点。

目标继续使用 Sia 的 `p=-0.5`、`lambda_n=1.1` 和 config-change restart factor。不同之处是容量约束从 Sia 的 GPU 类型总量约束变为逐节点 GPU/CPU/RAM 约束，因此即使 neutral，联合解也不保证逐轮 placement 或离散完成秒完全相同。

## 6. 两任务输入与资源模型

输入是 `workloads/saturn/workload-1.csv` 的精确行提取，没有改变 arrival、application、batch 或目标工作量：

| Job | arrival | application | 原字段 num_replicas | batch size |
|---|---:|---|---:|---:|
| `bert-28` | 4931 | bert | 12 | 384 |
| `cifar10-29` | 4935 | cifar10 | 6 | 2048 |

输入 SHA-256：`7E7BEC84471123AA098212EC5CF8EF622EECAC7A6C23C956A86A2C1E403EE100`。

Smoke 集群为受控的小型异构集群：

| GPU 类型 | 节点 | GPU/节点 | synthetic CPU/节点 | synthetic RAM/节点 |
|---|---:|---:|---:|---:|
| aws | 1 | 4 | 8 cores | 200 GB |
| dgx-ext | 1 | 8 | 18 cores | 400 GB |
| rtx | 1 | 8 | 16 cores | 500 GB |

这些 CPU/RAM 数值不是 AWS、DGX 或 RTX 真实主机规格。

响应映射采用 fixed hash、seed 1301、候选模板 `res18_1.py` 与 `m5_1.py`。该预注册哈希恰好让两个 Job 都映射到 `res18_1.py`；没有在看到结果后换 seed。响应形状来自 Synergy 官方矩阵，但 GPU 间迁移、需求缩放和绝对校准均为 synthetic cross-hardware transfer。

## 7. 执行命令

单元测试：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
..\..\.venv-sia-baseline\Scripts\python.exe -m unittest tests.test_joint_prototype tests.test_joint_integration -v
```

最终七配置运行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
..\..\.venv-sia-baseline\Scripts\python.exe run_joint_integration_smoke.py `
  --policy all --response both `
  --synergy-models ..\..\research_sources\Synergy\simulator\models `
  --output ..\..\research_runs\sia_synergy\joint_integration_smoke
```

日志审计：

```powershell
..\..\.venv-sia-baseline\Scripts\python.exe validate_joint_integration.py `
  ..\..\research_runs\sia_synergy\joint_integration_smoke
```

官方回归分别在 `official-artifacts/sia-simulator` 和 `sia-joint-experiments/sia-simulator` 执行：

```powershell
..\..\.venv-sia-baseline\Scripts\python.exe multi_simulator.py <two-job-workload> `
  --policy sia --oracle_num_nodes 1 --oracle_num_replicas 8 --output <log>
```

## 8. 每轮配置与训练事件

完整逐轮数据保存在各运行 JSON 的 `joint_round_logs` 中。每条记录明确分为本轮 `execution` 和下轮 `decision_for_next_interval`。空闲到达前的轮次也被保留。

敏感场景的配置 run-length 摘要如下；开头的 `UNALLOCATED×1` 是 4920–4980 秒间先推进时间、再进行首次调度，末尾的未分配轮次发生在任务已完成后。

| 策略 | Job | 配置序列（调度轮数） |
|---|---|---|
| B1 | bert-28 | unallocated×1 → rtx:1:1×3 → dgx-ext:1:4×8 → dgx-ext:1:8×5 |
| B1 | cifar10-29 | unallocated×1 → rtx:1:1×1 → dgx-ext:1:8×2 → dgx-ext:1:4×8 → completed×5 |
| B4 | bert-28 | 与 B1 完全相同 |
| B4 | cifar10-29 | 与 B1 完全相同 |
| B3 | bert-28 | unallocated×1 → aws:1:1×3 → dgx-ext:1:1×3 → dgx-ext:1:2×6 |
| B3 | cifar10-29 | unallocated×1 → dgx-ext:1:1×2 → dgx-ext:1:2×4 → completed×6 |

完成事件：

| 场景/策略 | bert-28 completion / JCT | cifar10-29 completion / JCT |
|---|---:|---:|
| sensitive B1 | 5835 / 904 s | 5616 / 681 s |
| sensitive B4 | 5832 / 901 s | 5609 / 674 s |
| sensitive B3 | 5612 / 681 s | 5302 / 367 s |
| neutral Sia/B1/B4 | 5714 / 783 s | 5499 / 564 s |
| neutral B3 | 5714 / 783 s | 5500 / 565 s |

进度审计确认所有 `progress_delta >= 0`；无训练秒数的区间没有进度；完整 60 秒 checkpoint 延迟区间没有进度。JCT 对每个任务均等于 `completion_time - submission_time`。

## 9. JCT 与资源使用

### 9.1 Sensitive synthetic response

| 策略 | 平均 JCT | 相对 B1 | GPU-hours | CPU-core-hours | RAM-GB-hours | rho 范围 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 792.5 s | — | 1.8067 | 4.0483 | 91.1667 | 0.7200–0.7583 |
| B4 | 787.5 s | -0.63% | 1.7922 | 2.2589 | 90.4444 | 0.7397–2.1833 |
| B3 | 524.0 s | -33.88% | 0.3967 | 2.9156 | 45.4167 | 2.1833–4.2981 |

B3 的收益同时来自较小 GPU 配置、较少 restart 和非常强的合成 response。由于 `rho` 上界过高，不能将 -33.88% 写成真实 Saturn 或真实集群收益。

### 9.2 Neutral response

| 策略 | bert JCT | cifar JCT | 平均 JCT | rho |
|---|---:|---:|---:|---:|
| 原始 Sia | 783 | 564 | 673.5 | 不适用 |
| B1 | 783 | 564 | 673.5 | 1 |
| B4 | 783 | 564 | 673.5 | 1 |
| B3 | 783 | 565 | 674.0 | 1 |

B1/B4 与原始 Sia 完全退化一致。B3 的 1 秒差异来自逐节点可行域、不同求解器和完成时间取整；没有任何辅助资源倍率收益。

## 10. 求解与重配置

| 策略/场景 | 联合求解总时间 | 联合求解 P99 | bert restarts | cifar restarts |
|---|---:|---:|---:|---:|
| B4 sensitive | 0.0193 s | 0.00228 s | 2 | 2 |
| B3 sensitive | 0.0446 s | 0.01353 s | 2 | 1 |
| B4 neutral | 0.0093 s | 0.00078 s | 2 | 2 |
| B3 neutral | 0.0189 s | 0.00177 s | 2 | 2 |

B1 的表中 `solve_seconds=0` 仅表示联合层没有再次计时；原 Sia CBC 求解仍在 `SiaPolicy.optimize()` 中执行。重配置延迟没有缩短，使用原应用的 120 秒 BERT、50 秒 CIFAR10 语义。

## 11. 全部测试与自动审计

最终单元测试：**19/19 通过，0.225 秒**。

新增覆盖：

1. checkpoint delay 不受倍率影响；
2. 倍率只乘一次训练进度；
3. 未分配任务只等待；
4. neutral Oracle 在任意资源点严格为 1；
5. direct 与 projected 来源保持可区分。

持久化输出审计：**15/15 通过**，包括完成事件、容量、进度单调性、单次 Goodput 倍率、checkpoint、neutral、B4 固定配置、B3 多候选、profile 来源和官方回归。

## 12. 原始 Sia 回归

同一 workload、同一 CLI、同一环境分别运行未修改的官方仓库和实验仓库：

```text
official SHA-256     24D8FC9AFF0412A79C222E7090952156E4877FE2E7FEB0DED3F4C41A8905FF35
experimental SHA-256 24D8FC9AFF0412A79C222E7090952156E4877FE2E7FEB0DED3F4C41A8905FF35
```

两个日志逐字节一致，JCT 均为 `bert-28=547 s`、`cifar10-29=410 s`。该回归使用原 CLI 默认 `p=0.5` 和官方默认集群；联合 smoke 使用预注册的 `p=-0.5` 与三节点小集群，两者不能混作同一性能对照。

## 13. 运行产物

目录：`research_runs/sia_synergy/joint_integration_smoke/`

- `run_manifest.json`：7 个最终配置、输入 hash 和运行时间；
- `b1/b4/b3.sensitive.json`：合成资源敏感端到端日志；
- `b1/b4/b3.neutral.json`：neutral 退化日志；
- `sia.control.json`：完全绕开联合策略的原始控制路径；
- `comparison_summary.json`：指标与活跃轮次摘要；
- `validation_summary.json`：15 项审计；
- `official_sia_regression.log`、`experimental_sia_regression.log`：原始回归。

## 14. 未解决问题与 20-job 决策

本轮证明了接入和状态推进正确，但**尚不批准直接开展 20-job 性能对照或性能声明**：

1. 最大 synthetic `rho=4.2981`，需要先冻结合理的 clip/敏感性范围，或获取可校准 profile。
2. 两个 Job 的 fixed hash 恰好映射到同一模板；下一阶段应预注册多 seed 或 category-mix，而不能事后只选有利映射。
3. projected path 只做了单元测试，未做端到端 projected smoke。
4. 当前只覆盖 adaptive DP Job；rigid、scale-gpus、PMP、多节点配置仍需集成测试。
5. B4 本次没有触发 placement 修复或 fallback；这些机制仅由原 2-job 优化器测试覆盖。
6. CPU/RAM 动态重分配未额外建模数据加载或 NUMA/内存迁移开销。
7. B3 使用 SciPy/HiGHS 的逐节点 MIP，而原 Sia 使用 CVXPY/CBC 类型总量 MIP；neutral 的 1 秒差异说明不能要求无条件逐轮相同。
8. Artifact 的 `Job.start_time` 没有被赋值，报告使用真实 `completion_time - submission_time` 计算 JCT；若后续需要 queueing/start-time 分解，应先补独立观测字段，不能改变原语义。

建议下一步仅做“集成覆盖扩展”：增加 projected、rigid/PMP、多节点与 B4 fallback 的小型测试，并预注册 response clip 和映射 manifest。通过后再批准固定 20-job **smoke**；真实性能结论仍需真实或可辩护的 CPU/RAM profile。
