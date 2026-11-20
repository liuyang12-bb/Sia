# Sia + Synergy 受控合成 S1/S2 设计

- 日期：2026-10-23（Asia/Shanghai）
- 分支：`sia-synergy-experiments/` / `exp/sia-synergy`
- 基线 commit：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 研究性质：**SYNTHETIC MECHANISM VALIDATION**
- 结论边界：**SYNTHETIC RESOURCE MODEL — NOT REAL SATURN PERFORMANCE**

## 1. 范围与非目标

本实验只研究：在 Sia 原 GPU 类型、GPU 数量、batch/Goodput MIP、GPU placement 和 checkpoint/restart 语义不变的前提下，增加一个独立 CPU/RAM 层能否产生可解释行为。

- S1：GPU-proportional CPU/RAM accounting，进度倍率恒为 1。
- S2：固定 Sia 当轮 GPU placement 后，离散选择每个运行作业的 CPU/RAM 比例，最大化当轮总 synthetic effective goodput。
- 不实现 S3、多资源 TUNE placement、完整 Synergy-OPT、真实 profiling、LAS/AFS/Lazer 组合或参数搜索。

S2 只借用 Synergy“辅助资源响应、GPU-proportional 下界、独立资源优化”的思想。Synergy 原论文固定 GPU demand、使用同构 V100 画像且还有第二阶段 placement；本实现的 GPU config 每轮来自 Sia，且不做第二阶段 placement。因此名称为本项目的 **synthetic resource-sensitive adaptation**，不是 Synergy-OPT 的直接移植。

## 2. 基线保护

任务开始时 `sia-synergy-experiments/` 位于 `exp/sia-synergy`，HEAD 为 `aaf7a0d`，工作区干净。实现只写入：

- `sia-synergy-experiments/sia-simulator/`；
- `research_notes/synergy/`；
- `research_runs/sia_synergy/synthetic/`。

未写入 `official-artifacts/`、`sia-experiments/`、`sia-afs-experiments/`、`sia-lazer-observe/` 或 `research_sources/Synergy/`。原 `--policy=sia` 不创建 auxiliary allocator；`Job.step()` 的新增参数默认是 1，故历史入口绕过全部合成逻辑。

## 3. 显式合成容量

唯一配置文件是 `sia-synergy-experiments/sia-simulator/synthetic_resource_model.json`，其中必须且已经写明：

```json
"resource_model_type": "synthetic"
```

CPU 单位统一为 `synthetic_cpu_core`。它是模拟标量，不声称等于 physical core 或 vCPU。RAM 单位为 GB。容量采用 Synergy 默认模拟器“每 8 GPU 配 24 CPU、500 GB RAM”的比例，即每 GPU 3 synthetic CPU cores 和 62.5 GB RAM；这是预注册合成假设，不是真实 AWS、DGX 或 RTX 主机规格。

| GPU 集群 | 节点数 | GPU/节点 | synthetic CPU/节点 | synthetic RAM/节点 |
|---|---:|---:|---:|---:|
| `aws` | 6 | 4 | 12 | 250 GB |
| `dgx-ext` | 2 | 8 | 24 | 500 GB |
| `rtx` | 3 | 8 | 24 | 500 GB |

节点数、类型和 GPU 容量保持 Sia Artifact 原值。allocator 初始化时逐项比对配置和 simulator；任一集群名、节点数、GPU/节点或 proportional capacity 不一致即拒绝启动。

## 4. 合成响应模型

### 4.1 作业 cohort

画像不使用 `CIFAR10`、`BERT` 等应用标签，也不读取未来到达、剩余时长或完成时间。只使用作业名末尾已有整数 `i`：

```text
cohort = i mod 3
0 -> cpu_sensitive
1 -> memory_sensitive
2 -> insensitive
```

若作业名没有末尾整数，使用作业名 SHA-256 的前四字节模 3；这仍只依赖当前静态标识。该标签只是合成实验分组，不表示 Saturn 作业具有对应真实性能曲线。

### 4.2 baseline、离散点和公式

GPU-proportional baseline：

```text
cpu_ratio = 1
ram_ratio = 1
rho = 1
```

S2 候选比例为 `1.0, 1.25, 1.5`。CPU-sensitive 作业只增加 CPU，memory-sensitive 作业只增加 RAM，不敏感作业只有 baseline 点。

CPU response：

```text
rho_cpu(c) = 1 + 0.30 * clip((c - 1) / (1.5 - 1), 0, 1)
```

- baseline：`rho_cpu(1)=1`；
- 饱和点：`c=1.5`；
- 最大倍率：1.30；
- RAM ratio 固定为 1。

Memory response：

```text
rho_mem(m) = 1 + 0.25 * clip((m - 1) / (1.5 - 1), 0, 1)
```

- baseline：`rho_mem(1)=1`；
- 饱和点：`m=1.5`；
- 最大倍率：1.25；
- CPU ratio 固定为 1。

这些曲线是本项目定义的分段线性合成曲线，没有复用 Synergy V100 profile 的绝对 iteration time 或矩阵值，因此也不是 borrowed synthetic response。

### 4.3 四种场景

| 场景 | cohort 0 | cohort 1 | cohort 2 |
|---|---|---|---|
| neutral | insensitive | insensitive | insensitive |
| cpu_sensitive | CPU response | insensitive | insensitive |
| memory_sensitive | insensitive | memory response | insensitive |
| mixed | CPU response | memory response | insensitive |

四个场景使用同一 GPU 集群、容量、Saturn 到达时间、训练总工作量、Sia Goodput 和推进规则；只改变哪个合成 cohort 可查询哪条响应曲线。

## 5. S1 算法

Sia 提交 GPU placement 后，对每个作业在每个节点实际占用的 GPU 数 `g` 计算：

```text
cpu_alloc = 3 * g
ram_alloc = 62.5 * g
effective_goodput = sia_goodput
```

多节点作业逐节点使用同一个 per-GPU 比率。因为一台节点所有作业的 GPU 总数不超过 `G`，CPU/RAM 总数按构造分别不超过 `3G` 和 `62.5G`。实现仍在每轮显式检查 GPU、CPU 和 RAM 三类容量，不依赖这一证明静默通过。

S1 在所有四个响应场景的倍率都严格为 1；场景标签不会改变进度。

## 6. S2 算法

S2 在 `Cluster.optimize_policy()` 完成原 Sia GPU MIP、node placement、`migrate()`/`reallocate()` 后执行。对当轮已获得 GPU 的作业 `j` 和合法候选 `q` 建二进制变量 `x[j,q]`：

```text
maximize sum(j,q) sia_goodput[j] * rho[j,q] * x[j,q]

subject to
  sum(q) x[j,q] = 1                         for each running job
  sum(j,q) cpu[j,node,q] * x[j,q] <= C[node]
  sum(j,q) ram[j,node,q] * x[j,q] <= M[node]
  rho[j,q] >= 1
  x[j,q] in {0,1}
```

GPU type、GPU 数和 node IDs 是常量，不进入此 MILP 的决策空间。多节点作业一个候选覆盖所有节点，所以每 GPU CPU/RAM 比天然一致。目标中的 `sia_goodput` 由当前 `Job` 配置按 Sia 原公式只读计算；训练推进时再执行一次 `effective_goodput = sia_goodput * rho`，不会把 `rho` 乘两次。

求解器为 SciPy 1.10.1 的 HiGHS MILP，零 relative gap，单次 30 秒上限。GPU-proportional baseline 对所有运行作业总是候选，因此容量不足时优化解回到 ratio 1；若求解器异常或未返回整数解，allocator 明确整体回退 S1 并记录 `fallback`，不提交超配结果。

## 7. 调度时序与状态语义

现有 Sia 一轮是：先用上一轮 allocation 推进 60 秒，再优化下一轮 allocation。合成层遵循同一时序：

```text
account previous auxiliary lease
Job.step(seconds, previous rho)
advance current_time
Sia GPU optimize and placement
original migrate/reallocate and restart accounting
S1/S2 auxiliary allocation on fixed placement
log the committed GPU and auxiliary state
```

`rho` 只乘 productive training goodput。checkpoint/restore delay 在计算 goodput 前消耗，因此不被加速；`used_gpu_seconds`、`wasted_gpu_seconds`、restart count 和 migration count 保持原语义。辅助资源比例变化本轮假设零迁移/数据加载开销，不触发 GPU restart；配置显式写明 `auxiliary_reallocation_overhead_modelled=false`。这是机制实验限制，不能外推为真实集群零成本。

## 8. 指标定义

- JCT：`completion_time - submission_time`；P95/P99 使用 NumPy 线性百分位。
- Makespan：最后完成时间减最早到达时间；另存 simulator end time。
- GPU-hours：`(used_gpu_seconds + wasted_gpu_seconds)/3600`；同时分开 productive/restart GPU-hours。
- CPU-core-hours、RAM-GB-hours：每轮已提交 auxiliary lease 的逐节点 allocation × 轮长积分。
- 每节点利用率：资源积分除以节点容量与 simulator end time。
- restart/migration：原 `Job` 计数。
- S2 auxiliary reallocation：同一存活作业的离散 CPU/RAM choice 改变次数；另存因 Sia GPU placement 变化引起的完整 resource-map change。
- Sia GPU decision change：`allocations[job]` 相对上一轮变化次数，包括首次分配。
- auxiliary solve time：只统计 S2 MILP，输出总时间和 P99。
- 容量违规：每轮所有节点均审计；发现一次即抛错，不能带病生成结果。

## 9. 文件与函数

| 文件 | 作用 |
|---|---|
| `synthetic_resource_model.json` | 合成容量、单位、曲线、场景、来源与开销声明。 |
| `synergy_synthetic.py:35-341` / `SyntheticAuxAllocator` | S1、S2 MILP、容量审计、fallback、重分配与求解时间统计；S2 求解在 174 行，提交在 282 行。 |
| `job.py:421` / `estimate_current_goodput()` | 当前 Sia Goodput 的无状态只读估计。 |
| `job.py:443` / `step(..., auxiliary_multiplier=1)` | 在 calibration 后只乘一次合成倍率。 |
| `cluster.py:188` / `optimize_policy()` | 原 GPU placement 提交后调用 auxiliary allocator。 |
| `cluster.py:179` / `_account_auxiliary_resources()` | CPU/RAM 时间积分。 |
| `cluster.py:564` / `get_synthetic_metrics()` | 生成绝对指标、逐节点利用率和任务级结果。 |
| `multi_simulator.py:49-63,175-183` | 新增显式 `--synthetic_*` 参数；默认 Sia 路径不启用。 |
| `tests/test_synergy_synthetic.py` | 15 项资源模型与状态语义测试。 |
| `run_synergy_synthetic_suite.py` | 2 GiB 门槛检查和 8 配置严格顺序运行。 |
| `analyze_synergy_synthetic.py` | neutral 逐轮一致性、容量、任务级差异和总表。 |

## 10. 预注册运行

固定 20-job 输入是官方 `workload-1.csv` 的前 20 个到达，只复用 arrival、application、GPU request 和 batch 字段。3-job 输入为同一序列前三项。二者位于 `research_runs/sia_synergy/synthetic/inputs/`，没有修改官方 workload。

完整 suite 顺序为：

```text
neutral/S1 -> neutral/S2
cpu_sensitive/S1 -> cpu_sensitive/S2
memory_sensitive/S1 -> memory_sensitive/S2
mixed/S1 -> mixed/S2
```

每个命令均固定：`--policy=sia --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 --interval=60 --project_throughputs --share_max_replicas`。不运行 160-job trace。

## 11. 判定规则

- `MECHANISM_VALID`：neutral 逐轮严格相同、容量与状态测试全部通过，且敏感场景出现与资源重分配一致的行为。
- `MIXED`：正确性通过但某些敏感场景有明显总体或任务级退化。
- `NO_BENEFIT`：正确性通过但四个 20-job 场景主要指标没有收益。
- `INVALID`：baseline 不为 1、neutral 不退化、重复倍率、容量超限、时间非单调或 restart 语义改变。

如果 20-job suite 因宿主机可用内存低于 2 GiB 未运行，则不对完整 suite 强行套用上述四类；只报告已完成的 manual mechanism gate，并标记 `NOT_RUN_LOW_MEMORY`。

## 12. 已知限制

1. CPU 和 RAM 容量、敏感度 cohort 与响应曲线全部是合成假设。
2. S2 不改变 GPU MIP 目标，但更快完成会改变后续 runnable set，因此敏感场景后期 GPU 决策轨迹可间接分叉；neutral 必须完全一致。
3. 辅助资源重分配开销未建模，不能将结果视为真实上线收益。
4. 没有 storage/network/cache 维度，也没有 Synergy TUNE placement。
5. Saturn 名称和应用只充当 workload timing/GPU 输入，不是 CPU/RAM 测量证据。
