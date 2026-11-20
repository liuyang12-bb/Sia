# Sia + Synergy 多资源适配设计与实施门槛

- 日期：2026-10-23
- 状态：**设计完成，未进入 Sia 实现**
- 原因：画像覆盖、节点容量和同基准 Goodput 归一化均未通过；继续实现会把缺失数据伪装成算法结果。

## 1. 设计原则

本设计严格分开三层：

1. **Sia 原有 GPU 决策**：GPU 类型、节点数、replicas、batch/Goodput MIP、reallocation cost 和 checkpoint 约束保持原义。
2. **辅助资源性能模型**：只有经过同应用、同 GPU 类型、同合法配置验证的 CPU×RAM response surface 才能改变训练进度。
3. **多资源 placement**：在既定 GPU 类型/数量上检查每节点 GPU/CPU/RAM 可行性；TUNE 只调 CPU/RAM 和 node mapping，不凭空改变 GPU demand。

Synergy 原论文固定 GPU demand；以下 S2/S3 是“每轮把 Sia 选定的 GPU config 当作该轮固定 demand”的异构组合适配，不是论文原算法完整复现。

## 2. 需要新增但本轮未创建的状态

只有数据门槛通过后，实验分支才应增加：

- `NodeInfo.resources`：`nvidia.com/gpu`、`cpu_cores`、`memory_gb`；CPU 单位必须明确是 physical core 或 vCPU，不能混用。
- job auxiliary profile key：`(application, gpu_type, num_nodes, num_replicas, local/global batch)`。
- baseline point：生成 Sia 原 throughput profile 时的 `baseline_cpu`、`baseline_memory`、cache/storage 条件。
- 每轮 job demand/allocation：总 CPU、总 RAM，以及按节点拆分的 allocation map。
- 日志：per-node capacity/usage、job CPU/RAM demand 与 allocation、profile key、归一化倍率、fallback 原因、peer downgrade、placement/restart 变化。
- 单独的 profile provenance metadata：来源 commit、机器、GPU、CPU、RAM、数据集、loader/cache、batch、测量单位。

不能把这些状态塞进现有 `used_gpu_seconds`、age 或 Goodput 字段；它们语义不同。

## 3. S1：Sia + GPU-proportional CPU/Memory

### 3.1 定义

对 GPU 类型 `t` 的每节点容量 `(G_t,C_t,M_t)`，每个 job 在该节点占 `g` 个 GPU 时获得：

```text
cpu_alloc = g * C_t / G_t
mem_alloc = g * M_t / G_t
```

多节点 job 在各节点按本节点 GPU 数分别计算。所有 job 的该节点 CPU/RAM 和不得超过 `C_t/M_t`；在相同 GPU placement 下，这一约束按构造应可行。

### 3.2 性能语义

S1 是后续同模型基线，归一化倍率必须恒为 1：

```text
effective_goodput_S1 = sia_goodput
```

不能把 Synergy 的 fair-share iteration time 替换 Sia Goodput。S1 只增加资源 accounting 和约束；在 CPU/RAM 不造成额外限制时，其逐轮 GPU decision、Goodput、progress、restart 与原始 Sia 应完全一致。

### 3.3 当前阻断

AWS 节点可从官方规格得到 CPU/RAM，DGX-A100 RAM 配置不唯一，RTX host 完全未知。若任意补成统一 CPU:GPU 比，S1 就是 synthetic 而非原 Saturn 集群；本轮没有授权/依据选择这种 synthetic cluster。

## 4. S2：Sia + Resource-Sensitive Allocation

### 4.1 来自 Synergy 的部分

- 使用离散 CPU×RAM response surface；
- 找到达到最大/饱和 throughput 的最小合理需求；
- 保证任何 job 不低于 GPU-proportional baseline；
- CPU/RAM 可每轮重新分配。

### 4.2 Sia 异构适配

对 Sia 已选择的 GPU config `k=(t,nodes,replicas,batch)`，画像必须是：

```text
R[j,t,k,c,m] = 在同一 Sia workload 与 GPU config 下，
               CPU=c、RAM=m 时相对训练 throughput
```

进度倍率定义为：

```text
rho = R[j,t,k,c_alloc,m_alloc] /
      R[j,t,k,c_baseline,m_baseline]
effective_goodput = sia_goodput(j,t,k) * rho
```

约束：

- baseline 点 `rho=1`；
- profile 外不外推，选择保守的最近可验证点或直接回退 `rho=1`，并写日志；
- 无 profile 的 job 必须 `rho=1`，不能用应用类别平均值；
- 若 CPU/RAM 超过 profile 饱和点，`rho` 保持饱和值，不能继续增长；
- 同一轮只能应用一次 `rho`；训练总样本/iteration/epoch 不变；
- 不读取未来到达、未来 runtime 或完成时间。

### 4.3 分配问题

在 GPU config 已固定的 job 集合上，可先做一个独立辅助资源优化：

```text
maximize  sum_j effective_goodput_j(c_j,m_j)
subject to per-node CPU/RAM capacity
           rho_j >= 1 for every running job
           one discrete (c,m) point per job/config
           multi-node per-GPU CPU/RAM ratios consistent
```

这相当于 Synergy-OPT 第一阶段的思想，但 Sia 版本必须按 GPU type/config 分别查画像。它不应直接进入 Sia 原 GPU MIP，否则 GPU 与辅助资源耦合后就不再满足“保持 GPU 配置选择不变”的 S2 定义。

### 4.4 当前阻断

当前 A 级 profile 覆盖为 0/480，且 baseline point 未知，无法计算可信 `rho`。所以没有实现 S2。

## 5. S3：Sia + Synergy-TUNE placement

### 5.1 候选算法

在 S2 已产生每个 running job 的 `(gpu_type, gpu_count, ideal_cpu, ideal_mem)` 后：

```text
输入：Sia 本轮选中的 GPU configs、旧 placement、每节点多资源容量、S2 demand
按 Sia 的本轮运行集合处理，不改变 job 的 GPU type/count
按 GPU demand、CPU demand、RAM demand 排序待 placement jobs

for job in order:
    优先复用可行的旧节点，避免无意义 restart
    否则选可完整容纳的最紧节点；多节点时选最小节点集合
    保证各节点 CPU/RAM 与其承担 GPU 数同比例
    若 ideal demand 不可行：
        把当前 job 降到该 GPU 类型的 proportional share，重试
    若仍不可行：
        只按 GPU 找可行节点集合
        在这些节点选择超过 proportional share 的 peers
        将足够 peers 降至 proportional share，再重新 placement
    若 proportional allocation 仍不可行：
        返回 INFEASIBLE 给上层；禁止 sys.exit，禁止事后超配

输出：联合可行的 GPU node IDs + per-node CPU/RAM map + fallback audit
```

### 5.2 与论文 TUNE 的区别

- GPU config 来自 Sia Goodput MIP，且可在下一轮改变；论文 GPU demand 生命周期固定。
- Sia 是 T4/A100/RTX 异构集群；论文主算法和 profiles 是同构 V100。
- 增加“优先复用旧 placement”是为了尊重 Sia checkpoint/restart 模型，不是论文原 TUNE 步骤，必须作为独立可关闭的适配开关。
- Sia 若改变 node IDs，即使 GPU 类型/数量相同也可能触发 reallocation；因此 placement 稳定性必须纳入测试和日志，不能把 TUNE 的 peer 调整视为零成本。
- 官方源码 `_tune()` 末端 `sys.exit(1)` 不适合嵌入 Sia；组合策略必须返回不可行状态并在同一轮重新求解/回退 S1。

### 5.3 可行性与 GPU 空闲保证

若每种节点的 S1 proportional share 定义完整，且 Sia 给出的 GPU node placement 本身满足 GPU 容量，则“所有 job 回退 S1”应构成联合可行解。这是 S3 不因辅助资源让 GPU 闲置的必要前提。

但该证明依赖：

- 所有节点的 CPU/RAM 容量已知；
- Sia GPU placement 可按节点表达；
- 每个 job 在各节点上的 proportional allocation 可切分；
- 资源单位和舍入规则不会造成总和超过容量。

当前 RTX/DGX 容量与单位未闭合，因此不能声称这一不变量已经成立。

## 6. 建议插入位置

如果未来数据门槛通过，建议独立策略文件而不是改写原 `SiaPolicy`：

1. `Cluster.step()` 推进上一轮后，仍调用 Sia GPU optimizer。
2. 在 Sia 产生 GPU type/count/config、执行最终 node placement之前，调用 auxiliary allocator。
3. auxiliary allocator 读取当轮 job state 与 GPU decision，输出联合 node placement 和 CPU/RAM map。
4. 只有完整联合解可行时提交；失败则使用 S1 proportional map 重新 placement，而不是提交后撤销。
5. `Job.step()` 训练阶段只通过已验证 `rho` 调整当轮 productive progress；checkpoint/restore 时间、总工作量和 GPU-seconds 语义不变。
6. `--policy=sia` 完全绕过该层；新入口才创建多资源状态。

具体候选文件（本轮均未修改）：

- 新增 `sia-simulator/sia_synergy.py`：S1/S2 resource allocator 与 S3 TUNE placement；
- `multi_simulator.py`：独立 `--policy=sia_synergy` 入口和显式 profile/capacity 文件参数；
- `cluster.py`：只在新策略路径记录 per-node CPU/RAM，并原子提交联合 allocation；
- `job.py`：只在训练推进阶段读取当轮经过验证的 `rho`；
- `tests/test_sia_synergy.py`：独立正确性测试。

## 7. 必须先通过的测试

| 测试 | 通过标准 |
|---|---|
| CPU 单调性 | 在同一已验证 profile 切片上，查询值与实测矩阵一致；不擅自强制原数据没有的单调性。 |
| RAM 响应 | MinIO/存储条件相同，离散点与 profile 一致；否则测试必须失败。 |
| baseline 对齐 | 每个 `(app,gpu_type,config)` 的 baseline `rho` 精确为 1。 |
| 无重复收益 | 充分资源只得到相对 baseline 的一次倍率，不再乘 profile 的绝对最大 speedup。 |
| 异构隔离 | 不同 GPU type 必须命中各自 profile；缺失时显式回退 1 并报警。 |
| 无画像应用 | YOLO/BERT 等不得静默继承 ResNet/类别平均曲线。 |
| 每节点容量 | 每轮 GPU/CPU/RAM 三维总量均不超限。 |
| 多节点一致性 | 每个 GPU process 的 CPU/GPU、RAM/GPU 比符合已定义模型。 |
| 原 GPU 约束 | Sia 合法配置、容量、checkpoint 保持约束不变。 |
| 无未来信息 | profile lookup 与 TUNE 只读取当前及历史状态。 |
| S1 退化 | 在 proportional baseline 下，S1 与原始 Sia 的逐轮 GPU allocation、progress、JCT 和 restart 一致。 |
| TUNE fallback | ideal 不可行时先降当前 job，再降 peers；最终 S1 可行，且不超配。 |
| restart accounting | 同配置但 node mapping 变化按 Sia 原逻辑计入，不被辅助 allocator 隐藏。 |

## 8. 本轮实施判定

| 门槛 | 状态 | 证据 |
|---|---|---|
| 正确识别 Synergy 原算法 | 通过 | 论文公式 (1)–(7)、TUNE fallback 和源码路径已核对。 |
| 原生 simulator 可运行 | 有条件通过 | 5-job CPU-only TUNE smoke 5/5 完成；需明确 Windows import 兼容层。 |
| Sia 模型画像覆盖 | 失败 | A 级 0/480；YOLO 缺失，BERT 未评测，其他存在数据/硬件差异。 |
| 节点 CPU/RAM 容量 | 失败 | RTX 未知，DGX RAM 配置不唯一。 |
| Goodput 同基准归一化 | 失败 | 产生 Sia throughput profile 的 CPU/RAM/cache baseline 未记录。 |
| S1 可验证 | 未开始/被上游阻断 | 无完整容量与性能 baseline。 |

最终决定：**STOP / INVALID FOR SIA PERFORMANCE CLAIMS**。本轮不创建 `--policy=sia_synergy`，不修改 `sia-synergy-experiments/`，不运行 20-job Sia smoke，更不运行三条 160-job traces。下一步只有在补齐真实资源配置与同平台 profiles 后才值得恢复；若只想研究算法行为，可以另立明确标为 synthetic 的研究任务，但不能与 Saturn 原始性能结果混称。
