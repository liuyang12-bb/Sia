# Sia + Synergy 联合调度器设计

- 日期：2026-10-23
- 阶段：算法级设计，**不实现、不运行完整实验**
- 推荐路线：方案 A 作为可实现主算法；方案 B 作为 20-job 小规模精确对照/oracle，达到求解预算后再决定是否工程化。
- 结果边界：使用跨硬件合成画像时，只能报告 `SYNTHETIC JOINT-RESOURCE EVALUATION — NOT REAL SATURN PERFORMANCE`。

## 1. 设计目标与不变量

目标是在保留 Sia 异构/弹性 GPU 核心的同时，让 CPU、RAM 性能响应和逐节点容量参与最终 allocation。以下不变量高于性能目标：

1. `--policy=sia` 的历史行为完全不变；新策略使用独立入口。
2. 合法 GPU 配置、PMP scale unit、checkpoint lock、restart factor、真实 rescale delay沿用 Sia。
3. 相同作业、到达、训练总工作量、GPU profile 和节点资源供所有对照共享。
4. GPU-proportional 点的辅助倍率严格为 1；倍率只乘一次。
5. 每轮每节点 GPU/CPU/RAM 均不超限，多节点 job 每 GPU 的辅助资源比例一致。
6. 只使用当前/历史状态和预注册 profile；不读未来到达、真实剩余运行时间或完成时间。
7. 求解失败不提交部分解：回退到经验证可行的 proportional 列或上一轮合法解，并记录原因。

## 2. 统一符号与性能模型

### 2.1 集合与候选

- `J_t`：调度轮 `t` 已到达、未完成作业。
- `V`：物理节点；节点 `v` 容量为 `(G_v,C_v,M_v)`，GPU 类型为 `type(v)`。
- `K_j`：Sia 为 `j` 生成的合法 GPU 候选；`k=(type,nodes,replicas,batch,scale_unit)`。
- `Q_jk`：候选 `k` 下的离散 CPU/RAM 点；`q=(cpu_per_gpu,ram_gb_per_gpu)`，必含 proportional 点 `q0(k)`。
- `H_jkq`：可实现的逐节点 placement pattern；记录每节点 GPU/CPU/RAM 使用量。

### 2.2 不重复计算的联合 Goodput

Sia 原始查询返回 `g_gpu(j,k)`。辅助资源画像返回同配置相对 baseline 的倍率：

```text
rho(j,k,q) = R(j,k,q) / R(j,k,q0(k))
rho(j,k,q0(k)) = 1
g_eff(j,k,q) = g_gpu(j,k) * rho(j,k,q)
```

固定的 Sia 归一化分母定义为：

```text
g_ref(j) = min { g_gpu(j,k) > 0 | k in K_j }
G_joint(j,k,q) = scale_unit(j) * g_eff(j,k,q) / g_ref(j)
```

关键点是 `g_ref` 只由原始 Sia Goodput 得到，不能因某策略拥有更多辅助候选而变化。这样 `rho=1` 时联合矩阵与 Sia 原矩阵逐元素相同，策略间效用口径也一致。

### 2.3 重配置语义

GPU 类型、GPU 数或 Sia 配置索引变化时，继续使用 Artifact 的候选 restart factor：

```text
R_gpu(j,k) = 1                         若保持当前 Sia GPU config
           = Sia restart factor       否则
```

当前 Artifact 的 restart factor 只跟踪 GPU config，`Job` 状态只看 `(cluster, placement-count tuple)`；相同 tuple 下更换实际 node ID 不会被单独计费。为保持基线，初版仍采用该语义，但每次 node-ID remap 必须记录为 `unmodeled_node_remap`，并在 tie-break 中优先旧节点。获得实测后才增加独立 `R_node(j,h_old,h_new)`。

同样，没有可靠数据表明 cgroup CPU/RAM quota 调整或缓存变化为零开销。初版令 `R_aux=1`，但必须记录 auxiliary reallocation，并将“未建模”写入结果；获得实测后才增加独立 `R_aux(j,q_old,q_new)`。不得借用失效的 `lambda_a` 充当上述任一开销。

### 2.4 公平目标

候选列 `a=(j,k,q,h)` 的效用为：

```text
U_a = [R_gpu(j,k) * G_joint(j,k,q)]^p
```

保持 Artifact 的取幂、两位小数和 clip 行为；`p<0` 最小化，`p>=0` 最大化。未分配项仍使用 `lambda_n*(|J|-num_allocs)`。所有运行作业只允许 `rho>=1` 的 q；不把“每个 job 必须运行”变成硬约束，因为作业数可能超过 GPU 数。

## 3. 总体架构

```text
当前 Job/Node 状态
        |
        v
Sia GPU Candidate Builder -----> 原始 g_gpu、合法 K_j、restart/checkpoint 元数据
        |
        v
Joint Profile Oracle ----------> Q_jk、rho、baseline/provenance
        |
        v
Placement/Column Builder ------> 逐节点可行列 (j,k,q,h)
        |
        +-------------------------------+
        |                               |
        v                               v
方案 A：有限列 Master MIP        方案 B：紧凑联合 MIP
        |                               |
        +---------------+---------------+
                        v
              解审计 + 原子提交
                        |
                        v
        Job.migrate/reallocate + auxiliary state
                        |
                        v
             下一轮 Job.step() 推进与日志
```

调度器不得先调用现有 `SiaPolicy.optimize()` 得出单一 GPU 解。它应复用候选生成和 raw Goodput 查询，直到联合求解后才提交一个 GPU+CPU+RAM+placement 组合。

## 4. 方案 A：分层候选列联合优化（推荐）

### 4.1 核心思想

“分层”指构造过程分层，而不是决策串行固定：

1. Sia 层枚举多个合法 GPU 候选 `K_j`，不选择最终候选。
2. Profile 层为每个 `k` 生成 auxiliary 候选 `Q_jk` 和 `rho`。
3. Placement 层为 `(k,q)` 生成有限个合法 pattern `H_jkq`，优先包含旧 placement、最紧装箱和低碎片 pattern。
4. Master MIP 在所有联合列中一次性选择；CPU/RAM 可以使它改选 GPU 类型/数量。

因此它不同于 S2：S2 的 `k` 已固定，方案 A 的 `k,q,h` 同时由 master 决定。

### 4.2 联合列 MIP

对每个可行列 `a=(j,k,q,h)` 定义二元变量 `x_a`；列中已有常数 `g_av,c_av,m_av` 表示其在节点 `v` 的占用。

```text
opt  sum_a w_j U_a x_a + lambda_n * (|J_t| - sum_a x_a)

s.t. sum_{a belongs to j} x_a <= 1                         for all j
     sum_a g_av x_a <= G_v                                 for all v
     sum_a c_av x_a <= C_v                                 for all v
     sum_a m_av x_a <= M_v                                 for all v
     x_a = 0                                                if rho_a < 1
     sum_{a preserving locked allocation of j} x_a = 1     for checkpoint-locked j
     x_a in {0,1}
```

`p<0` 时按 Sia 取最小化；候选的 GPU config 已提前保证 max replicas、scale unit 和 Sia placement 规则。可选的二级目标只在主目标相同的容差内最小化 GPU/CPU/RAM 碎片与 node-ID remap，不能改变主公平目标。

### 4.3 Pattern 生成与有界近似

完整枚举 placement 会爆炸。初版采用以下确定性、有上限的集合：

- `h_keep`：若旧 placement 对 `(k,q)` 可行，必须保留。
- `h_tight`：best-fit decreasing，先放剩余 GPU 最少且三维可容纳的节点。
- `h_cpu` / `h_mem`：分别按 CPU 或 RAM 剩余最紧生成一个候选。
- multi-node Sia config 只用同类型整节点；single-node partial config 不跨节点。
- 每 `(j,k,q)` 最多 `H_max=4` 个去重 pattern；该值是预注册工程参数，不在结果后搜索。

若 master infeasible，但 GPU-proportional 全局解理论上应可行，则增量生成缺失 pattern并重求解；仍失败标记 `PATTERN_INCOMPLETE`，不得静默超配。20-job 阶段可与方案 B oracle 比较最优差距，决定是否提高 `H_max`。

### 4.4 伪代码

```text
INPUT: current jobs J_t, nodes V, previous joint allocation A_{t-1}

for j in J_t:
    K_j <- Sia legal_gpu_candidates(j, current history)
    lock/remove candidates according to checkpoint state
    g_ref_j <- min positive raw Sia goodput over K_j
    for k in K_j:
        g_gpu <- query_sia_goodput(j,k)
        Q_jk <- profile_oracle.candidates(j,k), including q0
        for q in Q_jk with rho(j,k,q) >= 1:
            for h in generate_patterns(j,k,q,A_{t-1},limit=H_max):
                add column(j,k,q,h,g_gpu,rho,restart_factor)

solution <- solve_master(columns, node capacities, p, lambda_n)
if infeasible and proportional feasibility expected:
    add_patterns_or_no_good_diagnostics(); solve once more
audit solution: one-column/job, capacities, locks, rho, scale units
if audit fails: return explicit INFEASIBLE (no partial commit)
atomically apply GPU placement and auxiliary map
return allocation + solver/profile/provenance audit
```

### 4.5 复杂度

设每作业 `K` 个 GPU 候选、每候选 `Q` 个辅助点、每点 `H` 个 pattern，则变量约 `O(JKQH)`，节点约束 `O(V)`。问题仍是 NP-hard set packing，但 `K` 是 Sia 小候选集，`Q` 可做 Pareto/支配剪枝，`H` 被上限控制。候选生成约 `O(JKQH*V log V)`；实际 MIP 时间必须记录 total/P50/P99/max、gap、timeout 和 fallback。

## 5. 方案 B：紧凑联合 MIP

### 5.1 变量

- `x_jkq in {0,1}`：作业选择 GPU config `k` 与辅助点 `q`。
- `z_jkqv in Z_+`：该组合在节点 `v` 放置的 GPU 数。
- `b_jkv in {0,1}`：作业配置是否使用节点 `v`。
- `d_j in {0,1}`：是否改变 GPU 配置/placement；用于选择预计算 restart-adjusted utility，而非新增未经验证的线性 penalty。

仅允许 `type(v)=type(k)` 的变量存在。令 `r_k,n_k` 为 GPU 总数和节点数，`c_jkq,m_jkq` 为每 GPU 辅助资源。

### 5.2 约束

```text
sum_kq x_jkq <= 1
sum_v z_jkqv = r_k * x_jkq
0 <= z_jkqv <= G_v * b_jkv
sum_v b_jkv = n_k * x_jkq

sum_jkq z_jkqv <= G_v
sum_jkq c_jkq * z_jkqv <= C_v
sum_jkq m_jkq * z_jkqv <= M_v
```

对跨节点 Sia 配置再加 `z_jkqv = G_v*b_jkv`；单节点配置由 `n_k=1` 和候选合法性保证不拆分。checkpoint job 固定到原节点/配置。`rho<1` 的组合不建变量。若 RAM 或 CPU 是总量而不是 per-GPU 可分，需要增加 job-node 辅助变量；初版数据 schema 明确采用 per-GPU 一致分配以保持线性。

目标与方案 A 相同，`R_gpu` 依赖是否保持旧 Sia config；node-ID remap 只作可审计的二级 tie-break。为避免 `x*z` 双线性，所有 `g_eff`、资源系数和“保持/改变”状态都预计算为常数或使用标准 indicator 线性化。

### 5.3 优缺点与定位

紧凑 MIP 不会漏掉预先未生成的 placement pattern，精确性更好；代价是 `z/b` 变量约 `O(JKQV)`，约束和对称解大量增加。20-job 可用于验证方案 A 的 objective gap、可行率和 placement 质量；160-job 是否可用由预注册 60 秒/轮求解预算、MIP gap 和内存决定，不预设可扩展。

## 6. 两方案比较与选择

| 项目 | 方案 A：有限联合列 | 方案 B：紧凑联合 MIP |
|---|---|---|
| 是否真正联合 | 是；最终 `k,q,h` 一次选择 | 是；`k,q` 与逐节点 `z` 同时求解 |
| 决策空间 | 只含生成的 placement 列 | 所有线性约束内 placement |
| 精确性 | 受 `H_max` 与生成规则限制 | 给定离散画像与时间预算更接近精确解 |
| 工程侵入 | 中等；易复用 Sia 候选/Goodput | 高；需重写 placement 状态与大 MIP |
| 求解规模 | `O(JKQH)` binary | `O(JKQV)` mixed integer + 对称性 |
| 失败处理 | 增列/回退 proportional | timeout incumbent/显式 infeasible |
| 建议用途 | 主算法、160-job 候选 | 2-3 job 与 20-job oracle；规模化需后验评估 |

方案 A 的首要优势是工作量可控且仍是联合选择；方案 B 提供评审所需的“小规模精确参照”，避免只与自己的启发式比较。

## 7. 公平对照与消融

所有策略共享同一 `g_gpu`、`rho`、集群、到达、工作量、轮长、restart 和完成判定。

| ID | GPU 选择 | CPU/RAM | Placement | 作用 |
|---|---|---|---|---|
| B0 | FIFO，固定 demand/type | proportional | GPU-first 合法装箱 | 非 Sia 的基础 GPU 调度。 |
| B1 | 原始 Sia | proportional | 原始 Sia placement + 同构资源记账 | 核心 GPU 基线；`rho=1`。 |
| B2 | 固定 demand/type | 受限 TUNE | 逐节点 GPU/CPU/RAM | 对照 Synergy 的辅助资源/装箱能力。 |
| B3-A | 方案 A | 联合选择 | 联合列 pattern | 主联合算法。 |
| B3-B | 方案 B | 联合选择 | MIP node variables | 小规模 oracle。 |
| ABL-no-aux | 与 B3 相同框架 | 只保留 q0，`rho=1` | 联合容量 | 去掉 CPU/RAM 感知。 |
| ABL-fixed-GPU | 冻结 B2 的 GPU config | 仍优化 q | 联合 placement | 去掉异构/弹性配置选择。 |

### 7.1 B0/B2 的固定 GPU 映射规则

在作业 admission 时、独立于队列未来状态，按 trace 的 `num_replicas` 固定 GPU 数；GPU 类型取 proportional 条件下 `g_gpu/GPU` 最大者，若并列按 `aws < dgx-ext < rtx` 的预注册顺序。若该 GPU 数不是 Sia 合法 placement，则向上取能容纳该 demand 的最小合法节点集合，但 GPU demand 不变。映射结果写入输入 manifest，生命周期内不变。

B0 只按 FIFO 选择 GPU 能满足的前缀并给 proportional CPU/RAM。B2 使用完全相同的固定映射和 runnable set，再执行受限 TUNE；runnable set 保持 FIFO 优先级，集合内部按论文所述 GPU、CPU、RAM demand 排序。当前公开源码只按 arrival 排序的差异不复制，并在报告中标为“paper-defined restricted TUNE adaptation”。B2 不得改变 GPU 类型/数量，不调用源码中的 `sys.exit`，最终回退 proportional。

## 8. 预注册实验矩阵

设计阶段只注册，不执行：

1. 手工 2-3 job：构造“快 GPU 但 CPU 紧”“慢 GPU 但可供给”“RAM 碎片”三例，人工枚举最优解。
2. 20-job smoke：`B0/B1/B2/B3-A/B3-B/两消融 × low/medium/high`；顺序运行，B3-B 可设 60 秒/轮 timeout。
3. 通过 correctness gate 后：固定 Saturn `workload-1/5/10` 三条 160-job traces，先跑 B0/B1/B2/B3-A/两消融；不自动扩展到 10 traces或参数搜索。

主要指标：完成数、mean/P50/P95/P99 JCT、makespan、GPU-hours、CPU-core-hours、RAM-GB-hours、每节点三资源利用率、restart/migration、aux realloc、capacity violation、求解时间/gap/fallback、B3 对 B1 的 task-level gain/harm。额外报告：方案 A 相对 B3-B 的 objective gap 和不同解比例。

正确性门槛：neutral/`rho=1` 下 B3-no-aux 与 B1 逐轮核心日志一致；所有 baseline 倍率等于 1；不重复加速；所有容量合法；checkpoint 语义和模拟时间单调；B2/B3 不读取未来信息；求解失败可复核。

## 9. 预计代码修改点（后续实现阶段）

优先新建独立模块，不直接改写 `SiaPolicy`：

| 文件/结构 | 计划修改 | 依据接口 |
|---|---|---|
| 新 `joint_policy.py` | `JointPolicy`, candidate/column master, 方案 A/B | 复用 `sia.py:100-142,423-507` 的候选/Goodput语义。 |
| 新 `joint_profile.py` | schema 校验、baseline、rho、provenance、Pareto pruning | 禁止直接乘 Synergy absolute speedup。 |
| 新 `joint_placement.py` | pattern generator、capacity audit、原子解对象 | 替代 GPU-only `alloc_to_placement()` 的新策略路径。 |
| `utils.py:JobInfo/NodeInfo` | 新策略字段：profile key、current aux、CPU/RAM capacity | 原类资源 dict 可扩展，但原 Sia路径保持只读。 |
| `cluster.py:get_node_infos()` | 新策略才填 `cpu_cores`,`memory_gb` | 当前 `502-508` 只填 GPU。 |
| `cluster.py:get_sia_multi_job_info()` | adapter 增加 current joint allocation/profile id | 当前 `373-432` 是状态边界。 |
| `cluster.py:optimize_policy()` | 新策略解的三资源原子提交与审计 | 当前 `160-221` 只提交 GPU placement。 |
| `job.py` | 保存 aux map/rho；productive progress 乘一次 | rescale delay `414-434` 不得被加速。 |
| `multi_simulator.py` | 独立 `--policy=joint-*` 与显式 dataset/pressure 参数 | `--policy=sia` 不走新代码。 |
| 新 `tests/test_joint_*.py` | 退化、人工 oracle、容量、重配置、无未来信息 | 不复用历史结果作为“通过”替代。 |

建议数据结构：

```text
JointJobInfo:
  base JobInfo
  profile_id, current_gpu_config, current_nodes, current_aux_point
  gpu_candidates[], checkpoint_locked

JointCandidate:
  job_id, gpu_type, num_nodes, num_replicas, batch, scale_unit
  raw_sia_goodput, ref_goodput, rho, effective_goodput
  cpu_per_gpu, ram_gb_per_gpu, response_provenance

JointPattern:
  candidate_id
  node_alloc[{node_id,gpus,cpu_cores,ram_gb}]
  preserves_sia_config, preserves_node_ids, restart_factor

JointSolution:
  chosen_pattern_by_job
  node_usage, objective, status, gap, solve_ms, fallback_reason
```

## 10. 实现顺序与停止条件

1. 冻结数据 schema、provenance 和三档压力配置；生成 hash manifest。
2. 实现只读 `JointProfileOracle`，通过 baseline=1、无重复收益和跨硬件标签测试。
3. 实现 B0/B1 accounting 与 neutral 逐轮退化；失败即停止。
4. 实现 B2 受限 TUNE，先过 proportional fallback/容量测试。
5. 实现方案 A 的候选、pattern 和 master；先跑人工枚举。
6. 实现方案 B 小规模 oracle；比较 A 的可行性与 gap。
7. 仅在上述 gate 全通过后运行 20-job 预注册矩阵。
8. 结果有效且资源允许时，另行批准三条 160-job；本任务不执行。

停止条件包括：baseline 不是精确 1、neutral 不退化、任一容量违规、同一收益乘两次、checkpoint/restart 改义、pattern infeasible 被静默提交、B3 使用了对照不可见的资源/未来信息，或 synthetic 数据被标成实测。

## 11. 预期研究贡献与审慎表述

可主张的算法贡献（需实验支持）是：在 Sia 的异构/弹性候选空间内，加入配置条件化 CPU/RAM 响应与逐节点三资源 placement，并提供可扩展有限列算法及小规模精确 MIP 对照。不能预先声称性能一定更好，也不能把方案称为 Synergy-OPT 的直接移植。

如果 low/medium/high 三档中 B3 与 B1 无显著差异，正确结论是 `NO_BENEFIT under registered synthetic model`；若仅部分尾延迟改善则报告 mixed；只有正确性、容量和 oracle gap 均通过后，才讨论机制有效性。真实集群结论仍取决于重新 profiling。
