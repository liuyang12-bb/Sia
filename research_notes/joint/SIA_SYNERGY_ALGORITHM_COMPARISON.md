# Sia 与 Synergy 官方算法对照审计

- 日期：2026-10-23
- 状态：算法/数据设计输入；**未修改调度器，未运行实验**
- Sia Artifact：`siasosp23/artifacts`，commit `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- Synergy Artifact：`msr-fiddle/synergy`，commit `1bb2bed6edc9c5f953d0b7f0a40e4fdad053489c`
- 证据口径：论文公式与当前公开源码分别记录；两者不一致时，以“论文定义/Artifact 行为”两列并列，不互相替代。

## 1. 结论先行

Sia 决定“作业选哪一种 GPU、多少张 GPU、多少节点”，但当前模拟器只在 GPU 类型总量上做 MIP 约束，随后再映射节点；Synergy 假设 GPU demand 生命周期内固定，只决定 CPU、内存和物理装箱。两者的互补点真实存在，但接口并不直接相容。

真正的联合调度必须让 CPU/RAM 的性能收益和逐节点可行性参与 **GPU 候选的最终选择**。当前 S2 是“先固定 Sia GPU 解，再优化辅助资源”，不能通过增加一个目标权重升级为联合调度。

另有四个必须保留的审计结论：

1. Sia 论文说作业获得 CPU/GPU/网络 resource bundle，但当前 `NodeInfo` 和 MIP 仅填充/约束 `nvidia.com/gpu`。
2. Sia Artifact 的 `lambda_a` 只被读取，未进入最终目标；不能在新算法中假定它已惩罚 allocation change。
3. Synergy-OPT 论文的第二 LP 试图减少跨机碎片；当前源码 `SolveSecondLP()` 的目标是常数 `Minimize(1)`，只求可行解。
4. Synergy 当前 loader 对多 GPU job 仍强制加载 `_1` profile，CPU×RAM 矩阵不能无条件作为 Sia 多 GPU/异构 GPU 的实测画像。

## 2. Sia：论文公式、完整流程与源码

### 2.1 状态与性能模型

对作业 `j`、GPU 配置 `k=(t,n,r)` 和 batch `b`，Sia 的核心性能量为：

```text
g_gpu(j,k,b) = throughput(j,k,b) * statistical_efficiency(j,b)
```

`GoodputFunction.evaluate()` 在 `official-artifacts/sia-simulator/goodput.py:64-68` 实现该乘积；`throughput()` 在 `70-78` 建模计算/网络/梯度累积时间，`efficiency()` 在 `80-86` 使用梯度方差与平方范数。`Job.get_speedup_fn()` 在 `job.py:240-252` 将在线拟合参数包装成可查询的 `SpeedupFunction`。

`JobInfo` 的 Artifact 实际内容由 `cluster.py:373-432` 构造，包括：每 GPU 类型的 speedup function、min/max replicas、GPU used/wasted seconds、类别、restart/migration、age、跨类型 throughput ratio、checkpoint 状态和 PMP scale unit。节点状态由 `cluster.py:502-508` 构造，但只有 GPU 数量。

### 2.2 弹性 GPU 候选

论文 §3.2 将配置写作 `(n,r,t)`：节点数、资源/GPU 总数、GPU 类型。对同构子集的候选为：

```text
C_t = {(1, 2^i, t) | 2^i <= G_t}
      union {(n, n*G_t, t) | 2 <= n <= N_t}
```

即单节点取 1、2、4…张，跨节点配置占满所用节点。Artifact 的固定表与过滤在 `sia.py:100-142`；再按作业的 min/max replicas、scale unit 和 PMP 整除性过滤，见 `sia.py:652-669`。论文所述“每轮最多放大 2 倍”在当前模拟器 MIP 路径没有作为显式候选约束出现，属于论文/实现需要分开的地方。

### 2.3 异构 Goodput 与归一化

`SiaPolicy._compute_goodputs_v2()`（`sia.py:423-507`）直接查询目标 GPU 类型的模型；若缺失且启用 projection，则以其他类型模型和 throughput ratio 投影。

论文 §3.3 对每个作业按行最小值归一。Artifact 对 adaptive job 的确切定义可写为：

```text
g_ref(j) = min { g_gpu(j,k) | k 合法且 g_gpu(j,k) > 0 }
G(j,k)   = scale_unit(j) * g_gpu(j,k) / g_ref(j)
```

对应 `sia.py:673-705`；无模型类型的最小合法配置填为 scale unit（`708-715`），极小值截成 `ZERO_ALLOC_GAIN`（`717-719`）。rigid job 按固定 replicas 在 GPU 类型间归一，见 `sia.py:526-553`。

### 2.4 目标、约束与公平性

论文公式 (1) 先定义：

```text
maximize_A  sum_j [ sum_k A(j,k) G(j,k) + lambda * (1 - ||A_j||_1) ]
```

每个作业最多选一个配置，各 GPU 类型用量不超过容量。论文随后引入公平参数 `p`，将 `G` 逐元素变为 `G^p`；`p<0` 时改为最小化以保持偏好方向。论文正文默认 `p=0.5`，而本 Artifact `simulator_config.py:50-53` 默认 **`p=-0.5`**，正式实验必须记录实际 CLI 值。

Artifact 等价目标为：

```text
opt_X  sum_j sum_k w_j X(j,k) [R(j,k) G(j,k)]^p
       + lambda_n * (|J| - sum_jk X(j,k))
```

其中 `R(j,k)` 是下节 restart factor；`p<0` 用 `Minimize`，否则 `Maximize`。源码位置为 `sia.py:907-931,949-971`。约束为：

```text
0 <= X(j,k) <= 1, X 为整数
sum_k X(j,k) <= 1
sum_jk gpu_count(k,t) X(j,k) <= GPU_capacity(t)
checkpoint/restore 中的作业必须保持原配置
```

对应 `sia.py:939-980,982-1028`。求解器实际硬编码 CBC（`1030-1036`），不是配置文件所写 GLPK。`lambda_a` 在 `935-937` 赋给局部变量后不再使用；新设计不能把它算作既有机制。

### 2.5 重配置与状态推进

论文公式 (2) 的 restart factor 为：

```text
R_j = (T_j - N_j*S_j) / (T_j + S_j)
```

Artifact 默认取非负截断，并在年龄小于 penalty 时强制为 1（`sia.py:721-740`）。已有配置保持原 Goodput，其他配置乘该 factor（`873-900`）。checkpoint 中作业被固定（`763-785,1011-1028`）。

MIP 解在 `sia.py:1058-1120` 转成 GPU 类型/配置；`800-822` 调用 `alloc_to_placement()` 得到节点 ID。`Cluster.optimize_policy()`（`cluster.py:160-221`）比较新旧 placement 并调用 `Job.migrate()`/`reallocate()`。后两者设置 rescale delay、更新 placement 和 restart（`job.py:524-579`）。下一轮 `Job.step()` 先把 rescale delay 记为 wasted GPU seconds，再用 Goodput 推进 progress/epoch（`job.py:414-522`）。

一次完整轮次的实际时序是：

```text
上一轮 placement -> Job.step() 推进 -> current_time 增加
-> get_job_infos() -> SiaPolicy.optimize()
-> Goodput/归一化/restart factor/MIP
-> GPU node placement -> migrate/reallocate
-> step_log()
```

证据为 `cluster.py:117-153,160-221,340-432`。这意味着本轮求出的 allocation 从下一轮才产生训练进度。

## 3. Synergy：论文公式、完整流程与源码

### 3.1 固定 GPU demand 与画像

Synergy 将 `g_j` 视为生命周期内固定 GPU demand。每个 job 有离散矩阵：

```text
W_j[c,m] = 每轮分配 c CPU、m GB RAM 时的 progress/normalized throughput
```

源码坐标为 CPU `[1,2,3,4,5,6,9,12,24]`、RAM `[20.83,62.5,125,187.5,250]` GB（`jobs/job.py:96-97`）。`Workload.add_synergy_profile()` 将 per-GPU demand 乘 GPU demand（`jobs/workload.py:169-220`），并设置 iteration time/speedup（`242-245`）。

但 `jobs/model.py:169-183` 固定使用 `model_name + "_1"`，按 GPU demand 选择 `_2/_4` 的代码被注释。因此公开模拟器的矩阵不能被解释为完整的多 GPU profile 集。

### 3.2 GPU-proportional baseline

对节点 `i` 的容量 `(G_i,C_i,M_i)`，占用 `g` 张 GPU 的 baseline 为：

```text
C_g = (C_i/G_i) * g
M_g = (M_i/G_i) * g
```

源码 `resources/cluster.py:254-278` 按节点上 GPU 份额分 CPU/RAM/storage（CPU 向下取整）。默认配置是每节点 8 GPU、24 CPU、500 GB RAM；这只是 Synergy 模拟器配置，不是 Sia 三类节点的真实规格。

### 3.3 Synergy-OPT

论文公式 (1)-(7) 对当前 runnable set `J_t` 定义二元变量 `y_{c,m,j}`：

```text
max  sum_j sum_[c,m] W_j[c,m] y[c,m,j]                         (1)
s.t. sum_j sum_[c,m] c y[c,m,j] <= C                           (2)
     sum_j sum_[c,m] m y[c,m,j] <= M                           (3)
     sum_[c,m] y[c,m,j] = 1                                   (4)
     sum_[c,m] W_j[c,m] y[c,m,j] >= W_j[C_g,M_g]              (5)
     c*_j := c if y[c,m,j]=1                                  (6)
     m*_j := m if y[c,m,j]=1                                  (7)
```

第一阶段是把全体资源视为“超级服务器”的总量 ILP；源码对应 `opt_algo/synergy_opt.py:63-149`。第二阶段用 server×job 变量将 `(g_j,c*_j,m*_j)` 映射到节点。论文希望减少 fragmented jobs，但当前源码 `174-220` 的目标为 `Minimize(1)`，并允许分数 `X`；它不是可直接执行的整数联合 placement 最优器。

### 3.4 Synergy-TUNE

论文 §4.2 的 TUNE 保留基础 FIFO/SRTF/LAS/FTF 所选 runnable set，不能因 CPU/RAM 不足跳过 GPU 可满足的高优先作业。流程是：画像理想需求装箱；失败时先把当前 job 降为 proportional baseline；仍失败则只按 GPU 找节点并把相关 peer 降级；最坏所有 job 都回到 baseline。

源码链路为：

```text
FIFO.schedule()                                    fifo_synergy_new.py:24-102
  -> jobs_this_round 先按固定 GPU demand 选取       46-52
  -> Cluster.allocate(... tune=True)               cluster.py:281-299
  -> allocate_synergy_tune() / _tune()             575-669
  -> allocate_synergy_placement()                  382-431
  -> _top_synergy_gpus_placement()                 480-518
  -> _fits_in_server()                             550-562
  -> 当前 job / peer 回退 fair share               610-666,698-746
```

多节点作业按每 GPU 相同比例分 CPU/RAM；`_top_synergy_gpus_placement()` 每次按 per-GPU demand hold 资源。当前 `_tune()` 最终失败会 `sys.exit(1)`（`595-608`），联合调度实现必须改为返回 `INFEASIBLE`，不能复用该终止行为。

### 3.5 训练速度更新

`jobs/job.py:133-190` 在一次 lease allocation 时：跨机先除以 placement penalty，非 fair allocation 再除以 `synergy_speedup`，随后按 `floor(round_time/iteration_time)` 推进 iteration。这个绝对 iteration time 与 Sia 的 Goodput/训练工作量口径不同；联合模型只能借用归一化响应形状，不能用它替换 Sia 训练时长。

## 4. 两套算法逐项比较

| 维度 | Sia | Synergy | 联合设计含义 |
|---|---|---|---|
| 主决策 | GPU 类型、数量、节点数/配置 | 固定 GPU demand 下的 CPU/RAM 与 placement | GPU 候选不能先固定；最终变量须同时含 GPU 与辅助配置。 |
| 性能模型 | GPU/batch/placement Goodput | CPU×RAM progress matrix | `effective = Sia Goodput × 同基准相对响应`，只能乘一次。 |
| 异构性 | 三类 GPU，支持投影 | 论文/主实验同构 V100 | 跨 GPU CPU/RAM 响应只能显式标 synthetic，或重新测量。 |
| 弹性 | 每轮可改 GPU 类型/数量 | GPU demand 固定 | 这是新增决策能力的主要来源。 |
| 公平性 | 行归一化、`p` 次幂、未分配惩罚 | 基础策略顺序 + 每 job 不低于 proportional throughput | 联合策略保留 Sia `p`；`rho>=1` 只约束被运行作业的辅助候选。 |
| 重配置 | restart factor + 实际 rescale delay | lease/checkpoint；模拟器主要按轮重放 | GPU 变化沿用 Sia；辅助资源变化的真实开销必须单独测量。 |
| 容量 | Artifact 只约束每 GPU 类型总量 | 每节点 GPU/CPU/RAM | 联合策略必须逐节点约束三维容量。 |
| Placement | GPU MIP 后启发式映射 | TUNE 多维装箱/回退 | 必须原子提交，placement 失败应重求解，不能事后超配。 |
| 求解 | 单个配置选择 MIP，CBC | OPT 两阶段 LP/ILP；TUNE heuristic | 主方案使用有限联合列；全联合 MIP作小规模 oracle。 |

## 5. 组合机制是否确有新增价值

### 5.1 GPU 类型与 CPU/RAM 的相互影响

在明确的模型假设下，以下耦合是可检验而非预设结论：

- 更快 GPU 或更多 replicas 提高样本消耗速率，可能把 dataloader/pre-processing 变成瓶颈，从而提高 CPU 饱和点。
- 更大的 local/global batch 改变 host staging、prefetch 和缓存工作集，可能提高 RAM 饱和点；跨节点配置也可能改变每进程数据供给。
- 同一作业在不同 GPU 类型的 `g_gpu` 差距，只有在对应 CPU/RAM 能供应时才可实现；否则名义最快 GPU 未必具有最高 `effective_goodput`。
- CPU/RAM 不足可能让“更多 GPU”边际收益小于“较少 GPU + 足够辅助资源”，而 Synergy 的固定 `g_j` 无法选择后者。

这些关系在当前数据中尚未实测；数据文档将其注册成可关闭、可审计的 synthetic transfer 假设。

### 5.2 分层与联合解何时不同

它们只会在以下任一条件出现时不同：多个 GPU 候选的辅助资源需求不同；节点 CPU/RAM 碎片使某些 GPU placement 不可行；或给某作业更快 GPU 会挤占另一个作业的关键 CPU/RAM。若所有 `rho=1` 且 proportional baseline 总可行，联合解应严格退化为 Sia。

典型错配包括：Sia 选择 A100 配置却没有足够 CPU 达到画像 Goodput；两个 GPU 可行解在 GPU 总量层面相同，但只有一个能按节点装下内存；或 Sia 扩大某 job 的 GPU 数后，其 baseline CPU/RAM 占满节点，使高敏感 job 无法获得少量高收益资源。

反向地，Synergy 固定 GPU demand 会错过：改用另一 GPU 类型释放稀缺 CPU/RAM；缩小低扩展性 job 的 GPU 数以运行更多 job；或扩大高扩展性 job 并配合合适 batch 的方案。

### 5.3 新算法新增的决策能力

新算法增加的不是一个权重，而是四个联合选择维度：

1. 在多个 Sia 合法 GPU 类型/数量/节点数候选间保留选择权。
2. 对每个 GPU 候选选择一个 baseline 以上的 CPU×RAM 离散点。
3. 同时选择逐节点 placement，使 GPU/CPU/RAM 都合法。
4. 在上述联合空间内保留 Sia 公平性、未分配惩罚、checkpoint 锁和 GPU restart 语义。

收益是否存在由实验决定；在 `rho=1`、辅助资源充足或所有候选需求等比例时，合理结果就是无收益。

## 6. 不可混淆的边界

- 当前真实可迁移 Synergy profile 覆盖仍为 `0/480`；后续跨硬件构造必须标 `synthetic_cross_hardware_transfer`。
- Sia Goodput 已隐含原 profile 机器的 CPU/RAM 条件，联合响应必须用同一配置的 baseline 点归一为 1。
- 论文 Synergy-OPT、公开源码 OPT、TUNE 和本文联合算法是四个不同对象。
- 当前 S2 结果 `MIXED` 只证明固定 GPU 后的机制行为；不构成联合算法收益证据。
- 本文没有修改 `official-artifacts/`、`research_sources/Synergy/` 或正式实验分支。
