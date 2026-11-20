# Sia + Synergy 联合模型修订

- 日期：2026-10-24
- 对应分支：`sia-joint-experiments/`，`exp/sia-synergy-joint-prototype`
- 基线 commit：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 范围：修订数据/对照定义并实现 2-job 最小原型；不运行 20/160-job，不做参数搜索或真实 profiling。
- 声明：**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 1. 为什么修订

原 `JOINT_DATASET_DESIGN.md` 的 capacity-relative 定义把每个压力档的 proportional 点重新归一为 1。它适合研究“baseline 以上的相对增益”，但会隐藏一个重要事实：当节点实际 CPU/RAM 低于固定参考条件时，Sia profile 所表示的名义 Goodput 可能根本无法实现。

本次同时修正两个问题：

1. 增加 fixed-reference absolute 模型，使辅助资源不足可以令 `rho<1`。
2. 增加 B4 串行组合基线，分离“只需在固定 GPU 后重分配资源”与“必须联合改变 GPU 配置”两类收益。

## 2. 修订后的两个性能模型

### 2.1 共同原始量

对作业 `j`、Sia 合法 GPU 配置 `k` 和 auxiliary 点 `q=(c,m)`：

```text
g_sia(j,k) = Sia 原始 Goodput
F_s(c,m)   = 从 Synergy 官方 *_1.py 读取的 9x5 response matrix
```

矩阵坐标严格采用 Synergy 源码：CPU `[1,2,3,4,5,6,9,12,24]`，RAM `[20.83,62.5,125,187.5,250]` GB。原型通过 AST 只读解析官方文件并记录 SHA-256，不复制手写 1.30/1.25 曲线，也不导入 absolute iteration time。

配置条件化的 synthetic demand scale 为 `phi_c(j,k),phi_m(j,k)`。在完整数据生成器中它可由 per-GPU Sia Goodput 与 local batch 公式产生；最小场景直接把固定 scale 写入 JSON，以便人工复核。它不是实测量。

### 2.2 fixed-reference absolute 模型（主实验）

固定参考点：

```text
C_ref = 3 synthetic cores/GPU
M_ref = 62.5 GB/GPU
```

查询映射与归一化：

```text
c_src = 3 * (c / C_ref) / phi_c
m_src = 62.5 * (m / M_ref) / phi_m

rho_abs(j,k,c,m) = F_s(c_src,m_src) /
                   F_s(3/phi_c,62.5/phi_m)

g_eff(j,k,c,m) = g_sia(j,k) * rho_abs(j,k,c,m)
```

因此 `(3,62.5)` 对每个配置严格为 1；实际供给低于该点时允许 `rho_abs<1`。不同 GPU 配置通过不同 `phi` 具有不同曲线斜率/饱和位置。插值使用网格内双线性插值，范围外 clip，不外推。

这是 **synthetic calibration**：固定点借用 Synergy 模拟器 8 GPU:24 CPU:500 GB 比例，跨 GPU demand scale 不是 Saturn 实测。主结果必须给出 scale、矩阵文件和 hash。

### 2.3 capacity-relative 模型（保留诊断）

对 placement 所在节点的 proportional 点 `(C_v/G_v,M_v/G_v)`：

```text
rho_rel(j,k,c,m) = F_s(3*(c/c_prop)/phi_c,
                       62.5*(m/m_prop)/phi_m) /
                   F_s(3/phi_c,62.5/phi_m)
```

此时 proportional 点严格为 1，只比较其上的相对收益。它不会表达“节点本身低于固定参考”的 slowdown。因此报告将 absolute 作为瓶颈主口径，relative 仅用于和原设计对照。

### 2.4 统一执行模型

B1、B4、B3 均调用同一 `JointProfileOracle`。区别仅在调度决策可见的信息和可改变的变量：

- B1 目标使用 `g_sia`，但提交 proportional CPU/RAM 后，任务实际进度使用 `g_eff`。因此 B1 不会超配，也不能无视 slowdown。
- B4/B3 的决策目标使用同一个 `g_eff`。
- Oracle 不读取完成时间、未来 arrival 或真实 remaining runtime；输入状态仅含 arrival 已发生、observed progress、当前配置/checkpoint 状态。

## 3. B1、B4、B3 的精确定义

| 策略 | GPU config | CPU/RAM | Placement | 决策目标 | 执行 Goodput |
|---|---|---|---|---|---|
| B1 | GPU-only 选择 | proportional | GPU-only 合法 pattern | Sia raw Goodput | 统一 Oracle 的 actual `g_eff` |
| B4 | **固定 B1 config** | 离散优化 | 可在同 config 下重新装箱 | actual `g_eff` | actual `g_eff` |
| B3 | 可改类型/数量 | 离散优化 | 与 config/aux 联合选择 | actual `g_eff` | actual `g_eff` |

B1 原型是“GPU-only config + node pattern”对照，不宣称逐行复刻官方 `SiaPolicy` 的类型总量 MIP；正式 20-job 集成时 B1 必须调用原 Sia 路径并在提交后接统一执行 Oracle。

### 3.1 B4 修复与回退

对 B1 选中的每个 job：

1. 候选集合只保留相同 `config_id`，禁止换 GPU 类型、数量或节点数。
2. 有限列包含旧 placement 可行点、同 config 的其他合法 placement、所有合法 auxiliary 点以及 proportional 点。
3. MIP 可保持旧 placement，也可记录 `REPACK_FIXED_GPU_CONFIG`。
4. 若固定-config 联合解遗漏 B1 已运行作业，整体回退到 B1 的 proportional map；该 map 在提交前重新做三资源审计。
5. 若 proportional map 也不合法，返回错误且不提交；禁止部分提交或超配。

原型在每次 B4 结果中记录 `fixed_gpu_configs`、逐 job `repair_rule` 和 `proportional_fallback_constructively_valid`。

## 4. 方案 A 的有限列 MIP

联合列 `a=(j,k,q,h)` 包含 job、GPU 配置、CPU/RAM 点、逐节点 placement。令列在节点 `v` 的占用为 `(g_av,c_av,m_av)`，二元变量为 `x_a`：

```text
maximize  sum_a x_a [g_eff(a) + lambda_alloc]

subject to
  sum_{a in job j} x_a <= 1                      for all j
  sum_a g_av x_a <= G_v                          for all nodes v
  sum_a c_av x_a <= C_v                          for all nodes v
  sum_a m_av x_a <= M_v                          for all nodes v
  checkpoint-locked job 只能使用当前 config/node pattern
  x_a in {0,1}
```

`lambda_alloc=10` 仅用于三个小场景中对应 Sia 的“未分配惩罚”，所有策略一致。正式集成应恢复 Artifact 的 `p` 次幂、`lambda_n` 和 config restart factor。原型对已有 current config 的替代候选提供 state-only factor；三个场景均从 fresh state 开始，factor 不影响结果。

辅助点必须不低于该 `(config,placement)` 的 proportional actual progress；absolute 模型允许这个 floor 本身低于固定参考 1。多节点配置仅接受同 GPU 类型整节点 pattern，且各节点 CPU/GPU、RAM/GPU 比必须一致。

## 5. `rho=1` 退化条件的修正

不能无条件要求联合策略与 Sia 逐轮日志完全相同，需要区分三层命题：

1. **目标一致**：所有候选 `rho=1`、同 `p/lambda/restart` 时，联合列的 GPU utility 与 Sia 相同。
2. **存在等价可行解**：Sia 选中的类型/数量能生成至少一个满足逐节点 GPU/CPU/RAM 的联合 pattern。
3. **逐轮日志一致**：还要求候选集合、placement tie-break、求解器 tie-break、node-ID 语义和状态提交完全一致。

只有 1+2+3 同时满足，才应要求逐轮日志一致。原型测试分别检查 baseline 数值、合法 pattern 和 MIP/穷举目标；不把“目标相同”误写成“node placement 必然相同”。

## 6. 数据映射修订

保留两类 synthetic cross-hardware transfer：

- A / `fixed_hash`：`SHA256(seed:job_id) mod template_count`，不看应用标签。
- B / `category_mix`：预注册 CPU-sensitive、memory-sensitive、neutral 等比例及模板集合，再确定性分配。

两个映射函数均已实现和测试。三个机制场景使用 `category_mix` 并在 JSON 中显式写出模板，以便人工枚举；这不表示真实模型映射。后续 20-job 必须同时跑固定哈希与预注册混合，不得只选择有利映射。

## 7. 原型代码对应

| 功能 | 文件/位置 |
|---|---|
| 官方矩阵 AST loader 与 SHA | `joint_prototype/joint_profile.py:143` |
| `JointProfileOracle` | `joint_profile.py:187-283` |
| fixed-hash/category-mix | `joint_profile.py:286-326` |
| placement pattern | `joint_scheduler.py:104-131` |
| 联合列构造与 Pareto 剪枝 | `joint_scheduler.py:187-262` |
| B1 proportional 执行列 | `joint_scheduler.py:264-296` |
| 有限列 MIP | `joint_scheduler.py:316-354` |
| 独立穷举 oracle | `joint_scheduler.py:356-371` |
| 原子容量审计 | `joint_scheduler.py:373-404` |
| B1/B4/B3 | `joint_scheduler.py:406-454` |

## 8. 当前边界

- GPU Goodput 在三个最小 JSON 中是可人工核验的 synthetic Sia-like 数值，不是运行时从真实 `Job.get_speedup_fn()` 动态提取；正式集成仍需接 Sia candidate builder。
- 没有模拟 epoch/JCT，只比较单轮 allocation 的 actual effective Goodput。
- 尚未把 Sia `p=-0.5`、CBC config MIP、checkpoint delay 和 60 秒轮次接入完整 `Cluster.step()`；原型只验证联合决策核心。
- node remap 和 auxiliary reallocation 的真实开销仍未测量，只记录，不得宣称为零成本真实收益。
- Synergy `_1` 矩阵跨应用、跨 GPU 转移依旧是合成假设。
