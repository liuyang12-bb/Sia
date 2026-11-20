# B3 五策略实验可重复性审计与正式实验前检查

## 1. 结论先行

本轮状态为：

**`FULL_REPEAT_PASS — FIVE_POLICY_SIMULATION_TRACES_EXACT`**

首次审计时可用内存为 **0.792 GiB**，低于冻结的 2 GiB 门槛，因而安全停止。内存调整后，在全新目录 `cpu_contention_20job_repro_run2/` 使用完全相同的 Neutral 冻结输入顺序重跑 B1 → SY → B4 → B5 → B3；预检与每个策略启动前均满足 2 GiB 门槛，五个策略全部完成 20/20，统一验证器全部通过。没有修改调度算法、容量、workload、响应曲线或参数。

证据支持以下结论：

1. 修复后的完整五策略矩阵已独立复现。相对于 `cpu_contention_20job_repair_full/`，五种策略的确定性 summary、逐轮 GPU/CPU/RAM/placement 决策、求解状态、未分配集合、训练推进和完成事件均完全相同；差异只存在于墙钟与求解耗时等明确排除的非确定字段。
2. B3 仍为 552 轮、平均 JCT 3831.05 秒；B1、SY、B4、B5 分别为 5102.95、6783.25、5056.25、5053.30 秒。因此本冻结 cohort 上 B3 相对四个对照的观测差值已复现，但这仍只是 Neutral synthetic cross-hardware 机制实验，不构成真实 Saturn 性能证据。
3. B4 从 5083.75 秒变为 5056.25 秒的首个分叉发生在模拟时间 7500 秒。两次的输入、Sia GPU 意图、原始候选和进入该轮的执行记录一致，但联合接入代码 hash 不同。修复后的锁语义允许两个无当前配置的作业形成合法列，改变了 admission 结果。新旧记录解的目标分别为 124.316 和 115.131，不是同一个目标值下的并列最优分支。
4. 当前 B4 的 `fallback_count=473` 是旧统计口径错误：473 是带 `fallback_reason` 的正常 `OPTIMAL` admission 记录，真正 fallback 为 0。旧 B4 的 475 则由 473 个 admission reason 加 2 个 `FALLBACK_B1_PROPORTIONAL` 构成。
5. B3 的 53 个“CPU 饱和且 GPU 空闲”事件中，51 个没有等待作业，只有 2 个事件伴随 `bert-37` 等待，共 120 秒。53 是节点逐轮状态次数，不是作业等待次数。
6. B3 的低 RAM-GB-hours 可由统一积分公式和实际分配点完整复算：B3 始终使用 20.83 GB/GPU，而 B1 始终使用 125 GB/GPU。未发现 RAM 统计公式错误；这是 Neutral 下额外 RAM 没有收益时选择最低合法辅助资源点的结果。

综合判定：

**当前冻结实现已通过一次独立五策略全矩阵复跑，可进入下一轮预注册的多 workload 合成实验。** 放行仅限在提交并冻结可引用代码版本、继续记录逐文件 hash 与依赖版本、沿用统一 fallback 口径之后；不放行参数搜索、160-job 实验或真实集群性能结论。

> **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 2. 工程保护与版本

- 工作区：`D:\科研启航\Sia`
- 实验分支：`exp/synergy-baselines-cpu-bottleneck`
- HEAD：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 官方仓库：`official-artifacts`，`main...origin/main`，检查时干净。
- 实验工作树原有 `job.py` 修改及大量未跟踪联合实验文件均保留。
- 未执行 `reset`、`clean`、覆盖或删除。

本轮新增：

- 离线审计脚本：`sia-joint-experiments/sia-simulator/audit_b3_reproducibility.py`
- 权威机器可读结果：`research_runs/sia_synergy/b3_reproducibility_audit_canonical/audit_summary.json`

审计生成过程还保留了两个未覆盖的中间目录：`b3_reproducibility_audit/` 和 `b3_reproducibility_audit_final/`；最终报告只引用 `b3_reproducibility_audit_canonical/`。

## 3. 内存门槛与重复运行

首次尝试在可用内存 0.792 GiB 时按要求停止，没有构造 `JointCluster`。内存调整后的第二次尝试复用原计划的新目录：

`research_runs/sia_synergy/cpu_contention_20job_repro_run2/`

| 项目 | 结果 |
|---|---|
| 冻结门槛 | 2.000 GiB |
| 第二次预检可用内存 | 2.363 GiB |
| 五个策略启动前采样 | 2.357 / 2.666 / 2.429 / 2.538 / 2.744 GiB |
| 新五策略重复运行 | `SUCCESS`，5/5 策略均完成 20/20 |
| 历史结果是否覆盖 | 否 |
| 参数是否改变 | 否 |
| Medium / 160-job 是否运行 | 否 |

该目录在运行前不存在；运行器仍拒绝复用非空目录。首次低内存停止证据和历史运行均保留。

## 4. 输入、代码和求解环境对比

### 4.1 冻结输入

旧 run1 和修复后完整矩阵的输入完全相同：

| 对象 | SHA-256 |
|---|---|
| config | `2A4E3D7087D0BC66E3EB6987A6E25AC2E04C80B9D8CEF868399F5AAF39E53C11` |
| extracted workload | `46FAB9A9743719B56437E6FC35DADAA1CF397205DCF06483E6A12E7CA488627A` |
| source workload | `D80DC93782785AAB449D3109AA118F123EA799D891DF0AFE2B9DEE483EDA9468` |

五个 Synergy 模板 hash 在两次 manifest 中也逐项一致。因此 B4 差异不是 workload、容量、模板映射或响应参数变化造成的。

### 4.2 代码快照

两次均记录同一 dirty HEAD，但关键文件 hash 不同：

| 文件 | 旧 run1 | 修复后运行 |
|---|---|---|
| `joint_integration.py` | `513B2CB0...C68CA6A` | `5E1E60F7...8B175B` |
| `joint_scheduler.py` | `C199F6A8...9E0682` | `65933FFD...848E7` |
| `run_cpu_bottleneck_smoke.py` | `A15EFEAB...5105C` | `F125F7F3...D685E` |
| `run_cpu_contention_20job.py` | `986DDC77...C735ED` | `E2E01DC7...AA461` |

`joint_profile.py`、`synergy_adapter.py`、`job.py`、workload 和 config hash 未变化。

### 4.3 求解器

当前环境：

- Python 3.10.10；
- NumPy 1.23.5；
- SciPy 1.10.1；
- CVXPY 1.3.2；
- CyLP 0.91.6；
- 原 Sia GPU intent：CVXPY + CBC；
- B3/B4 有限列 MIP：`scipy.optimize.milp`，即 SciPy 的 HiGHS 后端，`mip_rel_gap=0`。

旧 run1 manifest 记录了 Python 和平台，但没有冻结 SciPy、CVXPY、CyLP/CBC 或 HiGHS 版本。因此不能仅从旧产物证明两次依赖版本完全相同，这是一个仍需修复的实验 provenance 风险。

不过，本次首个分叉的两个已记录解目标值不同，且分叉行为与已知的幽灵 checkpoint 锁修复直接吻合；现有证据不支持“HiGHS 在并列最优解间随机选择”作为主要原因。

## 5. B4 历史差异审计

### 5.1 汇总差异

| 指标 | 旧 run1 | 修复后运行 |
|---|---:|---:|
| 平均 JCT | 5083.75 | 5056.25 |
| 模拟轮数 | 575 | 573 |
| 总等待时间 | 29460 s | 28500 s |
| Restart | 68 | 70 |
| Wasted GPU-seconds | 14735 | 14835 |
| checkpoint/restore delay | 5610 s | 5685 s |
| GPU-hours | 91.676 | 91.773 |
| RAM-GB-hours | 1970.368 | 1911.639 |
| 真实 fallback | 2 | 0 |

两次共有 40 个时间戳的最终选择不同。

### 5.2 首个分叉

首个分叉为 `interval_end=7500`，round 66。进入该轮的 `execution` 数组逐字段一致；Sia 固定 GPU 意图一致；原始 candidate summary 一致。

旧结果：

- 选择 8 个作业；
- RTX 上选择 `deepspeech2-30` 的 4 GPU 与 `deepspeech2-31` 的 2 GPU；
- `bert-37`、`cifar10-39` 等待；
- recorded objective：115.131389551。

修复后结果：

- 选择 9 个作业；
- RTX 上选择 `deepspeech2-30` 的 4 GPU、`bert-37` 的 1 GPU、`cifar10-39` 的 1 GPU；
- `deepspeech2-31` 等待；
- recorded objective：124.315958851。

旧联合接入代码会在 `scaling_underway=True` 但当前配置为空时形成幽灵锁，使 `bert-37`、`cifar10-39` 的联合列被过滤。修复后的锁必须同时具备当前 config 和 node，二者因此进入可行域。

这不是“同一候选空间内两个等值解”的证据：新旧 recorded objective 相差约 9.185。根因判定为：

**主要原因已确认是联合接入代码版本及锁语义变化；不是输入变化。旧依赖版本未冻结，因此求解器版本差异只能标记为未完全排除的 provenance 风险，而不是当前最符合证据的原因。**

### 5.3 逐作业差异

仅 7 个作业的 JCT/完成时间发生变化：

| Job | 旧 JCT | 新 JCT | Δ新-旧 | 旧完成时间 | 新完成时间 |
|---|---:|---:|---:|---:|---:|
| deepspeech2-21 | 5632 | 5746 | +114 | 9230 | 9344 |
| imagenet-23 | 28914 | 28867 | -47 | 33220 | 33173 |
| deepspeech2-30 | 5188 | 5476 | +288 | 10205 | 10493 |
| deepspeech2-31 | 29157 | 29072 | -85 | 34448 | 34363 |
| cifar10-35 | 1247 | 961 | -286 | 8438 | 8152 |
| bert-37 | 2659 | 2779 | +120 | 9922 | 10042 |
| cifar10-39 | 1742 | 1088 | -654 | 9207 | 8553 |

其他 13 个作业的 JCT 和完成时间完全一致。首个 admission 分叉随后改变了 Sia 状态、重启、checkpoint 与完成顺序，因此不能把后续 40 个不同时间戳分别解释为独立求解器随机性。

## 6. 已有 B3 两次运行的一致性

比较：

1. `cpu_contention_20job_b3_repair_targeted/cpu_constrained.neutral.b3.json`；
2. `cpu_contention_20job_repair_full/cpu_constrained.neutral.b3.json`。

以下项目全部逐值相等：

- 20/20 完成；
- 平均及逐作业 JCT；
- 逐作业完成时间和等待时间；
- 552 轮 decision signature；
- 每轮 GPU config、CPU/RAM、placement、未分配作业；
- 每轮训练进度、checkpoint 消耗和完成事件；
- GPU/CPU/RAM-hours；
- 节点峰值、rho 范围和容量违规列表。

两次共有的 decision signature SHA-256：

`02C2AD51065556B4366D79E479985F043E1ED8D867C618D1F9AD08D752124D93`

需要保留的限制：两次运行使用相同输入和相同 dirty HEAD，且本次操作记录显示两次之间没有调度器修改；但针对性运行 manifest 没有自行嵌入逐文件调度器 hash。因此这是强结果级复现证据，不是两个干净 commit 的独立二进制快照证明。

## 7. 当前五策略统计口径

以下为已存在的修复后完整矩阵，并非本轮新重跑：

| 策略 | 完成 | Mean JCT | P95 | P99 | Makespan* | 等待 | GPU-h | CPU-core-h | RAM-GB-h | Restart | Wasted GPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 20/20 | 5102.95 | 28867.25 | 29266.25 | 31122 | 30000 | 91.362 | 91.362 | 11420.278 | 57 | 13775 |
| SY adapted | 20/20 | 6783.25 | 28142.55 | 29078.11 | 30263 | 57180 | 119.588 | 119.588 | 14417.784 | 4 | 10560 |
| B4 | 20/20 | 5056.25 | 28877.25 | 29033.05 | 30828 | 28500 | 91.773 | 91.773 | 1911.639 | 70 | 14835 |
| B5 | 20/20 | 5053.30 | 29048.05 | 29854.41 | 31812 | 29460 | 91.931 | 92.197 | 11474.652 | 62 | 14095 |
| B3 | 20/20 | 3831.05 | 14738.80 | 25968.56 | 29547 | 120 | 90.715 | 90.715 | 1889.593 | 73 | 13485 |

\* Makespan 在本审计中明确定义为 `最后完成时间戳 - 首个作业 arrival(3535)`，不是 `rounds × 60`。

当前完整矩阵中，B3 平均 JCT 相对 B1、B4、B5 的描述性差值分别为 -24.92%、-24.23%、-24.19%。但均值收益高度受到 `deepspeech2-31` 影响：该作业在 B3 中比 B1/B4/B5 分别少 24087/23793/24777 秒。多 workload 正式实验必须报告分布和逐作业结果，不能只报告平均值。

### 7.1 逐作业 JCT

| Job | B1 | SY | B4 | B5 | B3 | B3-B1 | B3-B4 | B3-B5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10-20 | 482 | 6386 | 482 | 482 | 484 | +2 | +2 | +2 |
| deepspeech2-21 | 6033 | 3299 | 5746 | 5443 | 4785 | -1248 | -961 | -658 |
| cifar10-22 | 515 | 8346 | 515 | 515 | 515 | 0 | 0 | 0 |
| imagenet-23 | 28841 | 29312 | 28867 | 28995 | 28776 | -65 | -91 | -219 |
| cifar10-24 | 929 | 1081 | 929 | 929 | 1007 | +78 | +78 | +78 |
| cifar10-25 | 1511 | 326 | 1511 | 1511 | 1511 | 0 | 0 | 0 |
| cifar10-26 | 904 | 9343 | 904 | 904 | 1313 | +409 | +409 | +409 |
| cifar10-27 | 1498 | 607 | 1498 | 1498 | 1475 | -23 | -23 | -23 |
| bert-28 | 1810 | 4274 | 1810 | 1810 | 1931 | +121 | +121 | +121 |
| cifar10-29 | 1435 | 1691 | 1435 | 1435 | 1780 | +345 | +345 | +345 |
| deepspeech2-30 | 4983 | 17661 | 5476 | 5567 | 5270 | +287 | -206 | -297 |
| deepspeech2-31 | 29366 | 7649 | 29072 | 30056 | 5279 | -24087 | -23793 | -24777 |
| cifar10-32 | 1169 | 2327 | 1169 | 1169 | 1258 | +89 | +89 | +89 |
| yolov3-33 | 13925 | 28081 | 13925 | 13925 | 14000 | +75 | +75 | +75 |
| cifar10-34 | 975 | 2890 | 975 | 975 | 975 | 0 | 0 | 0 |
| cifar10-35 | 2184 | 788 | 961 | 1217 | 1245 | -939 | +284 | +28 |
| cifar10-36 | 2142 | 1056 | 1205 | 860 | 900 | -1242 | -305 | +40 |
| bert-37 | 1437 | 422 | 2779 | 1567 | 1749 | +312 | -1030 | +182 |
| cifar10-38 | 826 | 8942 | 778 | 778 | 1248 | +422 | +470 | +470 |
| cifar10-39 | 1094 | 1184 | 1088 | 1430 | 1120 | +26 | +32 | -310 |

机器可读审计中同时保存五策略逐作业完成时间、配置/placement signature 和求解状态。

## 8. fallback 统计口径修正

统一定义：

- **真实 fallback**：`result.status` 明确包含 `FALLBACK`，或为 `ALL_WAIT*`、`NO_COLUMNS`、`MIP_NO*`。
- **正常成功**：`OPTIMAL`、`ORIGINAL_SIA_OPTIMAL`、`SYNERGY_TUNE_ADAPTED`。
- **admission reason**：`audit.fallback_reason` 仅说明某个 Sia 比例意图无法被辅助资源 admission 接纳，不等于求解器或策略回退。

修复后矩阵：

| 策略 | 成功状态 | 真实 fallback | admission reason rounds |
|---|---|---:|---:|
| B1 | `ORIGINAL_SIA_OPTIMAL=44`, `OPTIMAL=476` | 0 | 0 |
| SY | `SYNERGY_TUNE_ADAPTED=506` | 0 | 0 |
| B4 | `OPTIMAL=515` | 0 | 473 |
| B5 | `SYNERGY_TUNE_ADAPTED=532` | 0 | 0 |
| B3 | `OPTIMAL=494` | 0 | 0 |

B4 的 473 个 reason 均为 `PROPORTIONAL_INTENT_INFEASIBLE_NO_PARTIAL_COMMIT`。它们表示原 Sia intent 中至少一个作业因 CPU/placement 不可行而继续等待，但该轮其余完整分配仍由正常 OPTIMAL 解原子提交。

旧 B4 的真实 fallback 仅发生于 8460、9240 秒，共 2 次；旧 summary 的 475 等于 473 个 admission reason 加 2 个真实 fallback。原始 JSON、manifest 和 comparison summary 均未改写，本轮只在独立 audit summary 中给出一致口径。

## 9. B3 CPU 饱和/GPU 空闲事件

`cpu_saturated_gpu_idle_event_count` 的单位是“节点-调度轮状态事件”：若某节点 `free_gpu>0` 且 `free_cpu<1 core`，该轮记一次。它不要求存在等待作业。

B3 的 53 个事件全部发生在 `rtx:0`，每次均为：

- RTX 节点 6 CPU 已用满；
- 8 GPU 中仍有 2 GPU 空闲。

分解如下：

| 事件类别 | 次数 |
|---|---:|
| 有未分配作业 | 2 |
| 无未分配作业 | 51 |
| 合计 | 53 |

两次伴随等待的事件为 round 64/65，时间 7380/7440 秒，未分配作业均为 `bert-37`。每轮 60 秒，因此作业等待总量正好是 120 秒。

结论：53 与 120 秒并不矛盾。51 次事件只是 RTX 节点上的 CPU/GPU 资源形状不匹配，当时没有可调度等待作业；2 次事件才实际影响排队。该指标合法，但正式报告必须命名为 `node_round_cpu_saturated_gpu_idle_events`，避免被误解为 53 次作业阻塞。

## 10. B3 RAM-GB-hours 审计

统一公式位于 `run_cpu_bottleneck_smoke.py:100` 附近：

```text
RAM-GB-hours = Σ(node_usage.ram_gb × allocation_active_seconds) / 3600
```

若作业在轮次中间完成，active seconds 截断到完成时间；checkpoint/restore 期间资源仍被占用，因此资源时间继续累计。这与 GPU-hours、CPU-core-hours 使用相同时间口径。

实际辅助资源点统计：

| 策略 | 主要 RAM 配额 | RAM/GPU-hour | RAM-GB-hours |
|---|---|---:|---:|
| B1 | 始终 125 GB/GPU | 125.000 | 11420.278 |
| SY | 主要 125，少量 20.83 | 120.562 | 14417.784 |
| B4 | 始终 20.83 GB/GPU | 20.830 | 1911.639 |
| B5 | 主要 125，少量 62.5/20.83 | 124.819 | 11474.652 |
| B3 | 始终 20.83 GB/GPU | 20.830 | 1889.593 |

B3 复算：

```text
90.715000 GPU-hours × 20.83 GB/GPU
= 1889.593450 RAM-GB-hours
```

残差约 `1.2e-11`，属于浮点误差。因此 B3/B4 与 B1/B5 的 RAM 差异不是积分统计错误。

原因是 Neutral 下 `rho=1`，额外 RAM 不提高 Goodput；B3/B4 的离散优化可以选最低合法点 20.83 GB/GPU，而 B1 的 proportional allocation 固定为每 GPU 125 GB。这个行为满足容量约束且是当前模型定义下的合法解，但它意味着 Neutral 结果不能回答真实作业是否能以 20.83 GB/GPU 达到同样训练速度。正式非 Neutral 实验必须继续使用统一 Oracle，并把该合成执行假设作为外部有效性限制。

## 11. 通过项与风险

### 已通过

- 历史输入和结果均保留；官方仓库未修改。
- B4 首个分叉、逐作业影响和真实 fallback 已定位。
- 当前五策略的 JCT、等待、资源小时、restart、wasted GPU-seconds、状态口径可统一复算。
- 新的五策略全矩阵已完成；B1、SY、B4、B5、B3 的确定性 summary、逐轮决策和训练轨迹均与修复后参考矩阵完全一致。
- B3 CPU/GPU 事件和 RAM-hours 均能从逐轮数据复算。
- 新旧完整矩阵均保持 20/20、无容量违规、Neutral rho=1、checkpoint 与 Goodput 单次倍率审计通过。
- 当前完整测试套件为 38/38 通过；PMP 测试仍只有既有 autograd `invalid value encountered in log` RuntimeWarning。

### 仍有风险

1. dirty HEAD 不能唯一标识代码；旧 run1 与修复运行即使 HEAD 相同，关键文件 hash 已不同。进入下一轮前仍需冻结干净、可引用的 commit。
2. 旧 manifest 未冻结完整依赖/求解器版本；本次审计补记了 Python 3.10.10、NumPy 1.23.5、SciPy 1.10.1、CVXPY 1.3.2、CyLP 0.91.6，但 SciPy 公共接口未暴露其内嵌 HiGHS 的独立版本号。
3. B3 平均 JCT 优势主要由单个长尾作业 `deepspeech2-31` 驱动，跨 workload 稳定性未知。
4. B4 旧 `fallback_count` 字段语义混合，必须使用本报告的拆分口径。
5. 20.83 GB/GPU 在 Neutral 下不受性能惩罚属于合成模型假设，不是真实 Saturn 测量。

## 12. 是否允许进入下一轮多 workload 合成实验

当前判定：**有条件允许进入下一轮预注册的多 workload 合成实验。**

已满足：独立完整五策略 Neutral 复跑、五策略逐作业 JCT/完成时间/配置/placement 的确定性、统一验证器以及统一 fallback 口径。进入下一轮前还必须冻结一个干净、可引用的实验 commit，并把依赖版本、逐文件 hash、求解器参数和本报告的统计口径写入每个运行 manifest。

该放行不改变研究边界：结果仍是 synthetic cross-hardware transfer；不得据此声称真实 Saturn 性能，也不得跳过预注册直接进行参数搜索或 160-job 实验。本轮没有运行 Medium、160-job、参数搜索或任何新模块。

## 13. 可复核产物

- 权威机器可读审计：`research_runs/sia_synergy/b3_reproducibility_audit_canonical/audit_summary.json`
- 审计脚本：`sia-joint-experiments/sia-simulator/audit_b3_reproducibility.py`
- 独立重复运行：`research_runs/sia_synergy/cpu_contention_20job_repro_run2/`
- 重复运行统一验证：`research_runs/sia_synergy/cpu_contention_20job_repro_run2/validation_summary.json`
- 五策略逐轮对比：`research_runs/sia_synergy/b3_reproducibility_audit_repeat_canonical/repeat_comparison.json`
- 重跑测试摘要：`research_runs/sia_synergy/b3_reproducibility_audit_repeat_canonical/test_summary.json`
- 五策略对比脚本：`sia-joint-experiments/sia-simulator/compare_five_policy_repeats.py`
- 审计 JSON SHA-256：`1BBEFDAA3C6C23A0FFD91A684AC731C9153B46C3F542190D2AC2175F30813313`
- 审计脚本 SHA-256：`88ECB8F419660F0E02D9A22BA11E05CB69A47C371A3B07CE63FC4A60A84BD78C`
- 旧 run1 manifest SHA-256：`9A84279624880AB8D729B69E42BB00063D2EF5699498E41D8073EA8559D47492`
- 修复后 manifest SHA-256：`58FAF6C4B7D3FCBBBE71BFEAAC55918D07DEC7FD6A01C7F3BE22D39BF7EB006F`
- 重复运行 manifest SHA-256：`51738F03B943D74DFF9E9B4E7E1E0FBF51CD841C2E3953D822999EC386BFD771`
- 重复运行 validation SHA-256：`9A4733FF30F76B03E340CBB48BDE5243AE787F66E6F646BD8E72B70EF9EF8C11`
- 五策略逐轮对比 JSON SHA-256：`6273A9AA0559744EC25B6DBC0B3A51F1627B6442C02CCCDDB9F227621E01137A`
- 重跑测试摘要 SHA-256：`0D5EDBFE3530EF51A5D1D55BCFDA9D93798B774C1B3866D228E2B1A85691D8F9`
- 五策略对比脚本 SHA-256：`CC49BD766F5ABA7D52E818EAAF4A659A82BD8D62B713C37CACCED54883F03C09`

## 14. 内存恢复后的独立五策略复验

### 14.1 命令与环境

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe run_cpu_contention_20job.py `
  --config joint_prototype\cpu_contention_20job_config.json `
  --workload joint_prototype\cpu_contention_20job_workload.csv `
  --synergy-models ..\..\research_sources\Synergy\simulator\models `
  --output ..\..\research_runs\sia_synergy\cpu_contention_20job_repro_run2 `
  --capacities cpu_constrained --responses neutral `
  --policies b1,sy,b4,b5,b3
```

依赖审计：Python 3.10.10、NumPy 1.23.5、SciPy 1.10.1、CVXPY 1.3.2、CyLP 0.91.6；Sia GPU intent 使用 CVXPY+CBC，B3/B4 有限列 MIP 使用 `scipy.optimize.milp` 的 HiGHS 后端。输入、profile、代码 hash 与修复后参考矩阵一致，manifest identity 对比为 `true`。

### 14.2 完整结果

| Policy | 完成 | Avg JCT (s) | P95 (s) | P99 (s) | Makespan (s) | 等待 (s) | GPU-h | CPU-core-h | RAM-GB-h | Restart | Wasted GPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 20/20 | 5102.95 | 28867.25 | 29266.25 | 31122 | 30000 | 91.362 | 91.362 | 11420.278 | 57 | 13775 |
| SY adapted | 20/20 | 6783.25 | 28142.55 | 29078.11 | 30263 | 57180 | 119.588 | 119.588 | 14417.784 | 4 | 10560 |
| B4 | 20/20 | 5056.25 | 28877.25 | 29033.05 | 30828 | 28500 | 91.773 | 91.773 | 1911.639 | 70 | 14835 |
| B5 | 20/20 | 5053.30 | 29048.05 | 29854.41 | 31812 | 29460 | 91.931 | 92.197 | 11474.652 | 62 | 14095 |
| B3 | 20/20 | 3831.05 | 14738.80 | 25968.56 | 29547 | 120 | 90.715 | 90.715 | 1889.593 | 73 | 13485 |

在这个冻结 cohort 中，B3 平均 JCT 相对 B1、SY adapted、B4、B5 的观测降幅分别为 24.9248%、43.5219%、24.2314%、24.1872%。这些差值与参考完整矩阵完全相同。它们是可重复的机制观测，不是正式跨 workload 性能结论。

### 14.3 确定性判定

五种策略均满足：

- deterministic summary 完全一致；
- 逐轮状态、GPU 配置、CPU/RAM 点、placement 与未分配集合完全一致；
- 逐轮 progress、checkpoint delay、training seconds 与完成事件完全一致；
- 逐作业 JCT、完成时间、等待、resource-hours、restart 和 wasted GPU-seconds 完全一致；
- B4 仍为 515 个 `OPTIMAL` 状态、0 个真实 fallback；473 条 `fallback_reason` 仍只是 admission 原因，不能计作 fallback；
- B3 仍为 494 个 `OPTIMAL` 决策轮、53 个节点级 CPU 饱和/GPU 空闲事件及 120 作业等待秒。

原始 JSON 不是逐字节相同，因为包含墙钟和 optimizer duration；这些字段不进入模拟确定性签名。所有五个策略的决策签名与执行签名均逐字节相同。

### 14.4 验证结果

`validate_cpu_contention_20job.py` 返回 `PASS`，全部检查为 true，包括：5/5 矩阵完整、20/20 完成、无容量违规、Neutral `rho=1`、Goodput 倍率一次、checkpoint/restore 不推进训练、B4/B5 配置冻结、B3 多 GPU 候选、实际排队与 CPU 饱和/GPU 空闲、原始 Sia 回归 hash 一致。

当前测试套件结果为 **38 tests，38 passed**。因此可重复性阻塞已经解除；剩余风险是代码版本冻结与合成模型的外部有效性，而不是本次 Neutral 五策略运行的不稳定性。

