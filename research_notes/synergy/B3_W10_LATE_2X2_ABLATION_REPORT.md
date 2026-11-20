# B3 `w10_late` Medium 2×2 机制消融报告

## 1. 结论先行

本轮按预注册条件实现并执行了 B1、B4、B3-PROP、B3 四个实验单元，但最终状态是 **`INCOMPLETE_CENSORED_MATRIX`**，不是完整的 2×2 因子实验。

- B1、B4、B3 均完成 20/20 个作业。
- B3-PROP 在 6000 轮统一观察窗内完成 0/20；6000 轮均有非空合法候选集，MILP 均返回 `OPTIMAL`，但每轮都选择合法空集，20 个作业的训练进度均为 0。
- 因而 B3-PROP 没有可定义的 JCT，不能插补，也不能把它当作无穷大 JCT。B3-PROP 对 B1、B3 对 B3-PROP、完整 GPU 主效应、完整辅助资源主效应和交互效应均不可识别。
- 仍可识别的简单效应是：B4 相对 B1 平均 JCT 减少 4.95 s（0.13%）；B3 相对 B4 减少 985.10 s（26.21%）。后一差异的 90.08% 来自 `deepspeech2-159`；去除该作业后，B3 相对 B4 的诊断性均值改善降为 102.89 s（3.74%）。
- 这说明当前 B3 目标与 Medium 响应下，GPU 候选选择和辅助资源选择存在不可忽略的耦合；但由于一个因子单元失去活性，本轮**不能**把 B3 的收益因果归结为 GPU 选择、辅助资源优化或二者交互中的任何一个完整因子效应。

因此，当前结果**不支持直接扩展到九个 cohort**。下一步应先预注册并审查一个对 B3 与 B3-PROP 对称的活性/admission 语义；不能只为 B3-PROP 强制分配，也不能事后修改 `lambda_n`、响应曲线或目标权重。

> **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 2. 工程保护与冻结状态

- 隔离工作树：`sia-joint-medium-frozen`
- 分支：`exp/b3-medium-nonneutral`
- HEAD：`be03f95c7bc641cbef0e2a4452bc4c7a3b5404e4`
- 本任务没有创建或推送 Git commit。
- 没有修改 `official-artifacts/`、Synergy 官方源码或历史结果。
- 历史失败证据保留在 `research_runs/sia_synergy/b3_w10_late_ablation_2x2/run1/`；正式分析使用全新的 `run2/`。
- tracked diff 仅涉及 B3-PROP 接入和删失结果可序列化，共 82 insertions、5 deletions；未执行 reset、clean 或覆盖历史产物。

冻结哈希：

| 对象 | SHA-256 |
|---|---|
| preregistration | `553169E97081D0C57147A4FD108BEFC68A1FAFCAC6B4A77C5ECB0A29464B0CB1` |
| frozen config | `667D7717CF8FF09EEE6A82C9979180615B58B92670BF91A5D4E5728F4218DE03` |
| workload | `719267711948607055CAD9BA3F5966D63C65B4913693375D21B5F4E1F3675949` |
| Medium response manifest | `A46B5443F515DECA8FD992A7894BF67C1F93DFD7293AD6E985A7F395D5659DE5` |
| frozen tracked diff | `1E152D520625306320FEC8DB3A0DCE9FF69E62F45D3138F1A4A8E0B913224CE7` |

冻结时环境为 Windows 10 build 26200、Python 3.10.10、NumPy 1.23.5、SciPy 1.10.1、CVXPY 1.3.2、CyLP 0.91.6；CVXPY 可见 CBC、SCIPY 等求解器。联合 MIP 使用 SciPy MILP，`time_limit=30 s`、`mip_rel_gap=0`；原始 Sia 路径使用 CBC，8 threads、1% allowable gap。

## 3. 预注册 2×2 定义

| 策略 | GPU 配置因素 | CPU/RAM 与 placement 因素 |
|---|---|---|
| B1 | 原始 Sia 决策 | 确定性 proportional |
| B4 | 原始 Sia 决策并冻结 | 辅助资源与 placement 优化 |
| B3-PROP | 在所有合法 B3 GPU 候选中选择 | 确定性 proportional，仅一个辅助资源点 |
| B3 | 在所有合法 B3 GPU 候选中选择 | GPU、CPU、RAM、placement 联合优化 |

B3-PROP 对每个合法 `(GPU config, placement)` 只生成一个辅助资源点：

```text
cpu_per_gpu = max(node_cpu / node_gpu, profile_grid_min_cpu)
ram_per_gpu = max(node_ram / node_gpu, profile_grid_min_ram)
```

本集群 AWS 为 4 GPU/4 CPU/500 GB，DGX-ext 为 8 GPU/8 CPU/1000 GB，RTX 为 8 GPU/6 CPU/1000 GB。profile CPU floor 将 RTX 的 0.75 core/GPU 提升为 1 core/GPU；三类节点的 proportional RAM 均为 125 GB/GPU。优化器不能为提高 Goodput 选择其他 CPU/RAM 点，但仍可在合法 GPU 类型、GPU 数量和 placement 间选择。

四个单元共同使用：`w10_late` 的 job 140–159、Medium `alpha=0.5`、60 s 调度周期、6000 轮上限、相同 Sia Goodput、模板映射、checkpoint/restart 语义和 2 GiB 内存门槛。冻结时可用内存 2.887 GiB；四次正式启动前均超过门槛。

交互收益预注册为：

```text
I_gain = (JCT_B3-PROP - JCT_B3) - (JCT_B1 - JCT_B4)
       = JCT_B3-PROP + JCT_B4 - JCT_B3 - JCT_B1
```

`I_gain > 0` 表示辅助资源优化在联合 GPU 选择开启时节省的 JCT 更多。由于 B3-PROP 删失，本轮 `I_gain` 不可计算。

## 4. 实现变更

### 4.1 调度代码

- `joint_prototype/joint_scheduler.py::build_proportional_columns()`：为每个合法 GPU 配置和 placement 生成唯一 proportional 列；继续执行合法配置、逐节点容量和 checkpoint config/node lock 检查。
- `joint_prototype/joint_integration.py::JointSiaPolicy._b3_result()`：新增 `b3_prop` mode；复用 B3 的 Sia Goodput、restart factor、公平参数 `p=-0.5`、`lambda_n=1.1` 及同一三资源 MIP，只替换候选列集合。
- `run_cpu_bottleneck_smoke.py::summarize()`：将从未开始作业的 `num_restarts=None` 序列化为 0。此改动只保证删失运行可以保存，未改变调度或时间推进。

### 4.2 新增运行与审计文件

- `joint_prototype/b3_w10_late_ablation_preregistration.json`
- `freeze_b3_w10_late_ablation.py`
- `run_b3_w10_late_ablation_policy.py`
- `validate_b3_w10_late_ablation.py`
- `analyze_b3_w10_late_ablation.py`
- `tests/test_b3_prop_ablation.py`

运行后只对验证器/分析器增加了“保留并分析删失单元”的能力；四个 `result.json`、调度器目标和冻结输入未重写。冻结时与最终分析时脚本哈希均记录在 `audit_manifest.json`。

## 5. 测试与正确性审计

完整测试命令：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe `
  -m unittest discover -s tests -p test_*.py -v
```

结果为 **47/47 passed，0 failure，0 error，7.304 s**。新增测试覆盖：

1. B3-PROP 每个 GPU config/placement 只有一个确定性 proportional 资源点；
2. 小型可行实例中 B3-PROP 能在多个 GPU 候选间合法选择；
3. checkpoint lock 同时约束 GPU 配置与节点；
4. 从未启动、达到观察窗的作业仍能完整序列化。

运行验证器的 15 项检查中，除“全部完成”和“未触发 round limit”外均通过：

- 四策略输入与哈希可追溯；
- 每次提交均有原子审计；
- GPU/CPU/RAM 逐节点容量违规均为 0；
- Medium 响应公式逐次吻合；
- `effective_goodput = raw_goodput × rho` 只应用一次；
- checkpoint/restore 先于训练推进，progress 单调；
- B4 保持 Sia GPU config；
- B3-PROP 资源严格 proportional，并实际枚举多个 GPU 候选；
- B3-PROP 使用预期联合 MIP 路径。

唯一现存 RuntimeWarning 是既有 PMP 测试中的 autograd `invalid value encountered in log`；测试仍通过，与本轮结果无关。

## 6. 四单元结果

| 策略 | 完成 | Avg JCT (s) | Median | P95 | P99 | Makespan (s) | 轮数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 20/20 | 3763.20 | 428.0 | 21260.85 | 22565.77 | 30902 | 516 |
| B4 | 20/20 | 3758.25 | 426.5 | 21234.25 | 22560.45 | 30902 | 516 |
| B3-PROP | **0/20** | — | — | — | — | — | **6000（删失）** |
| B3 | 20/20 | 2773.15 | 474.0 | 12397.95 | 19069.99 | 27159 | 453 |

每个 cohort 只有 20 个作业，P95/P99 只是该固定样本的描述量，不是高精度总体尾延迟估计。

| 策略 | GPU-h | CPU-core-h | RAM-GB-h | Restart | Checkpoint/restore (s) | Wasted GPU-s | 总等待 (s) | rho 范围 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B1 | 79.314 | 79.314 | 9914.306 | 39 | 4405 | 17335 | 20700 | 0.548–1.512 |
| B4 | 79.221 | 80.854 | 9875.486 | 39 | 4405 | 17335 | 20700 | 0.548–1.512 |
| B3-PROP | 0 | 0 | 0 | 0 | 0 | 0 | 7113960 | 无训练样本 |
| B3 | 66.381 | 73.514 | 8270.486 | 50 | 4815 | 15845 | 120 | 0.738–1.804 |

B3 相对 B4 的描述性变化包括：makespan 减少 3743 s（12.11%），P95 减少 8836.30 s，P99 减少 3490.46 s，GPU-hours 减少 12.84，RAM-GB-hours 减少 1605.00；但 median 从 426.5 s 增至 474.0 s，restart 从 39 增至 50，checkpoint/restore 从 4405 s 增至 4815 s。这不是“所有作业均改善”。

## 7. B3-PROP 删失根因

该失败不是 `NO_COLUMNS`，也不是 `result.x is None`：

- 6000/6000 轮均有非空候选集；单轮最多 60 个候选列；
- 6000/6000 轮求解状态为 `OPTIMAL`；
- 6000/6000 轮 selected 为空、目标值为 0；
- 20 个作业 progress 均为 0。

B3/B3-PROP 复用的变换为：

```text
adjusted = effective_goodput × restart_factor
powered  = round(adjusted^p, 2),  p = -0.5
utility  = -(powered - lambda_n) = lambda_n - powered
lambda_n = 1.1
```

MIP 最大化列 utility，并只施加“每作业至多选一个”和资源容量上界；空分配目标为 0。忽略两位小数边界时，列只有在 `adjusted > (1/1.1)^2 ≈ 0.826` 才取得正增量效用。由每轮存在候选却以目标 0 选择空集可知，当轮全部 proportional 列的变换效用均非正；而 B3 的可变辅助资源列可跨过该阈值并维持活性。

因此，B3-PROP 揭示的是当前公平目标、no-allocation penalty 与 Medium fixed-reference 资源点之间的**活性耦合**。它不是一个可用的 JCT 对照单元。直接把空 audit 改为 valid、把空分配解释为完成，或只对 B3-PROP 强制至少分配一个作业都会改变预注册语义，均未执行。

还需区分 B1 与 B3-PROP 的 admission 口径：B1 的 GPU 决策由原始 Sia 基于归一化 Sia Goodput 作出，CPU/RAM `rho` 只在统一执行模型中影响实际推进；B3-PROP 则像 B3 一样把 fixed-point `effective_goodput` 放入联合候选目标。因此“原始 Sia GPU”与“B3 联合 GPU”这个因素同时改变了候选选择阶段是否看到辅助资源响应。B3-PROP 的全等待正是这种定义下的真实机制结果，也意味着这两个因素在当前实现中并非统计意义上完全正交。

## 8. 逐作业 JCT 与归因

| Job | B1 | B4 | B3-PROP | B3 | B4 对 B1 节省 | B3 对 B4 节省 |
|---|---:|---:|---:|---:|---:|---:|
| cifar10-140 | 343 | 341 | — | 438 | 2 | -97 |
| cifar10-141 | 313 | 311 | — | 410 | 2 | -99 |
| cifar10-142 | 283 | 281 | — | 378 | 2 | -97 |
| cifar10-143 | 316 | 314 | — | 413 | 2 | -99 |
| cifar10-144 | 316 | 314 | — | 534 | 2 | -220 |
| cifar10-145 | 302 | 300 | — | 455 | 2 | -155 |
| cifar10-146 | 447 | 442 | — | 394 | 5 | 48 |
| cifar10-147 | 420 | 416 | — | 493 | 4 | -77 |
| cifar10-148 | 324 | 322 | — | 419 | 2 | -97 |
| yolov3-149 | 11124 | 11092 | — | 9767 | 32 | 1325 |
| cifar10-150 | 290 | 288 | — | 399 | 2 | -111 |
| cifar10-151 | 468 | 465 | — | 432 | 3 | 33 |
| cifar10-152 | 431 | 429 | — | 429 | 2 | 0 |
| yolov3-153 | 12864 | 12859 | — | 11959 | 5 | 900 |
| imagenet-154 | 21175 | 21147 | — | 20738 | 28 | 409 |
| cifar10-155 | 425 | 424 | — | 605 | 1 | -181 |
| cifar10-156 | 1200 | 1198 | — | 821 | 2 | 377 |
| cifar10-157 | 626 | 625 | — | 603 | 1 | 22 |
| cifar10-158 | 705 | 705 | — | 631 | 0 | 74 |
| deepspeech2-159 | 22892 | 22892 | — | 5145 | 0 | 17747 |

B4 相对 B1 为 18 个改善、2 个相同、0 个退化。B3 相对 B4 为 9 个改善、1 个相同、10 个退化。因此 B3 的平均值收益明显依赖长作业，而不是多数作业普遍改善。每个作业在四策略下的 completed、JCT、completion time 和 waiting time 均保存在 `ablation_analysis.json::per_job`；逐轮配置、placement、CPU/RAM、rho、raw/effective Goodput 和 checkpoint 时间保存在各策略 `result.json::joint_round_logs`。

## 9. `deepspeech2-159` 机制追踪

| 策略 | JCT | 完成时刻 | 等待 | 训练秒 | Restart | Checkpoint | 加权 raw GP | 加权 effective GP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 22892 | 30902 | 18960 | 3827 | 2 | 75 | 11.884 | 7.456 |
| B4 | 22892 | 30902 | 18960 | 3827 | 2 | 75 | 11.884 | 7.456 |
| B3-PROP | — | — | 351960 | 0 | 0 | 0 | 0 | 0 |
| B3 | 5145 | 13155 | 60 | 4980 | 2 | 75 | 7.297 | 5.730 |

B1/B4 使用 `rtx:1:1 → rtx:1:4 → dgx-ext:1:8`，CPU 恒为 1 core/GPU；B4 的 RAM 点为 62.5 或 125 GB/GPU。B3 使用 `rtx:1:1 → rtx:1:2 → rtx:1:4`，CPU 为 1/3/6 core/GPU、RAM 为 62.5/125 GB/GPU，rho 范围 0.738–1.789。

值得注意的是，B3 的训练期加权 effective Goodput 反而低于 B4；收益主要来自等待由 18960 s 降到 60 s，而不是该作业每个训练秒都更快。`deepspeech2-159` 对 B3−B4 总 JCT 节省贡献 17747 job-seconds，占总节省 19702 job-seconds 的 **90.08%**。去除此作业后，B1/B4/B3 的诊断性平均 JCT 分别为 2756.42/2751.21/2648.32 s，B3 对 B4 只改善 3.74%。该去除结果仅用于诊断，不替代全 20 作业主结果。

## 10. 主效应与交互效应

| 对比 | 绝对平均 JCT 变化 | 相对变化 | 是否可识别 |
|---|---:|---:|---|
| B4 vs B1：Sia GPU 下开启辅助优化 | -4.95 s | -0.13% | 是 |
| B3-PROP vs B1：proportional 下开启联合 GPU 选择 | — | — | 否，B3-PROP 删失 |
| B3 vs B4：辅助优化下开启联合 GPU 选择 | -985.10 s | -26.21% | 是 |
| B3 vs B3-PROP：联合 GPU 下开启辅助优化 | — | — | 否，B3-PROP 删失 |
| 2×2 interaction | — | — | 否，B3-PROP 删失 |

这里负号表示 JCT 降低。B3−B4 是在“辅助优化开启”切片上的 GPU 选择简单效应；不能用它代替跨两种辅助资源水平平均后的 GPU 主效应。类似地，B4−B1 也不能代表联合 GPU 水平下的辅助资源效应。

## 11. 运行产物与复核入口

正式运行根目录：`research_runs/sia_synergy/b3_w10_late_ablation_2x2/run2/`

- `frozen_manifest.json`：冻结配置、依赖、源文件和输入哈希；
- `preregistration.json`、`frozen_config.json`、`workload.csv`：冻结输入快照；
- `policies/{b1,b4,b3_prop,b3}/result.json`：逐轮执行证据；
- `policies/*/policy_manifest.json`、`console.log`：命令、内存、状态和日志；
- `validation_summary.json`：统一正确性审计；
- `ablation_analysis.json`：指标、逐作业表、`deepspeech2-159` segments 和删失诊断；
- `test_summary.json`：47 项测试；
- `audit_manifest.json`：执行结果与最终分析哈希。

四个结果 SHA-256：

| 策略 | SHA-256 |
|---|---|
| B1 | `1D5D95D1C435BF2286060952C4C8626ED1986476AA858A5286A0E6CBE1216FD3` |
| B4 | `23A71F89477913CF8871831402209902D5FF9AD36BC36A106B12330C60882962` |
| B3-PROP | `E36BFBA0A770B639584BD54EBFA34F6E953ED6F2C0491827B33D483647E913DC` |
| B3 | `FB3878006D80EB60312EA2DF35787BF195A14082B8A85691CF32C4A16B864335` |

## 12. 局限与下一步判断

1. CPU/RAM 矩阵单元来自 Synergy，但 application 映射、跨 GPU 迁移、固定参考点、插值和与 Sia Goodput 的乘法均为合成假设。
2. 单一 `w10_late` cohort 不能支持跨 workload 稳定性结论。
3. B3-PROP 的全等待使完整因子分解失败；这比单纯 JCT 数字更重要，说明当前算法的活性依赖辅助资源候选。
4. B3 对 B4 的平均收益高度集中在一个 DeepSpeech2 长尾作业，同时 10/20 作业退化、median 变差、restart 增加。
5. 本轮没有修改目标权重、Medium `alpha`、模板映射、容量或求解参数，也没有运行其他 cohort。

**是否值得扩展到全部九个 cohort：当前不值得直接扩展。** 应先对以下二选一研究口径做设计审查和预注册：

- 保留当前目标，并把“proportional 条件下无活性”作为机制结论；此时不再声称能完成标准 2×2 JCT 因子分解；或
- 引入对 B3 与 B3-PROP 完全对称的最低活性/admission 规则，再作为新算法版本重跑四单元。该规则会改变策略语义，必须单独审查，不能在本报告结果上追补。

本报告不宣称 B3 在真实 Saturn 集群上有性能收益，也不宣称已经识别 GPU 选择与辅助资源优化各自的独立因果贡献。
