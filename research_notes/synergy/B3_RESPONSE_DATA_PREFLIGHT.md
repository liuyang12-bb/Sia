# 响应数据预检：phi 来源、覆盖与 clipping

本报告只读分析已有配置、Sia profile CSV、Synergy CPU×RAM 矩阵和旧 run2 日志；没有运行 Job/Cluster 训练仿真，没有生成新 phi、替换缺失数据或调整执行响应。

最终机器可读结果：`research_runs/sia_synergy/correctness_stage1/response_preflight_final/response_preflight.json`，逐策略细表为同目录 `{b1,b4,b3_prop,b3}_coverage.json`。脚本为新树 `sia-simulator/preflight_stage1_response.py`；没有 import Job/Cluster，也没有调用性能参数拟合。首次输出因 NumPy 整数 JSON 序列化失败，日志保留；v2/v3 是逐步补齐统计维度的只读分析，最终版本不覆盖它们。

## 1. 沿用模型，未修改

冻结配置 hash：`667D7717CF8FF09EEE6A82C9979180615B58B92670BF91A5D4E5728F4218DE03`。

响应 manifest hash：`A46B5443F515DECA8FD992A7894BF67C1F93DFD7293AD6E985A7F395D5659DE5`。

CPU 网格：`[1,2,3,4,5,6,9,12,24]`；RAM 网格：`[20.83,62.5,125,187.5,250] GB`。参考配额为 `(3 cores/GPU,62.5 GB/GPU)`。

对当前候选的未归一化 Sia Goodput `g(j,c,t)`：

```text
r(j,c,t) = (g(j,c,t)/GPU(c)) / min_positive_candidate(g(j,c',t)/GPU(c'))
phi_c = clip(r, 0.5, 4)
phi_m = clip(sqrt(r), 0.5, 2)
numerator coordinates   = (CPU_per_GPU / phi_c, RAM_per_GPU / phi_m)
denominator coordinates = (3 / phi_c, 62.5 / phi_m)
rho = [T_bilinear(clipped numerator) / T_bilinear(clipped denominator)]^0.5
```

这里有两层不同的 clip：phi 需求倍率本身的范围约束，以及响应查询坐标落到源矩阵边界。两者都没有改动。对实际正候选比率，`r>=1`，因此实际 `phi_c∈[1,4]`、`phi_m∈[1,2]`。缺文件/非法矩阵仍硬错误，未补值。

## 2. 哪些数据可以作为冻结 phi 的来源

| 数据 | 实际覆盖 | 能否直接作为独立冻结 phi | 未来信息/路径依赖 |
|---|---|---|---|
| Sia `traces/<application>/placements-{aws,dgx-ext,rtx}.csv` 与 scalability CSV | 5个映射应用×3 GPU类型，共30文件；AWS 1/2/4 GPU、另两类1/2/4/8 GPU的单节点几何均有记录，共55个 application/type/config 几何组合 | **不能直接**。提供吞吐/profile 原料，仍缺统一 batch、梯度/效率参考状态和标定定义 | 离线画像可作为预先固定先验；不能按本次策略最终轨迹选择参考点 |
| Sia validation/batch 数据和 `Job.seed_profiles(1,8)` | 有离线训练/效率相关资料；seed 只拟合 throughput/perf 参数，不等于已经有完整当前梯度状态 | 需先决定固定参考 epoch/batch 及数据许可口径，不能自动取未来实际训练状态 | 离线资料与当前在线观测不同；用未来真实状态决定当前需求会泄漏，需避免 |
| 旧 run2 当前 selected config 的 provenance | B1/B4 各53、B3 63个不同 job/config 有 phi；其余配置大面积缺失 | 只适合诊断，不适合事后选取“有利”的统一冻结表 | 由过去实际训练和候选集合决定；同一 key 多轮变化、策略路径依赖 |
| 旧 candidate_summary | B1/B4/B3 各记录220个 job/config ID；B3-PROP记录60个冷启动ID | 只有 ID/来源类别，不含全部 raw Goodput/phi，不能恢复完整数值表 | 无法从选中列推断所有未选配置 |
| Synergy 官方 `models/*_1.py` | 12个9×5正数矩阵；当前映射用其中5个 | 矩阵给 CPU/RAM 响应形状，不给 Saturn 跨GPU需求倍率 phi | 原矩阵与跨应用映射必须分开；后者是 synthetic transfer |

“有几何记录”不意味着每个 batch/epoch 都有测量，也不意味着真实 Saturn 联合 CPU/RAM 画像覆盖已闭合。详细文件 hash、行数、seed 子集行数、batch范围、node/GPU组合都在 JSON 的 `sia_source_coverage` 中。

当前映射保持不变：CIFAR10→alexnet；DeepSpeech2→deepspeech；ImageNet→res50；BERT→bert；YOLOv3→mobilenet。w10_late 没有 BERT 作业，但本次仍核验整个5应用映射的源文件。

全部官方矩阵数值范围：alexnet `0.1–2.94`；deepspeech `0.1–1`；m5 `0.1–2.5`；mobilenet `0.1–1.55`；res18 `0.1–2.4`；res50 `0.1–1.15`；shufflenet `0.1–2.89`；bert/gnmt/lstm/transformer/vae 均为 `1–1`。这只是官方源码矩阵中的数据范围，不把恒1矩阵或模型对应关系重新宣称为 Saturn 实测。

## 3. 分子与基准坐标的越界统计

### 3.1 历史 selected 列观测：按 job-round 计数

| 策略 | 观测列数 | 分子CPU越界 | 分子RAM越界 | 基准CPU越界 | 基准RAM越界 | 任一越界 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 927 | 901 | 0 | 439 | 0 | 901 |
| B4 | 925 | 899 | 0 | 438 | 0 | 899 |
| B3-PROP | 0 | — | — | — | — | — |
| B3 | 932 | 848 | 0 | 597 | 0 | 899 |

B3-PROP 的“0观测”不是“没有 clipping”，而是6000轮未选择任何列；没有记录可用于此表的 selected 配额查询。

### 3.2 配置级：按唯一 `(job, GPU-config)` 计数

| 策略 | 有phi的配置 / 枚举ID | 完全缺phi的配置 | 曾分子CPU越界的配置 | 曾基准CPU越界的配置 |
|---|---:|---:|---:|---:|
| B1 | 53 / 220 | 167 | 36 | 27 |
| B4 | 53 / 220 | 167 | 36 | 27 |
| B3-PROP | 0 / 60 | 60 | 未知 | 未知 |
| B3 | 63 / 220 | 157 | 42 | 47 |

同一配置可能部分轮越界、部分轮不越界；“曾越界”不等于永久不可行。两类 RAM 的配置级越界计数在这个观测子集中均为0。

### 3.3 条件候选级网格预检：不是完整原 MIP 候选复原

只对已经观察到的唯一 `(job, config, phi_c, phi_m)` 状态，展开原有 CPU/RAM 网格，检查容量与 proportional progress floor，统计 Pareto 剪枝前的候选点。未画像配置不填充、不参与分母。

| phi 来源策略 | 状态×配额原始点数 | 容量可行 | progress floor 后候选 | 任一坐标越界 | 分子CPU | 基准CPU |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 12000 | 2065 | 1191 | 791 | 755 | 312 |
| B4 | 12000 | 2065 | 1191 | 791 | 755 | 313 |
| B3 | 12960 | 3571 | 2172 | 1745 | 1424 | 1349 |

这些条件网格中分子RAM/基准RAM越界计数均为0。进度 floor 判断以相同候选的公共 Sia 因子约掉后的 rho 比较；脚本的 unit-goodput 仅用于此代数比较，不是补造 Sia profile，也不是训练仿真。此表不宣称恢复了原求解器实际列顺序或全部候选；B1/B4/B3 不能被当作相同覆盖样本比较越界比例。

## 4. 严格禁止 clipping 会失去什么

当前三种节点的比例执行点均为 `(CPU=1, RAM=125)`，RTX 的原0.75 CPU/GPU由既有 profile floor 升到1。于是：

- 比例分子的 CPU 坐标为 `1/phi_c`，在实际 `phi_c∈[1,4]` 下，严格落在源矩阵CPU网格内 **当且仅当 phi_c=1**。
- 基准 CPU 为 `3/phi_c`，要求 `phi_c<=3`；所以 `phi_c>3` 时即使给更多CPU让分子合法，基准仍会越界。
- 比例分子 RAM 为 `125/phi_m∈[62.5,125]`；基准 RAM 为 `62.5/phi_m∈[31.25,62.5]`，此范围本身没有RAM越界。

因此，“直接禁止 clipping”会剔除 **所有 phi_c>1 的 proportional 配置状态**，不是一个中性的日志开关。不能在未批准的数据定义下自动启用。

对历史已观测 phi 状态作反事实比例查询：

| 来源 | 唯一 job/config/phi 状态 | strict clipping 会剔除 | 保留 |
|---|---:|---:|---:|
| B1 | 300 | 280 | 20 |
| B4 | 300 | 280 | 20 |
| B3 | 324 | 304 | 20 |

每个策略的详细 `strict_proportional_on_observed_phi_states.by_job` 列出具体失去/保留的 config ID。所有20个作业在其**整个历史观测子集**中至少有一个曾合法的状态；这不保证同一调度轮仍有合法候选，更不能断言没有作业会失去全部候选。

例如 B3 的 `deepspeech2-159` 观测到78个 config/phi 状态，只有1个通过严格比例边界；失去的状态涉及 `rtx:1:1`、`rtx:1:2`、`rtx:1:4`。其 `rtx:1:1` 在早期另一个phi状态又曾可保留。同一个 ID 同时出现于 lost/surviving 列表是路径变化，不是统计错误。

**哪些作业会全局失去全部合法配置，目前不能可靠判定。** 每策略仍缺157–167个枚举配置的完整phi，B3-PROP缺全部60个。这就是必须补齐的覆盖缺口；没有自动补 phi=1，也没有删除任何作业。

## 5. 在线路径依赖与未来信息

| 策略 | 同一 job/config 出现多个phi | 同一 job/config/CPU/RAM 出现多个rho |
|---|---:|---:|
| B1 | 21 | 21 |
| B4 | 21 | 18 |
| B3 | 22 | 13 |

当前代码不是读取未来到达或未来真实完成时间来构造phi；但它使用已训练得到的 online profile/梯度状态，以及当轮可用候选中的最小 per-GPU Goodput。因而它依赖当前训练路径、候选范围与 batch 选择。用旧结果中挑出的phi作为“固定数据”会把策略结果反馈进新实验，不能未经预注册就这样做。

不能用“存在 offline placement 文件”直接证明已经得到唯一合理 phi。还缺：固定的参考 batch/梯度状态、跨GPU需求缩放假设、未覆盖项的处理规则，以及 response numerator 和 denominator 的边界方案。

## 6. 需要研究者决定的数据方案

本轮不选新模型。下一阶段至少需把以下内容作为同一个“响应边界模型”决定：

1. 采用哪种策略无关参考状态，将现有 Sia throughput/profile 转成冻结需求倍率；给每个 job/config 输出来源、batch、参考状态、适用范围和缺失标记。
2. 对分子越界与基准越界分别选择预注册处理：补测/补数据、有限支持并明确不可行，或经论证的边界/外推假设。不能默认选择对 B3 有利的方式。
3. 先生成完整覆盖表和 lost-all-config 作业列表，研究者确认后才接入执行模型；不靠删作业、改容量或默认 phi=1 消除缺失。

本预检没有给出新的执行参数。所有跨应用、跨GPU需求迁移仍是 **synthetic cross-hardware transfer，不是真实 Saturn CPU/RAM 测量**。
