# B3 多 workload Neutral 五策略对照实验报告

## 1. 结论先行

本轮最终状态为 **`MULTI_WORKLOAD_NEUTRAL_SMOKE_VALID`**：3 条官方 Saturn trace 的 Early/Middle/Late 共 9 个 20-job cohort、5 种策略，共 **45/45** 次运行全部完成；每次均为 20/20 作业完成，9/9 统一验证为 `PASS`，38/38 测试通过。所有运行均使用 `cpu_constrained` 合成节点容量和 Neutral 响应，观测到的 `rho` 严格为 1，因此 B3 的差异来自 GPU 配置、admission 与节点放置的联合决策，而不是 CPU/RAM 性能倍率。

以 cohort 为分析单位，B3 的平均 JCT 相对结果如下：

| 对照 | B3 胜/平/负 cohort | 相对变化中位数 | 范围 | 最差 cohort |
|---|---:|---:|---:|---|
| B1 | 8/0/1 | +7.51% | -1.57% ～ +32.50% | `w5_early`，-1.57% |
| SY adapted | 9/0/0 | +58.20% | +24.76% ～ +76.31% | `w5_late`，+24.76% |
| B4 | 8/0/1 | +7.51% | -1.20% ～ +33.15% | `w5_early`，-1.20% |
| B5 | 9/0/0 | +7.48% | +1.12% ～ +32.56% | `w5_early`，+1.12% |

“+”表示对照平均 JCT 减去 B3 后为正，即 B3 更低。结论不是“B3 在所有 workload 都获胜”：`w5_early` 中 B3 分别比 B1/B4 高 1.57%/1.20%；部分大幅收益又明显受单个 DeepSpeech2 长尾作业影响。B3 对 SY adapted 的优势也不能解释为对未经修改的官方 Synergy 的优势，因为 SY 包含本项目为异构 GPU 增加的固定 demand/type 映射。

本轮支持进入下一阶段的**预注册非 Neutral 合成机制实验**，但不支持真实集群性能或 SOTA 结论。下一阶段必须继续冻结 workload、模板映射和响应档位，并把长尾敏感性作为主审计项。

> **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 2. 现场保护、冻结版本与两次尝试

- 实验分支：`exp/b3-multiworkload-neutral`
- 初始冻结 commit：`85431a3b054359443e9b8c5c843111202919f7f0`
- 最终执行 commit：`e0dd17538851dc100be6170b21402bce18294628`
- 最终隔离 worktree：`D:\科研启航\Sia\sia-joint-multiworkload-frozen-v2`
- 最终产物：`research_runs/sia_synergy/b3_multiworkload_neutral/attempt2/`
- 官方 `official-artifacts` 检查时仍为 `main...origin/main` 且干净。

首次正式尝试在父目录中完成了 15/45 次运行，但旧验证器错误地把“每个 cohort 都必须同时出现未分配作业与 CPU-saturated/GPU-idle”当成正确性条件；`w1_late` 的全部容量、进度、Neutral 和 checkpoint 检查均通过，只是该观测未触发。首次尝试及原验证结果没有覆盖，保留在父目录；修复后的 post-hoc 记录也单独保留。

commit `e0dd175` 只修改：

- `run_b3_multiworkload_neutral.py`：多 workload 运行显式使用 `record-only` contention 观测模式；
- `validate_cpu_contention_20job.py`：将 cohort-specific contention 现象改为记录项，并支持独立验证输出目录。

该 commit 没有修改调度器、目标函数、容量、workload、响应曲线或求解参数。最终 `attempt2` 从空目录重新运行全部 45 项，没有复用首次尝试的策略结果。

运行前和每个策略前共记录 46 次内存采样，范围为 **3.052–4.370 GiB**，均高于 2 GiB 门槛。总编排墙钟时间为 **5343.962 秒**，策略严格顺序执行。

## 3. 输入预注册与哈希

三条官方输入均存在且各含 160 个作业：

| Trace | SHA-256 |
|---|---|
| `workload-1.csv` | `D80DC93782785AAB449D3109AA118F123EA799D891DF0AFE2B9DEE483EDA9468` |
| `workload-5.csv` | `71AA3B0C4F4F5D69B6BFCDBB33772A57BD9C15F4262A132250E8CC86F1AADE6B` |
| `workload-10.csv` | `FECC829DC1899159D987ED0595329439D223B7206550BDB641EAA57DD29262B0` |

提取规则在任何新策略运行前冻结：按到达时间稳定排序；同到达时间保留原始行顺序；Early=1–20、Middle=71–90、Late=141–160；每个 cohort 独立从空集群启动，把首个 arrival 平移为 0，并保持相对间隔、任务顺序、应用、GPU demand、batch 与工作量不变。

| Cohort | 稳定位置 | 首/末作业 | 提取 workload SHA-256 | 派生 config SHA-256 |
|---|---:|---|---|---|
| `w1_early` | 1–20 | `cifar10-0` / `cifar10-19` | `6F6AED923A37A066373E891A029682FEF0FB2C6E04973CB041A8C96E35D77E3E` | `15FAA75742C9ACC24D15BF319AD75710AC9D08CE4311F354E69A8B2B802B5C8A` |
| `w1_middle` | 71–90 | `cifar10-70` / `imagenet-89` | `F504037755969FB2B6FB8F13630E3DABF97FFF846290DA468C025905D98187B7` | `645E73D3217F9FE826672224A6BB246DD2723E28DBB0B08DFBFCA1ACDE2E2FBA` |
| `w1_late` | 141–160 | `cifar10-140` / `bert-159` | `BB5E0BDBD7B54CFD944D6854651AE9DA3AD131ACED6592CF4D18CA9DB6945A3A` | `85941A24CAD82E03BD1EFB99532DCC0979CA525487FC75B127DE3E64E301DCAF` |
| `w5_early` | 1–20 | `cifar10-0` / `cifar10-19` | `3FC7BDF3CD4E7E46AD7CA3447448B39EC7483418A6FEA13974D07E8F000DBF82` | `E7243186187FD5DA20351E9C3AE28579D4AECEA32A96686590D17394A1A21336` |
| `w5_middle` | 71–90 | `deepspeech2-70` / `cifar10-89` | `33E18019362ED683608D0BACDD3EF018EA7C40B1088BA6CDE9BC5FC18CECB4D0` | `366750ABED8F068F76D05C3BB4185BECF1BAA6C05AF6E3C5255E131280800495` |
| `w5_late` | 141–160 | `cifar10-140` / `cifar10-159` | `59433BC932BA55AD57C0698F1B508DF39DCC8552B805A3E4124B7E9E23AF4EF2` | `EDC5130DC3C22D9B949AE5860C5C49F27A8200050981E1118F10DEF996F739A1` |
| `w10_early` | 1–20 | `cifar10-0` / `cifar10-19` | `EE163619135530E274DCE928C9D29CEF340A0ADE08FAD4B5A5715662DF3BF513` | `DB677D5C646B4EA6E6C5057B8E46C092C6F5B2BD5BA21832686BCF26AE084966` |
| `w10_middle` | 71–90 | `cifar10-70` / `cifar10-89` | `219A77FE23DAE6ADD334BD6815A647976C51EE0C91DFBAF4D9DE017A8C74E8A2` | `A4D923D08143138E372EF5F91985E3CC6A3D7CA83BBA4BF88F60698A751EF105` |
| `w10_late` | 141–160 | `cifar10-140` / `deepspeech2-159` | `719267711948607055CAD9BA3F5966D63C65B4913693375D21B5F4E1F3675949` | `2286786565BE339F44FD5BFD622E2584EB9D8A89CDFDFD3843FB3A3662CE00CA` |

应用到 Synergy 模板的映射和 SY 的 GPU type/demand 映射均在各 cohort manifest 中逐作业冻结。它们是 deterministic synthetic transfer，不是 Saturn CPU/RAM 实测。

## 4. 环境与固定矩阵

| 组件 | 版本/设置 |
|---|---|
| OS / Python | Windows 10 build 26200 / Python 3.10.10 |
| NumPy / SciPy | 1.23.5 / 1.10.1 |
| CVXPY / CyLP | 1.3.2 / 0.91.6 |
| Sia GPU intent | CVXPY + CBC；8 threads；允许 0.01% gap |
| B3 finite-column MIP | `scipy.optimize.milp`，HiGHS backend；30 s；`mip_rel_gap=0` |
| CBC / HiGHS 独立版本 | 当前 API 未暴露，明确记录为 unavailable |
| 拓扑 | AWS 4 GPU/4 CPU/500 GB；DGX-ext 8/8/1000；RTX 8/6/1000；均为 synthetic capacity |
| 响应 | Neutral，`rho=1` |
| 周期 | 60 s |
| 顺序 | B1 → SY adapted → B4 → B5 → B3 |

执行命令：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe run_b3_multiworkload_neutral.py `
  --preregistration joint_prototype\multiworkload_neutral_preregistration.json `
  --synergy-models D:\科研启航\Sia\research_sources\Synergy\simulator\models `
  --previous-regression D:\科研启航\Sia\research_runs\sia_synergy\cpu_bottleneck_smoke `
  --output D:\科研启航\Sia\research_runs\sia_synergy\b3_multiworkload_neutral\attempt2
```

## 5. 正确性结果

九份验证均通过以下 19 类检查：manifest 完整、冻结矩阵一致、内存门槛、20/20 完成、无 round limit、无容量违规、Neutral `rho=1`、Goodput 倍率只应用一次、瞬时与时间加权 GPU/CPU/RAM 容量、B1 intent/admission、B4/B5 GPU config 冻结、SY adapted TUNE core、B3 多 GPU 候选、checkpoint/restore 先消耗时间再训练、progress 单调、contention 字段完整、B3 与 B1 intent 存在差异、原始 Sia 历史回归 hash 保持一致。

所有 cohort 都观察到实际排队和至少一个策略上的 CPU-saturated/GPU-idle 节点事件；“未分配作业与该节点事件必须同轮发生”只在 4/9 cohort 出现，因此按观测项记录，不作为每个 cohort 必须满足的正确性条件。

聚合求解状态：B1 `OPTIMAL=1484`、`ORIGINAL_SIA_OPTIMAL=1545`；SY adapted `SYNERGY_TUNE_ADAPTED=7358`；B4 `OPTIMAL=3034`；B5 `SYNERGY_TUNE_ADAPTED=3040`；B3 `OPTIMAL=2905`。按修订后的统一口径，45 次运行均没有真实 fallback status；没有用含 `fallback_reason` 的普通 audit 字段冒充 fallback 次数。

冻结 commit 上再次执行完整测试：**38 tests，38 passed，0 failure，0 error，8.523 s**。唯一警告是既有 PMP/autograd `invalid value encountered in log` RuntimeWarning，对应测试通过。

## 6. 绝对性能结果

单位均为秒。每个 cohort 只有 20 个作业，P95/P99 仅是该小样本的描述统计，不能解释为高精度总体尾延迟估计。

| Cohort | Policy | Avg JCT | Median | P95 | P99 | Makespan |
|---|---|---:|---:|---:|---:|---:|
| w1_early | B1 | 2393.90 | 940.50 | 9820.25 | 9869.65 | 11570 |
|  | SY adapted | 4298.00 | 2105.00 | 12256.90 | 28124.18 | 33138 |
|  | B4 | 2393.90 | 940.50 | 9820.25 | 9869.65 | 11570 |
|  | B5 | 2281.90 | 795.50 | 9457.20 | 10326.64 | 11890 |
|  | B3 | 2036.40 | 892.00 | 5013.65 | 8428.33 | 10329 |
| w1_middle | B1 | 2733.75 | 1586.50 | 3328.60 | 21684.12 | 28067 |
|  | SY adapted | 4875.25 | 2279.50 | 13700.55 | 23680.11 | 27969 |
|  | B4 | 2771.70 | 1566.50 | 3579.05 | 21911.01 | 28288 |
|  | B5 | 2734.35 | 1509.00 | 3306.30 | 21763.66 | 28172 |
|  | B3 | 2631.50 | 1435.00 | 3097.80 | 22079.56 | 28619 |
| w1_late | B1 | 1048.90 | 684.00 | 1764.80 | 6336.96 | 12872 |
|  | SY adapted | 2813.25 | 1333.50 | 6353.25 | 19915.45 | 28698 |
|  | B4 | 1048.90 | 684.00 | 1764.80 | 6336.96 | 12872 |
|  | B5 | 1048.90 | 684.00 | 1764.80 | 6336.96 | 12872 |
|  | B3 | 984.25 | 575.50 | 1358.15 | 6254.83 | 12871 |
| w5_early | B1 | 1733.25 | 891.00 | 7550.35 | 10276.47 | 12384 |
|  | SY adapted | 7430.55 | 3520.00 | 30411.05 | 46204.61 | 51678 |
|  | B4 | 1739.50 | 891.00 | 7607.15 | 10254.23 | 12441 |
|  | B5 | 1780.35 | 891.00 | 8443.65 | 10216.73 | 12086 |
|  | B3 | 1760.45 | 882.50 | 9835.20 | 10203.04 | 11721 |
| w5_middle | B1 | 1911.80 | 1773.50 | 2607.80 | 4337.56 | 4770 |
|  | SY adapted | 4390.75 | 2499.00 | 12193.40 | 17717.08 | 19098 |
|  | B4 | 1859.50 | 1719.00 | 2577.70 | 4473.14 | 4947 |
|  | B5 | 1885.60 | 1779.00 | 2522.50 | 4460.50 | 4945 |
|  | B3 | 1835.45 | 1812.50 | 2182.10 | 4038.02 | 4502 |
| w5_late | B1 | 2129.35 | 829.50 | 2896.60 | 21677.72 | 32429 |
|  | SY adapted | 2617.60 | 1080.00 | 4959.45 | 21458.29 | 31639 |
|  | B4 | 2129.35 | 829.50 | 2896.60 | 21677.72 | 32429 |
|  | B5 | 2128.70 | 828.50 | 2885.20 | 21675.44 | 32429 |
|  | B3 | 1969.40 | 719.50 | 2248.75 | 21244.95 | 32050 |
| w10_early | B1 | 1656.20 | 1038.00 | 7545.65 | 8011.53 | 11045 |
|  | SY adapted | 3521.30 | 1717.50 | 8365.75 | 22482.75 | 27438 |
|  | B4 | 1653.25 | 986.50 | 7545.30 | 8005.86 | 11038 |
|  | B5 | 1656.20 | 1038.00 | 7545.65 | 8011.53 | 11045 |
|  | B3 | 1377.85 | 951.00 | 3225.95 | 7329.19 | 9781 |
| w10_middle | B1 | 4136.85 | 1557.00 | 27032.55 | 28211.31 | 29594 |
|  | SY adapted | 4479.80 | 3342.50 | 10043.80 | 25179.96 | 29949 |
|  | B4 | 4176.70 | 1581.00 | 27322.30 | 28208.46 | 29518 |
|  | B5 | 4140.15 | 1526.00 | 27407.20 | 28595.84 | 29981 |
|  | B3 | 2792.25 | 1450.00 | 5756.80 | 22534.56 | 27714 |
| w10_late | B1 | 4761.90 | 568.00 | 30232.20 | 30645.64 | 38759 |
|  | SY adapted | 13599.85 | 1077.00 | 48485.45 | 157749.89 | 191487 |
|  | B4 | 4761.90 | 568.00 | 30232.20 | 30645.64 | 38759 |
|  | B5 | 4781.10 | 568.00 | 30228.50 | 30585.70 | 38685 |
|  | B3 | 3300.45 | 578.50 | 14043.00 | 26765.40 | 36367 |

## 7. 资源、等待与求解开销

下表列出 B3 的完整资源汇总；五策略逐 cohort 的同口径绝对值、逐作业 JCT/完成时间/等待、逐轮 placement 与 solve status 均在 `aggregate_summary.json` 和各 `result.json` 中。

| Cohort | Wait (s) | GPU-h | CPU-core-h | RAM-GB-h | Restarts | Wasted GPU-s | CPU-sat/GPU-idle events |
|---|---:|---:|---:|---:|---:|---:|---:|
| w1_early | 420 | 37.348 | 37.348 | 778.0 | 56 | 8450 | 77 |
| w1_middle | 780 | 67.709 | 67.709 | 1410.4 | 60 | 11760 | 33 |
| w1_late | 60 | 25.193 | 25.193 | 524.8 | 45 | 12780 | 0 |
| w5_early | 60 | 34.853 | 34.853 | 726.0 | 49 | 10210 | 2 |
| w5_middle | 2520 | 16.826 | 16.826 | 350.5 | 59 | 7165 | 36 |
| w5_late | 60 | 65.958 | 65.958 | 1373.9 | 44 | 11700 | 2 |
| w10_early | 60 | 28.797 | 28.797 | 599.8 | 56 | 11535 | 10 |
| w10_middle | 360 | 70.337 | 70.337 | 1465.1 | 55 | 9225 | 34 |
| w10_late | 60 | 94.803 | 94.803 | 1974.8 | 46 | 16245 | 9 |

B3 的优化器九个 cohort 累计求解时间为 4.348 秒，最大 cohort 内 P99 单轮求解时间为 8.6 ms；B1/SY/B4/B5 的对应累计值为 0.968/5.784/1.893/2.005 秒。模拟器墙钟累计不等于优化器时间，45 次策略运行及验证总编排耗时约 89 分钟。

B3/B4 的 RAM-GB-hours 显著低于 B1/B5：B3 是 B1 的 16.00%–16.81%。这不是 Neutral 下的训练加速证据。原因是 B1 的 proportional allocation 约为每 GPU 125 GB，而 Neutral 下所有辅助资源点 `rho=1`，B3/B4 的目标对更多 RAM 没有收益，因而合法地选择最小约 20.83 GB/GPU 的候选。该指标反映合成配额占用，不等价于真实物理内存能耗；进入非 Neutral 阶段后必须重新审计响应函数是否让 RAM/CPU 供给真实影响 Goodput。

节点级 CPU-saturated/GPU-idle event 是“节点×轮”计数，而等待是“作业×秒”累计，两者不能直接相减。例如 `w1_early/B3` 有 77 个节点事件但只有 420 waiting-seconds，说明若干轮中节点存在 CPU 约束下的空闲 GPU，并不意味着同样数量的作业各等待 60 秒。

## 8. 跨 workload 与长尾敏感性

### 8.1 逐 cohort 平均 JCT 相对变化

| Cohort | B3 vs B1 | B3 vs SY | B3 vs B4 | B3 vs B5 |
|---|---:|---:|---:|---:|
| w1_early | +14.93% | +52.62% | +14.93% | +10.76% |
| w1_middle | +3.74% | +46.02% | +5.06% | +3.76% |
| w1_late | +6.16% | +65.01% | +6.16% | +6.16% |
| w5_early | **-1.57%** | +76.31% | **-1.20%** | +1.12% |
| w5_middle | +3.99% | +58.20% | +1.29% | +2.66% |
| w5_late | +7.51% | +24.76% | +7.51% | +7.48% |
| w10_early | +16.81% | +60.87% | +16.66% | +16.81% |
| w10_middle | +32.50% | +37.67% | +33.15% | +32.56% |
| w10_late | +30.69% | +75.73% | +30.69% | +30.97% |

跨 180 个 cohort-job 实例，B3 相对 B1 为 93 个改善、71 个退化、16 个相等；相对 SY 为 119/61/0；相对 B4 为 96/70/14；相对 B5 为 95/65/20。因此收益既不是“所有作业普遍改善”，也不只是每个 cohort 单一长尾：多数比较中改善作业数略占优势，但幅度分布高度不均。

### 8.2 单个高影响作业诊断

| Cohort / 对照 | 主结果差值 | 最大贡献作业（对照 JCT → B3 JCT） | 去除该作业后的差值 |
|---|---:|---|---:|
| w1_early / B1 | +357.50 s | `deepspeech2-12`，9817 → 4767；贡献 +5050 s | +110.53 s (+5.52%) |
| w10_early / B1 | +278.35 s | `deepspeech2-16`，8128 → 2956；贡献 +5172 s | +20.79 s (+1.58%) |
| w10_middle / B1 | +1344.60 s | `deepspeech2-83`，28506 → 4653；贡献 +23853 s | +159.95 s (+5.60%) |
| w10_late / B1 | +1461.45 s | `deepspeech2-159`，30749 → 4150；贡献 +26599 s | +138.42 s (+4.08%) |
| w5_early / B1 | -27.20 s | `yolov3-11`，7371 → 9811；贡献 -2440 s | +99.79 s (+6.95%) |
| w5_middle / B4 | +24.05 s | `cifar10-77`，2208 → 1461；贡献 +747 s | -14.00 s (-0.76%) |
| w1_early / B5 | +245.50 s | `deepspeech2-9`，10544 → 4789；贡献 +5755 s | -44.47 s (-2.41%) |

表中“去除”仅是事后诊断，主结果仍保留全部作业。预注册的 9 个 cohort 不包含 job ID 31，所以历史 `deepspeech2-31` 不可能再次主导本轮结果；取而代之的是 `deepspeech2-12/16/83/159` 在多个 cohort 成为最大贡献项。相对 B1 的 9 个 cohort 在移除各自最大绝对贡献作业后仍全部为正，但优势中位数明显收缩；相对 B4/B5 则各有一个 cohort 在移除最大贡献作业后转为负值。

注：`aggregate_summary.json` 保存了上述每个比较的完整逐作业 signed contribution 和准确 JCT；报告不删除任何作业，也不以去除结果替代主分析。

### 8.3 平均值、尾部和 makespan 冲突

- `w1_middle`：B3 平均 JCT 优于 B1/B4/B5，但 P99 和 makespan 均更差；相对 SY 平均 JCT 更低但 makespan 更高。
- `w5_early`：B3 平均 JCT 优于 B5 1.12%，但 P95 更差（9835.20 vs 8443.65）。
- `w5_late`：B3 平均 JCT 优于 SY adapted，但 makespan 更高（32050 vs 31639）。

因此不能用平均 JCT 单一指标宣称全面改善。B3 的联合选择可显著提前部分长作业，但也会改变公平性、重启和尾部资源顺序；当前日志支持“存在这种权衡”，不支持把某一个 placement 变化断言为唯一因果来源。

## 9. 五个研究问题的回答

1. **B3 平均 JCT 优势能否跨 cohort 复现？** 相对 B1/B4 为 8/9，相对 B5/SY adapted 为 9/9；因此可复现，但不是无例外。
2. **相比 B4/B5 是否普遍？** cohort 胜率较高，但幅度受长尾影响。B3 相对 B4/B5 各有约半数作业没有改善，且去除单一高影响作业后各出现一个负 cohort。
3. **平均 JCT 与尾部/makespan 是否冲突？** 是，明确出现在 `w1_middle`、`w5_early` 和 `w5_late`。
4. **哪里表现不好，可能原因是什么？** `w5_early` 相对 B1/B4 退化，主要由 YOLOv3 长作业变慢驱动；`w1_middle` 平均值改善但尾部和 makespan 退化。日志表明联合 placement/配置改变了资源时序与重配置，但当前实验没有因果消融，不能进一步断言唯一原因。
5. **能否进入非 Neutral 阶段？** 可以进入严格预注册的小规模/20-job 合成响应实验，因为代码冻结、容量和时间语义、Neutral 单倍率及多 workload 稳定性均通过；仍不得直接进入 160-job 或真实性能结论。

## 10. 局限与下一阶段约束

- 节点 CPU/RAM 容量是 synthetic schema，不是真实 AWS/DGX/RTX 主机规格。
- Neutral 场景不验证 Synergy CPU×RAM 矩阵在跨硬件迁移后的数值真实性，只验证硬容量与联合 placement 机制。
- SY 是 Synergy-TUNE adapted，不是未经修改的官方 Synergy；其固定 GPU demand/type mapping 会显著影响结果。
- 9 个 cohort 来自 3 条 trace 的不同区段，分析单位是 cohort；不能把 180 个作业当成 180 次独立实验。
- 下一阶段必须在运行前冻结 Low/Medium/High 响应档位、模板映射与归一化，不得根据本轮 JCT 挑档。
- 非 Neutral 阶段需重点复核 RAM 最小候选、CPU/RAM 响应对实际 `Job.step()` 的单次作用，以及长尾 DeepSpeech2/YOLOv3 的贡献。

## 11. 机器可读产物与哈希

| 文件 | SHA-256 |
|---|---|
| `attempt2/preflight.json` | `8F667400900247E3CE199F811C3268B5988345F54EA37729706DC6D99262DE2A` |
| `attempt2/run_manifest.json` | `BAD173A51469299C49BDC29C7A8FACF8560D246FD9C8F9C31E2A75E84B59AFCA` |
| `attempt2/aggregate_summary.json` | `C26B3B78DE20D1C11A53119E5422AF3CB918CCDDB00CE51E1CB79876204A8F83` |
| `attempt2/test_summary.json` | `7EED8D0AC8714E5A2A4115FF11B678D4ED968C61E23B5C235E5566F0B29D150E` |
| preregistration JSON | `50D16F42C2A127DEF745AA7EC1015558C2222EF54238A2F4CD5A5F2CAB4B52C6` |

`run_manifest.json` 还保存了每个策略的源码、输入、profile、console log、result 和 validation hash；各 cohort 的 `policies/<policy>/result.json` 保存逐轮分配、Goodput、进度、checkpoint/restart、等待和求解状态。历史内存阻塞、首次 15/45 尝试及旧验证失败证据均保留，未被最终运行覆盖。
