# B3 Medium 非 Neutral `w10_late` 6000 轮补充报告

## 1. 结论

本次补充实验状态为 **`PASS_COMPLETE_PAIR`**。在完全相同的冻结输入下，B1、SY adapted、B4、B5、B3 按预注册顺序、以相互独立的 Python 进程运行，五种策略均在统一的 6000 轮观察窗内完成 20/20 个作业。

- 唯一曾删失的 `w10_late / SY adapted` 在第 5041 轮完成最后一个作业 `imagenet-154`；它距离原 5000 轮上限仅差 41 轮。
- B1、B4、B5、B3 的 20 个 JCT 与历史 5000 轮结果逐项完全相同；SY adapted 原先已经完成的 19 个 JCT 及完成时间也逐项完全相同。提高观察窗没有改变已经观测到的完成事件。
- 统一验证器的 23 项检查全部通过：无 GPU/CPU/RAM 容量违规，无 round-limit，无求解器失败，Medium 响应公式一致，Goodput 倍率只应用一次，checkpoint/restore 不被 `rho` 加速。
- B3 在 `w10_late` 的平均 JCT 为 2773.15 秒，相对 B1、SY adapted、B4、B5 分别改善 26.309%、86.771%、26.212%、26.890%。但相对 B1/B4/B5 时，B3 的 median JCT 更差，平均值收益主要由 `deepspeech2-159` 的长尾压缩贡献。
- 补齐后，B3 相对四个对照均为 9/9 cohort 平均 JCT 更低。不过这仍是 **合成跨硬件迁移机制实验**，不是 Saturn 或真实联合 GPU/CPU/RAM 性能测量。

> **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

本轮没有运行 Low、High、参数搜索或 160-job 实验，也没有修改调度算法。

## 2. 工程保护与冻结证据

实验工作树为 `D:\科研启航\Sia\sia-joint-medium-frozen`，分支为 `exp/b3-medium-nonneutral`。正式执行 commit：

```text
be03f95c7bc641cbef0e2a4452bc4c7a3b5404e4
```

该 commit 只增加补充实验的预注册、冻结、单策略运行、汇总脚本和测试，不修改 B1、SY、B4、B5、B3、Sia Goodput 或 `Job.step()`。核心文件 SHA-256 与原 Medium 正式执行 commit 完全相同：

| 文件 | SHA-256 |
|---|---|
| `joint_prototype/joint_integration.py` | `BD850D88C2F31113ECB2FAF7CF2E1E8D712760A6E2601C96ED75CCC1F7D9AB21` |
| `joint_prototype/joint_profile.py` | `1BE6E99706B7140C26F350308E0FF6078CF6BC8B772A7ACEF2C994AE8CCE0F0E` |
| `joint_prototype/joint_scheduler.py` | `3A9A81C544A92FF21A350ED3466AAB3F4FEA6FF01DE6F18B2D92F17AD43BC964` |
| `joint_prototype/synergy_adapter.py` | `E2CCACA1B246B956F2D6C5C0EEA9FDE721752B5A6CF354713C3235B8455CECBD` |
| `job.py` | `FAAF92275359A94ED9CA0C8D011C663B37E297C59D7A334856E1B3C7C0369E7E` |

官方 Sia 仓库检查时干净。Synergy 源码没有被修改；其仓库仍只有此前已经存在的未跟踪论文 PDF `osdi22-mohan.pdf`。隔离工作树运行后仅产生 Python `__pycache__`，没有覆盖或清理用户已有文件。

原报告 `B3_MEDIUM_NON_NEUTRAL_REPORT.md` 未修改，SHA-256 仍为 `30D7F2F94D33751D48282883E99E79B52CFC0D866757BF779CA7E3AE457F1E5E`。

## 3. 冻结配置

补充实验根目录：

```text
research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/
```

冻结器对原 `w10_late` 配置做结构化比较，唯一差异为：

```text
simulation.max_rounds: 5000 -> 6000
```

其他输入保持不变：

| 项目 | 冻结值/哈希 |
|---|---|
| Cohort | `workload-10.csv` 稳定到达顺序第 141–160 个，即 job ID 140–159 |
| Workload CSV | `719267711948607055CAD9BA3F5966D63C65B4913693375D21B5F4E1F3675949` |
| 原 5000 轮 config | `A2DA69825FCE7122815CB2481E3362A4610C763676620B406C6AC3CA5167DB64` |
| 新 6000 轮 config | `667D7717CF8FF09EEE6A82C9979180615B58B92670BF91A5D4E5728F4218DE03` |
| Medium response manifest | `A46B5443F515DECA8FD992A7894BF67C1F93DFD7293AD6E985A7F395D5659DE5` |
| Medium preregistration | `9A2F49964F09B523BF28F2B3B64D3B5930922A4943F215B4EA65B81228CE2353` |
| Response | `medium`, `alpha=0.5` |
| Capacity | `cpu_constrained` |
| Interval | 60 秒 |
| Policy order | B1 → SY adapted → B4 → B5 → B3 |
| 内存门槛 | 2 GiB |

冻结时可用内存为 2.472 GiB。五个策略启动前分别为 2.324、2.410、2.663、2.850、2.777 GiB，均满足门槛。每种策略由单独 Python 进程运行。

环境为 Python 3.10.10、NumPy 1.23.5、SciPy 1.10.1、CVXPY 1.3.2、CyLP 0.91.6。联合 MIP 保持 `time_limit=30s`、`mip_rel_gap=0`；Sia CBC 保持 8 threads、0.01% allowable gap。安装环境未通过公共接口暴露 CBC/HiGHS 的独立版本号，因此不作猜测。

## 4. 运行结果

### 4.1 完成性与 JCT

| Policy | 完成 | 轮数 | Avg JCT (s) | Median (s) | P95 (s) | P99 (s) | Makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 20/20 | 516 | 3763.20 | 428.0 | 21260.85 | 22565.77 | 30902 |
| SY adapted | 20/20 | 5041 | 20963.30 | 1084.0 | 67011.40 | 250223.08 | 302447 |
| B4 | 20/20 | 516 | 3758.25 | 426.5 | 21234.25 | 22560.45 | 30902 |
| B5 | 20/20 | 521 | 3793.10 | 428.0 | 21572.45 | 22871.29 | 31206 |
| B3 | 20/20 | 453 | **2773.15** | 474.0 | **12397.95** | **19069.99** | **27159** |

P95/P99 均只由 20 个作业计算，不应解释为高精度总体尾延迟估计。SY adapted 的极端 P99 和 makespan 由 `imagenet-154` 主导；这也是原 5000 轮窗口删失该策略的原因。

### 4.2 B3 直接对照

正值表示 B3 的 JCT 更低。

| 对照 | Avg 差值 (s) | Avg 改善率 | Median 差值 (s) | P95 差值 (s) | P99 差值 (s) | Makespan 差值 (s) | B3 更好/相同/更差作业数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 990.05 | 26.309% | -46.0 | 8862.90 | 3495.78 | 3743 | 10/0/10 |
| SY adapted | 18190.15 | 86.771% | 610.0 | 54613.45 | 231153.09 | 275288 | 11/0/9 |
| B4 | 985.10 | 26.212% | -47.5 | 8836.30 | 3490.46 | 3743 | 9/1/10 |
| B5 | 1019.95 | 26.890% | -46.0 | 9174.50 | 3801.30 | 4047 | 10/0/10 |

B3 相对 B1/B4/B5 的平均、P95、P99 与 makespan 更好，但 median 更差，且只有约一半作业改善。因此不能表述为“所有作业普遍加速”。相对 SY adapted 时 median 也改善，但 20 个作业中仍有 9 个退化。

### 4.3 逐作业 JCT

| Job | B1 | SY adapted | B4 | B5 | B3 |
|---|---:|---:|---:|---:|---:|
| cifar10-140 | 343 | 290 | 341 | 343 | 438 |
| cifar10-141 | 313 | 263 | 311 | 313 | 410 |
| cifar10-142 | 283 | 1053 | 281 | 283 | 378 |
| cifar10-143 | 316 | 266 | 314 | 316 | 413 |
| cifar10-144 | 316 | 3586 | 314 | 316 | 534 |
| cifar10-145 | 302 | 1072 | 300 | 302 | 455 |
| cifar10-146 | 447 | 6687 | 442 | 447 | 394 |
| cifar10-147 | 420 | 269 | 416 | 420 | 493 |
| cifar10-148 | 324 | 1394 | 322 | 324 | 419 |
| yolov3-149 | 11124 | 30093 | 11092 | 11917 | 9767 |
| cifar10-150 | 290 | 240 | 288 | 290 | 399 |
| cifar10-151 | 468 | 285 | 465 | 468 | 432 |
| cifar10-152 | 431 | 7670 | 429 | 431 | 429 |
| yolov3-153 | 12864 | 54958 | 12859 | 12053 | 11959 |
| imagenet-154 | 21175 | **296026** | 21147 | 21487 | 20738 |
| cifar10-155 | 425 | 274 | 424 | 425 | 605 |
| cifar10-156 | 1200 | 9189 | 1198 | 1200 | 821 |
| cifar10-157 | 626 | 286 | 625 | 626 | 603 |
| cifar10-158 | 705 | 1096 | 705 | 705 | 631 |
| deepspeech2-159 | 22892 | 4269 | 22892 | 23196 | **5145** |

### 4.4 资源与执行开销

| Policy | GPU-h | CPU-core-h | RAM-GB-h | Restart | Wasted GPU-s | Checkpoint/restore (s) | Waiting (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 79.314 | 79.314 | 9914.306 | 39 | 17335 | 4405 | 20700 |
| SY adapted | 412.977 | 412.977 | 51622.083 | 1 | 6960 | 1315 | 52920 |
| B4 | 79.221 | 80.854 | 9875.486 | 39 | 17335 | 4405 | 20700 |
| B5 | 78.530 | 79.113 | 9806.875 | 42 | 18295 | 4645 | 21480 |
| B3 | **66.381** | **73.514** | **8270.486** | 50 | **15845** | 4815 | **120** |

B3 的资源小时和等待更低，但 restart 与 checkpoint/restore 时间更高，不能把资源下降描述为无代价收益。SY adapted 的资源小时主要被 `imagenet-154` 的超长运行放大。

## 5. 长尾贡献审计

相对 B1 和 B4，`deepspeech2-159` 单作业贡献 17747 秒 JCT 差；相对 B5 贡献 18051 秒。移除该单作业后，仅作诊断的平均 JCT 改善率分别从 26.309%、26.212%、26.890% 降为 3.922%、3.740%、4.458%。因此 B3 对三条 Sia 路径的 `w10_late` 平均收益仍然主要由一个 DeepSpeech2 长尾作业驱动，虽然移除后仍保留小幅正收益。

相对 SY adapted，最大贡献者不是 DeepSpeech2，而是 `imagenet-154`：该作业贡献 275288 秒差值。移除它后，B3 相对 SY adapted 的诊断改善仍为 71.823%。这一差距包含 SY adapted 的固定 GPU demand、确定性异构 GPU 映射及 TUNE 适配语义，不能据此宣称官方 Synergy 性能低于 B3。

历史特别关注的 `deepspeech2-31` 不在 `w10_late` cohort 中；本 cohort 的对应长尾是 `deepspeech2-159`。

## 6. 补齐后的九 cohort 汇总

其余 8 个 cohort 沿用原 5000 轮完整结果；仅 `w10_late` 五策略统一使用本次 6000 轮结果。因为历史已完成 JCT 全部逐项相等，混合观察窗只改变了此前被右删失的 SY `imagenet-154` 是否可观测完成，没有改变其他完成事件。

### 6.1 等权平均 JCT 与胜负

| 对照 | 对照九 cohort 等权 Avg JCT (s) | B3 (s) | 差值 (s) | 等权均值改善 | 改善率中位数 | 改善率范围 | B3 win/tie/loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 2020.21 | 1716.38 | 303.83 | 15.039% | 10.378% | 0.874%–26.309% | 9/0/0 |
| SY adapted | 7244.44 | 1716.38 | 5528.06 | 76.308% | 71.578% | 50.143%–86.771% | 9/0/0 |
| B4 | 2027.87 | 1716.38 | 311.48 | 15.360% | 9.848% | 0.870%–26.212% | 9/0/0 |
| B5 | 2009.39 | 1716.38 | 293.01 | 14.582% | 10.352% | 0.305%–26.890% | 9/0/0 |

这是以 cohort 为分析单位的结果，不把 cohort 内 20 个作业当作 20 次独立实验。B3 的最小改善分别出现在：B1/B5 的 `w1_middle`、B4 的 `w5_middle`、SY adapted 的 `w5_late`。

### 6.2 九 cohort 尾部、makespan 与资源

下表为九个 cohort 指标的等权平均；restart、wasted GPU-s、waiting 为九 cohort 总和。

| Policy | Mean median | Mean P95 | Mean P99 | Mean makespan | Mean GPU-h | Mean CPU-h | Mean RAM-GB-h | Restart | Wasted GPU-s | Waiting (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 701.94 | 8827.98 | 13367.46 | 17749.11 | 43.214 | 43.214 | 5401.779 | 426 | 105640 | 74340 |
| SY adapted | 2310.39 | 23689.60 | 54497.30 | 64925.78 | 108.963 | 108.963 | 13599.536 | 14 | 64520 | 504840 |
| B4 | 697.22 | 9136.78 | 13527.89 | 17872.44 | 43.339 | 44.320 | 5341.670 | 431 | 108105 | 75780 |
| B5 | 696.17 | 8799.19 | 13586.68 | 17897.89 | 43.138 | 43.270 | 5386.979 | 431 | 105475 | 74220 |
| B3 | 745.56 | **4932.29** | **11888.24** | **16365.67** | **34.459** | **41.971** | **4245.251** | 449 | 91600 | **6000** |

B3 的跨 cohort 平均 median 仍高于 B1/B4/B5，而 P95、P99、makespan 更低，再次说明主要机制是压缩尾部而不是普遍改善短作业。既有冲突仍存在：相对 SY adapted 的 `w5_late` 中，B3 平均 JCT 更低但 makespan 更高。

### 6.3 Neutral 与 Medium

补齐后 Medium 相对 Neutral 的 B3 跨 cohort 改善率中位数变化如下：

| Comparator | Neutral median | Medium median | 变化 |
|---|---:|---:|---:|
| B1 | 7.512% | 10.378% | +2.866 pp |
| SY adapted | 58.197% | 71.578% | +13.380 pp |
| B4 | 7.512% | 9.848% | +2.337 pp |
| B5 | 7.483% | 10.352% | +2.869 pp |

Medium 并非对所有策略都加速。相对 Neutral 的九 cohort 等权平均 JCT变化为：B1 改善 19.213%、B4 改善 19.010%、B5 改善 19.400%、B3 改善 17.340%，而 SY adapted 退化 35.759%。这属于冻结响应与调度行为共同产生的结果，不是选参结果。

## 7. 正确性、容量与求解审计

统一验证状态为 `PASS`，通过的关键检查包括：

- 五策略完整、均为 20/20、均未触发 round limit；
- 每轮和时间加权 GPU/CPU/RAM 容量均合法，容量违规数为 0；
- `rho` 为正且有限，并严格满足冻结的 `rho = raw_rho ** 0.5`；
- `effective_goodput = raw_goodput * rho`，最大浮点残差仅为舍入量级，倍率没有重复施加；
- checkpoint/restore 完整延迟轮不推进训练，progress 单调；
- B1 admitted config 与 Sia intent 一致；B4/B5 固定 Sia GPU config；SY 走 adapted TUNE core；B3 枚举多个 GPU 候选；
- 原始 Sia official/experimental 回归 hash 均为 `F81E0B22C76734E5264742A7231687F366CE77B4D3448155E444A64D42E02DFF`。

运行时 `rho` 范围为：B1/B4/B5 0.5477–1.5118，SY adapted 0.5477–1.4142，B3 0.7382–1.8037。它们都来自同一冻结 Oracle，差异来自策略实际选择的 GPU 与 CPU/RAM 配额。

求解状态均为成功状态：B1 为 `ORIGINAL_SIA_OPTIMAL=179`、`OPTIMAL=337`；SY 为 `SYNERGY_TUNE_ADAPTED=5041`；B4 为 `OPTIMAL=516`；B5 为 `SYNERGY_TUNE_ADAPTED=521`；B3 为 `OPTIMAL=453`。没有 `NO_COLUMNS`、`MIP_NO_*` 或求解失败。旧汇总字段把 B4 337 个带 `fallback_reason` 元数据的轮次记入宽口径 `fallback_count`，但逐轮 `status` 没有真实 fallback；本报告以状态字段作为真实 fallback 口径，真实 fallback 为 0。

节点级 CPU 饱和且 GPU 空闲事件分别为 B1=1、SY=0、B4=19、B5=5、B3=55；真正同时存在未分配作业的对应等待轮分别为 0、0、1、0、1。节点事件数和作业等待秒数是不同统计单位，不构成容量违规。

最终完整测试：43/43 通过，0 failure、0 error，耗时 7.365 秒。PMP 测试仍出现既有 autograd `invalid value encountered in log` RuntimeWarning，但测试通过。

## 8. 产物与可复核路径

主要机器可读产物：

- 预注册：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/followup_preregistration.json`
- 冻结配置：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/frozen_config_6000.json`
- 冻结 manifest：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/frozen_manifest.json`
- 五策略统一 manifest：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/run_manifest.json`
- 各策略结果与日志：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/policies/{b1,sy,b4,b5,b3}/`
- 验证结果：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/validation_summary.json`
- 直接比较：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/comparison_summary.json`
- 新九 cohort 汇总：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/aggregate_summary_9cohort_followup.json`
- 测试摘要：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/test_summary.json`
- 哈希审计：`research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/audit_manifest.json`

关键摘要 SHA-256：验证结果 `B07645F80A9F2EB5FA6F7B63E352D87DE97D3478CCC8FF79B71D41868F093B16`；新九 cohort 汇总 `EF964ADCBEFCA1F8BE3EEFA2C8A0BEFA68BE807B19CE406BCAB83A9D92B89223`。各策略结果 hash 已完整记录在 `audit_manifest.json`。

## 9. 最终判断

`w10_late` 现已形成公平、完整的五策略配对，原 5000 轮删失不再阻止九 cohort 汇总。证据支持：在冻结的 Medium 合成响应与 CPU 受限拓扑下，B3 在 9/9 cohort 的平均 JCT 低于四个对照，并明显压缩 P95/P99 与 makespan。

证据同样表明，这一优势并非多数作业一致加速：B3 的 median 对 B1/B4/B5 更差，`w10_late` 的均值优势主要由 `deepspeech2-159` 长尾压缩贡献；SY 差距又被 `imagenet-154` 极端长尾放大。因此本结果适合支持“联合选择在预注册合成机制下能稳定压缩部分长尾”的结论，不足以支持真实集群普遍性能优势或所有作业公平改善。

本任务到此停止，不自动开展 Low/High、参数搜索、160-job 或新算法实验。
