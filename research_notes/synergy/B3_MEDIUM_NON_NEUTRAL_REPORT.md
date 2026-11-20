# B3 Medium 非 Neutral 五策略实验报告

## 1. 结论先行

本轮状态为 **`FAIL_INCOMPLETE_MATRIX`**，不能标记为完整通过。

- Medium 响应模型已按既有 `alpha=0.5` 定义冻结，没有设计新曲线、调参或根据 JCT 改映射。
- 单 cohort `w1_early` 的 B1、SY adapted、B4、B5、B3 全部完成 20/20，并通过容量、响应公式、Goodput 单次倍率及 checkpoint/restore 审计。
- 完整矩阵的 45 次策略进程均已执行。44 次完成 20/20；B3 在 9/9 cohort 均完成 20/20。
- 唯一未完成项是 `w10_late / SY adapted`：在冻结的 5000 轮上限处完成 19/20。未完成作业 `imagenet-154` 仍在推进，不是求解器崩溃或永久无分配。
- 因预注册规则禁止在看到结果后修改其他实验参数，本轮没有把 5000 轮提高到 6000，也没有给未完成作业插补 JCT。因此不能给出 9-cohort 的完整 SY 平均 JCT 对照。
- 在可以完整比较的范围内，B3 相对 B1、B4、B5 的平均 JCT 在 9/9 cohort 更低；相对 SY adapted 在 8/8 完整 cohort 更低。但是这些均为 **合成跨硬件迁移机制证据**，且多数改善受单个长尾作业显著影响。

> **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

本报告没有启动 Low、High、参数搜索或 160-job 实验。

## 2. 冻结版本与工程保护

实验在隔离工作树 `D:\科研启航\Sia\sia-joint-medium-frozen` 和分支 `exp/b3-medium-nonneutral` 中进行。官方 Sia、官方 Synergy、Neutral 历史结果均未修改或覆盖。

| 用途 | Commit | 内容 |
|---|---|---|
| Medium 初始冻结 | `8bd66113f5a0ff857237621b194e54b6a5d45946` | 预注册、响应 manifest、Medium 验证器和分析器 |
| 正式隔离执行 | `ba6095d2e57eee21658500582a713d9d5b9e1a55` | 仅增加逐 cohort 新进程编排和 hash 清单；调度算法不变 |
| 删失结果审计 | `6caa73370164d8a8e5e26ab728104aed6629e3b0` | 仅增加不插补未完成 JCT 的离线分析器 |

核心调度与性能查询文件在正式执行 commit 中的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `joint_prototype/joint_integration.py` | `BD850D88C2F31113ECB2FAF7CF2E1E8D712760A6E2601C96ED75CCC1F7D9AB21` |
| `joint_prototype/joint_profile.py` | `1BE6E99706B7140C26F350308E0FF6078CF6BC8B772A7ACEF2C994AE8CCE0F0E` |
| `joint_prototype/joint_scheduler.py` | `3A9A81C544A92FF21A350ED3466AAB3F4FEA6FF01DE6F18B2D92F17AD43BC964` |
| `joint_prototype/synergy_adapter.py` | `E2CCACA1B246B956F2D6C5C0EEA9FDE721752B5A6CF354713C3235B8455CECBD` |
| `job.py` | `FAAF92275359A94ED9CA0C8D011C663B37E297C59D7A334856E1B3C7C0369E7E` |

依赖版本：Python 3.10.10、NumPy 1.23.5、SciPy 1.10.1、CVXPY 1.3.2、CyLP 0.91.6；Sia GPU intent 使用 CVXPY+CBC，联合列 MIP 使用 `scipy.optimize.milp` 的 HiGHS 后端。CyLP 未暴露 CBC 版本，SciPy 公共 API 未暴露 HiGHS 版本，因此没有猜测版本号。联合 MIP 保持 `time_limit=30s`、`mip_rel_gap=0`，Sia CBC 保持 8 threads、0.01% allowable gap。

## 3. Medium 响应模型冻结

### 3.1 唯一既有定义

预注册文件 SHA-256：`9A2F49964F09B523BF28F2B3B64D3B5930922A4943F215B4EA65B81228CE2353`。

固定参考点为每 GPU 3 CPU core、62.5 GB RAM，响应指数为 `alpha=0.5`。对一个 Sia 合法 GPU 候选，定义：

```text
q = (raw_sia_goodput / gpu_count)
    / minimum_positive_candidate_per_gpu_goodput

phi_c = clip(q, 0.5, 4.0)
phi_m = clip(sqrt(q), 0.5, 2.0)

baseline = bilinear(T, clip(3 / phi_c), clip(62.5 / phi_m))
observed = bilinear(T,
                    clip(3 * (cpu_per_gpu / 3) / phi_c),
                    clip(62.5 * (ram_per_gpu / 62.5) / phi_m))

raw_rho = observed / baseline
rho = raw_rho ** 0.5
effective_goodput = raw_sia_goodput * rho
```

`T` 是 Synergy 官方 CPU×RAM 矩阵；查询使用双线性插值。超出网格的坐标保守 clip，不外推；缺失模板或无效矩阵直接报错，不插补。固定参考配置的 `raw_rho` 和 `rho` 均严格为 1。

CPU 候选点为 `1, 2, 3, 4, 6, 9, 12, 24 core/GPU`；RAM 候选点为 `20.83, 62.5, 125, 187.5, 250 GB/GPU`。任务映射固定为：CIFAR10→`alexnet_1.py`、BERT→`bert_1.py`、DeepSpeech2→`deepspeech_1.py`、ImageNet→`res50_1.py`、YOLOv3→`mobilenet_1.py`。

矩阵单元来自 Synergy 源码；任务到模板映射、固定参考归一化、`phi` 与 GPU Goodput 的耦合均是本项目的合成跨硬件迁移，不是 Saturn 实测。

### 3.2 模板范围与边际收益审计

| Template | Matrix min–max | Sampled Medium rho | CPU 严格边际递减 | RAM 严格边际递减 |
|---|---:|---:|---|---|
| alexnet | 0.10–2.94 | 0.3162–3.4155 | 否，7 个斜率反例 | 是 |
| bert | 1.00–1.00 | 1.0000–1.0000 | 是（平坦） | 是（平坦） |
| deepspeech | 0.10–1.00 | 0.3162–2.5819 | 是 | 是 |
| mobilenet | 0.10–1.55 | 0.3162–3.0418 | 否，4 个斜率反例 | 是 |
| res50 | 0.10–1.15 | 0.3162–2.5633 | 是 | 是 |

所有矩阵沿 CPU/RAM 轴均非递减，但 AlexNet 和 MobileNet 的 CPU 轴不是全局严格凹函数。审计保留官方矩阵形状，没有为了制造更平滑结果而拟合、平滑或截断 `rho`。

预注册采样包络为 0.3162–3.4155，但这不是运行时观测。正式 45 次运行的实际总体范围为 0.5477–1.8037；此前历史 4.2981 不在本轮 Medium 路径中出现。

响应 manifest：`research_runs/sia_synergy/b3_multiworkload_medium/response_model_manifest_isolated.json`，SHA-256 `A46B5443F515DECA8FD992A7894BF67C1F93DFD7293AD6E985A7F395D5659DE5`。

## 4. Goodput 与时间语义

所有策略共用同一 `JointProfileOracle`。每轮分配完成后，`JointCluster` 只设置一次 `job.goodput_multiplier=rho`；`Job.step()` 先消耗 checkpoint/restore delay，再把 `rho` 作用到实际训练 progress。等待时间、checkpoint/restore 倒计时没有乘 `rho`。

Medium 验证器逐轮检查：

- `rho > 0` 且有限；
- `rho == raw_rho ** 0.5`；
- `effective_goodput == raw_goodput * rho`，没有重复倍率；
- checkpoint/restore 完整延迟轮 progress 不增加；
- progress 单调；
- 每节点 GPU、CPU、RAM 容量合法；
- B4/B5 保持 Sia 选定的 GPU config；
- B3 确实枚举多个 GPU 候选；
- 原始 Sia 历史回归 hash 保持一致。

## 5. 单 cohort 验证

预注册的 `w1_early` 先行验证在 commit `8bd6611` 上完成，5/5 策略均 20/20，验证状态 `PASS`。

| Policy | Avg JCT (s) | Rounds | Runtime rho min–max |
|---|---:|---:|---:|
| B1 | 2500.50 | 239 | 0.5477–1.4862 |
| SY adapted | 5780.60 | 706 | 0.5477–1.4142 |
| B4 | 2437.50 | 250 | 0.5477–1.4862 |
| B5 | 2403.35 | 253 | 0.5477–1.4862 |
| B3 | 2154.55 | 202 | 0.5477–1.7888 |

该结果仅证明 Medium 路径能够生成随 CPU/RAM 配额变化的合法响应，并能完成端到端训练；它不是选参依据。正式隔离重跑的 `w1_early` 数值与上述结果完全一致。

## 6. 完整矩阵执行过程

### 6.1 保留的内存门槛停止证据

第一次单进程全矩阵执行在完成 32/45 次策略后，于 `w10_early / B4` 前检测到可用内存 1.772 GiB，严格按 2 GiB 门槛停止。状态为 `ABORTED_MEMORY_GATE`，没有降低门槛。证据目录为 `research_runs/sia_synergy/b3_multiworkload_medium/full_run1/`。

之后只改变实验编排：每个 cohort 使用新的 Python 进程，cohort 内仍严格按 B1→SY→B4→B5→B3 顺序运行。算法、workload、容量、Medium 参数、solver 参数和每策略 5000 轮上限均未改变。隔离运行目录为 `research_runs/sia_synergy/b3_multiworkload_medium/isolated/`。

### 6.2 每 cohort 平均 JCT

| Cohort | Validation | B1 | SY adapted | B4 | B5 | B3 |
|---|---|---:|---:|---:|---:|---:|
| w1_early | PASS | 2500.50 | 5780.60 | 2437.50 | 2403.35 | 2154.55 |
| w1_middle | PASS | 1830.25 | 6383.15 | 2005.00 | 1819.80 | 1814.25 |
| w1_late | PASS | 936.40 | 3533.50 | 934.60 | 936.05 | 919.35 |
| w5_early | PASS | 1719.00 | 9548.10 | 1708.90 | 1719.00 | 1540.60 |
| w5_middle | PASS | 1472.45 | 6531.15 | 1448.85 | 1492.30 | 1436.25 |
| w5_late | PASS | 1473.30 | 2739.85 | 1466.40 | 1473.25 | 1366.00 |
| w10_early | PASS | 1647.25 | 4492.10 | 1652.95 | 1656.75 | 1315.40 |
| w10_middle | PASS | 2839.55 | 5228.20 | 2838.35 | 2790.90 | 2127.90 |
| w10_late | **FAIL** | 3763.20 | **6486.32（仅 19 个已完成作业）** | 3758.25 | 3793.10 | 2773.15 |

`w10_late / SY` 的 6486.32 秒不能与其他完整均值直接比较，也没有用于 B3 vs SY 的九 cohort 汇总。

## 7. B3 跨 cohort 对照

### 7.1 完整配对结果

| Comparator | 完整 cohort | B3 win/tie/loss | 改善率中位数 | 范围 | 最差完整 cohort |
|---|---:|---:|---:|---:|---|
| B1 | 9 | 9/0/0 | 10.378% | 0.874%–26.309% | w1_middle，0.874% |
| B4 | 9 | 9/0/0 | 9.848% | 0.870%–26.212% | w5_middle，0.870% |
| B5 | 9 | 9/0/0 | 10.352% | 0.305%–26.890% | w1_middle，0.305% |
| SY adapted | 8 | 8/0/0 | 71.147% | 50.143%–83.865% | w5_late，50.143% |

SY 的计数是 8 个完整 pair，不能写成 9/9。SY adapted 与 B3 的差距同时包含固定 GPU demand、确定性异构 GPU 类型映射和 TUNE 适配语义的差异，不能归因于“官方 Synergy 不如 B3”。

九 cohort 等权平均 JCT：B1 2020.21 秒、B4 2027.87 秒、B5 2009.39 秒、B3 1716.38 秒。SY 仅在 8 个完整 cohort 上为 5529.58 秒，不能与上述九 cohort 均值作严格等样本比较。

### 7.2 Neutral 与 Medium 的相对变化

相对 Neutral，Medium 下 B3 对各基线的“跨 cohort 改善率中位数”变化：

| Comparator | Neutral median | Medium median | 增量 |
|---|---:|---:|---:|
| B1 | 7.512% | 10.378% | +2.866 pp |
| B4 | 7.512% | 9.848% | +2.337 pp |
| B5 | 7.483% | 10.352% | +2.869 pp |
| SY adapted | 58.197%（9 cohort） | 71.147%（8 完整 cohort） | +12.950 pp，但样本集合不同 |

Medium 不会对所有策略或 cohort 单调加速。相对 Neutral 的九 cohort 等权平均 JCT 变化为：B1 改善 19.21%、B4 改善 19.01%、B5 改善 19.40%、B3 改善 17.34%。`w1_early` 中所有 Sia 路径反而略慢；SY adapted 在 8 个完整 cohort 中均更慢，等权均值退化 28.50%。这说明响应模型与资源分配、GPU 选择和任务长尾共同作用，不能把 `rho>1` 简化成所有作业都加速。

## 8. 尾部、makespan 与长尾敏感性

每个 P95/P99 只基于 20 个作业，不是高精度总体尾延迟估计。

九 cohort 等权的统计均值：

| Policy | Cohorts | Median JCT | P95 JCT | P99 JCT | Makespan |
|---|---:|---:|---:|---:|---:|
| B1 | 9 | 701.94 | 8827.98 | 13367.46 | 17749.11 |
| B4 | 9 | 697.22 | 9136.78 | 13527.89 | 17872.44 |
| B5 | 9 | 696.17 | 8799.19 | 13586.68 | 17897.89 |
| B3 | 9 | **745.56** | **4932.29** | **11888.24** | **16365.67** |
| SY adapted | 8 完整 | 2463.69 | 18274.38 | 30031.58 | 35235.63 |

B3 的平均 median JCT 高于 B1/B4/B5，但 P95、P99 和 makespan 更低。这意味着 B3 的均值优势主要来自压缩长尾，而不是让多数短作业普遍变快。

逐作业计数也支持这一点。例如：

- `w10_middle` 中 B3 相对 B1 仅改善 6/20 个作业、退化 14/20 个，但平均 JCT 改善 25.06%；`deepspeech2-83` 单作业贡献 16687 秒。移除该作业后诊断改善率变为 **-6.91%**。
- `w10_early` 相对 B1 的 20.15% 改善主要由 `deepspeech2-16` 的 6887 秒贡献；移除后变为 **-1.15%**。
- `w10_late` 相对 B1 的 26.31% 改善主要由 `deepspeech2-159` 的 17747 秒贡献；移除后仍有 3.92% 改善。
- `w1_early` 相对 B5 的 10.35% 改善主要由 `deepspeech2-9` 的 8429 秒贡献；移除后变为 **-10.07%**。
- 历史关注的 `deepspeech2-31` 不属于本轮九个 cohort，因此没有再次主导本轮结果。

对 B1/B4/B5 的完整配对中，没有出现“B3 平均 JCT 改善但 P95/P99/makespan 任一变差”的 cohort。相对 SY adapted，`w5_late` 出现平均 JCT 改善但 makespan 变差的冲突。

## 9. 资源、等待与调度事件

以下资源小时为完整 cohort 的等权平均；SY 只有 8 个完整 cohort：

| Policy | Cohorts | GPU-h | CPU-core-h | RAM-GB-h | Restart 总数 | Wasted GPU-s | 总等待 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 9 | 43.214 | 43.214 | 5401.779 | 426 | 105640 | 74340 |
| SY adapted | 8 | 70.961 | 70.961 | 8846.718 | 14 | 64520 | 504840（含删失 run 的等待累计） |
| B4 | 9 | 43.339 | 44.320 | 5341.670 | 431 | 108105 | 75780 |
| B5 | 9 | 43.138 | 43.270 | 5386.979 | 431 | 105475 | 74220 |
| B3 | 9 | 34.459 | 41.971 | 4245.251 | 449 | 91600 | 6000 |

B3 更低的 GPU/RAM 小时和等待伴随更高的 restart 总数，不能只把资源小时下降解读为无代价效率提升。

节点级 `CPU 饱和且 GPU 空闲` 是逐节点逐轮事件，不等同于作业等待秒数。累计事件数为 B1 57、SY 0、B4 190、B5 75、B3 788；对应真正有未分配作业且原因为 CPU 饱和/GPU 空闲的等待轮为 103、0、128、113、91。B3 可以同时出现大量节点级局部饱和事件和很少的作业等待，这是统计单位不同，不构成容量违规。

TUNE 累计操作：

- SY adapted：`ALLOCATE_PROFILE_DEMAND=15`、`DOWNGRADE_CURRENT_TO_FAIR=13276`、`WAIT_FINAL_RETRY_INFEASIBLE=8405`；
- B5：`ALLOCATE_PROFILE_DEMAND=84`、`ATOMIC_PEER_DOWNGRADE_TO_FAIR=48`、`DOWNGRADE_CURRENT_TO_FAIR=4782`、`WAIT_FINAL_RETRY_INFEASIBLE=1184`。

通过原始逐轮 JSON 对 `status` 字段计数：B1 为 `ORIGINAL_SIA_OPTIMAL=3006`、`OPTIMAL=2328`；B4 为 `OPTIMAL=5370`；B5 为 `SYNERGY_TUNE_ADAPTED=5382`；B3 为 `OPTIMAL=4916`；SY 为 `SYNERGY_TUNE_ADAPTED=19402`。44 个完整结果为 `SUCCESS`，唯一顶层非成功状态为 `ROUND_LIMIT_WITH_UNFINISHED_JOBS`。本轮未观察到名为 `FALLBACK`、`NO_COLUMNS` 或 `MIP_NO*` 的真实 fallback status。

## 10. `w10_late / SY` 删失根因

冻结上限处状态：

| 项目 | 数值 |
|---|---:|
| 完成作业 | 19/20 |
| 轮数 / 模拟时间 | 5000 / 300000 s |
| 未完成作业 | `imagenet-154` |
| epoch | 89 |
| 当前 progress | 571457.7119 |
| 完成 progress | 576525.5949 |
| 尚差 | 5067.8830 |
| 该作业 rho | 0.6324555320（运行期固定） |
| 等待时间 | 10140 s |
| GPU config changes | 0 |

该作业已进入最后一个 epoch，仍有正进度，不是死锁。失败类型是“冻结观察窗不足以覆盖 SY adapted 的这一长尾”，不是容量超配、响应公式错误、checkpoint 误推进或 solver exception。

科学处理规则：

1. 保留 5000 轮失败产物；
2. 不把 19 个完成作业的均值当成完整 cohort 均值；
3. 不为未完成作业估算或插补 JCT；
4. 不在看到结果后提高轮数并混入主实验；
5. 若研究者希望补齐 SY，必须另行预注册统一的新观察窗，并对五策略同样适用，而不是只续跑 SY。

## 11. 正确性与测试结果

最终分析 commit 上执行完整测试：**41 tests，41 passed，0 failure，0 error，7.335 s**。PMP 测试仍出现既有 autograd `invalid value encountered in log` RuntimeWarning，但测试通过。

8 个完整 cohort 的验证器均为 `PASS`。`w10_late` 验证器只有以下两项失败：

- `all_20_jobs_complete=false`；
- `no_round_limit=false`。

其余检查均通过，包括：无容量违规、Medium 元数据、`rho` 正值有限、公式严格一致、Goodput 单次倍率、逐节点瞬时与时间加权容量、B1 intent、B4/B5 GPU config 冻结、SY adapted core、B3 多 GPU 候选、checkpoint/restore、progress 单调、实际非恒一 `rho`、多个辅助资源点、竞争证据字段和原始 Sia 回归。

45 次运行的 `capacity_violations` 总数为 0。原始 Sia official/experimental 历史回归 SHA-256 仍一致：`F81E0B22C76734E5264742A7231687F366CE77B4D3448155E444A64D42E02DFF`。

## 12. 运行命令

响应冻结：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe freeze_medium_response_manifest.py `
  --preregistration joint_prototype\multiworkload_medium_preregistration.json `
  --synergy-models D:\科研启航\Sia\research_sources\Synergy\simulator\models `
  --output D:\科研启航\Sia\research_runs\sia_synergy\b3_multiworkload_medium\response_model_manifest_isolated.json
```

每个 cohort 在新 Python 进程中运行；`<cohort>` 依次为九个预注册 ID：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe run_b3_multiworkload_neutral.py `
  --preregistration joint_prototype\multiworkload_medium_preregistration.json `
  --synergy-models D:\科研启航\Sia\research_sources\Synergy\simulator\models `
  --previous-regression D:\科研启航\Sia\research_runs\sia_synergy\cpu_bottleneck_smoke `
  --output D:\科研启航\Sia\research_runs\sia_synergy\b3_multiworkload_medium\isolated\<cohort> `
  --cohort-ids <cohort>
```

删失审计使用 `analyze_b3_multiworkload_medium_censored.py` 读取九个独立根目录；该工具明确拒绝插补未完成 JCT。

## 13. 关键产物与 hash

- `research_runs/sia_synergy/b3_multiworkload_medium/response_model_manifest_isolated.json`：`A46B5443F515DECA8FD992A7894BF67C1F93DFD7293AD6E985A7F395D5659DE5`
- `research_runs/sia_synergy/b3_multiworkload_medium/aggregate_summary_censored.json`：`481E7B4B1E9B8B6748FFB966AE2605DC1CBFA78D7A864313F5D8427AF09CE527`
- `research_runs/sia_synergy/b3_multiworkload_medium/audit_manifest.json`
- `research_runs/sia_synergy/b3_multiworkload_medium/test_summary.json`
- `research_runs/sia_synergy/b3_multiworkload_medium/single_w1_early/`：单 cohort 通过证据
- `research_runs/sia_synergy/b3_multiworkload_medium/full_run1/`：2 GiB 门槛安全停止证据
- `research_runs/sia_synergy/b3_multiworkload_medium/isolated/`：九 cohort、45 次策略的逐轮 JSON、日志与 manifest

各 cohort 的 workload/config/validation/result hash 保存在对应 `run_manifest.json`、`policy_manifest.json` 和机器可读汇总中，没有在 Markdown 中重复 45 组长 hash。

## 14. 最终判断

本轮已经证明：

1. 既有 `alpha=0.5` Medium 模型计算路径可审计，响应确实进入实际训练推进且只应用一次；
2. B3 在九个 cohort 均稳定完成，资源约束与 checkpoint 语义通过；
3. B3 相对 B1/B4/B5 的平均 JCT 改善在九个 cohort 都出现，Medium 下改善率中位数比 Neutral 高约 2.3–2.9 个百分点；
4. 改善主要集中在长尾，不能表述为多数作业普遍提速；B3 的平均 median JCT 反而略高；
5. 当前 Medium 五策略矩阵仍因一个 SY adapted 长尾被 5000 轮截断而不完整。

因此，**不建议把本轮升级为正式性能结论，也不建议自动进入 Low/High 或 160-job**。下一步首先需要研究者决定是否接受右删失分析，或另行预注册对所有策略一致、更长的观察窗后重跑完整 Medium 矩阵。
