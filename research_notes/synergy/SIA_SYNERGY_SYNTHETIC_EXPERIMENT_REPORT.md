# Sia + Synergy 受控合成 S1/S2 实验报告

- 日期：2026-10-23（Asia/Shanghai）
- 报告状态：**完成；20-job 8/8 配置成功**
- 已完成：实现、15 项单元测试、3-job 手工门槛、20-job S1/S2 × 四场景、统一正确性和结果分析
- 结论边界：**SYNTHETIC RESOURCE MODEL — NOT REAL SATURN PERFORMANCE**

## 1. 仓库和保护状态

`sia-synergy-experiments/`：

- 分支：`exp/sia-synergy`
- 基线 HEAD：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 开始状态：干净
- 当前改动：只包含本任务的 synthetic config、S1/S2、测试和运行/分析脚本

复核时 `official-artifacts/` 仍为干净的 `main...origin/main`。LAS、AFS、Lazer 和官方 Synergy 源目录只保留任务开始前已经存在的状态；本任务没有写入这些目录。

## 2. 环境

- OS：Windows，工作区 `D:\科研启航\Sia`
- Python：`.venv-sia-baseline`，Python 3.10.10
- NumPy 1.23.5、Pandas 1.5.3、SciPy 1.10.1、CVXPY 1.3.2
- Sia GPU MIP：CBC 2.10.10（Artifact 当前源码实际硬编码路径）
- S2 auxiliary MILP：SciPy/HiGHS
- 所有 20-job 运行器强制串行，并设置 BLAS/OMP 线程变量为 1

2026-10-23 初次安全检查依次得到 1.890、1.122 和 1.609 GiB 可用内存，runner 按规则拒绝启动。内存回升到 2.429 GiB 后才开始 20-job suite。前 7 项成功后，`mixed/S2` 启动前内存降到 1.830 GiB，runner 记录 `SKIPPED_LOW_MEMORY` 并停止；内存回升到 2.343 GiB 后用 `--resume` 只补跑最后一项，未覆盖前 7 项。没有运行任何 160-job trace，也没有终止用户/系统进程腾内存。

## 3. 实现摘要

完整模型见 `SIA_SYNERGY_SYNTHETIC_DESIGN.md`。核心行为是：

- S1 按每 GPU 3 synthetic CPU cores、62.5 GB RAM 记账，倍率恒 1。
- S2 固定 Sia GPU type/count/node IDs，以每作业一个离散候选、逐节点 CPU/RAM 容量、`rho>=1` 为约束，最大化当前总 `sia_goodput*rho`。
- CPU curve 在 1.5× CPU 饱和到 1.30×；RAM curve 在 1.5× RAM 饱和到 1.25×。
- profile 由末尾 job index 模 3 分组，不用应用标签或未来信息。
- `rho` 只在 productive progress 上乘一次；checkpoint/restart 不加速。
- auxiliary reallocation overhead 当前显式为未建模、0 秒，不能外推真实成本。

修改/新增文件：

| 文件 | 修改 |
|---|---|
| `sia-simulator/synthetic_resource_model.json` | synthetic schema、容量、响应曲线与来源。 |
| `sia-simulator/synergy_synthetic.py` | `SyntheticAuxAllocator`、S1、S2 MILP、容量审计、fallback。 |
| `sia-simulator/job.py` | `estimate_current_goodput()`；`step()` 增加默认 1 的倍率并只应用一次。 |
| `sia-simulator/cluster.py` | allocator 插入、CPU/RAM 积分、日志和汇总指标。 |
| `sia-simulator/multi_simulator.py` | `--synthetic_mode/scenario/resource_config/summary`。 |
| `sia-simulator/tests/test_synergy_synthetic.py` | 15 项测试。 |
| `sia-simulator/run_synergy_synthetic_suite.py` | 20-job 串行 runner 与逐次内存门槛。 |
| `sia-simulator/analyze_synergy_synthetic.py` | 8 配置比较和有效性校验。 |

## 4. 正确性测试

命令：

```powershell
cd D:\科研启航\Sia\sia-synergy-experiments\sia-simulator
..\..\.venv-sia-baseline\Scripts\python.exe -m unittest discover -s tests -v
```

结果：最终退出码 0，15/15 通过，墙钟 2.78 秒（测试本体 0.020 秒）。覆盖：

1. 配置必须显式 synthetic，baseline 倍率为 1。
2. S1 GPU-proportional CPU/RAM。
3. neutral 中 S1/S2 resource map 一致。
4. CPU-sensitive 额外 CPU 和 1.30× 上限。
5. memory-sensitive 额外 RAM 和 1.25× 上限。
6. insensitive 无额外资源和性能收益。
7. GPU/CPU/RAM 每节点不超限。
8. 多节点 per-GPU 比一致。
9. 资源饱和后倍率不增长。
10. `effective_goodput = sia_goodput*rho` 只乘一次。
11. 满节点无 spare resource 时回到 baseline choice。
12. profile 分配与应用标签无关。
13. 集群 schema 不匹配时拒绝启动。
14. checkpoint delay 下 1.3× multiplier 不改变时间、wasted GPU-seconds 或 restart count。
15. auxiliary MILP 失败时显式记录原因并整体回退 S1。

## 5. 3-job 手工实验

输入：`research_runs/sia_synergy/synthetic/inputs/saturn_workload_3.csv`，即官方 Saturn workload-1 的前三个到达。固定 Sia 参数：

```text
--policy=sia --policy_p_val=-0.5
--mip_lambda_a=0.01 --mip_lambda_n=1.1
--interval=60 --project_throughputs --share_max_replicas
```

运行清单与退出码保存在 `research_runs/sia_synergy/synthetic/manual/manual_run_manifest.json`。第一次原始对照的模拟完成 3/3，但最终写日志时发现 `manual/` 尚未创建，退出码 1；创建目录后重跑成功。该错误没有修改算法或输出数据。

### 5.1 Neutral 退化门槛

原始 Sia、S1-neutral、S2-neutral 都有 23 个调度轮。去除仅供合成审计的附加字段后：

```text
original core JSONL == S1-neutral core JSONL: true
original core JSONL == S2-neutral core JSONL: true
```

这不是只比较最终均值，而是逐轮比较所有原字段，包括 GPU allocation、placement、progress、epoch、batch、completion、restart 和 cluster。

| 指标 | S1 neutral | S2 neutral |
|---|---:|---:|
| 完成作业 | 3/3 | 3/3 |
| 平均 JCT | 584.333 s | 584.333 s |
| P95 JCT | 607.600 s | 607.600 s |
| P99 JCT | 609.520 s | 609.520 s |
| Makespan | 932.000 s | 932.000 s |
| GPU-hours | 2.616667 | 2.616667 |
| CPU-core-hours | 7.850000 | 7.850000 |
| RAM-GB-hours | 163.541667 | 163.541667 |
| 重启 / 迁移 | 10 / 0 | 10 / 0 |
| Sia GPU decision changes | 14 | 14 |
| auxiliary choice reallocations | 0 | 0 |
| 容量违规 | 0 | 0 |
| auxiliary solve total / P99 | 0 / 0 | 0.019701 / 0.006823 s |

因此满足“若 S1 不退化则停止”的进入条件，也验证原 `--policy=sia` 保持历史行为。

### 5.2 Mixed 行为

| Job | 合成标签 | observed choices | 最大倍率 | S1 JCT | S2 mixed JCT | 变化 |
|---|---|---|---:|---:|---:|---:|
| `cifar10-0` | CPU-sensitive | baseline / CPU 1.5× | 1.30 | 557 s | 541 s | -16 s |
| `cifar10-1` | memory-sensitive | baseline / RAM 1.5× | 1.25 | 586 s | 566 s | -20 s |
| `cifar10-2` | insensitive | baseline only | 1.00 | 610 s | 610 s | 0 s |

应用名称只用于显示；分组来自末尾索引 0/1/2，不代表 CIFAR10 实测敏感度。

| 指标 | S1 neutral | S2 mixed | 变化 |
|---|---:|---:|---:|
| 平均 JCT | 584.333 s | 572.333 s | -12.000 s (-2.05%) |
| P95 JCT | 607.600 s | 605.600 s | -2.000 s |
| P99 JCT | 609.520 s | 609.120 s | -0.400 s |
| Makespan | 932.000 s | 932.000 s | 0 |
| GPU-hours | 2.616667 | 2.583333 | -0.033333 |
| CPU-core-hours | 7.850000 | 8.025000 | +0.175000 |
| RAM-GB-hours | 163.541667 | 171.354167 | +7.812500 |
| 重启 / 迁移 | 10 / 0 | 10 / 0 | 0 / 0 |
| Sia GPU decision changes | 14 | 14 | 0 |
| S2 choice reallocations | — | 2 | — |
| 容量违规 | 0 | 0 | 0 |

收益只出现在获得额外对应资源的 CPU/RAM cohort；不敏感作业完全不变。重启和 GPU decision change 没有增加，说明该手工案例的变化来自 auxiliary resource multiplier，而不是额外 GPU 迁移。

S2 mixed 的逐节点时间平均利用率：

| 节点 | CPU | RAM |
|---|---:|---:|
| aws/0–5 | 0 | 0 |
| dgx-ext/0 | 55.978% | 53.261% |
| dgx-ext/1 | 29.348% | 34.239% |
| rtx/0 | 1.902% | 1.902% |
| rtx/1–2 | 0 | 0 |

所有调度轮的容量审计均通过。S2 mixed 调用 auxiliary MILP 16 次，总计 0.020459 秒，P99 0.002934 秒；发生 2 次 choice change 和 11 次完整 resource-map change，后者包括原 Sia GPU placement 变化。

## 6. 20-job 预注册 suite

输入已创建并核对 SHA-256：

```text
research_runs/sia_synergy/synthetic/inputs/saturn_workload_20.csv
SHA-256 4F5A276525EBDA40A8243D92AF1A83C4323C859FEC65DE023E748BDFDBB2FE46
```

它是官方 workload-1 的表头和前 20 行；四个场景不会改变 arrival、训练量或 GPU 参数。预注册运行器为：

```powershell
cd D:\科研启航\Sia\sia-synergy-experiments\sia-simulator
..\..\.venv-sia-baseline\Scripts\python.exe run_synergy_synthetic_suite.py `
  --workspace D:\科研启航\Sia --minimum-available-gib 2
```

runner 在每个配置前检查门槛，严格串行运行并保存 command、exit code、duration、stdout、JSONL 和 summary。第八项使用恢复命令，只跳过 manifest 中已有的 SUCCESS 项：

```powershell
..\..\.venv-sia-baseline\Scripts\python.exe run_synergy_synthetic_suite.py `
  --workspace D:\科研启航\Sia --minimum-available-gib 2 --resume

..\..\.venv-sia-baseline\Scripts\python.exe analyze_synergy_synthetic.py `
  ..\..\research_runs\sia_synergy\synthetic `
  --historical-sia-log ..\..\research_runs\sia_afs\smoke\sia\workload-1-first20.log
```

### 6.1 命令状态

| 配置 | 退出码 | 墙钟 | 启动前 / 结束后可用内存 |
|---|---:|---:|---:|
| neutral / S1 | 0 | 67.300 s | 2.354 / 2.232 GiB |
| neutral / S2 | 0 | 75.071 s | 2.232 / 2.171 GiB |
| cpu_sensitive / S1 | 0 | 82.315 s | 2.168 / 2.778 GiB |
| cpu_sensitive / S2 | 0 | 68.785 s | 2.778 / 2.718 GiB |
| memory_sensitive / S1 | 0 | 71.550 s | 2.718 / 2.467 GiB |
| memory_sensitive / S2 | 0 | 82.745 s | 2.467 / 2.040 GiB |
| mixed / S1 | 0 | 75.738 s | 2.040 / 1.830 GiB |
| mixed / S2（resume） | 0 | 40.335 s | 2.302 / 2.617 GiB |

完整参数数组、时间戳、日志路径和中途低内存事件在 `run_manifest.json`。

### 6.2 正确性与退化

统一分析器的 12 个 suite validation 全部为 `true`：

- neutral S1/S2 逐轮 core JSON 完全相同；
- neutral S1 与已有未修改 `--policy=sia` 20-job 历史日志逐轮完全相同，均 149 轮、最后时间戳 8940；
- 四个 S1 场景的 core JSON 完全相同；
- 8/8 均完成 20/20 作业；
- 所有轮次零 CPU/RAM/GPU 容量违规；
- S1、neutral 和 insensitive 倍率始终为 1；
- 三个敏感场景均实际出现倍率大于 1 的 allocation；
- 零 solver fallback；
- 所有运行模拟时间严格单调递增。

因此原 Sia GPU allocator、neutral progress/JCT/restart 历史行为与 S1 退化门槛通过。

### 6.3 绝对结果与 S1/S2 对比

S1 在四个场景完全相同，以下重复列出以保持 8 配置口径。

| 场景 | 模式 | 完成 | 平均 JCT | P95 | P99 | Makespan | GPU-h | CPU-core-h | RAM-GB-h |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | S1 | 20 | 1393.40 | 2944.75 | 6573.75 | 8528 | 64.6500 | 193.9500 | 4040.6250 |
| neutral | S2 | 20 | 1393.40 | 2944.75 | 6573.75 | 8528 | 64.6500 | 193.9500 | 4040.6250 |
| cpu_sensitive | S1 | 20 | 1393.40 | 2944.75 | 6573.75 | 8528 | 64.6500 | 193.9500 | 4040.6250 |
| cpu_sensitive | S2 | 20 | 1352.55 | 3076.45 | 7111.29 | 9167 | 60.2833 | 182.7250 | 3767.7083 |
| memory_sensitive | S1 | 20 | 1393.40 | 2944.75 | 6573.75 | 8528 | 64.6500 | 193.9500 | 4040.6250 |
| memory_sensitive | S2 | 20 | 1385.60 | 2938.95 | 6479.79 | 8412 | 64.1167 | 192.3500 | 4042.1875 |
| mixed | S1 | 20 | 1393.40 | 2944.75 | 6573.75 | 8528 | 64.6500 | 193.9500 | 4040.6250 |
| mixed | S2 | 20 | 1347.90 | 3110.90 | 6559.78 | 8469 | 59.9167 | 181.3750 | 3785.4167 |

相对 S1：

| 场景 | 平均 JCT | P95 | P99 | Makespan | GPU-hours |
|---|---:|---:|---:|---:|---:|
| neutral | 0 | 0 | 0 | 0 | 0 |
| cpu_sensitive | -2.93% | **+4.47%** | **+8.18%** | **+7.49%** | -6.75% |
| memory_sensitive | -0.56% | -0.20% | -1.43% | -1.36% | -0.82% |
| mixed | -3.27% | **+5.64%** | -0.21% | -0.69% | -7.32% |

CPU-sensitive 的均值和 GPU-hours 改善，但尾部与 makespan 明显退化；mixed 的均值、P99、makespan 和 GPU-hours 改善，但 P95 退化。这些相反方向不能用单个平均数掩盖。

### 6.4 Restart、GPU 决策和 auxiliary 开销

| 场景 | S1 restart | S2 restart | S1/S2 GPU decision changes | S2 choice realloc | S2 map realloc | solve calls | solve total / P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 87 | 87 | 133 / 133 | 0 | 113 | 142 | 0.173095 / 0.001621 s |
| cpu_sensitive | 87 | 73 | 133 / 111 | 11 | 94 | 153 | 0.171191 / 0.001815 s |
| memory_sensitive | 87 | 87 | 133 / 133 | 18 | 123 | 140 | 0.196835 / 0.002335 s |
| mixed | 87 | 77 | 133 / 117 | 32 | 110 | 141 | 0.100191 / 0.001108 s |

没有场景增加 GPU restart；neutral 完全相同。敏感场景完成时间变化会改变后续 runnable set，所以 CPU/mixed 的 Sia GPU decision count 和 restart count 间接下降；这不是 S2 修改 GPU MIP 目标。`map realloc` 包含 Sia placement 改变引起的 node map 变化，不能当作纯 S2 choice change。

### 6.5 任务级收益和损害

完整 20 作业 × 四场景数据位于 `synthetic_comparison.json -> task_level`。直接敏感 cohort 的 JCT 如下（单位秒，括号为 S2-S1）：

| 场景 | 直接敏感作业结果 |
|---|---|
| CPU | `cifar10-0` 557→541 (-16)；`-3` 674→632 (-42)；`-6` 668→615 (-53)；`-15` 668→650 (-18)；`-18` 676→642 (-34)；`deepspeech2-12` 2569→2344 (-225)；`deepspeech2-9` 2706→2811 (**+105**) |
| Memory | `cifar10-1` 586→566 (-20)；`-4` 663→667 (+4)；`-10` 621→594 (-27)；`-13` 793→796 (+3)；`-19` 679→679 (0)；`deepspeech2-16` 2409→2409 (0)；`yolov3-7` 7481→7365 (-116) |
| Mixed | `cifar10-0` -16；`-1` -20；`-3` -2；`-4` -35；`-6` -4；`-10` -27；`-13` -175；`-15` **+213**；`-18` -26；`-19` -42；`deepspeech2-9` **+178**；`-12` -368；`-16` **+145**；`yolov3-7` -59 |

任务级计数：CPU 场景 15 改善、2 不变、3 变差；memory 场景 3 改善、15 不变、2 变差；mixed 场景 14 改善、3 不变、3 变差。CPU 场景中两个 insensitive 作业也变差（`deepspeech2-8 +188`、`yolov3-7 +639`）；它们的倍率始终为 1，JCT 变化来自其他作业提前完成后引起的 Sia runnable-state/placement 轨迹变化，而不是错误赋予性能倍率。

CPU 场景的 7/7 CPU cohort 曾获得 >1 倍率；memory 场景 6/7 memory cohort 曾获得 >1（一个作业始终没有节点 spare RAM）；mixed 的 7/7 CPU 与 7/7 memory cohort 都曾获得 >1；所有 insensitive cohort 的最大倍率均为 1。

### 6.6 每节点 CPU/RAM 利用率

以下为 S2 全模拟期时间平均 CPU/RAM 利用率；每个 summary JSON 保留未舍入值。

| 场景 | aws/0..5 CPU/RAM (%) | dgx-ext/0,1 CPU/RAM (%) | rtx/0..2 CPU/RAM (%) |
|---|---|---|---|
| neutral | 14.09/14.09, 12.75/12.75, 12.08/12.08, 10.91/10.91, 6.88/6.88, 6.71/6.71 | 92.62/92.62, 85.23/85.23 | 42.87/42.87, 40.02/40.02, 32.97/32.97 |
| CPU | 14.92/14.84, 4.30/4.22, 1.41/1.41, 1.25/1.25, 1.25/1.25, 0/0 | 84.14/83.59, 75.55/74.53 | 46.13/45.55, 43.48/42.97, 24.65/24.45 |
| Memory | 14.29/14.37, 12.93/12.93, 12.24/12.24, 11.05/11.05, 6.97/7.06, 6.80/6.80 | 92.52/92.52, 85.03/87.41 | 43.45/43.54, 40.56/40.56, 33.42/33.72 |
| Mixed | 9.63/9.54, 5.83/5.83, 4.22/4.22, 4.05/4.05, 4.05/4.05, 0.17/0.25 | 93.16/92.57, 85.14/87.42 | 41.93/41.89, 39.10/38.94, 33.07/32.14 |

这些是 synthetic lease utilization，不是物理主机监控值。所有逐轮峰值均由 capacity audit 验证不超过 100%。

## 7. 最终判定

- 正确性：通过；机制层可解释且所有不变量通过。
- 综合结果：**MIXED**。
- 判定理由：memory 场景整体小幅改善；CPU 和 mixed 的均值/GPU-hours 有收益，但 CPU 的 P99/Makespan 与 mixed 的 P95 出现超过 5% 的明显退化，并存在敏感和不敏感任务级损害。
- 正式 Saturn 性能主张：**禁止**。这里只证明在预设合成曲线中，资源层能正确退化、满足容量并产生可解释但非一致有利的行为。

为把任务定义中的“明显退化”操作化，汇总器保守地把任一敏感场景 P95、P99 或 Makespan 增加超过 5% 判为 `MIXED`；该阈值只用于结果分类，不参与调度、倍率或任何实验参数。

没有证据表明真实 CIFAR10、BERT、ImageNet、YOLOv3 或 DeepSpeech2 在真实 Saturn 主机上会得到这些倍率，也不能据此声称 SOTA。

## 8. 产物

- 设计：`research_notes/synergy/SIA_SYNERGY_SYNTHETIC_DESIGN.md`
- 本报告：`research_notes/synergy/SIA_SYNERGY_SYNTHETIC_EXPERIMENT_REPORT.md`
- 合成配置与代码：`sia-synergy-experiments/sia-simulator/`
- 输入、手工 JSONL/summary、manifest：`research_runs/sia_synergy/synthetic/`
- 20-job `runs/`、`run_logs/`、`summaries/` 含 8 套 JSONL/stdout/summary；`run_manifest.json` 记录首次低内存停止和 resume；`synthetic_comparison.json` 含完整对比、任务级结果和验证。

## 9. 是否值得收集真实 Profile

**值得收集，但不应立即进入 S3。** 本轮证明了接口和约束可用，也暴露了重要风险：即使每个 running job 的瞬时 effective goodput 不低于 baseline，GPU 调度状态反馈仍可能让个别作业和尾延迟变差。真实 profiling 应先回答这些 synthetic gains 是否存在，并测量辅助资源重分配开销。最低数据要求仍是：三类真实节点的同口径 CPU/RAM 容量，以及每个应用 × GPU 类型 × 合法 GPU config 的同基准 CPU/RAM response。没有这些数据前，不运行 160-job trace、不实现 S3。
