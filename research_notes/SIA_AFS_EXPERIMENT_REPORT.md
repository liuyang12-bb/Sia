# Sia + AFS-P 模块组合实验报告

- 实验日期：2026-10-23（Asia/Shanghai）
- 研究基准：Sia SOSP 2023 Artifact，commit `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 实验 worktree：`sia-afs-experiments/`，分支 `exp/sia-afs`
- AFS 论文：Changho Hwang et al., *Elastic Resource Sharing for Distributed Deep Learning*, NSDI 2021
- Python：`.venv-sia-baseline/Scripts/python.exe`（Python 3.10.10）
- 预注册 trace：Saturn `workload-1.csv`、`workload-5.csv`、`workload-10.csv`

> 本报告中的 `Sia + AFS-P` 是“论文驱动的同构 AFS-P 份额分配 + Sia 异构合法配置映射”。它不是作者官方实现，也不是把 AFS-P 简化成 Goodput 权重。本轮不包含 LAS、参数搜索或双模块实验。

## 1. 基线保护与实验边界

执行前确认：

- `official-artifacts/` 位于指定 commit，工作区干净；
- LAS worktree `sia-experiments/` 及 `research_runs/sia_las_weighted/` 保留原状；
- AFS 从原始 commit 新建独立 worktree `sia-afs-experiments/`，没有叠加 LAS；
- 三条官方 trace 均存在且各含 160 个作业，没有按结果替换；
- 沿用既有隔离 Python、profile、CBC 与单线程数值库设置；
- 每条完整模拟串行运行，启动前要求可用内存不低于 2 GiB；
- 未修改 trace、性能画像或 `Job.step()` 的实际运行时间推进逻辑。

阶段 0 的机器记录为 8 个物理核、16 个逻辑核、13.69 GiB 内存，D 盘可用空间约 29.97 GiB。为读取和渲染论文 PDF，仅在既有隔离环境增加 `pypdf==6.21.2`、`PyMuPDF==1.28.2` 和依赖 `typing_extensions==4.16.0`；没有修改系统 Python。

详细算法审计见 `research_notes/AFS_MODULE_AUDIT.md`。

## 2. 原始 AFS-P 与本实验适配

### 2.1 论文算法

AFS 的基本思想不是最大化当前总吞吐量，而是用“当前份额将延续到未来”的保守假设，比较额外 GPU 对两项作业完成时间的影响。论文 Section 3.2（p. 725）式 (3) 为：

```text
(p'_b - p_b) / p'_b > (p'_a - p_a) / p_a
```

两侧分母有意不对称。AFS-L 的 Algorithm 1（p. 726）先给作业各一张 GPU，再逐张调用 `TopPriority` 分配剩余 GPU。

AFS-P 在不知道作业长度时，对式 (3) 的两种长短次序都进行判断；若两种方向都不能决定，随机选择。它另用 processor sharing 模拟 AFS-L 的第一阶段：计数器累计“被调度的墙钟时间”而非 GPU-seconds；当作业数不少于 GPU 数时，只给最低计数作业各一张 GPU，并在一个 unit time 内不抢占。论文完整 trace 的 unit time 为 2 小时（7200 秒）。当作业数少于 GPU 数时忽略计数器，先每作业一张 GPU，再按修改后的 `TopPriority` 分配剩余 GPU。

论文假设同构 GPU、BSP 作业、任意整数 GPU 份额和低开销弹性调整；未定义异构 GPU 类型选择，也没有 Sia 的 checkpoint/restart 代价。没有找到可可靠归属给作者的公开 CoDDL/AFS 实现，因此本实现以论文公式、伪代码和系统章节为依据，并明确标记新增适配层。

### 2.2 两层实现

第一层 `allocate_afs_p_homogeneous()` 是不依赖 GPU 类型的参考核心：输入每个作业的整数 Goodput 曲线、虚拟 GPU 总数、已调度墙钟时间、时间片和随机种子，输出整数份额、阶段与 dispatch counters。Sia 的 Goodput 用作论文吞吐量 `p_j(r)` 的进度率代理，因为模拟器的训练进度正由 Goodput 推进；这不是对 Sia 目标函数加权。

第二层把虚拟份额映射到 Sia 合法配置：

1. 从各 GPU 类型可观测或可投影的 Goodput 构造虚拟份额曲线；冷启动按论文 Section 4.5 线性外推，并在尚无模型时限制到 `1/n` 公平份额。
2. 先运行同构 AFS-P 核心得到目标份额。
3. 用整数规划为每个正份额作业选择一个合法 `(GPU 类型, 节点数, GPU 数)` 配置。
4. 目标优先级依次是最小份额偏差、充分利用不超过虚拟总份额的 GPU、复用原配置，最后才用归一化 Goodput 解决异构候选并列。
5. 硬约束每类 GPU 容量、每作业单一配置、scale unit、最大副本数、虚拟总份额，以及 checkpoint/restore 中配置保持。
6. 最终 placement 复用 Sia 原逻辑；实际迁移/重分配仍由原 `Job.reallocate()` / `Job.migrate()` 收取代价。

合并三类 GPU 为虚拟池、Goodput 包络、异构 MIP、checkpoint 硬保持和 profile-report 事件均为本实验新增，不属于原始 AFS-P。

## 3. 代码修改

| 文件 | 位置 | 修改 |
|---|---:|---|
| `sia-simulator/sia_afs.py` | 28-172 | processor-sharing 计数、精确式 (3)、修改后的 `TopPriority` 与同构 AFS-P 参考分配器。 |
| `sia-simulator/sia_afs.py` | 175-313 | `SiaAFSPolicy`、事件状态、冷启动及异构 Goodput 包络。 |
| `sia-simulator/sia_afs.py` | 315-445 | 合法配置枚举、容量/checkpoint 约束及映射 MIP。 |
| `sia-simulator/sia_afs.py` | 447-598 | placement、事件式重算和结构化 `AFS_STATE` 审计记录。 |
| `sia-simulator/cluster.py` | 396-400 | 只读附加既有 `execution_time` 与 profile 计数到 `JobInfo`。 |
| `sia-simulator/multi_simulator.py` | 14, 38-49 | 新增独立 `--policy=sia_afs` 构造路径。 |
| `sia-simulator/multi_simulator.py` | 61-66, 145-170 | 保持原 batch-size/rigid 初始化，并新增 AFS 参数。 |
| `sia-simulator/tests/test_sia_afs.py` | 1-285 | 同构核心、异构映射、容量、checkpoint 与事件状态测试。 |
| `research_notes/calc_sia_afs_stats.py` | 仓库外 | 统一计算 JCT、容量、GPU 类型、任务级与调度时间统计。 |
| `research_notes/compare_sia_afs_stats.py` | 仓库外 | 对三条预注册 trace 生成绝对值、相对变化及逐任务对照。 |

参数：

| 参数 | 默认/正式值 | 含义 |
|---|---:|---|
| `--afs_processor_share_unit_seconds` | 7200 | AFS-P processor-sharing 时间片，单位秒。 |
| `--afs_seed` | 0 | 论文规定无法判断时随机选择的可复现实验种子。 |

为保持命令可比性，AFS 运行仍接受 `policy_p_val=-0.5`、`mip_lambda_a=0.01`、`mip_lambda_n=1.1`，但策略启动时明确记录这些 Sia 目标参数不进入 AFS 份额公式。

## 4. 正确性测试

最终命令：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe -m unittest discover -s sia-simulator/tests -p test_sia_afs.py -v
```

- 13/13 通过，退出码 0；测试框架报告 0.073 秒，完整命令墙钟 2.609 秒。
- 红灯阶段按预期因 `ModuleNotFoundError: sia_afs` 失败，随后才实现策略。
- 覆盖式 (3) 的不对称分母、单/双作业、高竞争、GPU 不足、非抢占时间片、scheduled-wall-time 语义、冷启动公平份额、异构合法配置、配置复用优先级、容量、无法映射、checkpoint 保持/成本字段不变和无事件复用。
- 日志：`research_notes/run_logs/sia_afs_unit_tests.stdout.log`。

原始 Sia 回归：同一 20 作业 trace 在 AFS worktree 下运行 `--policy=sia`，输出与上一轮保存的原始 Sia smoke JSONL 字节完全一致，SHA-256 均为 `FE5B7EF8D5C9D8A759B017F0EE9866D14415DB14B91E8A12572F23E0813D4B9A`。

## 5. 20 作业 smoke test

固定 trace 为官方 `workload-1.csv` 的表头加前 20 条作业：`research_runs/sia_afs/traces/workload-1-first20.csv`。

| 策略 | 完成 | 平均 JCT (s) | P95/P99 (s) | Makespan (s) | 重启/作业 | GPU-h/作业 | 容量违规 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sia | 20/20 | 1393.40 | 7481 / 7481 | 8528 | 4.35 | 3.2325 | 0 |
| Sia + AFS-P | 20/20 | 2442.35 | 21506 / 21506 | 22553 | 7.85 | 4.2083 | 0 |

最终 AFS smoke 有 376 轮状态记录、74 次实际映射、0 个未映射事件，全部为 `efficiency` 阶段；映射平均 47.049 ms，policy 平均 22.622 ms。平均 JCT、尾部、Makespan、重启和 GPU-hours 已显示明显退化，但 smoke 只作为正确性门槛，不能替代预注册三 trace 结论。

实现审查期间还保留了两个较早 smoke 产物。它们在“异构 Goodput 优先于配置复用”的目标顺序下完成 20/20，但该顺序不符合论文 Section 4.4 的先复用 placement 语义；修正目标层级后重新测试并生成上述正式 smoke。旧产物不纳入正式性能比较，也未被删除。

## 6. 三条完整 trace 对照

三条 Sia 基线已在本 worktree 重新运行。`workload-1` 还与历史基线 JSONL 字节一致，SHA-256 为 `C2EC4462ED57135770BCD33D4DB41CE91990E63DE03B2D8D6C3B29BB52DD699C`。

统计定义固定为：P95/P99 使用排序后 `int(q*N)` 下标，GPU-hours 按每轮日志分配乘 60 秒，和 Sia 基线使用相同定义。括号内是 AFS 相对 Sia 的变化；JCT、Makespan、重启和 GPU-hours 越低越好。

### 6.1 总体指标

| Trace | 策略 | 完成 | 平均 JCT (s) | P95 JCT (s) | P99 JCT (s) | Makespan (s) | 重启/作业 | GPU-h/作业 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| workload-1 | Sia | 160/160 | 2091.04 | 14365 | 27765 | 44859 | 4.1188 | 3.6116 |
|  | AFS-P | 160/160 | 3042.51 (+45.50%) | 18039 (+25.58%) | 41578 (+49.75%) | 64823 (+44.50%) | 8.1625 (+98.18%) | 4.7568 (+31.71%) |
| workload-5 | Sia | 160/160 | 2391.98 | 14469 | 29285 | 51088 | 4.3250 | 4.5446 |
|  | AFS-P | 160/160 | 3422.43 (+43.08%) | 19192 (+32.64%) | 38847 (+32.65%) | 75767 (+48.31%) | 7.6625 (+77.17%) | 5.7022 (+25.47%) |
| workload-10 | Sia | 160/160 | 2866.59 | 14444 | 45086 | 61391 | 4.6813 | 5.5378 |
|  | AFS-P | 160/160 | 3772.41 (+31.60%) | 18785 (+30.05%) | 67303 (+49.28%) | 99210 (+61.60%) | 7.3875 (+57.81%) | 7.1228 (+28.62%) |

候选没有在任何预注册 trace 上改善主要 JCT 指标。三条 trace 的平均 JCT 均退化 31.60% 以上，P99 退化 32.65%～49.75%，Makespan 退化 44.50%～61.60%。同时重启次数增加 57.81%～98.18%，GPU-hours 增加 25.47%～31.71%。因此结果不是“用资源换 JCT”的权衡，而是 JCT 和资源开销同时变差。

### 6.2 queue、contention 与容量

| Trace | 策略 | queue mean / max | contention mean / max | 最大 AWS / DGX-ext / RTX | 容量违规 |
|---|---|---:|---:|---:|---:|
| workload-1 | Sia | 0.00265 / 1 | 7.372 / 21 | 24 / 16 / 24 | 0 |
|  | AFS-P | 0 / 0 | 7.455 / 30 | 24 / 16 / 24 | 0 |
| workload-5 | Sia | 0.00939 / 1 | 7.480 / 38 | 24 / 16 / 24 | 0 |
|  | AFS-P | 0 / 0 | 7.222 / 46 | 24 / 16 / 24 | 0 |
| workload-10 | Sia | 0.00098 / 1 | 7.457 / 22 | 24 / 16 / 24 | 0 |
|  | AFS-P | 0 / 0 | 6.077 / 26 | 24 / 16 / 24 | 0 |

AFS-P 把极小的 queue 指标降为 0，并在 workload-5/10 降低 contention mean；但三条基线的 queue max 本来都只有 1，这些改善的绝对量很小。与此同时 contention max 全部升高，且 JCT、尾部和资源开销明显恶化，不能把 queue 的 -100% 相对值解释成有意义的总体收益。

三条 AFS 运行分别产生 1081、1263、1653 条状态记录，实际映射 512、498、541 次；所有映射状态均为 `optimal`，未映射事件和容量违规均为 0。三条运行的所有记录都处在 `efficiency` 阶段：每轮活跃作业数始终小于 64 张物理 GPU，正式 trace 没有触发 `n >= M` 的 processor-sharing 分支。该分支的正确性由单元/集成测试覆盖，但本轮性能结论只适用于实际触发的 efficiency 路径。

### 6.3 各应用平均 JCT

| Trace | 应用 | Sia (s) | AFS-P (s) | 变化 |
|---|---|---:|---:|---:|
| workload-1 | bert | 1051.83 | 3121.00 | +196.72% |
|  | cifar10 | 816.10 | 1262.60 | +54.71% |
|  | deepspeech2 | 2570.50 | 3476.20 | +35.23% |
|  | imagenet | 30807.00 | 44108.00 | +43.18% |
|  | yolov3 | 14396.00 | 18828.71 | +30.79% |
| workload-5 | bert | 1234.75 | 1638.75 | +32.72% |
|  | cifar10 | 971.58 | 1459.51 | +50.22% |
|  | deepspeech2 | 3248.25 | 5355.38 | +64.87% |
|  | imagenet | 29611.25 | 37784.75 | +27.60% |
|  | yolov3 | 14320.00 | 21014.00 | +46.75% |
| workload-10 | bert | 1032.86 | 1291.71 | +25.06% |
|  | cifar10 | 872.40 | 1209.37 | +38.63% |
|  | deepspeech2 | 3133.45 | 4609.55 | +47.11% |
|  | imagenet | 39201.00 | 51022.83 | +30.16% |
|  | yolov3 | 13493.20 | 15855.00 | +17.50% |

所有 15 个“trace × 应用”分组都退化，不存在靠某一应用类别改善而被总体均值掩盖的情况。

### 6.4 任务级差异

| Trace | 对比作业 | 改善 | 不变 | 退化 | 平均 JCT 差 (s) | 中位 JCT 差 (s) |
|---|---:|---:|---:|---:|---:|---:|
| workload-1 | 160 | 36 | 0 | 124 | +951.47 | +379.5 |
| workload-5 | 160 | 38 | 1 | 121 | +1030.45 | +354.5 |
| workload-10 | 160 | 52 | 0 | 108 | +905.83 | +322.5 |

虽然每条 trace 都有少数任务改善，但 67.5%～77.5% 的作业退化，且中位作业也退化 322.5～379.5 秒。完整逐任务数据保存在 `research_notes/run_logs/sia_afs_full_comparison.json`。

### 6.5 GPU 类型分配

| Trace | 策略 | AWS GPU-h（占比） | DGX-ext GPU-h（占比） | RTX GPU-h（占比） |
|---|---|---:|---:|---:|
| workload-1 | Sia | 130.83 (22.64%) | 189.55 (32.80%) | 257.47 (44.56%) |
|  | AFS-P | 189.38 (24.88%) | 179.40 (23.57%) | 392.30 (51.54%) |
| workload-5 | Sia | 217.07 (29.85%) | 218.88 (30.10%) | 291.18 (40.05%) |
|  | AFS-P | 410.03 (44.94%) | 218.82 (23.98%) | 283.50 (31.07%) |
| workload-10 | Sia | 276.93 (31.25%) | 255.87 (28.88%) | 353.25 (39.87%) |
|  | AFS-P | 480.93 (42.20%) | 236.92 (20.79%) | 421.80 (37.01%) |

异构映射显著改变了类型组合：workload-1 更偏向 RTX，workload-5/10 更偏向 AWS，同时 DGX-ext 占比下降。由于所有 trace 总 GPU-hours 都上升，这种重分配没有转化为整体效率收益。

### 6.6 调度时间

| Trace | 策略 | 轮数 | Policy 平均 / P99 (ms) | Policy 总计 (s) | 实际映射轮数 | 映射平均 / P99 (ms) | 映射总计 (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| workload-1 | Sia | 748 | 22.53 / 49.81 | 16.85 | — | — | — |
|  | AFS-P | 1081 | 169.50 / 664.29 | 183.23 | 512 | 131.50 / 312.14 | 67.33 |
| workload-5 | Sia | 851 | 22.10 / 60.34 | 18.80 | — | — | — |
|  | AFS-P | 1263 | 162.74 / 919.56 | 205.54 | 498 | 154.84 / 444.27 | 77.11 |
| workload-10 | Sia | 1023 | 22.03 / 50.33 | 22.54 | — | — | — |
|  | AFS-P | 1653 | 102.71 / 542.84 | 169.77 | 541 | 119.16 / 273.03 | 64.47 |

Sia 的 `Solver took` 平均为 14.51～14.98 ms/轮；AFS 使用独立映射计时，不能把缺少该字符串解释为 0 ms。AFS 的事件式 no-op 轮仍计入 policy 轮数，实际 MIP 映射轮数单列。即使考虑定义差异，候选的 policy 总耗时也明显更高。

## 7. 运行命令与产物

共同环境变量：`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`。所有命令从 `sia-afs-experiments/sia-simulator` 串行执行。

```powershell
# 20 作业 smoke：原始 Sia
& $PY multi_simulator.py <workload-1-first20.csv> --policy=sia `
  --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 `
  --interval=60 --project_throughputs --share_max_replicas `
  --output=<smoke/sia/workload-1-first20.log>

# 20 作业 smoke：最终 AFS-P
& $PY multi_simulator.py <workload-1-first20.csv> --policy=sia_afs `
  --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 `
  --interval=60 --project_throughputs --share_max_replicas `
  --afs_processor_share_unit_seconds=7200 --afs_seed=0 `
  --output=<smoke/sia_afs_reuse_final/workload-1-first20.log>

# 原始 Sia；WORKLOAD 依次为 1、5、10
& $PY multi_simulator.py <workload-WORKLOAD.csv> --policy=sia `
  --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 `
  --interval=60 --project_throughputs --share_max_replicas `
  --output=<full/sia/workload-WORKLOAD.log>

# Sia + AFS-P；WORKLOAD 依次为 1、5、10
& $PY multi_simulator.py <workload-WORKLOAD.csv> --policy=sia_afs `
  --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 `
  --interval=60 --project_throughputs --share_max_replicas `
  --afs_processor_share_unit_seconds=7200 --afs_seed=0 `
  --output=<full/sia_afs/workload-WORKLOAD.log>
```

| 运行 | 退出码 | 墙钟 (s) | JSONL | stdout |
|---|---:|---:|---|---|
| unit tests | 0 | 2.609 | — | `run_logs/sia_afs_unit_tests.stdout.log` |
| smoke Sia | 0 | 40.993 | `smoke/sia/workload-1-first20.log` | `run_logs/sia_afs_smoke_sia.stdout.log` |
| smoke AFS-P（最终） | 0 | 43.107 | `smoke/sia_afs_reuse_final/workload-1-first20.log` | `run_logs/sia_afs_smoke_afs_reuse_final.stdout.log` |
| full Sia workload-1 | 0 | 309.692 | `full/sia/workload-1.log` | `run_logs/sia_afs_full_sia_workload1.stdout.log` |
| full Sia workload-5 | 0 | 275.742 | `full/sia/workload-5.log` | `run_logs/sia_afs_full_sia_workload5.stdout.log` |
| full Sia workload-10 | 0 | 316.044 | `full/sia/workload-10.log` | `run_logs/sia_afs_full_sia_workload10.stdout.log` |
| full AFS-P workload-1 | 0 | 710.423 | `full/sia_afs/workload-1.log` | `run_logs/sia_afs_full_afs_workload1.stdout.log` |
| full AFS-P workload-5 | 0 | 726.296 | `full/sia_afs/workload-5.log` | `run_logs/sia_afs_full_afs_workload5.stdout.log` |
| full AFS-P workload-10 | 0 | 695.745 | `full/sia_afs/workload-10.log` | `run_logs/sia_afs_full_afs_workload10.stdout.log` |

表中 JSONL 路径均相对 `research_runs/sia_afs/`，stdout 路径均相对 `research_notes/`。每条完整统计分别为 `research_notes/run_logs/sia_afs_full_{sia|afs}_workload{1|5|10}.stats.json`，统一对照为 `research_notes/run_logs/sia_afs_full_comparison.json`。

统一运行清单为 `research_notes/run_logs/sia_afs_run_manifest.json`。为避免对未跟踪文件做暂存操作，审计 diff 按来源拆为 `sia_afs_tracked.patch`、`sia_afs_new_policy.patch` 和 `sia_afs_new_tests.patch`，均位于 `research_notes/run_logs/`。

## 8. 限制与未核实事项

- 论文没有公开、可可靠归属的作者实现；这是论文驱动参考实现，不是源码移植。
- 三条预注册 Saturn trace 的总 GPU 数都是 64。是否触发 `n >= M` processor-sharing 分支由每轮活跃作业数决定，不能由总作业数 160 推断。
- 异构虚拟曲线取各类型合法/投影 Goodput 的包络；它不能表达同时占用多种 GPU 类型的单个作业，因为原 Sia 也把单个作业放在一个集群类型上。
- AFS 原系统强调低开销弹性调整，而 Sia 模拟器显式收取 checkpoint/restore 与迁移成本；本实验保留后者，因此重启与 GPU-hours 是关键副作用指标。
- 本轮只对纯同构分配器做了单类型参考测试，没有额外运行完整单 GPU 类型 trace；它不与异构主结果混算。
- 没有进行 AFS 参数搜索、LAS 组合或任何未来信息读取。
- AFS 正式 stdout 因 Artifact 的 CVXPY/CBC verbose 设置很大；解析只使用结构化 `AFS_STATE`、policy 计时和 JSONL，不把控制台文本量作为性能指标。
- 第一次为 workload-1 调用统计脚本时误传了不存在的 `--trace` 参数，辅助命令退出码 2 且未生成输出；随后按脚本实际接口重跑成功。模拟 JSONL/stdout 未受影响。

## 9. 最终判断

**结论：FAIL。**

这不是 `INVALID`：三条 trace 都完成 160/160，所有实际映射均为 `optimal`，没有未映射、容量违规、永久饥饿或不终止；原始 Sia 回归也保持字节一致。因此数据足以判断当前实现。

它也不是 `MIXED`：三条预注册 trace 的平均/P95/P99 JCT、Makespan、重启和 GPU-hours 全部退化，15 个应用级分组全部退化，且多数任务逐项退化。queue mean 降为 0 和两个 trace 的 contention mean 降低不足以抵消主要指标损失。

主要原因与证据一致：

1. AFS 原算法依赖同构、低开销弹性调整；Sia 的 checkpoint/restart 成本使频繁事件重映射代价显著，平均重启增加 58%～98%。
2. 把同构虚拟份额再映射为单一异构合法配置会产生份额离散化和类型选择损失；三条 trace 的 GPU 类型占比均明显改变，但没有改善总效率。
3. 预注册负载没有进入 `n >= M` processor-sharing 路径，因而本轮主要评估的是修改后的 TopPriority/efficiency 路径；原 AFS-P 用于过载公平轮转的机制没有机会抵消重配置与映射代价。
4. 当前事件定义会在 throughput profile 和 checkpoint 状态变化时重算；这保证适配使用最新可得信息，却增加映射与重配置频率。

不建议继续扩大相同算法与参数的实验，也不建议自动开始参数搜索或与 LAS 组合。若未来另立任务继续，首先应把“减少重配置/显式考虑 checkpoint 代价”作为算法设计约束，并用能实际触发 `n >= M` 的单类型参考负载单独验证 processor-sharing 分支；这些都不在本轮继续实现。

## 10. 最终审计清单

- [x] `official-artifacts/` 保持指定 commit 且工作区干净。
- [x] LAS worktree、测试、报告和实验数据未被修改或覆盖。
- [x] AFS worktree 只包含独立策略入口、只读 JobInfo 状态、策略实现和测试。
- [x] 三条原始 Sia 与三条 AFS-P 正式运行均串行完成，退出码均为 0。
- [x] 运行产物在 `research_runs/sia_afs/`，日志、统计、对照和 diff 在 `research_notes/run_logs/`。
- [x] 未引入 LAS、Lucid、Synergy、参数搜索或双模块实验。

