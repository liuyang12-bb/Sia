# Sia + LAS 派生优先机制：第一轮对照实验报告

- 实验日期：2026-10-23（Asia/Shanghai）
- 原始 Sia Artifact：`official-artifacts/`，commit `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 实验 worktree：`sia-experiments/`，分支 `exp/sia-las-weighted`
- Tiresias 官方源码：`research_sources/Tiresias/`，commit `959f9b08f44fa1b4f2b3aed79b01e4d439264a94`
- Python：`.venv-sia-baseline/Scripts/python.exe`（Python 3.10.10）
- 结论摘要：实现和退化验证通过，但第一轮单 trace 结果不足以支持继续扩大相同参数实验。`factor=2.0` 的平均 JCT 仅改善 0.47%，同时 P99 JCT、Makespan、GPU-hours/作业和重启数分别退化 11.24%、4.36%、5.85% 和 3.03%；`factor=1.5` 连平均 JCT 也退化 1.09%。

## 1. 基线保护与实验边界

执行前确认：

- `official-artifacts` 的 HEAD 是指定 commit，`git status --short --branch` 为 `## main...origin/main`；
- 原始 160 作业日志 `research_runs/sia_saturn_workload1/workload-1.log`、统计 JSON `research_notes/run_logs/sia_saturn_workload1_stats.json` 和隔离 Python 环境均存在；
- `sia-experiments/` 与 `exp/sia-las-weighted` 均不存在，随后用 `git worktree add ..\sia-experiments -b exp/sia-las-weighted HEAD` 创建；
- 调度代码只在 `sia-experiments/` 修改；trace、运行产物、统计和报告位于 worktree 外；
- 所有完整模拟严格串行，BLAS/OpenMP 线程数均限制为 1；未修改系统 Python，未安装新依赖；
- 第一轮保持 Artifact 的 CBC、Goodput、checkpoint/restart、配置集合、容量约束、`ZERO_ALLOC_GAIN`、未分配惩罚和无效 `mip_lambda_a` 行为。

## 2. Tiresias 原始算法核查

论文依据为 [USENIX NSDI 2019 论文](https://www.usenix.org/system/files/nsdi19-gu.pdf)，源码依据为 [SymbioticLab/Tiresias](https://github.com/SymbioticLab/Tiresias)。本节区分论文算法、当前官方源码和本实验适配。

### 2.1 论文中的 Discretized 2D-LAS

- attained service 是 `W_J × t_J`：作业占用 GPU 数乘实际执行时间，即 GPU-time；它不是提交后的总年龄，也不包含等待时间。
- 无先验分布时 LAS 优先级为 `-W_J × t_J`，服务量越少优先级越高。
- 论文使用离散 MLFQ：`K` 个队列按 attained service 区间划分；作业跨越阈值时降级。
- LAS 队列内部按第一次开始运行时间 FIFO；高优先级作业若因 GPU 不足无法启动会被跳过，以避免 head-of-line blocking。
- 每轮重新考虑作业；应运行但资源不足的已运行作业会被加入抢占集合。
- 饥饿任务在等待超过阈值后被提升至最高优先级队列，并将用于优先级的执行/等待计数重置。论文给出 `δ_J ≥ PROMOTEKNOB × t_J`，`PROMOTEKNOB=∞` 禁用提升。
- 论文说明实践中采用 `K=2`；敏感性实验用 `(2, 1h)` 表示 1 小时 GPU-time 阈值。

### 2.2 官方源码的实际位置与行为

| 项目 | 文件与函数 | 源码实际行为 |
|---|---|---|
| 服务量状态 | `simulator/jobs.py:432-448` `move_to_runnable()` | `total_executed_time` 和用于队列判断、可被提升重置的 `executed_time` 分开维护。 |
| 2D GPU-time | `simulator/run_sim.py:1034-1054` `dlas_sim_jobs()` | RUNNING 时间累加到 `executed_time`，`gputime=True` 时使用 `executed_time × num_gpu`。 |
| 默认队列 | `simulator/jobs.py:102-104` | 当前主 `_TFJobs` 是 3 个队列，阈值数组为 `[3250, 7200, 18000]`；由于只有 3 个队列，实际降级使用前两个阈值，区间为 `[0,3250)`、`[3250,7200)`、`[7200,∞)`。 |
| README 示例 | `simulator/README.md:64-75` | 文档给出 2 队列/3600 秒示例，同时另一个示例把 `num_queue=2` 与三个阈值并列，和当前源码不一致。 |
| 队内次序 | `simulator/run_sim.py:1132-1143` | 每轮把 PENDING 作业移到队尾，保留已运行作业在前；对应源码对 start-time FIFO 的实现方式，而不是显式按字段排序。 |
| 跳过与抢占 | `simulator/run_sim.py:1092-1127` | 从高到低遍历队列；GPU 不足时 `continue` 跳过；原 RUNNING 作业若不再入选则转 PENDING 并增加抢占计数。 |
| 饥饿提升 | `simulator/run_sim.py:1060-1074` | `last_pending_time ≥ executed_time × solve_starvation` 时把 `executed_time` 清零并移回 Q0。 |
| 默认入口 | `simulator/run_sim.py:2045-2089` | `dlas-gpu` 调用 `dlas_sim_jobs(True)`，因此默认 `solve_starvation=0`、提升关闭；`dlas-gpu-1/2/4/...` 才显式启用对应系数。 |

因此，论文的 `K=2, 1h`、README 示例和仓库当前主对象的 `K=3, 3250/7200` 不是同一个固定配置；报告不把任一描述替代另一方。官方源码默认 `dlas-gpu` 也没有启用论文讨论的饥饿提升。

### 2.3 与本实验组合适配的区别

本实验只借用“按已获得 GPU 服务量提高短服务作业偏好”的思想：

- 只有高/普通两个逻辑类别，不建立 Tiresias MLFQ；
- 不使用 Tiresias 的严格跨队列优先、start-time FIFO、资源不足跳过顺序或显式抢占集合；
- 不实现饥饿提升、Gittins Index、placement profile、PACKLIMIT 或网络策略；
- 保留 Sia 的异构 GPU 配置搜索和 Goodput MIP，仅把高优先级作业的有效 Goodput 乘以实验系数。

所以这是 **Sia + LAS 派生的 weighted-Goodput 组合适配**，不是 Tiresias 或 Discretized 2D-LAS 的完整复现。

## 3. Sia 服务量与目标函数核查

### 3.1 调用链与状态

- `Cluster.step()`：`cluster.py:117-134`，先推进上一轮分配，再调用调度器并写日志。
- `Cluster.get_sia_multi_job_info()`：`cluster.py:373-432`，向策略提供 speedup、上下限、类别、年龄、重启/迁移数、checkpoint 状态、throughput ratio、scale unit，以及 `used_gpu_seconds`/`wasted_gpu_seconds`。
- `SiaPolicy.optimize()`：`sia.py:1157-1159`，输入 `{jobname: JobInfo}`、按 GPU 类型组织的节点信息和旧分配，输出 `{jobname: (cluster, node_ids)}`。

### 3.2 服务量语义

`job.py:73-78,414-434,489-518` 已有三个相关量：

- `attained_service`：所有已分配 GPU-seconds，包含 checkpoint/restore 延迟；
- `wasted_gpu_seconds`：checkpoint/restore 期间浪费的 GPU-seconds；
- `used_gpu_seconds`：实际进入训练阶段的 `allocated_gpu_count × elapsed_seconds`，不含等待，也不含 checkpoint/restore 浪费。

策略使用 `used_gpu_seconds`。它与本实验所需的生产性累计 GPU 服务语义等价，不重复增加新的累计字段，也没有使用 `age`。实现对活跃作业是准确的；作业在最后一个 60 秒离散轮次中若提前完成，`used_gpu_seconds` 会先按整轮增加，但该作业随即完成并不再参与后续优先级决策，因此不影响调度分类。

### 3.3 Goodput、目标方向与插入点

- adaptive 作业通过 `_compute_goodputs_v2()`（`sia.py:432-516`）生成配置 Goodput；每个作业按最小非零 Goodput/scale unit 归一化，缺失或极小值保留 `ZERO_ALLOC_GAIN=0.0001`。
- rigid 作业走 `_compute_rigid_job_goodput_single()`（`sia.py:536-562`），按固定副本数和 GPU 类型 throughput 归一化；两条路径的原始目标构造不同。
- 两类作业随后都应用原有重分配折扣；checkpoint 中作业通过 `scaling_underway` 进入硬保持约束（`sia.py:772-792,1028-1045`）。
- `__solve_mip_rescaled()` 在 `sia.py:853-1155`：Goodput 在 `sia.py:927` 取 `p` 次幂；`p<0` 用 `Minimize`，否则用 `Maximize`（`sia.py:986-988`）。未分配惩罚是 `lambda_n × (num_jobs-num_allocs)`（`sia.py:975-982`）。
- 新权重插入点为归一化和原重分配折扣之后、`p` 次幂之前。对 `p=-0.5`，Goodput 乘大于 1 的 factor 会降低 `goodput^p` 成本，因而在最小化中提高偏好；对 `p>0`，同一乘法提高最大化收益。没有假定两种 `p` 的目标方向相同。
- `ZERO_ALLOC_GAIN` 对应的无效/极小配置不乘 factor，未分配仍由原来的“整行没有选择变量 + `lambda_n`”定义处理。
- `mip_lambda_a` 在 `sia.py:953` 只赋给局部变量 `opt_lambda_alloc_change`，之后没有进入目标或约束。第一轮没有修复此 Artifact 行为。

## 4. 实现

### 4.1 修改文件与行号

| 文件 | 行 | 修改 |
|---|---:|---|
| `sia-simulator/sia_las.py` | 6-17 | 从 `JobInfo.used_gpu_seconds` 读取服务量；实现阈值分类。 |
| `sia-simulator/sia_las.py` | 20-39 | 只缩放有效 Goodput，保持 `ZERO_ALLOC_GAIN`；factor=1 直接返回原数组。 |
| `sia-simulator/sia_las.py` | 42-99 | 新增 `SiaLASPolicy`；记录每轮高优先级集合，覆盖目标 Goodput hook。注释明确不是 Tiresias 原公式。 |
| `sia-simulator/sia.py` | 100-108 | 为派生策略增加默认 no-op hook；原 `SiaPolicy` 返回输入原对象。 |
| `sia-simulator/sia.py` | 913-920 | 在既有重分配折扣之后、`p` 次幂之前调用 hook。 |
| `sia-simulator/multi_simulator.py` | 14,46-55 | 新增独立 `--policy=sia_las` 创建路径；`--policy=sia` 仍构造原 `SiaPolicy`。 |
| `sia-simulator/multi_simulator.py` | 141-154 | CLI 增加策略选择与两个参数。 |
| `sia-simulator/tests/test_sia_las.py` | 1-185 | 9 项单元/求解器集成测试。 |

### 4.2 参数

| 参数 | 默认值 | 定义 |
|---|---:|---|
| `--las_threshold_gpu_seconds` | 3600 | `used_gpu_seconds < threshold` 为高优先级；等于阈值即普通类。 |
| `--las_priority_factor` | 1.0 | 高优先级有效 Goodput 乘数；必须大于 0。普通类不变。 |

`factor=1.0` 的实现是显式 no-op，以保证精确退化而不是数值上近似。

## 5. 测试

最终命令：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe -m unittest discover -s sia-simulator/tests -p test_sia_las.py -v
```

- 结果：9/9 通过，退出码 0，耗时 2.579 秒。
- 日志：`research_notes/run_logs/sia_las_unit_tests.stdout.log`
- 元数据：`research_notes/run_logs/sia_las_unit_tests.run.json`

覆盖项：

1. 使用生产性 `used_gpu_seconds`，而不是年龄或 `attained_service` 中的 restart waste；
2. 3599/3600 GPU-seconds 的阈值边界；
3. factor=1 的 Goodput 矩阵 bitwise identity；
4. `p=-0.5` 最小化及 `p=0.5` 最大化的偏好方向分别验证；
5. `ZERO_ALLOC_GAIN` 不变；
6. 1-GPU 集成场景不超过容量；
7. factor=1 的 Sia/LAS 分配和目标值完全一致；
8. checkpoint 中作业保持原分配；
9. 未分配惩罚与原重分配折扣保持一致。

测试先于实现创建；红灯阶段按预期因 `ModuleNotFoundError: sia_las` 失败，随后实现并修复一个测试期望的 list/tuple 表示差异，最终全部通过。

## 6. 20 作业 smoke test

trace 是官方 `workload-1.csv` 的表头加前 20 条作业：`research_runs/sia_las_weighted/traces/workload-1-first20.csv`。

| 策略 | 完成 | 平均 JCT (s) | P99 JCT (s) | Makespan (s) | 重启/作业 | GPU-h/作业 | 容量违规 | 退出码 | 耗时 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sia | 20/20 | 1393.4 | 7481 | 8528 | 4.35 | 3.2325 | 0 | 0 | 54.743 |
| Sia-LAS factor=1 | 20/20 | 1393.4 | 7481 | 8528 | 4.35 | 3.2325 | 0 | 0 | 69.915 |
| Sia-LAS factor=1.5 | 20/20 | 1372.7 | 8234 | 9281 | 3.75 | 3.4092 | 0 | 0 | 63.810 |

退化验证：Sia 与 factor=1 的完整 JSONL 字节完全相同，二者 SHA-256 均为：

```text
FE5B7EF8D5C9D8A759B017F0EE9866D14415DB14B91E8A12572F23E0813D4B9A
```

因此没有出现需要借助“并列最优解”解释的差异：逐轮分配、完成时间、重启和所有日志字段完全一致。factor=1.5 的 smoke 平均 JCT 改善 1.49%，但 P99、Makespan、GPU-hours/作业分别退化 10.07%、8.83%、5.47%，已经显示尾部代价。

## 7. 完整 Saturn workload-1 对照

统计沿用官方 `print_run_stats.py` 的单 trace 定义：P99 使用排序后 `int(0.99*N)` 下标；GPU-hours 按每轮日志分配乘 60 秒估算，不扣 checkpoint 浪费。扩展脚本只新增应用平均 JCT、容量扫描和 stdout 调度耗时解析。

### 7.1 总体指标

括号内为相对原始 Sia 的变化；JCT、Makespan、重启、GPU-hours 和 contention 越低越好。

| 指标 | 原始 Sia | Sia-LAS 1.5 | Sia-LAS 2.0 |
|---|---:|---:|---:|
| 完成作业 | 160/160 | 160/160 (0.00%) | 160/160 (0.00%) |
| 平均 JCT (s) | 2091.0375 | 2113.8250 (+1.09%) | 2081.3063 (-0.47%) |
| P99 JCT (s) | 27765 | 28242 (+1.72%) | 30887 (+11.24%) |
| Makespan (s) | 44859 | 45879 (+2.27%) | 46816 (+4.36%) |
| 平均重启/作业 | 4.11875 | 4.43125 (+7.59%) | 4.24375 (+3.03%) |
| GPU-hours/作业 | 3.61156 | 3.54354 (-1.88%) | 3.82292 (+5.85%) |
| contention mean | 7.37219 | 7.29793 (-1.01%) | 7.04828 (-4.39%) |
| contention max | 21 | 22 (+4.76%) | 23 (+9.52%) |
| queue mean | 0.002649 | 0 (-100%) | 0 (-100%) |
| queue median | 0 | 0（基线为 0，比例不定义） | 0（基线为 0，比例不定义） |
| queue max | 1 | 0 (-100%) | 0 (-100%) |
| 容量违规轮次 | 0 | 0 | 0 |
| 最大 AWS / DGX-ext / RTX GPU | 24 / 16 / 24 | 24 / 16 / 24 | 24 / 16 / 24 |

queue 改善的绝对量极小：原始 Sia 只有一次最大队列长度 1，不能把 -100% 解释成重要收益。

### 7.2 各应用平均 JCT

| 应用（作业数） | Sia (s) | LAS 1.5 (变化) | LAS 2.0 (变化) |
|---|---:|---:|---:|
| bert (6) | 1051.83 | 1090.67 (+3.69%) | 1126.83 (+7.13%) |
| cifar10 (134) | 816.10 | 827.69 (+1.42%) | 834.03 (+2.20%) |
| deepspeech2 (10) | 2570.50 | 2830.50 (+10.11%) | 2460.40 (-4.28%) |
| imagenet (3) | 30807.00 | 31162.67 (+1.15%) | 31723.33 (+2.97%) |
| yolov3 (7) | 14396.00 | 14137.71 (-1.79%) | 13530.57 (-6.01%) |

factor=1.5 只改善 YOLOv3；factor=2.0 改善 DeepSpeech2 和 YOLOv3，但 BERT、CIFAR10、ImageNet 都退化。由于 CIFAR10 占 134/160，factor=2.0 的总体均值改善仍只有 0.47%。

### 7.3 调度求解时间

从 stdout 中可靠解析每轮 `Solver took` 和 `Policy optimization took`：

| 策略 | 求解轮数 | CBC 平均/轮 (ms) | CBC P99 (ms) | CBC 总计 (s) | Policy 平均/轮 (ms) | Policy 总计 (s) |
|---|---:|---:|---:|---:|---:|---:|
| Sia 基线 | 748 | 31.713 | 52.42 | 23.722 | 45.348 | 33.920 |
| LAS 1.5 | 765 | 25.980 | 53.54 | 19.875 | 40.147 | 30.712 |
| LAS 2.0 | 780 | 26.025 | 65.67 | 20.300 | 39.424 | 30.750 |

单轮均值未增加，但 LAS 运行轮数增加。墙钟时间不能直接归因于策略：基线来自较早一次运行，控制台渲染、系统负载和 PowerShell 日志捕获也会影响总耗时。

## 8. 运行命令、退出码与产物

共同环境变量：`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`。以下命令均从 `sia-experiments/sia-simulator` 执行，`PY` 为工作区 `.venv-sia-baseline/Scripts/python.exe`。

```powershell
# smoke: original Sia
& $PY multi_simulator.py <first20.csv> --policy=sia --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 --interval=60 --project_throughputs --share_max_replicas --output=<smoke/sia/workload-1-first20.log>

# smoke: LAS factor=1
& $PY multi_simulator.py <first20.csv> --policy=sia_las --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 --interval=60 --project_throughputs --share_max_replicas --las_threshold_gpu_seconds=3600 --las_priority_factor=1 --output=<smoke/sia_las_factor1/workload-1-first20.log>

# smoke: LAS factor=1.5
& $PY multi_simulator.py <first20.csv> --policy=sia_las --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 --interval=60 --project_throughputs --share_max_replicas --las_threshold_gpu_seconds=3600 --las_priority_factor=1.5 --output=<smoke/sia_las_factor1_5/workload-1-first20.log>

# full: LAS factor=1.5 / 2.0（两次顺序执行）
& $PY multi_simulator.py <official workload-1.csv> --policy=sia_las --policy_p_val=-0.5 --mip_lambda_a=0.01 --mip_lambda_n=1.1 --interval=60 --project_throughputs --share_max_replicas --las_threshold_gpu_seconds=3600 --las_priority_factor=<1.5|2.0> --output=<对应 full 目录/workload-1.log>
```

| 运行 | 退出码 | 耗时 (s) | JSONL | stdout/stderr | 元数据 |
|---|---:|---:|---|---|---|
| smoke Sia | 0 | 54.743 | `research_runs/sia_las_weighted/smoke/sia/workload-1-first20.log` | `run_logs/sia_las_smoke_sia.stdout.log` | `sia_las_smoke_sia.run.json` |
| smoke LAS 1 | 0 | 69.915 | `research_runs/sia_las_weighted/smoke/sia_las_factor1/workload-1-first20.log` | `run_logs/sia_las_smoke_factor1.stdout.log` | `sia_las_smoke_factor1.run.json` |
| smoke LAS 1.5 | 0 | 63.810 | `research_runs/sia_las_weighted/smoke/sia_las_factor1_5/workload-1-first20.log` | `run_logs/sia_las_smoke_factor1_5.stdout.log` | `sia_las_smoke_factor1_5.run.json` |
| full LAS 1.5 | 0 | 578.135 | `research_runs/sia_las_weighted/full/sia_las_factor1_5/workload-1.log` | `run_logs/sia_las_full_factor1_5.stdout.log` | `sia_las_full_factor1_5.run.json` |
| full LAS 2.0 | 0 | 531.420 | `research_runs/sia_las_weighted/full/sia_las_factor2_0/workload-1.log` | `run_logs/sia_las_full_factor2_0.stdout.log` | `sia_las_full_factor2_0.run.json` |

路径表中的 `run_logs/` 均指 `research_notes/run_logs/`。完整统计为 `sia_las_full_factor1_5.stats.json`、`sia_las_full_factor2_0.stats.json`；统一相对变化为 `sia_las_full_comparison.json`。

## 9. 失败、异常与不确定性

1. 第一次 smoke 的日志包装器把 Pandas `FutureWarning` 当作 PowerShell 可终止原生命令错误，约 39 秒后中断；没有生成模拟 JSONL。失败日志保留为 `sia_las_smoke_sia.failed_wrapper.stdout.log`。修复仅把原生 stderr 作为日志继续捕获，未改模拟器。
2. 成功日志中的同一 Pandas `FutureWarning` 被 Windows PowerShell 渲染为 `NativeCommandError/RemoteException` 记录，但 Python 最终退出码为 0、无 traceback，JSONL 完整。这是捕获格式，不是模拟失败。
3. 只有一个 Saturn trace；应用级样本极不均衡，尤其 ImageNet 仅 3 个作业。不能外推到论文 Table 3 或所有 10 条 trace。
4. GPU-hours 是官方离散日志定义，不是 `used_gpu_seconds` 的生产性服务量；两者用途不同。
5. factor 是组合适配参数，没有 Tiresias 原论文公式依据。阈值 3600 来自论文的 `(2,1h)` 敏感性设置，而非当前官方源码的默认 3250。
6. `lambda_a=0.01` 仍未进入 Artifact 的最终目标；本轮刻意保持基线行为。
7. CBC 的并列最优与主机负载可能影响墙钟时间，但 factor=1 的 JSONL 字节级一致排除了本次退化验证中的并列解差异。

## 10. 是否继续

不建议立即把相同的 `threshold=3600, factor∈{1.5,2.0}` 扩展到全部 10 条 trace：

- factor=1.5 没有平均 JCT 收益；
- factor=2.0 的平均 JCT 收益只有 0.47%，却付出明显 P99、Makespan、GPU-hours 和重启退化；
- 更大的 factor 已表现出强化短服务偏好、损伤尾部的趋势，而第一版按要求没有饥饿提升。

若进入下一轮，建议先作为单独任务设计更低 factor（例如 1.1/1.25）或更平滑的服务量权重，并明确尾部约束；不要在本轮代码中继续引入饥饿提升、Gittins、网络感知或任务打包。

## 11. 最终审计清单

- [x] 原始 `official-artifacts` 保持 commit `aaf7a0d...`。
- [x] 原始工作区干净。
- [x] 实验 worktree 的改动只涉及独立策略入口、目标 hook 和测试。
- [x] 运行产物全部在 `research_runs/sia_las_weighted/`；日志/统计/diff 在 `research_notes/run_logs/`。
- [x] 未覆盖或删除已有基线日志、统计或源码。
- [x] 未引入第一版范围外模块。
