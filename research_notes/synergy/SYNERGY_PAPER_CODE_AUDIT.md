# Synergy 论文与官方源码审计

- 审计日期：2026-10-23（Asia/Shanghai）
- 论文：*Resource Elasticity for Distributed Deep Learning* / Synergy，OSDI 2022（任务描述中的核心机制与该论文一致）
- 本地论文：`research_sources/Synergy/osdi22-mohan.pdf`
- 论文 SHA-256：`B698E3C3F06EFD1E17EF1B93926B337DB5540C7CA9AC4A469CD7D86D1756E1C1`
- 官方源码：`https://github.com/msr-fiddle/synergy`
- 本地源码：`research_sources/Synergy/`，commit `1bb2bed6edc9c5f953d0b7f0a40e4fdad053489c`
- 总结：论文算法和公开模拟器足以核查 Synergy 的核心机制，但当前源码与论文/AE 文档有数处重要差异。它是同构 V100、固定 GPU demand、模型级 CPU/内存画像下的辅助资源调度器，不能把其画像或 TUNE 逻辑不加验证地套到 Sia 的异构、GPU 弹性 Goodput 模型中。

## 1. 论文原方法

### 1.1 问题与输入输出

Synergy 不重新决定一个作业的 GPU demand。论文假设 GPU demand 在作业整个生命周期固定；每个调度轮由既有策略（FIFO、SRTF、LAS、FTF）先选出 GPU demand 能由集群满足的 runnable jobs，Synergy 再为这些作业选择 CPU、内存及物理 placement。

输入包括：

- runnable job 的调度优先顺序；
- 每个 job 的固定 GPU demand `g_j`；
- 每台同构服务器的 GPU、CPU、RAM 容量；
- 每个 job 的 CPU × RAM 离散资源敏感性矩阵 `W_j[c,m]`；
- GPU-proportional 的 CPU/RAM 基线份额。

输出包括：

- 当前轮每个 runnable job 的 CPU、RAM 配额；
- job 到一个或多个物理服务器的 GPU/CPU/RAM placement；
- 当理想需求无法装箱时，将当前 job 或已放置 peer 降至 GPU-proportional 份额的调整结果。

论文明确把 GPU 看成不可替换的主资源，把 CPU/RAM 看成每轮可调整的 fungible auxiliary resources。该假设位于 PDF 第 6–9 页（论文页 583–586），固定 GPU demand 的限制再次在 PDF 第 14–15 页（论文页 591–592）说明。

### 1.2 Optimistic Profiling

论文 §3.1（PDF 第 6 页，论文页 583）的画像过程为：

1. 作业到达时只画像一次，短暂运行若干 iteration 测量 throughput。
2. 对 CPU 和内存的离散组合构造 `W_j[c,m]`。
3. 利用 MinIO 提供的应用级缓存隔离与可预测 cache hit，固定 CPU 后用已知存储带宽解析估计不同内存配额下的 throughput。
4. 因此只实测“满内存”时随 CPU 变化的点，其他内存列由模型估计；论文称 24 CPU × 10 个内存点的朴素 240 分钟画像可降至约 24 分钟。
5. CPU 轴再从最大 CPU 开始二分选择画像点；若新点 throughput 改善小于示例阈值 10%，继续搜索低 CPU 半区，否则补测高 CPU 半区。论文的 1-GPU ResNet18 示例只测 8 个 CPU 点，约 8 分钟。
6. 画像输出是离散 throughput/progress matrix，不是单一权重。

这个过程依赖 MinIO、数据集大小、数据预处理、存储带宽和 GPU 类型。论文 §6 明确说不用 MinIO 时需要实测离散内存点，矩阵趋势也可能改变。

### 1.3 Synergy-OPT

论文 §4.1（PDF 第 7–8 页，论文页 584–585）先构造理想化“超级服务器”ILP。对每个 job 和离散配置定义布尔变量 `y_{c,m,j}`：

- 公式 (1)：最大化 `sum_j sum_[c,m] W_j[c,m] y_{c,m,j}`；
- 公式 (2)、(3)：总 CPU、总 RAM 不超过集群容量；
- 公式 (4)：每个 job 恰选一个 CPU/RAM 配置；
- 公式 (5)：每个 job 的 progress 不低于 GPU-proportional 基线 `W_j[C_g,M_g]`；
- 公式 (6)、(7)：从选中项得到 `c*_j`、`m*_j`。

第二个 LP 把 `(g_j,c*_j,m*_j)` 映射到真实服务器，论文目标是尽量减少跨服务器碎片，并允许 `x_{i,j}` 为分数；这也是 OPT 难以落地的原因之一，因为可能产生 3.3/2.7 GPU 之类不可物理实现的分数 placement。另一个原因是每轮求两个 LP 成本高。

### 1.4 Synergy-TUNE

论文 §4.2（PDF 第 8–9 页，论文页 585–586）的机制不是 Goodput 权重：

```text
R <- 基础策略按优先级选出的 runnable jobs，GPU demand 可满足
按 GPU demand、CPU demand、RAM demand 排序 R
for j in R:
    尝试把 j 放到“剩余资源最少但仍可完整容纳”的服务器
    多 GPU job 若单机放不下，选择 GPU 足够的最小服务器集合
    各服务器上 CPU/RAM 与该服务器承担的 GPU 数成比例

    if CPU/RAM 导致放不下且 j 高于 GPU-proportional share:
        把 j 降至 GPU-proportional share 后重试
    if 仍放不下:
        暂时忽略 CPU/RAM，只找 GPU placement
        找出该 placement 上占用超过 proportional share 的已放置 jobs
        将足够多的 peer 降至 proportional share，释放刚好所需资源
        递归继续

最坏情况下所有 runnable jobs 都回到 proportional share
```

其两条不变量是：不能因辅助资源导致有 GPU 可用却跳过高优先级 job；任一 job 的 throughput 不低于 GPU-proportional 基线。单 GPU job 的三类资源必须在同一服务器；多 GPU job 跨服务器时，每个 GPU 进程的 CPU/RAM 比例必须一致。

### 1.5 时机、抢占和开销假设

- 论文采用 round-based/lease-based 调度；作业到达后画像，调度轮开始时重算 runnable set、辅助资源和 placement。
- 公开代码 `schedulers/scheduler.py:44-52` 与 `runner.py:76-91` 的默认 round duration 是 300 秒；`configs/default_cluster.ini:11` 记录 `lease_time=5`。
- lease 不续期时 Synergy iterator checkpoint 到共享存储，使用 gRPC 与调度器交互（论文 §4.3，PDF 第 9 页）。
- 论文实验使用同构服务器、固定 GPU demand、数据并行和本地数据集；没有显式联合调度共享网络带宽。

## 2. 官方源码对应位置

| 机制 | 文件/类/函数 | 当前源码行为 |
|---|---|---|
| 调度轮与 runnable set | `simulator/schedulers/fifo_synergy_new.py:9-99`，`FIFO.schedule()` | FIFO 按到达时间排序；先按剩余 GPU 选/放 job，再调用 cluster allocator。 |
| GPU-proportional / fair | `simulator/resources/cluster.py:253-279`，`get_allocation_map()` | 按 job 在每台 server 占用的 GPU 比例分配该机 CPU、RAM 和 storage；`runner.py:419-427` 解析 `scheduler+fair`。 |
| Synergy-GREEDY | `simulator/schedulers/fifo_synergy_new.py:73-97` + `resources/cluster.py:382-431`，`allocate_synergy_placement()` | AE 中无 `+fair/+tune` 的基础 scheduler 即 GREEDY；直接尝试画像需求的 first-fit/tight placement，失败时跳过该 job，没有 TUNE 的 peer 回退。 |
| OPT 第一阶段 | `simulator/opt_algo/synergy_opt.py:16-149`，`Solver.SolveFirstLP()` | `Y_j` 与画像矩阵同形；最大化总 normalized throughput，约束 CPU、RAM、单配置和不低于 fair share。 |
| OPT 第二阶段 | `simulator/opt_algo/synergy_opt.py:174-212`，`SolveSecondLP()` | 建立 server × job 的 `X` 和逐服务器 GPU/CPU/RAM 约束。当前代码目标实际是常数 `Minimize(1)`。 |
| 画像装载 | `simulator/jobs/model.py:169-183`，`Model.use_scores_from_tput()` | 从 Python profile class 复制 CPU、RAM、batch、iteration time、speedup 和 9×5 throughput matrix。 |
| 默认模型集合 | `simulator/jobs/model_zoo.py:178-214` | 激活 5 个 image、3 个 language、2 个 speech 模型；BERT 只在注释候选中。 |
| job demand 初始化 | `simulator/jobs/workload.py:169-245`，`Workload.add_synergy_profile()` | 将画像的 per-GPU CPU/RAM/storage 乘 GPU demand；必要时按单机容量截断并线性调整 speedup。 |
| 多维 placement | `simulator/resources/cluster.py:382-431`，`allocate_synergy_placement()` | 优先单机合并；否则按 per-GPU demand 在多个 server 选择 GPU 并同步构造 CPU/RAM allocation map。 |
| 容量检查 | `simulator/resources/cluster.py:480-562` | `_top_synergy_gpus_placement()` 使用临时 hold；`_fits_in_server()` 检查所有资源维度。 |
| TUNE 入口 | `simulator/resources/cluster.py:575-586` | `allocate_synergy_tune()` 进入递归 `_tune()`。 |
| 当前 job 降级 | `simulator/resources/cluster.py:589-666,733-746` | 首次失败后 `_make_fair_share()` 把 CPU/RAM/storage 降至 GPU-proportional，speedup 置 1。 |
| peer 降级 | `simulator/resources/cluster.py:671-731` | 忽略辅助资源找 GPU server；选择超出 fair share 的 peer，回滚当前轮后改为 fair share，再重放 allocation。 |
| 性能推进 | `simulator/jobs/job.py:133-176` | 跨机应用 placement penalty；非 fair allocation 用 `iteration_time / synergy_speedup` 推进。 |
| 离散坐标 | `simulator/jobs/job.py:96-97` | CPU 轴 `[1,2,3,4,5,6,9,12,24]`，RAM 轴 `[20.83,62.5,125,187.5,250]` GB。 |

### 2.1 仓库结构、画像格式与 profiler

- 模拟入口是 `simulator/runner.py`；策略在 `simulator/schedulers/`，资源与 placement 在 `simulator/resources/`，OPT 在 `simulator/opt_algo/`。
- 可执行训练模型在仓库根目录 `models/`，部署代码在 `simulator/deployment/`，离线 profiler 位于 `src/profiler/offline_profiler.py`。
- 当前 simulator 实际读取 `simulator/models/*.py` 中的 Python class；每个 class 保存 `cpu_per_gpu`、`mem_per_gpu`、`sspeed_per_gpu`、`batch`、`iter_time`、`speedup` 和二维 `tput` 数组。`models-4/5/6` 是另存画像目录，不是当前默认 loader 路径。
- `src/profiler/README.md` 说明 profiler 接收 Docker image、mount、GPU/节点数、CPU、memory、per-GPU batch 和最大 iteration，并要求训练脚本支持 batch、workers、max_iterations、data 参数。其 TODO 明确写着当前只打印 performance matrix、JSON 导出尚未完成，多服务器 profiling 也未测试。
- 依赖文件没有提供可直接复刻的现代 Windows lockfile；README 记录测试 Python 3.6.8/3.8.10，部署路径还需要 Docker、GPU/NUMA/gRPC 等。本轮只为纯 simulator 建立隔离 Python 3.10 环境，没有启动真实 profiling、Kubernetes 或部署系统。

## 3. 论文、AE 文档与当前源码的差异

1. `ae-readme.md` 写明默认 smoke 是 `FIFO+tune`，但当前 commit 的 `runner.py:393-394` 实际硬编码 `FIFO+fair`；CLI 的 `--tune` 会在 `benchmark()` 内被该 scheduler 字符串重新覆盖。因此不能不改源码就复现文档给出的 TUNE 期望数值。
2. 论文说 TUNE 在 runnable set 内按 GPU、CPU、RAM demand 排序；当前 FIFO 实现仅在 `fifo_synergy_new.py:39` 按 arrival time 排序，未找到论文描述的三关键字 job sort。cluster allocator 会选择较紧的 server，但这不等价于 job 排序。
3. 仓库包含 `_1/_2/_4` 等画像类，但 `model.py:169-175` 强制构造 `model_name + "_1"`；按 `gpu_demand` 选画像的代码被注释。当前模拟路径因此对 1/2/4/8/16 GPU demand 都复用 1-GPU profile。
4. 论文第二 LP 目标是尽量减少 fragmented jobs；当前 `synergy_opt.py:205` 是常数目标 `Minimize(1)`，只求可行解，不能声称源码实现了论文的碎片最小化目标。
5. `Model.update_res_score_from_json()`（`model.py:62-72`）把解析结果赋给局部变量且文件未导入 `json`；默认路径实际使用硬编码 Python profile classes。审计未把 JSON loader 视为可靠可用画像入口。
6. `_tune()` 在所有递归 fallback 后仍失败时于 `cluster.py:595-608` 调用 `sys.exit(1)`，不是把 infeasible 结果返回给上层重新求解。
7. Windows 不能直接导入 Linux `numa`，公开 checkout 也未包含生成后的 protobuf Python stubs。它们影响开箱运行，但不改变论文算法定义。

这些差异均按“当前官方源码实际行为”记录；本实验不会用论文描述覆盖源码，也不会把源码偶然行为反称为论文公式。

## 4. 公开实现与原生最小复现

存在作者公开实现：Microsoft Research `msr-fiddle/synergy`。本地 checkout 未改源码，只有按任务要求下载到同目录的论文 PDF 是 untracked 文件。

为避免修改 Sia 的 `.venv-sia-baseline`，新建 `.venv-synergy`（Python 3.10.10），只安装 CPU-only simulator 所需的 `numpy`、`pandas`、`matplotlib`、`cvxpy` 及其依赖。官方 README 说明测试过 Python 3.6.8/3.8.10，故这里是兼容性复现，不是原作者锁定环境的逐包复刻。

运行脚本：`research_notes/synergy/native_synergy_smoke.py`。

```powershell
$env:MPLBACKEND='Agg'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
D:\科研启航\Sia\.venv-synergy\Scripts\python.exe research_notes\synergy\native_synergy_smoke.py --jobs 5
```

结果：退出码 0，5/5 jobs 完成，pending=0，模拟 makespan 74400 秒；使用官方默认 16 × 8 GPU、每机 24 CPU/500 GB 的模拟容量。元数据位于 `research_runs/sia_synergy/native_reference/native_synergy_smoke.run.json`。

兼容层只用于 import：Python 3.10 的 `collections.Mapping` 别名、Windows 的空 NUMA 模块、纯模拟路径下空 deployment RPC 模块。没有执行 NUMA 或部署 RPC，也没有修改官方源码。这个 smoke 只证明当前 TUNE 模拟路径能够完成小型 workload，不证明论文 Figure 8 的数值复现。

只读回归测试：

```powershell
D:\科研启航\Sia\.venv-synergy\Scripts\python.exe -m unittest -v test_synergy_reference.py
```

4/4 通过：默认容量、激活 ModelZoo、9×5 profile 形状、TUNE 超额需求降至 proportional share。

## 5. 对 Sia 组合实验的审计结论

Synergy 可以作为“先由 Sia 选 GPU 类型/数量，再做辅助资源分配和 placement”的概念来源，但不能直接移植：

- 论文原方法的 GPU demand 固定，Sia 每轮可改变 GPU 类型和数量；
- 论文画像只在同构 V100 + 特定数据栈上成立，论文 §6 对异构扩展要求额外增加 GPU-type 画像维；
- 当前参考代码甚至对多 GPU demand 复用 `_1` profile；
- Sia Goodput 已包含特定 GPU 类型、batch size、节点数和通信的性能，若再乘不匹配的 Synergy speedup，会产生不可解释的重复建模。

因此论文/源码审计本身完成，但 Sia 正式实现门槛取决于画像覆盖和资源模型审计；结论见 `SIA_SYNERGY_PROFILE_COVERAGE.md` 与 `SIA_SYNERGY_MODEL_DESIGN.md`。
