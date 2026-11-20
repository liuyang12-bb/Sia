# Sia 原始基线审计报告

- 审计日期：2026-10-20（Asia/Shanghai）
- 工作区：`D:\科研启航\Sia`
- 审计范围：官方 Artifact 的模拟器；未部署 Kubernetes、Docker 或真实 GPU 集群；未修改调度算法。
- 结论：**单作业 smoke test 与一个完整的官方 Saturn trace（160 个作业）均成功运行**。官方 Table 3 的全部 10 个 trace 未执行。

## 1. 仓库来源与原始版本

克隆命令：

```powershell
git clone https://github.com/siasosp23/artifacts.git official-artifacts
```

克隆前的工作区检查结果：

- 绝对路径：`D:\科研启航\Sia`
- 工作区不是 Git 仓库。
- 工作区中没有既有项目文件。
- `official-artifacts/` 不存在，因此没有覆盖或复用已有目录。
- 首次克隆因受限代理 `127.0.0.1` 无法连接 GitHub；取得联网权限后成功。

版本记录：

```text*
origin  https://github.com/siasosp23/artifacts.git (fetch)
origin  https://github.com/siasosp23/artifacts.git (push)
branch  main
HEAD    aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70
log     aaf7a0d added workloads used for scalability analysis
status  ## main...origin/main
```

审计结束时再次检查为干净工作区。运行产生的 Python 3.10 `__pycache__` 已按精确路径删除；没有执行 `git clean`、`git reset`、`git push`，也没有修改任何被跟踪文件。

## 2. 主机与 Python 环境

主机信息（报告生成前复核）：

```text
OS               Windows 10.0.26200, 64 bit
逻辑 CPU          16
物理内存          13.69 GiB
当时可用内存      1.46 GiB
工作盘可用空间    30.76 GiB
系统默认 Python   3.13.9
官方要求 Python   3.10
实际隔离环境      .venv-sia-baseline, Python 3.10.10
```

系统还安装了 Python 3.11、3.10 和旧 Anaconda Python，但 `conda` 不在 PATH。为避免修改系统 Python，本次只使用工作区内的 `.venv-sia-baseline/`。

官方环境说明见 `sia-simulator/README.md:9-14`：Python 3.10、NumPy 1.23（MKL）、Pandas、SciPy、Pymoo 0.4.2.1、带 CBC/GLPK 的 CVXPY。`sia-simulator/requirements.txt:1-4` 实际只列出 NumPy、Pandas、Pymoo、SciPy；源码还需要：

- `cvxpy` 与 CBC/GLPK：`sia.py:5`、`sia.py:1030-1036`；README 有说明但 requirements 未列。
- `rich`：`multi_simulator.py:8-9`；README/requirements 均未列。
- `psutil`：`utils_gavel.py:6`；入口无条件导入 Gavel，因此运行 Sia 也会触发；README/requirements 均未列。
- `matplotlib`：`utils/print_run_stats.py:3`；统计脚本实际未画图，但导入是无条件的。

最终用于成功基线的关键版本：

```text
Python 3.10.10
numpy 1.23.5
pandas 1.5.3
scipy 1.10.1
cvxpy 1.3.2
cylp 0.91.6 (CBC 2.10.10)
pymoo 0.4.2.1
psutil 7.2.2
rich 15.0.0
matplotlib 3.10.9
```

Windows 兼容处理及其边界：

1. 当前 `cvxpy[CBC,GLPK]` 会拉取 `cylp 0.94.0` 源码包，在 Windows 上因没有 COIN-OR 安装和 `COIN_INSTALL_DIR` 而失败；改用仍满足 `cylp>=0.91.5` 的 Windows wheel `cylp==0.91.6`。
2. Pandas 2.3.3 删除了 `DataFrame.append`，而 `applications.py:282` 仍调用该 API；不改源码，改用 Pandas 1.5.3。
3. 当前 CVXPY/OSQP 二进制与官方 NumPy 1.23 组合在此 Windows 路径中触发原生 access violation。Sia 的源码只调用 CBC，故在隔离环境中移除了不用的 OSQP，使 CVXPY 跳过该可选求解器探测。`pip check` 因此会提示 CVXPY 声明的 OSQP 缺失；**这不是完整、通用的 CVXPY 环境，只是可复核 Sia/CBC 路径的隔离环境**。
4. 默认 `num_parallel_traces=4` 在 Windows 同时初始化多个 OpenBLAS 线程池时发生内存分配失败/栈溢出。smoke 的目录模式通过运行时把 trace 并行度降为 1 并设置 `OPENBLAS_NUM_THREADS=1` 等环境变量完成；没有改源码。

因此，源码基线有效，但若追求与论文环境严格一致，仍应在 Linux/Conda 中按 README 使用 NumPy+MKL，并锁定完整依赖版本。

## 3. 官方项目结构

```text
official-artifacts/
├── README.md                 # 顶层说明；模拟器与真实集群实验入口
├── sia-sosp23.pdf            # SOSP 2023 论文
├── sia-simulator/            # 本次审计重点：离散事件模拟器
│   ├── multi_simulator.py    # 异构 Sia/Gavel 主入口
│   ├── cluster.py            # 调度轮、JobInfo、分配应用与日志
│   ├── job.py                # 作业状态、训练推进、重启/迁移开销
│   ├── applications.py       # profile/validation/scalability trace 读取
│   ├── goodput.py            # Goodput 模型
│   ├── speedup.py            # Goodput/基准 Goodput 与缓存
│   ├── sia.py                # 原始异构 Sia MIP 策略
│   ├── sia_fix.py            # 固定规模变体
│   ├── multi_simulator_unaware.py / sia_unaware.py / pollux_*.py
│   ├── gavel.py / shockwave.py
│   ├── simulator_config.py   # 集群、Sia、并行与日志默认值
│   ├── run_experiment.sh     # Saturn 上 Sia/Gavel/Pollux 示例命令
│   ├── workloads/            # saturn、philly、newtrace、scale 工作负载
│   ├── traces/               # 各模型 validation/placement/scalability profile
│   └── utils/print_run_stats.py
└── adaptdl/                  # 真实集群实现；本次不部署、不运行
```

仓库约 28.12 MiB；模拟器的 `workloads/` 与 `traces/` 共 266 个文件、22,403,738 字节，数据已随仓库提供，无额外大型下载。

## 4. 模拟器入口与命令行流程

### 4.1 主入口

异构 Sia 的主入口是 `sia-simulator/multi_simulator.py`：

- `multi_simulate(args)`：`multi_simulator.py:18-113`
- CLI 定义与 `__main__`：`multi_simulator.py:129-187`
- `--policy` 可选 `sia`、`gavel`、`sia_fix`：`multi_simulator.py:131-134`
- 调度间隔 `--interval`：`multi_simulator.py:152-154`
- 输出 `--output`：`multi_simulator.py:154`

异构无感知 Pollux 使用另一个入口 `multi_simulator_unaware.py`。真实集群不使用此模拟器入口，而使用顶层 README 所述 AdaptDL/Kubernetes 流程。

### 4.2 官方 Sia 命令

`run_experiment.sh:1-4` 给出的 Saturn Sia 参数是：

```bash
python multi_simulator.py workloads/saturn/ \
  --policy=sia \
  --policy_p_val=-0.5 \
  --mip_lambda_a=0.01 \
  --mip_lambda_n=1.1 \
  --interval=60 \
  --project_throughput \
  --share_max_replicas \
  --output=/tmp/sia-saturn
```

解析器定义的是复数 `--project_throughputs`（`multi_simulator.py:142-143`）。脚本中的单数形式是它的唯一长选项前缀，当前 argparse 默认允许缩写，所以可被接受；为避免依赖缩写，本次运行使用完整的 `--project_throughputs`。

完整流程：

1. 创建 Python 3.10 隔离环境并安装依赖。
2. 从 `sia-simulator/` 启动 `multi_simulator.py`。
3. 若参数是单个 CSV，运行一条 trace 并写一个 JSONL 日志；若参数是目录，枚举其中全部 `*.csv`，每条写 `${output}/<workload>.log`，最终写 `${output}/summary.json`（`multi_simulator.py:163-185`）。
4. 用 `utils/print_run_stats.py --workload_dir=... --output_dir=... --interval=...` 汇总（`sia-simulator/README.md:19-22`）。

输出位置完全由 `--output` 决定。官方脚本使用 `/tmp`；本次所有输出都放在工作区的 `research_runs/` 和 `research_notes/run_logs/`，位于官方仓库之外。

### 4.3 策略、轮次和随机种子

- 策略：`--policy=sia`。
- 公平性：`--policy_p_val`，默认 `-0.5`。
- 未分配惩罚：`--mip_lambda_n`，官方值 `1.1`。
- 分配变化参数：`--mip_lambda_a`，官方值 `0.01`；但见第 6 节，当前 `sia.py` 中该值未进入最终目标函数。
- 调度轮长度：`--interval`，Sia 官方值 60 秒。模拟器没有“固定轮数”参数，而是 `while not cluster.all_complete()` 运行到全部作业结束（`multi_simulator.py:64-67`）。
- 重复实验数：传目录时由目录中的 CSV 数量决定；Saturn 有 10 条 trace。
- 模拟器主入口没有 `--seed`。Sia MIP 路径未设置随机种子；预生成 workload 已固定。只有 workload 生成器提供 `--seed`（`utils/generate_workload.py:99-100`，默认 0）；Pollux 的遗传搜索另有 NumPy 随机调用。故原始 Sia trace 重放不能从 CLI 设置 seed。

## 5. 工作负载、profile 与集群配置

### 5.1 工作负载

主实验使用 `workloads/saturn/` 的 10 个 CSV，每个 160 个作业。每行字段是 `name,time,application,num_replicas,batch_size`，由 `multi_simulator.py:19` 读入，`cluster.py:50-64` 转为 `Job`。

本次正式基线的 `workload-1.csv` 含：

| 应用 | 作业数 |
|---|---:|
| cifar10 | 134 |
| deepspeech2 | 10 |
| yolov3 | 7 |
| bert | 6 |
| imagenet | 3 |
| 合计 | 160 |

`applications.py:25-69` 从各应用目录读取：

- `validation-*.csv`：batch size 对应的训练进度、指标和梯度统计；
- `placements-<gpu>.csv`：placement、local batch size、step/sync time；
- `scalability-<gpu>.csv`：跨节点/副本扩展 profile。

GPU 类型对应的应用集合在 `applications.py:341-392` 构造，包括 `aws`、`dgx`、`rtx`、`dgx-ext`、`azure`、`quad`；默认集群只选其中三类。

### 5.2 默认异构集群

`simulator_config.py:1-15`：

| 集群标签 | 含义 | 节点数 | 每节点 GPU | 总 GPU |
|---|---|---:|---:|---:|
| `aws` | AWS g4dn.12xlarge | 6 | 4 | 24 |
| `dgx-ext` | DGX-A100，扩展 profile | 2 | 8 | 16 |
| `rtx` | RTX 2080 Ti | 3 | 8 | 24 |
| **合计** |  | 11 |  | **64** |

配置集合在 `sia.py:13-17` 静态定义：4-GPU 节点使用 GPU 数 `1,2,4,8,...,64` 的合法组合；8-GPU 节点使用 `1,2,4,8,16,...,64`。`populate_valid_configs()`（`sia.py:100-142`）按集群节点数和 `cluster_max_physical_nnodes` 过滤，建立 `(GPU 类型, 节点数, GPU 数)` 与索引的双向映射，并建立每 GPU 类型的资源消耗矩阵。

本次默认集群得到 19 个候选配置：AWS 8 个、DGX-ext 5 个、RTX 6 个。

## 6. 一次完整调度轮的调用链

### 6.1 入口到状态读取

1. `multi_simulator.multi_simulate()` 读取 CSV（`multi_simulator.py:18-19`），实例化 `SiaPolicy`（`28-44`），调用 `populate_valid_configs()`（`45-46`），再创建 `Cluster`（`47-48`）。
2. `Cluster.__init__()` 为每行创建 `Job`（`cluster.py:46-69`），预计算跨 GPU 类型的 throughput ratio/throughput（`cluster.py:104-105`）。
3. 主循环每 60 秒调用 `Cluster.step()`，直到 `all_complete()`（`multi_simulator.py:64-67`）。
4. `Cluster.step()` **先**用上一轮分配推进作业状态，再求下一轮分配，最后写日志（`cluster.py:117-134`）。因此第一个调度周期在无分配状态下推进，首个到达作业随后才被调度。
5. `get_job_infos()` 只选择已到达且未完成的作业（`cluster.py:340-354`）。`get_sia_multi_job_info()` 附加每 GPU 类型的 SpeedupFunction、最大副本数、已用/浪费 GPU 秒、类别、重启/迁移数、年龄、throughput ratio、checkpoint 状态与 scale unit（`cluster.py:373-432`）。

### 6.2 Goodput 与归一化

1. `Job.get_speedup_fn()`（`job.py:240-252`）把在线拟合的 performance/gradient 参数包装为 `SpeedupFunction`。
2. `GoodputFunction.evaluate()`（`goodput.py:61-68`）计算 `throughput × efficiency`；效率中的统计增益见 `goodput.py:81-86`。`SpeedupFunction.get_goodput()` 以 1 节点/1 副本 Goodput 为基准，生成 speedup/Goodput（`speedup.py:20-80`）。
3. `SiaPolicy._compute_goodputs_v2()`（`sia.py:423-507`）为一个作业在一个 GPU 类型的合法配置计算 Goodput；若本类型模型缺失且启用 `--project_throughputs`，通过其他 GPU 类型的模型与 throughput ratio 投影。
4. `optimize_mip()` 为每个 adaptive 作业构造按 GPU 类型分块的矩阵（`sia.py:600-669`）。对每个作业取各类型所有有效配置中的最小非零 Goodput，再缩放整行，使最慢基准配置的值等于该作业的 scale unit（`sia.py:673-705`）。无模型的 GPU 类型把其最小合法配置填成 scale unit（`708-715`），极小值截为 `ZERO_ALLOC_GAIN=0.0001`（`717-719`）。
5. rigid 作业按 GPU 类型的固定副本 throughput 计算，并把最慢类型归一为 `num_replicas`（`sia.py:526-553`）。

### 6.3 公平性、目标与未分配惩罚

- 公平性变换：对归一化 Goodput/Speedup 矩阵逐元素取 `goodput^p`，四舍五入到两位并截断到 3000（`sia.py:907-931`）。默认 `p=-0.5`（`simulator_config.py:50-53`）。
- 基础目标：选择变量与 `goodput^p` 的加权和（所有 job weight 默认 1），见 `sia.py:607-609, 949-956`。
- 未分配惩罚：`lambda_n × (num_jobs - num_allocs)`（`sia.py:958-965`）。默认 `lambda_n=1.1`；当 `p<0` 时使用 `Minimize`，否则 `Maximize`（`sia.py:967-971`）。
- `ZERO_ALLOC_GAIN` 不是独立的“未分配配置列”；未分配由某作业整行选择变量全为 0 表示，并由 `lambda_n` 项处罚。

### 6.4 重分配/restart factor

- 每类应用的 restart penalty 在 `sia.py:23-35` 硬编码：DeepSpeech2 25 秒、ImageNet 250 秒、CIFAR 30 秒、BERT 120 秒、YOLOv3 45 秒、其他 30 秒。
- adaptive 作业默认使用 Pollux 形式：`max(age - num_restarts × penalty, 0) / (age + penalty)`，年龄小于 penalty 时强制为 1（`sia.py:721-740`）。rigid 作业使用相同思想（`sia.py:543-550`）。
- 若已有配置，所有替代配置的 Goodput 乘 restart factor，当前配置保持原值（rigid：`sia.py:873-883`；adaptive：`890-900`）。因此重分配成本作为候选 Goodput 的乘性阻尼进入目标。
- checkpoint/restore 尚未完成的作业被视为不可抢占，约束保持原分配（`sia.py:763-785, 1011-1028`）。
- 真实状态开销在 `Job.reallocate()`/`migrate()` 设置 `rescale_time` 并增加 restart 计数（`job.py:524-579`）；`Job.step()` 先消耗该延迟，把对应 GPU 秒计入 wasted service（`job.py:414-434`）。应用级实际延迟由 `applications.py:56-69` 定义，例如 CIFAR 50 秒、ImageNet 250 秒、BERT 120 秒、YOLOv3 80 秒。

**源码异常：** `lambda_a` 被解析、保存在 `SiaPolicy`，并在 `sia.py:936` 赋给局部变量 `opt_lambda_alloc_change`，但后续目标与约束没有引用该局部变量。对 `sia.py` 全文检索也没有其他有效引用。因此当前 `--mip_lambda_a=0.01` 对 `SiaPolicy` 的最终 MIP **没有直接作用**；重分配抑制来自上述 restart factor。后续做模块实验前必须决定是保持这一 artifact 行为，还是依据论文/其他分支恢复 `lambda_a` 项，二者不能混作同一基线。

### 6.5 约束、求解与分配应用

- 变量为整数矩阵并约束在 `[0,1]`（`sia.py:939-980`），等价于二进制选择。
- 每类 GPU 的资源使用通过 rigid 副本矩阵和 adaptive reduce matrix 累加，不得超过各类型总 GPU（`sia.py:982-1008`）。
- 每个作业最多选择一个配置；checkpoint 中作业必须保持原配置（`sia.py:1010-1028`）。
- `simulator_config.py:57-58` 虽设置 `sia_solver="glpk"`，但实际求解器在 `sia.py:1034-1036` **硬编码为 `cp.CBC`**；GLPK 调用只保留为注释（`1032-1033`）。
- 解被四舍五入并转为 GPU 类型/配置（`sia.py:1065-1120`），再由 `alloc_to_placement()` 生成节点 ID 列表（`sia.py:800-822`）。
- `Cluster.optimize_policy()` 把节点 ID 列表压缩为 placement，调用 `Job.migrate()` 或 `Job.reallocate()`，并保存新分配供下一轮使用（`cluster.py:160-221`）。
- `Job.step()` 根据 placement、batch size、profile throughput、梯度增益和 slowdown calibration 推进 progress/epoch，设置 completion time（`job.py:414-522`）。
- `Cluster.step_log()` 记录 timestamp、分配、placement、batch size、restart、提交/开始/完成时间等（`cluster.py:306-339`）；`output_logs()` 输出 JSONL（`cluster.py:513-517`）。

## 7. 指标定义与统计脚本

`utils/print_run_stats.py` 的真实实现：

- Makespan：对每条 workload，`max(submission_time + JCT) - min(submission_time)`（`86-98`）。
- 平均 JCT：把所有 workload 的全部 job JCT 合并后取平均；打印的 std 是“每条 workload 的平均 JCT”的标准差（`101-107`）。
- P99 JCT：排序后取 `int(0.99*N)` 下标，不是带插值的 NumPy percentile（`110-116`）。
- 重启：最终日志中 `num_restarts` 的整体与按应用均值（`137-155`）。
- contention：每个时间点 running+queued，先按 workload 求均值，再对 workload 均值取平均；max 取全局最大（`158-171`）。
- queue：类似 contention，同时汇总全部时间点的中位数（`175-191`）。
- GPU-hours：对每个尚未完成且有分配的作业，用 `len(gpu_list) × interval` 累加，再除 3600；整体 mean 是总 GPU-hours/作业数，按应用/GPU 类型也除以该应用作业数（`12-43, 195-227`）。这是按离散日志分配近似，未扣除 checkpoint 的浪费时间。
- FTF：仅当输出目录存在 `ftf_summary_true_share.json` 时计算 JCT/FTF-JCT 比率、unfair ratio 和分位数（`230-259`）。本次没有生成该文件。

注意：目录模式才自动写 `summary.json`。单 CSV 模式只写 JSONL，因此本次完整单 trace 用仓库外的 `research_notes/calc_single_trace_stats.py` 按官方脚本同一语义复算；smoke 目录模式则直接成功运行了官方 `print_run_stats.py`。

## 8. 执行记录

### 8.1 入口和求解器 smoke

```powershell
python multi_simulator.py --help
```

- 首次退出码 1：`ModuleNotFoundError: psutil`，确认未声明依赖。
- 在隔离环境安装 `psutil` 后退出码 0。

独立 CVXPY CBC 整数规划 smoke：状态 `optimal`，目标值 `1.0`。

### 8.2 第一次官方 trace 尝试（失败但有完整 Python 错误）

```powershell
python multi_simulator.py workloads/saturn/workload-1.csv \
  --policy=sia --policy_p_val=-0.5 \
  --mip_lambda_a=0.01 --mip_lambda_n=1.1 \
  --interval=60 --project_throughputs --share_max_replicas \
  --output=../../research_runs/sia_saturn_workload1/workload-1.log
```

- 环境：Pandas 2.3.3。
- 退出码：1。
- 实耗：24.290 秒。
- 阻塞点：模拟时间 2880 秒，`applications.py:282` 调用已删除的 `DataFrame.append`。
- 完整 stdout/stderr：`research_notes/run_logs/sia_saturn_workload1.stdout.log`。
- 没有伪造或补写结果。

核心 traceback：

```text
File "applications.py", line 282, in get_throughput
    df = df.append(self.scalability, ignore_index=True)
AttributeError: 'DataFrame' object has no attribute 'append'
```

### 8.3 依赖漂移与 Windows 原生错误

- 按当前最新 CVXPY 安装后，把 Pandas 降到 1.5.3 再运行，出现 `-1073741819 / 0xC0000005`。
- `PYTHONFAULTHANDLER=1` 定位到 CVXPY 自动探测 OSQP 时加载原生扩展发生 access violation。
- 默认目录模式在沙箱中先因 Windows multiprocessing 匿名管道权限失败；允许本地进程后，4 个 worker 初始化 OpenBLAS 又出现 `Memory allocation still failed` 和 `StackOverflowException`。
- 这些错误均未通过修改 Sia 源码规避；最终采用第 2 节所述隔离环境与单进程/单线程运行参数。

### 8.4 单作业 smoke（成功）

输入是官方 `workloads/saturn/workload-1.csv` 第一行的仓库外副本：

```text
cifar10-0,401.0,cifar10,12,4096
```

同一官方 Sia 参数的单文件运行：

- 退出码：0
- 实耗：6.393 秒
- JCT：533 秒
- 原始日志：`research_runs/sia_smoke_one_job/workload-smoke.log`
- stdout：`research_notes/run_logs/sia_smoke_one_job_compat.stdout.log`

目录模式把 `num_parallel_traces` 在进程内临时设为 1 后成功：

- 退出码：0
- 实耗：19.370 秒
- 输出：`research_runs/sia_smoke_dir_output/summary.json` 和 `workload-smoke.log`
- stdout：`research_notes/run_logs/sia_smoke_dir_singleprocess.stdout.log`

官方统计脚本成功，退出码 0、实耗 2.274 秒：

```text
Avg Makespan = 0.1480555556 h
Avg JCT = 0.1480555556 h
P99 JCT = 0.1480555556 h
Average restarts/job = 4.000
Contention mean/max = 0.5625 / 1
Queue mean/median/max = 0 / 0 / 0
GPU-hours mean/job = 1.1166666667
```

完整统计输出：`research_notes/run_logs/sia_smoke_stats_compat.stdout.log`。

### 8.5 完整官方 Saturn workload-1（成功基线）

命令与 8.2 相同，额外仅把 `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 设为 1，以限制 Windows CPU 线程池；没有改调度参数或源码。

- 输入：官方 `workloads/saturn/workload-1.csv`
- 作业数：160
- 退出码：0
- 实耗：488.068 秒
- 调度间隔：60 秒
- JSONL 记录数：755
- 最终模拟时间：45,300 秒；按首个提交到最后完成计算的 Makespan 为 44,859 秒
- 原始 JSONL：`research_runs/sia_saturn_workload1/workload-1.log`（27,689,067 字节）
- 完整 stdout/stderr：`research_notes/run_logs/sia_saturn_workload1_compat.stdout.log`（14,207,680 字节）
- 复算指标：`research_notes/run_logs/sia_saturn_workload1_stats.json`

原始指标：

| 指标 | 结果 |
|---|---:|
| 完成作业 | 160 / 160 |
| 平均 JCT | 2,091.0375 s = 0.58084375 h |
| P99 JCT（官方下标法） | 27,765 s = 7.7125 h |
| Makespan | 44,859 s = 12.46083333 h |
| 平均重启/作业 | 4.11875 |
| contention mean / max | 7.37218543 / 21 |
| queue mean / median / max | 0.00264901 / 0 / 1 |
| 平均 GPU-hours/作业 | 3.6115625 |

按应用的平均重启次数：

```text
bert 2.0
cifar10 3.4328358209
deepspeech2 7.9
imagenet 17.3333333333
yolov3 8.0
```

这是一条官方 trace 的原始基线，不应与 README 示例中的 Gavel 10-trace 结果或论文 Table 3 的全量平均直接比较。

## 9. Windows、Linux、GPU 与集群边界

普通 Windows 电脑可运行：

- `multi_simulator.py` 的单 CSV Sia 模拟；不需要物理 GPU，profile 已随仓库提供。
- 在依赖兼容、内存足够并降低并行度后，可运行目录模式。
- `utils/print_run_stats.py` 可跨平台运行。

需要 Bash/Linux 替代或命令翻译：

- `run_experiment.sh` 使用 `export`、`mkdir -p` 和 `/tmp`，Windows PowerShell 不能原样执行，但等价 Python 命令可以运行。
- `workloads/scale/README.md` 的 `grep/cut/sed/datamash` 运行时统计流水线是 Linux 工具链；Windows 需要 WSL 或等价脚本。
- 默认多进程 trace 模式在本机 Windows/OpenBLAS 组合不稳定；推荐 Linux/Conda+MKL。

必须使用 Linux/集群/GPU 或外部许可：

- 顶层 README 的真实集群实验：`make deploy`、Kubernetes、PVC、Docker registry、真实训练数据与 GPU 节点（`README.md:16-32`）。
- AdaptDL profile/placement 的真实采集需要对应 GPU 集群。
- Shockwave 当前需要 Gurobi 许可，官方 README 明确标为 disabled（`sia-simulator/README.md:6`）。

## 10. 后续模块组合实验的关键接口

在建立实验分支前，建议把以下点作为最小改动面：

1. **调度器接口**：`SiaPolicy.optimize(jobs, nodes, base_allocations)`（`sia.py:1140-1142`），输入是 `JobInfo`/`NodeInfo` 和旧分配，输出是 `{job: (cluster, node_id_tuple)}`。
2. **状态适配层**：`Cluster.get_sia_multi_job_info()`（`cluster.py:373-432`）。外部算法若需要 age、service、restart、GPU 类型性能等，应从这里显式传入，避免直接耦合 `Job`。
3. **候选配置**：`populate_valid_configs()` 与 `idx_to_config_map`/`adaptive_reduce_matrix`（`sia.py:100-142`）。改变 placement 搜索空间时必须保持 GPU 类型容量矩阵一致。
4. **性能模型**：`Job.get_speedup_fn()`、`GoodputFunction`、`_compute_goodputs_v2()`。模块组合时应区分原始 Goodput、按 1-GPU 标准化的 speedup、跨类型投影后 Goodput、以及 Sia 再归一化后的矩阵。
5. **目标与约束**：`__solve_mip_rescaled()`（`sia.py:844-1138`）。新的优先级、等待时间或公平性模块应明确插入在 Goodput 归一化前、`p` 次幂前、还是目标加权处。
6. **开销模型**：`get_restart_penalty()`、restart factor、`Job.reallocate()/migrate()/step()`。不要只改目标里的惩罚而遗漏真实模拟状态中的 checkpoint 延迟。
7. **调度轮与日志**：`Cluster.step()`/`optimize_policy()`/`step_log()`。所有实验分支应保留相同 `--interval`、同一 trace、同一 profile、同一指标脚本。
8. **输出契约**：保留 JSONL schema 与 `summary.json`，否则官方 `print_run_stats.py` 无法直接比较。

推荐后续在新 Git 分支或官方仓库的独立工作副本中实现模块；当前 `official-artifacts/` 应继续作为只读基线锚点。

## 11. 尚未确认及必须先决策的问题

1. **完整依赖锁文件缺失**：官方只固定 NumPy 大版本和 Pymoo；Pandas、SciPy、CVXPY、CyLP 的论文期准确版本待确认。建议先在 Linux/Conda 导出可复现 lock file。
2. **`lambda_a` 无效**：当前 artifact 的 `sia.py` 没有把 `opt_lambda_alloc_change` 加入目标。需对论文公式、作者其他 commit/真实集群 `mip` 实现进行交叉核对；在确认前不得把该参数当作有效控制量。
3. **solver 配置不一致**：配置写 GLPK，源码硬编码 CBC。应决定后续实验统一哪个求解器并记录版本、线程数、MIP gap。
4. **随机性接口缺失**：Sia CLI 无 seed；如果后续模块引入随机策略，必须新增统一 seed 契约，并保持原始 Sia 路径无随机或固定随机。
5. **全量 Table 3 未运行**：只完成 `workload-1.csv`；还需在推荐 Linux 环境跑完 10 条 Saturn trace，才能形成论文表格级平均值与方差。
6. **Windows 多进程不稳定**：默认 4-trace 并行失败；本报告的目录 smoke 使用单进程运行时覆盖。Linux 下默认并行行为待验证。
7. **FTF 基线缺失**：未生成 `ftf_summary_true_share.json`，因此没有 FTF unfair ratio。
8. **GPU-hours 的离散近似**：官方脚本按日志中的下一轮分配乘 interval 计费，且未扣 checkpoint 时间；后续若引入 Dally 等延迟机制，需要决定继续沿用官方指标还是增加精确 used/wasted GPU-hours，但两者应并列而非替换。
9. **profile 投影误差**：`--project_throughputs` 对缺失 GPU 类型模型按均值比例投影；模块组合实验需要记录哪些 job/type 是实测、哪些是投影。

## 12. 基线判定

**基线状态：成功（限定为一个完整官方 Saturn trace）。**

- 仓库来源、提交和洁净状态已确认。
- 模拟器入口、数据、配置、Goodput、MIP、状态推进与指标链路已从源码核对。
- 单作业 smoke、目录统计 smoke、完整 160 作业官方 trace 均有真实成功产物。
- 没有修改 Sia 调度源码或引入 Tiresias、Gavel、Dally、Cuckoo 等新模块。
- 尚不能声明“复现论文 Table 3”，因为没有在官方推荐 Linux/Conda 环境跑完 10 条 trace，且官方依赖未完整锁定。
