# Sia Simulator 中文注释版

从 [入口 multi_simulator.py](multi_simulator.py) 开始，沿着 [Cluster](cluster.py)、[Job](job.py) 和 [SiaPolicy](sia.py) 看一次调度。每个函数上方的「导读」给出职责、输入、调用和状态变化，较长函数内部按步骤解释关键变量和分支。用 [函数索引](函数索引.md) 查询注释版行号。

| 想追踪什么 | 源码 |
|---|---|
| 命令行参数、作业 CSV、如何开始或结束模拟 | [multi_simulator.py](multi_simulator.py)、[simulator_config.py](simulator_config.py) |
| 怎样推进一轮，怎样把状态交给策略 | [cluster.py](cluster.py)、[utils.py](utils.py) |
| 单个作业的时钟、GPU、batch、恢复开销和进度 | [job.py](job.py) |
| 计时与验证资料怎样进入执行和预测 | [applications.py](applications.py)、[goodput.py](goodput.py)、[speedup.py](speedup.py) |
| 怎样枚举、打分、求解和放到节点 | [sia.py](sia.py) |
| 对照策略、其它入口与工具 | [函数索引](函数索引.md) |

一次异构 Sia 运行从入口读取 CSV 和集群配置开始。入口创建策略与 Cluster，Cluster 创建每个 Job。每次 Cluster.step 先用已有分配推进 Job，再通过 JobInfo 与 NodeInfo 把当前状态交给 SiaPolicy，应用它返回的新分配并记日志。Job 使用 Application 的记录计算模拟执行进度，期间取得的观测会进入性能模型，供后续调度查询 Goodput。SiaPolicy 求解配置后再安排具体节点。

## 版本与核对

本目录覆盖 **40 个 Python 文件、373 个函数和方法**，共 **2220 行导读**。核心文件还对关键语句逐处说明。来源是官方仓库当前工作树，Git 基础提交为 aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70，包含工作树中 cluster.py 已有的排版改动。

每个副本去掉带「导读」的新增行后，与对应原文件逐字节一致。另逐个比较 Python 语法树并进行编译检查。行号随注释插入而变化，函数注释中保留原源码定义行号。此目录按原结构保留 Python 源码供对照阅读，运行时使用 [原版模拟器及其数据](../../official-artifacts/sia-simulator/README.md)。
