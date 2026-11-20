# Sia + Synergy Pilot 报告

- 日期：2026-10-23（Asia/Shanghai）
- 判定：**STOP / INVALID FOR SIA PERFORMANCE CLAIMS**
- 实际完成范围：官方 Synergy 原生 simulator 的最小参考复现、代码级回归测试、Sia 画像/容量/归一化可行性审计
- 未执行范围：Sia S1/S2/S3 实现、20-job Sia smoke、三条 160-job Saturn trace、参数搜索

## 0. 基线保护

- `official-artifacts/`：HEAD `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`，`main...origin/main`，工作区干净。
- 独立 worktree：`sia-synergy-experiments/`，分支 `exp/sia-synergy`，从上述 commit 创建且最终无代码 diff。
- LAS、AFS 和 Lazer worktree 的最终 Git 状态与本任务开始时的既有修改清单一致；本任务没有写入这些目录，也没有覆盖其日志或报告。
- 审计时主机为 8 physical / 16 logical CPU、13.69 GiB RAM（当时约 2.30 GiB 可用）；工作区卷约 29.28 GiB 可用。这里只用于运行安全判断，未作为模拟集群配置。
- `.venv-sia-baseline` 未安装或升级任何包；Synergy 使用单独的 `.venv-synergy`。

## 1. 为什么在 Sia pilot 前停止

任务要求在画像、节点容量或 Goodput 一致性不足时停止，不得用 synthetic 假设补齐。三项硬门槛均失败：

1. 三条预注册 Saturn trace 共 480 个作业，严格同应用、同数据集/流水线、同 GPU 类型和可映射配置的 A 级 Synergy 画像覆盖为 **0/480**。仅 ImageNet/ResNet50 的 13/480 作业可算 B 级模型/数据集候选，但 Synergy 是 V100 画像，Sia 是 T4/A100/RTX 2080 Ti，不能直接作为数值曲线。
2. Sia 的 RTX 节点没有 CPU/RAM 主机规格；DGX-A100 存在 1 TB/2 TB 配置差异；CPU physical core 与云端 vCPU 也不能静默混用。因此逐节点 CPU/RAM 容量约束无法完整建立。
3. Sia Goodput 已含原 throughput profile 所在数据流水线和主机条件；这些 profile 的 CPU/RAM/cache 基准点未记录。直接再乘 Synergy 的 V100 speedup 会跨平台并可能重复计算收益。

因此没有创建 `--policy=sia_synergy`，实验 worktree 保持原始 commit。这里的“INVALID”指当前数据不足以支持 Sia+Synergy 性能结论，不是对 Synergy 论文机制本身的否定。

## 2. 原生参考复现环境

- 官方源码：`research_sources/Synergy/`，commit `1bb2bed6edc9c5f953d0b7f0a40e4fdad053489c`
- Python：`.venv-synergy/Scripts/python.exe`，Python 3.10.10
- 数值库：单线程运行，`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`
- 模拟集群：16 台同构服务器，共 128 GPU、384 CPU、8000 GB RAM；每台 8 GPU、24 CPU、500 GB RAM
- 工作负载：固定随机种子 42 的 5-job synthetic trace；两次运行使用完全相同的 job 序列

官方源码以 Linux/旧版 Python 环境为主。Windows 参考 wrapper 只增加 `collections.Mapping`、NUMA 和 deployment RPC 的 import compatibility shim；后两条路径在 CPU-only simulator 中未执行。没有修改官方 Synergy 文件。

## 3. 正确性测试

命令：

```powershell
D:\科研启航\Sia\.venv-synergy\Scripts\python.exe -m unittest -v test_synergy_reference.py
```

从 `research_notes/synergy/` 执行，退出码 0，墙钟 0.706 秒，4/4 通过：

- 默认集群容量与官方配置一致；
- active ModelZoo 与源码一致；
- CPU×RAM profile 矩阵为 9×5；
- TUNE 在理想资源不可行时能够回退到 GPU-proportional share。

## 4. 原生 fair 与 TUNE 对照

命令：

```powershell
# GPU-proportional baseline
D:\科研启航\Sia\.venv-synergy\Scripts\python.exe research_notes\synergy\native_synergy_smoke.py --mode fair

# Synergy-TUNE
D:\科研启航\Sia\.venv-synergy\Scripts\python.exe research_notes\synergy\native_synergy_smoke.py --mode tune
```

两次退出码均为 0，分别耗时 4.298 秒和 4.238 秒，均完成 5/5 作业。

| 指标 | GPU-proportional | Synergy-TUNE | 相对变化 |
|---|---:|---:|---:|
| 平均 JCT (s) | 20622.2 | 19362.2 | -6.11% |
| 中位 JCT (s) | 5291 | 3501 | -33.83% |
| P95 JCT (s) | 61951.0 | 61951.0 | 0.00% |
| P99 JCT (s) | 71221.4 | 71221.4 | 0.00% |
| Makespan (s) | 74400 | 74400 | 0.00% |

P95/P99 使用线性百分位，仅有 5 个作业，不具有统计意义。该微型实验只用于确认官方 TUNE 路径可以执行，并非 Sia/Saturn 性能证据。

### 4.1 任务级结果

| Job | 模型 | GPU | profile CPU/RAM | GPU-proportional CPU/RAM | profile speedup | fair JCT (s) | TUNE JCT (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | alexnet | 1 | 12 / 125 GB | 3 / 62.5 GB | 2.94 | 5291 | 1991 |
| 1 | gnmt | 1 | 1 / 20 GB | 3 / 62.5 GB | 1.00 | 73539 | 73539 |
| 2 | transformer | 1 | 1 / 20 GB | 3 / 62.5 GB | 1.00 | 15599 | 15599 |
| 3 | m5 | 4 | 24 / 500 GB | 12 / 250 GB | 2.50 | 5181 | 2181 |
| 4 | lstm | 8 | 8 / 160 GB | 24 / 500 GB | 1.00 | 3501 | 3501 |

TUNE 的均值改善完全来自官方画像中 speedup 大于 1 的 AlexNet 和 M5；GNMT、Transformer、LSTM 不变，最长的 GNMT 决定 makespan，因此 makespan 没有改善。这符合源码使用画像倍率推进 iteration 的行为。

## 5. 与预定 S1/S2/S3 的关系

| 阶段 | 状态 | 结论 |
|---|---|---|
| S1：Sia + GPU-proportional CPU/RAM | 未实现 | 三类节点完整容量与资源单位未闭合，无法做真实约束和退化验证。 |
| S2：Resource-Sensitive Allocation | 禁止进入 | 0/480 A 级画像覆盖，且缺少与 Sia Goodput 同基准的归一化点。 |
| S3：Synergy-TUNE placement | 禁止进入 | S2 demand 不成立；强行运行只会把虚构需求装箱。 |

所以没有 Sia 的 CPU-core-hours、memory-GB-hours、GPU 类型分配、重启次数或 Saturn JCT 指标；把原生 synthetic 结果填入这些栏位会构成错误比较。

## 6. 产物

- 论文/源码审计：`research_notes/synergy/SYNERGY_PAPER_CODE_AUDIT.md`
- 画像覆盖审计：`research_notes/synergy/SIA_SYNERGY_PROFILE_COVERAGE.md`
- S1/S2/S3 设计与门槛：`research_notes/synergy/SIA_SYNERGY_MODEL_DESIGN.md`
- 本报告：`research_notes/synergy/SIA_SYNERGY_PILOT_REPORT.md`
- 原生运行 wrapper：`research_notes/synergy/native_synergy_smoke.py`
- 参考测试：`research_notes/synergy/test_synergy_reference.py`
- TUNE 元数据：`research_runs/sia_synergy/native_reference/native_synergy_smoke.run.json`
- fair 元数据：`research_runs/sia_synergy/native_reference/native_synergy_fair_smoke.run.json`
- 汇总：`research_runs/sia_synergy/native_reference/native_synergy_comparison.json`
- 测试元数据：`research_runs/sia_synergy/native_reference/native_synergy_unit_tests.run.json`

此外新建隔离环境 `.venv-synergy/`，并在 `research_sources/Synergy/` 保存官方源码 checkout 与论文 `osdi22-mohan.pdf`。Sia Synergy worktree 没有修改文件，因此不存在 Sia 策略 diff。

## 7. 是否值得继续

当前不值得继续实现或运行 Sia+Synergy。只有同时补齐以下证据后才应恢复：

1. Sia 三类节点的真实、同口径 CPU/RAM 容量与资源分配粒度；
2. 五个 Saturn 应用在 T4/A100/RTX、各合法 Sia 配置上的 CPU×RAM response surface；
3. 生成现有 Sia throughput profile 时的 CPU/RAM/cache 基准点；
4. 能证明 `effective_goodput = sia_goodput × relative_auxiliary_multiplier` 只计算一次收益的校准测试。

在此之前，任何 S2/S3 结果最多只能标作另一个 synthetic 算法实验，不能与官方 Saturn 基线混算。本轮按停止条件结束，不启动参数搜索或完整 trace。
