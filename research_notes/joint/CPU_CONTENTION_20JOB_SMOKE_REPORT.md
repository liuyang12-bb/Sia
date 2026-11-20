# Sia + Synergy 20-job 稳定性 smoke 报告

## 1. 结论先行

首次尝试因 0.804 GiB 可用内存触发 2 GiB 门槛，证据保存在 `cpu_contention_20job/`。释放内存后的第二次执行使用新目录 `cpu_contention_20job_run1/`，预检内存为 3.069 GiB，B1、SY、B4、B5 均完成 20/20，但 B3 在运行中触发原子提交保护：

```text
RuntimeError: refusing non-atomic joint allocation: {}
```

因此本轮最终状态是 **`FAIL_B3_EXECUTION`**，五策略矩阵只完成 **4/5**。这是已观测到的 20-job 稳定性失败，不能做 B3 性能比较，也不能声称已具备正式评估条件。

四个已完成策略均通过以下审计：20/20 完成、无 GPU/CPU/RAM 超配、Neutral `rho=1`、Goodput 倍率只施加一次、checkpoint/restore 不错误推进训练、B4/B5 GPU config 冻结。运行期也真实观察到排队和 CPU 饱和但 GPU 空闲。30 项既有单元与集成测试仍全部通过，原始 Sia 回归 hash 未改变。

> **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 2. 工程保护与版本

- 工作区：`D:\科研启航\Sia`
- 实验仓库：`sia-joint-experiments`
- 分支：`exp/synergy-baselines-cpu-bottleneck`
- HEAD：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 官方仓库：`official-artifacts`，`main...origin/main`，检查时干净。
- 实验仓库原本已有未提交的联合调度代码和产物；本轮未覆盖或清理这些内容。
- 20-job 输出使用独立目录 `research_runs/sia_synergy/cpu_contention_20job/`。运行器现在拒绝复用非空输出目录，后续正式重试必须使用新目录，从而保留本次阻塞证据。

四作业基线关键 hash 未被本轮覆盖：

| 文件 | SHA-256 |
|---|---|
| `cpu_contention_4job_config.json` | `2F4D6C1AE765DD68DB1CAFAF0223CFFAD852F5094B9EBE30054A3325DE0D0CE9` |
| `cpu_contention_4job_workload.csv` | `43EE4622CF80096847BDF4FAF9A032EE9BB56A03FA070214473E176C4017A07B` |
| `run_cpu_contention_4job.py` | `2B275ABB586396BBEE82B78F1704DA24015E0DFC023F996B62580C8DCFF26A63` |
| `validate_cpu_contention_4job.py` | `01F69EFBDCABB72990799A4FD40C93711297A313B601056E9C044B178A88E140` |
| 四作业 `run_manifest.json` | `ACD7E3BAC0E925FF69A2CCE15B2F82A9C2AAA95A53D2E43347EB47EE2CDEDDFE` |

## 3. 冻结的 20-job 输入

### 3.1 选择规则

在任何 20-job 策略运行前预注册如下唯一规则：

> 从官方 `workloads/saturn/workload-1.csv` 选择数值 job ID 为 20–39 的连续 20 条记录。

该区间自然包含既有四作业 cohort 28–31。没有根据调度器结果、JCT 或模板响应更换样本。提取文件与源数据按顺序逐单元格相等；源文件 SHA-256 为 `D80DC93782785AAB449D3109AA118F123EA799D891DF0AFE2B9DEE483EDA9468`。

| Job | Arrival (s) | Application | GPU demand | Batch |
|---|---:|---|---:|---:|
| cifar10-20 | 3535 | cifar10 | 10 | 4096 |
| deepspeech2-21 | 3598 | deepspeech2 | 8 | 320 |
| cifar10-22 | 3685 | cifar10 | 12 | 4096 |
| imagenet-23 | 4306 | imagenet | 10 | 1600 |
| cifar10-24 | 4465 | cifar10 | 4 | 2048 |
| cifar10-25 | 4783 | cifar10 | 8 | 2048 |
| cifar10-26 | 4788 | cifar10 | 10 | 4096 |
| cifar10-27 | 4862 | cifar10 | 12 | 4096 |
| bert-28 | 4931 | bert | 12 | 384 |
| cifar10-29 | 4935 | cifar10 | 6 | 2048 |
| deepspeech2-30 | 5017 | deepspeech2 | 12 | 640 |
| deepspeech2-31 | 5291 | deepspeech2 | 6 | 320 |
| cifar10-32 | 5379 | cifar10 | 6 | 2048 |
| yolov3-33 | 5717 | yolov3 | 4 | 64 |
| cifar10-34 | 5896 | cifar10 | 4 | 2048 |
| cifar10-35 | 7191 | cifar10 | 10 | 4096 |
| cifar10-36 | 7233 | cifar10 | 8 | 2048 |
| bert-37 | 7263 | bert | 16 | 384 |
| cifar10-38 | 7289 | cifar10 | 10 | 4096 |
| cifar10-39 | 7465 | cifar10 | 12 | 4096 |

到达跨度为 3930 秒；应用组成是 CIFAR10×13、DeepSpeech2×3、BERT×2、ImageNet×1、YOLOv3×1。输入有两个明显到达段，但因正式模拟未启动，本报告不把“会形成多少实际重叠”当作已验证事实。

### 3.2 性能模板映射

映射按 application 固定，而非逐 job 挑选：

| Saturn application | Synergy matrix template |
|---|---|
| cifar10 | `alexnet_1.py` |
| deepspeech2 | `deepspeech_1.py` |
| imagenet | `res50_1.py` |
| bert | `bert_1.py` |
| yolov3 | `mobilenet_1.py` |

这些映射全部是 **synthetic cross-hardware transfer**。只有模板矩阵单元来自 Synergy 官方源码；application 对应关系、固定参考点、插值、`phi` 缩放及其与 Sia Goodput 的乘法都不是 Saturn 实测。

### 3.3 SY 的固定 GPU 适配

规则只读取 workload GPU demand 和数值 job ID，不读取 CPU/RAM 响应或 JCT：

- demand ≤ 6：RTX；
- demand = 8：DGX-ext；
- demand ≥ 10：偶数 ID 使用 AWS，奇数 ID 使用 DGX-ext；
- 随后把 workload demand 截到该类型不超过 demand 的最大合法 Sia 配置。

冻结后的类型数为 AWS×6、DGX-ext×8、RTX×6。该规则属于本项目异构适配，不是官方 Synergy 的 GPU 类型选择算法。

## 4. 冻结资源与响应场景

| Node | GPU | CPU cores | RAM (GB) | 性质 |
|---|---:|---:|---:|---|
| aws:0 | 4 | 4 | 500 | synthetic |
| dgx-ext:0 | 8 | 8 | 1000 | synthetic |
| rtx:0 | 8 | 6 | 1000 | synthetic |

最小合法辅助资源为每 GPU 1 CPU core、20.83 GB RAM。因此 RTX 节点虽然暴露 8 GPU，但最小 CPU floor 至多允许同时使用 6 GPU，静态上至少有 2 GPU 会在“节点 GPU 全满”的假设下被 CPU 约束排除。该静态事实不能替代运行时的“CPU 饱和且 GPU 空闲”证据；后者本轮未运行、未验证。

首轮正式矩阵在运行前冻结为：

- capacity：`cpu_constrained`；
- response：`neutral`，指数 0，`rho` 必须恒等于 1；
- policies：B1、SY、B4、B5、B3；
- 顺序运行，不并行；
- interval：60 秒；max rounds：5000；
- 可用内存门槛：2 GiB。

`medium`（指数 0.5）只登记为后续可选敏感性档。在看到任何 20-job 结果前，因为初始内存余量已经很窄，决定不在首轮运行 Medium；这不是按 JCT 结果选择档位。

冻结输入 hash：

| 对象 | SHA-256 |
|---|---|
| 20-job config | `2A4E3D7087D0BC66E3EB6987A6E25AC2E04C80B9D8CEF868399F5AAF39E53C11` |
| 20-job extracted workload | `46FAB9A9743719B56437E6FC35DADAA1CF397205DCF06483E6A12E7CA488627A` |
| alexnet template | `5BE97BBBABDA711E43903C29EB914C4B45DEC3B7997E3EA60F7ACEAB99943635` |
| bert template | `BB3B0DA2CF646A4EF72D8AA4F23D01D1BE23CA21243FC625576FE51BECF44CE2` |
| deepspeech template | `CB64753CAB8AA32325588B3E56F0D677379BBD0423678D5404B03295CA381443` |
| mobilenet template | `8F6DA649F3A05309A90DF11C8047665BE5CCB53E2DCBDD2D26F990FA4E4A8B7B` |
| res50 template | `699ADE948408057199B50E6F34B98D71A37BC16638E12FE2EA144DF11D53FD4D` |

## 5. 新增实现

### 5.1 文件

- `sia-joint-experiments/sia-simulator/joint_prototype/cpu_contention_20job_workload.csv`
- `sia-joint-experiments/sia-simulator/joint_prototype/cpu_contention_20job_config.json`
- `sia-joint-experiments/sia-simulator/run_cpu_contention_20job.py`
- `sia-joint-experiments/sia-simulator/validate_cpu_contention_20job.py`
- `research_runs/sia_synergy/cpu_contention_20job/` 下的预检、阻塞 manifest、测试与验证摘要

没有修改官方 Sia/Synergy 仓库，也没有修改 B1/SY/B4/B5/B3 的算法实现。

### 5.2 运行器保护

`run_cpu_contention_20job.py`：

1. 逐行核对冻结 CSV 是官方 workload 的精确有序提取；
2. 检查 20–39 连续 ID、完整模板映射和完整 SY 映射；
3. 检查请求矩阵与预注册矩阵完全一致；
4. 检查所有模板文件并记录 hash；
5. 在总运行前及每个策略前检查 2 GiB 门槛；
6. 五策略严格顺序运行，每个策略结束后写进度 manifest；
7. 拒绝覆盖任何非空输出目录；
8. 每轮保留原始执行日志，并追加资源利用、等待原因、CPU 饱和/GPU 空闲及 Goodput 单次倍率证据。

### 5.3 验证器覆盖

如果五策略完整运行，`validate_cpu_contention_20job.py` 会检查：

- 20/20 完成、无 round limit；
- 逐轮及时间加权 GPU/CPU/RAM 容量；
- Neutral `rho == 1`；
- `effective_goodput == raw_goodput * rho`，且倍率只施加一次；
- B1 admitted config 等于 Sia intent；
- B4/B5 不改变冻结的 Sia GPU config；
- SY 走 `SYNERGY_TUNE` adapted core；
- B3 为作业枚举多个 GPU 候选；
- checkpoint/restore 时间先于训练时间，完整延迟轮不得推进 progress；
- progress 单调；
- 实际排队、CPU 饱和且 GPU 空闲；
- B3 与 B1 Sia intent 的同时间轴配置差异（并明确状态可能已在先前轮次分叉）；
- 求解状态、fallback、TUNE action、资源小时、restart/migration/wasted GPU-seconds。

阻塞时验证器不会把这些检查判成成功，而是写成 `null`。

## 6. 首次执行与内存停止点（历史保留）

### 6.1 测试

执行：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe -m unittest discover -s tests -p test_*.py -v
```

结果：**30 tests，30 passed，0 failure，0 error，15.902 s**。PMP 测试仍出现既有 autograd `invalid value encountered in log` RuntimeWarning，但测试通过。

### 6.2 20-job 命令

执行：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe run_cpu_contention_20job.py `
  --config joint_prototype\cpu_contention_20job_config.json `
  --workload joint_prototype\cpu_contention_20job_workload.csv `
  --synergy-models ..\..\research_sources\Synergy\simulator\models `
  --output ..\..\research_runs\sia_synergy\cpu_contention_20job `
  --capacities cpu_constrained --responses neutral `
  --policies b1,sy,b4,b5,b3
```

退出码：1。明确错误：

```text
RuntimeError: available memory 0.804 GiB is below frozen 2.000 GiB gate
```

运行器在构造任何 `JointCluster` 或启动任何策略前退出。因此没有部分调度解、没有被推进的训练状态，也没有输出任何策略结果 JSON。

### 6.3 当前结果表

| Policy | 运行状态 | Completed | Avg JCT | 运行期容量审计 | TUNE/solver 事件 |
|---|---|---:|---:|---|---|
| B1（带辅助资源 admission 的 Sia 基线） | NOT RUN | — | — | — | — |
| SY（Synergy-TUNE 适配版） | NOT RUN | — | — | — | — |
| B4 | NOT RUN | — | — | — | — |
| B5 | NOT RUN | — | — | — | — |
| B3 | NOT RUN | — | — | — | — |

所以以下问题均 **待运行验证**：死锁/求解失败、真实排队、CPU/RAM 不足、CPU 饱和且 GPU 空闲、B3 改变 GPU 配置、B1/B4/B5 admission/fallback、checkpoint/restore、TUNE current-to-fair/final-wait/peer downgrade、Neutral 单次倍率及逐节点容量。

## 7. 原始 Sia 回归

未重跑或覆盖历史回归日志；验证器核对已有两份日志：

- official：`F81E0B22C76734E5264742A7231687F366CE77B4D3448155E444A64D42E02DFF`
- experimental：`F81E0B22C76734E5264742A7231687F366CE77B4D3448155E444A64D42E02DFF`

两者逐字节相同。该证据只说明已有原始 Sia CLI 回归仍被保留，不代替 20-job 联合模式验证。

## 8. 机器可读产物

- `preflight.json`：冻结 cohort、内存、拓扑、hash 与静态 CPU floor 审计；
- `run_manifest.json`：`ABORTED_MEMORY_GATE`，完成运行 0/5；
- `validation_summary.json`：`BLOCKED_MEMORY_GATE`，运行期检查为 `null`；
- `comparison_summary.json`：空结果表，明确禁止作性能推断；
- `test_summary.json`：30 项测试摘要。

首次内存阻塞目录：`research_runs/sia_synergy/cpu_contention_20job/`。

第二次执行目录：`research_runs/sia_synergy/cpu_contention_20job_run1/`，包含四个完整策略 JSON、`failure_event.json`、`comparison_summary.json` 和 `validation_summary.json`。两个目录均保留，未覆盖。

## 9. 是否具备进入下一阶段的条件

**不具备。** 内存阻塞已经解除，但 B3 在 20-job 状态空间中发生未审计结果并被原子保护拒绝。必须先补齐失败分支的状态、轮次与候选数持久化，区分 `NO_COLUMNS` 和 `milp result.x is None`，然后设计不破坏 checkpoint 锁与原子提交语义的修复，并在新目录重跑相同冻结矩阵。

在五策略 Neutral 都完成且验证器全部通过之前，不运行 Medium、正式 20-job 性能评估或 160-job 实验。

## 10. 释放内存后的 `run1` 实际结果

### 10.1 执行状态

第二次执行仍使用相同 config/workload/profile hash，未更改 cohort、资源容量、响应参数、映射或调度目标。内存采样均高于 2 GiB：预检 3.069 GiB；B1/SY/B4/B5 开始前分别为 3.222/3.258/3.948/2.852 GiB。

| Policy | 状态 | 完成 | 平均 JCT (s) | 总等待 (s) | 轮数 | 墙钟 (s) |
|---|---|---:|---:|---:|---:|---:|
| B1（带辅助资源 admission 的 Sia 基线） | SUCCESS | 20/20 | 5102.95 | 30000 | 578 | 97.56 |
| SY（Synergy-TUNE 适配版） | SUCCESS | 20/20 | 6783.25 | 57180 | 564 | 66.09 |
| B4 | SUCCESS | 20/20 | 5083.75 | 29460 | 575 | 111.06 |
| B5 | SUCCESS | 20/20 | 5053.30 | 29460 | 590 | 100.55 |
| B3 | **FAILED** | 未形成结果文件 | — | — | — | — |

这些数值只用于稳定性与机制诊断。矩阵不完整，且全部为 Neutral 合成跨硬件资源模型，不能将前三个 JCT 差异解释成正式性能优劣。

### 10.2 逐作业 JCT

| Job | B1 | SY adapted | B4 | B5 |
|---|---:|---:|---:|---:|
| cifar10-20 | 482 | 6386 | 482 | 482 |
| deepspeech2-21 | 6033 | 3299 | 5632 | 5443 |
| cifar10-22 | 515 | 8346 | 515 | 515 |
| imagenet-23 | 28841 | 29312 | 28914 | 28995 |
| cifar10-24 | 929 | 1081 | 929 | 929 |
| cifar10-25 | 1511 | 326 | 1511 | 1511 |
| cifar10-26 | 904 | 9343 | 904 | 904 |
| cifar10-27 | 1498 | 607 | 1498 | 1498 |
| bert-28 | 1810 | 4274 | 1810 | 1810 |
| cifar10-29 | 1435 | 1691 | 1435 | 1435 |
| deepspeech2-30 | 4983 | 17661 | 5188 | 5567 |
| deepspeech2-31 | 29366 | 7649 | 29157 | 30056 |
| cifar10-32 | 1169 | 2327 | 1169 | 1169 |
| yolov3-33 | 13925 | 28081 | 13925 | 13925 |
| cifar10-34 | 975 | 2890 | 975 | 975 |
| cifar10-35 | 2184 | 788 | 1247 | 1217 |
| cifar10-36 | 2142 | 1056 | 1205 | 860 |
| bert-37 | 1437 | 422 | 2659 | 1567 |
| cifar10-38 | 826 | 8942 | 778 | 778 |
| cifar10-39 | 1094 | 1184 | 1742 | 1430 |

### 10.3 资源竞争与 TUNE 事件

| Policy | CPU 饱和且 GPU 空闲事件 | CPU-idle-GPU 等待轮 | Restart | Wasted GPU-s |
|---|---:|---:|---:|---:|
| B1 | 23 | 45 | 57 | 13775 |
| SY adapted | 0 | 0 | 4 | 10560 |
| B4 | 34 | 49 | 68 | 14735 |
| B5 | 15 | 25 | 62 | 14095 |

峰值审计中，AWS/DGX-ext 分别达到 4/8 GPU 与 4/8 CPU；B1/B4/B5 的 RTX 峰值为 6 GPU、6 CPU，验证了 8 GPU RTX 节点受 6 CPU floor 限制。四策略都存在实际等待；SY 的 953 个等待轮被分类为 placement/config fragmentation，而不是“CPU 饱和且 GPU 空闲”。

TUNE 事件：

- SY adapted：`ALLOCATE_PROFILE_DEMAND=74`、`DOWNGRADE_CURRENT_TO_FAIR=1236`、`WAIT_FINAL_RETRY_INFEASIBLE=953`；
- B5：`ALLOCATE_PROFILE_DEMAND=62`、`ATOMIC_PEER_DOWNGRADE_TO_FAIR=50`、`DOWNGRADE_CURRENT_TO_FAIR=1082`、`WAIT_FINAL_RETRY_INFEASIBLE=479`。

B4 的求解状态为 `OPTIMAL=515`、`FALLBACK_B1_PROPORTIONAL=2`。现有 summary 的 `fallback_events=475` 是按 audit 中存在 `fallback_reason` 的轮次累计，与真正的 fallback status 次数不是同一口径；在正式实验前必须澄清/重命名该指标。

### 10.4 B3 失败审计

失败调用链：

```text
run_cpu_contention_20job.py -> run_one -> JointCluster.step
-> Cluster.optimize_policy -> JointSiaPolicy.optimize
-> atomic audit guard
```

`JointSiaPolicy.optimize()` 在提交资源前执行：

```python
if not result.audit.get("valid", False):
    raise RuntimeError(f"refusing non-atomic joint allocation: {result.audit}")
```

本次 `result.audit == {}`，保护逻辑正确拒绝提交，因此没有超配或部分 B3 解被写入。但 `JointScheduler.solve()` 有两个会返回默认空 audit 的分支：候选列为空时的 `NO_COLUMNS`，以及 SciPy MILP 没有 `result.x` 时的失败分支。异常发生在 `_commit_metadata()` 之前，原运行器又只在一个策略完成后更新 manifest，所以失败轮次、result status、候选列数量和当轮 checkpoint 锁状态没有持久化。现有证据无法在两条分支之间作可靠判定，标记为 **待验证**。

这同时暴露两个问题：

1. B3 对无候选/无解状态没有形成带审计信息的原子全量回退或明确等待结果；
2. 运行器在策略异常时缺少 per-round failure snapshot。

不得简单把空 audit 改成 `valid=true`：若无列由 checkpoint 锁与当前配置不可行共同造成，直接提交空分配可能破坏原始配置锁。修复前必须先记录失败状态，再决定是“合法全等待”“保留锁定配置”还是“求解失败且整轮保持上一分配”。

### 10.5 验证结论

`validation_summary.json` 状态为 `FAIL_B3_EXECUTION`。已通过：

- 冻结输入有效、内存门槛满足；
- 四个已完成策略均 20/20；
- 四策略无容量违规；
- Neutral `rho` 严格为 1；
- Goodput 倍率只应用一次；
- B4/B5 GPU config 冻结；
- SY 使用 adapted TUNE core；
- checkpoint/restore 语义；
- 实际排队与 CPU 饱和/GPU 空闲事件。

未通过/未验证：五策略矩阵完整性、B3 完成性、B3 与 B1/SY/B4/B5 的任何比较，以及 B3 是否在受限状态选择不同 GPU 配置。

